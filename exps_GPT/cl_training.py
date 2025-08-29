# finetune_cnn_dm_gpt.py
# pip install -U "transformers>=4.42.0" "datasets>=2.19.0" "accelerate>=0.30.0" "evaluate>=0.4.2" sentencepiece
# Optional (for ROUGE): pip install rouge-score

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np

from datasets import load_dataset
import evaluate
from datasets import load_dataset, DatasetDict
import time
import re, glob

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


# -----------------------------
# Collator that preserves labels and pads them with -100
# -----------------------------
@dataclass
class DataCollatorForCausalLMWithLabelMask:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            # round up to a multiple of pad_to_multiple_of
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            L = len(f["input_ids"])
            pad_len = max_len - L
            input_ids.append(torch.tensor(f["input_ids"] + [pad_id] * pad_len, dtype=torch.long))
            attention_mask.append(torch.tensor(f["attention_mask"] + [0] * pad_len, dtype=torch.long))
            # labels are already -100 for context; pad labels with -100
            labels.append(torch.tensor(f["labels"] + [-100] * pad_len, dtype=torch.long))

        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }
        return batch


# -----------------------------
# Preprocessing helpers
# -----------------------------
def build_prompt(article: str) -> str:
    return (
        "Summarize the following article.\n\n"
        "Article:\n"
        f"{article.strip()}\n"
    )

def preprocess_example(
    ex: Dict[str, Any],
    tokenizer: AutoTokenizer,
    sep_token: str,
    max_source_len: int,
    max_target_len: int,
    max_seq_len: int,
) -> Dict[str, Any]:
    # Tokenize source (prompt + article)
    src_text = build_prompt(ex["article"])
    src_toks = tokenizer(
        src_text,
        truncation=True,
        max_length=max_source_len,
        add_special_tokens=True,
    )

    # Tokenize target summary (+ EOS)
    tgt_text = ex["highlights"].strip() + tokenizer.eos_token
    tgt_toks = tokenizer(
        tgt_text,
        truncation=True,
        max_length=max_target_len,
        add_special_tokens=False,
    )

    # Insert a separator token between source and target to make boundaries explicit
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    input_ids = src_toks["input_ids"] + [sep_id] + tgt_toks["input_ids"]
    attention_mask = src_toks["attention_mask"] + [1] + [1] * len(tgt_toks["input_ids"])

    # Ensure final length does not exceed model max
    if len(input_ids) > max_seq_len:
        # Prefer to trim from the *source* portion
        overflow = len(input_ids) - max_seq_len
        # We keep the last part of the source so that the end aligns with the target start.
        src_ids = src_toks["input_ids"]
        src_am = src_toks["attention_mask"]
        # Trim source only if possible
        if len(src_ids) > overflow:
            src_ids = src_ids[overflow:]
            src_am = src_am[overflow:]
            input_ids = src_ids + [sep_id] + tgt_toks["input_ids"]
            attention_mask = src_am + [1] + [1] * len(tgt_toks["input_ids"])
        else:
            # Fallback: hard truncate to max_seq_len
            input_ids = input_ids[-max_seq_len:]
            attention_mask = attention_mask[-max_seq_len:]

    # Labels: ignore loss on context (source + sep), train on target tokens only
    labels = [-100] * (len(input_ids) - len(tgt_toks["input_ids"])) + tgt_toks["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset(
    ds,
    tokenizer: AutoTokenizer,
    sep_token: str,
    max_source_len: int,
    max_target_len: int,
    max_seq_len: int,
    num_proc: int = 4,
):
    
    if isinstance(ds, dict):
        ds = DatasetDict(ds)
     
    def _map_fn(batch):
        out = [preprocess_example(
            {"article": a, "highlights": h},
            tokenizer, sep_token, max_source_len, max_target_len, max_seq_len
        ) for a, h in zip(batch["article"], batch["highlights"])]
        return {
            "input_ids": [o["input_ids"] for o in out],
            "attention_mask": [o["attention_mask"] for o in out],
            "labels": [o["labels"] for o in out],
        }

    # return ds.map(_map_fn, batched=True, num_proc=num_proc, remove_columns=ds["train"].column_names)
    # pick columns from the first split (works for any split names like canary/non_canary)
    first_split = next(iter(ds.keys()))
    base_cols = ds[first_split].column_names
    return ds.map(_map_fn, batched=True, num_proc=num_proc, remove_columns=base_cols)


# -----------------------------
# ROUGE evaluation on generated summaries
# -----------------------------
def evaluate_rouge(model, tokenizer, dataset, device, num_samples=80, gen_max_new_tokens=128):
    rouge = evaluate.load("rouge")
    model.eval()
    preds, refs = [], []
    # Use a small subset to keep it fast
    subset = dataset.select(range(min(num_samples, len(dataset))))
    for ex in tqdm(subset):
        # Build generation prompt: source only, no labels
        prompt = build_prompt(tokenizer.decode(ex["input_ids"], skip_special_tokens=True)) \
            if "article" in ex else None  # fallback not used since we've removed columns
        # Reconstruct prompt from original tokens: instead, rebuild directly from stored input_ids up to sep_token
        ids = ex["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long)
        # Find sep position
        sep_id = tokenizer.convert_tokens_to_ids("<|sep|>")
        sep_pos = (ids == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_pos) == 0:
            # If separator missing (shouldn't happen), take half split
            cut = len(ids) // 2
        else:
            cut = int(sep_pos[0])

        gen_input_ids = ids[:cut+1].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(gen_input_ids, dtype=torch.long)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=gen_input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=gen_max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        summary = tokenizer.decode(out_ids[0][gen_input_ids.shape[1]:], skip_special_tokens=True)
        # Reference: decode labels where != -100
        labels = torch.tensor(ex["labels"])
        ref_ids = labels[labels != -100]
        reference = tokenizer.decode(ref_ids, skip_special_tokens=True)

        preds.append(summary.strip())
        refs.append(reference.strip())

    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return scores


# -----------------------------
# Single-example inference + loss + gradient (for MIA)
# -----------------------------
def single_example_inference_and_gradient(model, tokenizer, device, article: str, reference_summary: str,
                                          max_source_len=800, max_target_len=128, max_seq_len=1024):
    model.train()  # enable grad
    sep_token = "<|sep|>"
    example = preprocess_example(
        {"article": article, "highlights": reference_summary},
        tokenizer, sep_token, max_source_len, max_target_len, max_seq_len
    )

    input_ids = torch.tensor(example["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.tensor(example["labels"], dtype=torch.long, device=device).unsqueeze(0)

    # Forward + backward to get gradients
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss
    loss.backward()

    # Collect a flat gradient vector (careful: huge for big models)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    flat_grad = torch.cat(grads) if grads else torch.tensor([], device=device)
    grad_norm = flat_grad.norm().item()

    # Also produce a generated summary from the source only
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    sep_pos = (input_ids[0] == sep_id).nonzero(as_tuple=True)[0]
    cut = int(sep_pos[0]) if len(sep_pos) else input_ids.shape[1] // 2
    gen_inp = input_ids[:, :cut+1]
    gen_att = attention_mask[:, :cut+1]

    model.eval()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=gen_inp,
            attention_mask=gen_att,
            do_sample=False,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(gen_out[0][gen_inp.shape[1]:], skip_special_tokens=True)

    return {
        "loss": float(loss.item()),
        "grad_norm": grad_norm,
        "flat_grad": flat_grad,  # torch.Tensor on device
        "generated_summary": generated.strip(),
    }


# -----------------------------
# Membership Inference Attack (MIA) via loss thresholding
# -----------------------------
@torch.no_grad()
def per_example_losses(model, dataset, device, collator, batch_size=4):
    """
    Compute per-example average NLL (token-level CE) ignoring -100 labels.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    losses = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        # HF causal LM uses shift; mimic it
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab = shift_logits.size(-1)
        token_losses = ce(shift_logits.view(-1, vocab), shift_labels.view(-1)).view(shift_labels.size())
        mask = (shift_labels != -100)

        # mean over valid tokens per example
        seq_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        losses.extend(seq_loss.detach().cpu().tolist())

    return losses


def run_simple_mia(model, device, collator, canary_ds, non_canary_ds, k_frac=1/3, batch_size=4):
    """
    Loss-based MIA:
      - Compute per-example loss on canary (IN) and non-canary (OUT) sets.
      - Sort ascending (lower loss => more likely IN).
      - Predict IN for lowest k_frac, OUT for highest k_frac; report accuracy on those 2k.
    """
    in_losses  = per_example_losses(model, canary_ds, device, collator, batch_size)
    out_losses = per_example_losses(model, non_canary_ds, device, collator, batch_size)

    y = np.array([1]*len(in_losses) + [0]*len(out_losses), dtype=int)
    scores = np.array(in_losses + out_losses, dtype=float)  # lower = more member-like

    order = np.argsort(scores)  # ascending
    k = int(len(scores) * k_frac)
    pred = np.full_like(y, fill_value=-1)  # -1 = abstain
    pred[order[:k]]  = 1   # IN
    pred[order[-k:]] = 0   # OUT

    idx = np.where(pred != -1)[0]
    acc = float((pred[idx] == y[idx]).mean()) if len(idx) else float("nan")

    print(f"[MIA] members={len(in_losses)} | non-members={len(out_losses)} | total={len(scores)}")
    print(f"[MIA] k={k} ({k_frac:.2f} of total); evaluated={len(idx)} samples")
    print(f"[MIA] mean loss (IN)={np.mean(in_losses):.4f} | (OUT)={np.mean(out_losses):.4f}")
    print(f"[MIA] accuracy (on lowest/highest thirds) = {acc*100:.2f}%")

    # Optional: threshold values for analysis
    thr_in  = scores[order[k-1]] if k > 0 else None
    thr_out = scores[order[-k]]  if k > 0 else None
    print(f"[MIA] threshold IN@k: {thr_in} | threshold OUT@k: {thr_out}")

    return {"accuracy": acc, "k": k, "thr_in": thr_in, "thr_out": thr_out}

def find_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    cand = sorted(
        [p for p in glob.glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(p)],
        key=lambda p: int(re.findall(r"checkpoint-(\d+)", p)[0]),
        reverse=True,
    )
    return cand[0] if cand else None


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2-xl",
                        help="gpt2-xl or EleutherAI/gpt-j-6B")
    parser.add_argument("--output_dir", type=str, default="./outputs_gpt_cnn_dm")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_source_len", type=int, default=800)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=1024)  # gpt2 context=1024; GPT-J supports 2048
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5) # try smaller e.g., 1e-5
    parser.add_argument("--warmup_ratio", type=float, default=0.03) # try larger e.g., 0.06
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--eval_rouge_samples", type=int, default=20)
    parser.add_argument("--device_idx", type=int, default=0, help="GPU device index (if using CUDA)")
    parser.add_argument("--tot_samples", type=int, default=1000, help="Total samples to use from CNN/DM")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run eval/MIA on a trained checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to a HF checkpoint dir to load (e.g., ./outputs_gpt_cnn_dm/checkpoint-86)")
    args = parser.parse_args()

    # Set total number of samples and split proportions
    total_samples = args.tot_samples
    train_prop = 0.7
    val_prop = 0.15
    
    set_seed(args.seed)
    device_index = args.device_idx
    device = f"cuda:{device_index}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # ----------------- Tokenizer & model -----------------
    # Decide load path
    load_path = None
    if args.skip_train:
        load_path = args.checkpoint_path or find_latest_checkpoint(args.output_dir)
        if load_path:
            print(f"Loading from checkpoint: {load_path}")
        else:
            print("No checkpoint found, falling back to base model.")
    # Load tokenizer (from checkpoint if available to keep special tokens consistent)
    tok_src = load_path if load_path else args.model_name
    print(f"Loading tokenizer from: {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    # Ensure pad token and a separator token exist
    added_tokens = {}
    if tokenizer.pad_token is None:
        added_tokens["pad_token"] = tokenizer.eos_token
    if "<|sep|>" not in tokenizer.get_vocab():
        added_tokens.setdefault("additional_special_tokens", [])
        added_tokens["additional_special_tokens"].append("<|sep|>")
    if added_tokens:
        tokenizer.add_special_tokens(added_tokens)

    # sep_token = tokenizer.additional_special_tokens[0]
    sep_token = "<|sep|>"   # avoid relying on index 0
    pad_id = tokenizer.pad_token_id

    # Load model (from checkpoint if provided/found)
    model_src = load_path if load_path else args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_src,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    # Resize embeddings if we added tokens
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = pad_id
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)

    # ----------------- Load or preprocess dataset -----------------
    print("\033[93m\nPreparing CNN/DailyMail datasets...\033[0m")
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_paths = {
        "train": os.path.join(args.output_dir, f"train_dataset_{total_samples}.pt"),
        "validation": os.path.join(args.output_dir, f"val_dataset_{total_samples}.pt"),
        "test": os.path.join(args.output_dir, f"test_dataset_{total_samples}.pt"),
        "canary": os.path.join(args.output_dir, f"canary_dataset_{total_samples}.pt"),
        "non_canary": os.path.join(args.output_dir, f"non_canary_dataset_{total_samples}.pt"),
    }

    train_count = int(total_samples * train_prop)
    val_count = int(total_samples * val_prop)
    test_count = total_samples - train_count - val_count  # ensures sum == total_samples

    datasets_exist = all(os.path.exists(path) for path in dataset_paths.values())
    if datasets_exist:
        print("Found preprocessed datasets. Loading from disk...")
        train_ds     = torch.load(dataset_paths["train"], weights_only=False)
        val_ds       = torch.load(dataset_paths["validation"], weights_only=False)
        test_ds      = torch.load(dataset_paths["test"], weights_only=False)
        canary_ds    = torch.load(dataset_paths["canary"], weights_only=False)
        non_canary_ds= torch.load(dataset_paths["non_canary"], weights_only=False)
        raw = load_dataset("cnn_dailymail", "3.0.0")
    else:
        print("Loading CNN/DailyMail...")
        raw = load_dataset("cnn_dailymail", "3.0.0")
        
        # raise error if not enough samples
        total_available = sum(len(raw[split]) for split in ["train", "validation", "test"])
        if total_samples > total_available:
            raise ValueError(f"Requested total_samples={total_samples} exceeds available={total_available} in CNN/DM.")

        # Split into train/val/test
        train_data = raw["train"].shuffle(seed=args.seed).select(range(train_count))
        val_data = raw["validation"].shuffle(seed=args.seed).select(range(val_count))
        test_data = raw["test"].shuffle(seed=args.seed).select(range(test_count))

        # Canary = subset of *training* data; Non-canary = disjoint from training
        canary_raw     = train_data.select(range(min(val_count, len(train_data))))
        non_canary_raw = raw["train"].shuffle(seed=args.seed).select(
            range(train_count, train_count + min(val_count, len(raw["train"]) - train_count))
        )

        # Tokenize main splits
        raw_limited = DatasetDict({"train": train_data, "validation": val_data, "test": test_data})
        print("\033[93m\nTokenizing main splits...\033[0m")
        proc = tokenize_dataset(
            raw_limited, tokenizer, sep_token=sep_token,
            max_source_len=args.max_source_len, max_target_len=args.max_target_len,
            max_seq_len=args.max_seq_len, num_proc=args.num_proc,
        )
        train_ds, val_ds, test_ds = proc["train"], proc["validation"], proc["test"]

        # Tokenize canary/non-canary
        print("\033[93m\nTokenizing canary/non-canary...\033[0m")
        proc_mia = tokenize_dataset(
            DatasetDict({"canary": canary_raw, "non_canary": non_canary_raw}),
            tokenizer, sep_token=sep_token,
            max_source_len=args.max_source_len, max_target_len=args.max_target_len,
            max_seq_len=args.max_seq_len, num_proc=args.num_proc,
        )
        canary_ds     = proc_mia["canary"]
        non_canary_ds = proc_mia["non_canary"]

        # Save all
        torch.save(train_ds,      dataset_paths["train"])
        torch.save(val_ds,        dataset_paths["validation"])
        torch.save(test_ds,       dataset_paths["test"])
        torch.save(canary_ds,     dataset_paths["canary"])
        torch.save(non_canary_ds, dataset_paths["non_canary"])
        print("Saved tokenized datasets.")

    collator = DataCollatorForCausalLMWithLabelMask(tokenizer, pad_to_multiple_of=8)

    # ----------------- Training -----------------
    total_train_steps = (len(train_ds) // (args.train_batch_size * max(1, args.gradient_accumulation_steps))) * math.ceil(args.num_train_epochs)
    print(f"\nTrain examples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # fp16=(args.fp16 and device.startswith("cuda")),
        # bf16=(args.bf16 and device != "cpu"),
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ----------------- Eval: test loss & perplexity -----------------
    print("\033[93m\nEvaluating on test set (loss/perplexity) before training...\033[0m")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    test_loss = test_metrics.get("eval_loss", float("nan"))
    test_ppl = math.exp(test_loss) if test_loss < 20 else float("inf")
    print(f"[Test] loss: {test_loss:.4f} | ppl: {test_ppl:.2f}")

    # ----------------- Eval: ROUGE on generated summaries (subset) -----------------
    # print("\033[93m\nEvaluating ROUGE (subset) before training...\033[0m")
    # rouge_scores = evaluate_rouge(model, tokenizer, test_ds, device, num_samples=args.eval_rouge_samples)
    # print("ROUGE:", {k: round(v, 4) for k, v in rouge_scores.items()})

    print("\033[93m\nStarting training...\033[0m")
    if not args.skip_train:
        trainer.train()

        # ----------------- Eval: test loss & perplexity -----------------
        print("\033[93m\nEvaluating on test set (loss/perplexity)...\033[0m")
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        test_loss = test_metrics.get("eval_loss", float("nan"))
        test_ppl = math.exp(test_loss) if test_loss < 20 else float("inf")
        print(f"[Test] loss: {test_loss:.4f} | ppl: {test_ppl:.2f}")

        # # ----------------- Eval: ROUGE on generated summaries (subset) -----------------
        # print("\033[93m\nEvaluating ROUGE (subset)...\033[0m")
        # rouge_scores = evaluate_rouge(model, tokenizer, test_ds, device, num_samples=args.eval_rouge_samples)
        # print("ROUGE:", {k: round(v, 4) for k, v in rouge_scores.items()})

    # # ----------------- Single-example inference + loss + gradient -----------------
    # print("\033[93m\nRunning single-example inference & gradient...\033[0m")
    # # Take one test sample
    # ex = raw["test"][0]
    # article = ex["article"]
    # reference = ex["highlights"]

    # single = single_example_inference_and_gradient(
    #     model, tokenizer, device,
    #     article=article,
    #     reference_summary=reference,
    #     max_source_len=args.max_source_len,
    #     max_target_len=args.max_target_len,
    #     max_seq_len=args.max_seq_len,
    # )
    # print(f"Single-example loss: {single['loss']:.6f}")
    # print(f"Single-example grad L2-norm: {single['grad_norm']:.4f}")
    # print("Generated summary:")
    # print(single["generated_summary"][:1000])

    # # Optionally save the flat gradient vector for MIA experiments
    # os.makedirs(args.output_dir, exist_ok=True)
    # grad_path = os.path.join(args.output_dir, "single_example_flat_grad.pt")
    # torch.save(single["flat_grad"].detach().cpu(), grad_path)
    # print(f"Saved flat gradient vector to: {grad_path}")

    # ----------------- Membership Inference Attack (MIA) -----------------
    print("\033[93m\nRunning loss-based MIA (⅓ lowest vs ⅓ highest)...\033[0m")
    _ = run_simple_mia(
        model=model,
        device=device,
        collator=collator,
        canary_ds=canary_ds,
        non_canary_ds=non_canary_ds,
        k_frac=1/3,
        batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\033[90mTotal training time: {end_time - start_time:.2f} seconds\033[0m")

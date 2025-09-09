import os
import math
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import evaluate
from datasets import DatasetDict
from torch.utils.data import Dataset

import re, glob
from collections import OrderedDict
from functools import reduce
from typing import Tuple
import pandas as pd
import shutil

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

try:
    from opacus.accountants.utils import get_noise_multiplier
    from opacus.accountants.rdp import RDPAccountant
    _HAS_OPACUS = True
except Exception:
    _HAS_OPACUS = False


# -----------------------------
# FL helpers: partitioning, params, FedAvg, local trainer, eval
# -----------------------------
def split_even_hf_dataset(ds, n_parts: int, seed: int) -> List:
    """Even random split of a HF dataset into n_parts (±1 sample)."""
    ds = ds.shuffle(seed=seed)
    N = len(ds)
    base = N // n_parts
    rem  = N % n_parts
    parts = []
    start = 0
    for i in range(n_parts):
        end = start + base + (1 if i < rem else 0)
        if end > start:
            parts.append(ds.select(range(start, end)))
        else:
            parts.append(ds.select([]))
        start = end
    return parts

def remove_empty_dirs(root_dir, patterns):
    for pat in patterns:
        for d in glob.glob(os.path.join(root_dir, pat)):
            if os.path.isdir(d) and not os.listdir(d):
                try:
                    shutil.rmtree(d)
                    print(f"Removed empty folder: {d}")
                except Exception as e:
                    print(f"Could not remove {d}: {e}")
                        
def select_fraction_subset(ds, frac: float, min_size: int = 1, seed: int = 0):
    """Return a shuffled subset containing frac of the dataset (at least min_size)."""
    if len(ds) == 0 or frac <= 0:
        return ds.select([])
    k = max(int(len(ds) * frac), min_size)
    k = min(k, len(ds))
    return ds.shuffle(seed=seed).select(range(k))

def get_parameters_from_model(model) -> List[np.ndarray]:
    # Ordered by state_dict() insertion order (stable across same model)
    return [t.detach().cpu().numpy() for _, t in model.state_dict().items()]

def set_parameters_to_model(model, parameters: List[np.ndarray]):
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state_dict, strict=True)

def fedavg_weighted(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """List[(weights_as_np, num_samples)] -> weighted average parameter list."""
    total = sum(n for _, n in results) or 1
    weighted = [[w * n for w in weights] for weights, n in results]
    avg: List[np.ndarray] = [reduce(np.add, layer_updates) / total
                             for layer_updates in zip(*weighted)]
    return avg

def make_local_trainer(local_model, train_ds, collator, tokenizer, args, round_idx: int, client_id: int):
    out_dir = os.path.join(args.output_dir, f"fl_r{round_idx}_c{client_id}")
    train_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )
    return Trainer(
        model=local_model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[],
    )

def evaluate_weighted_val_loss(global_model, val_splits: List, collator, tokenizer, args, round_idx: int) -> float:
    """Compute weighted mean eval_loss across client validation splits."""
    out_dir = os.path.join(args.output_dir, f"fl_eval_round_{round_idx}")
    eval_args = TrainingArguments(
        output_dir=out_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=global_model, args=eval_args, data_collator=collator, processing_class=tokenizer)
    losses, sizes = [], []
    for ds in val_splits:
        if len(ds) == 0:
            continue
        m = trainer.evaluate(eval_dataset=ds)
        losses.append(m.get("eval_loss", float("nan")))
        sizes.append(len(ds))
    if not sizes:
        return float("nan")
    return float(sum(l * s for l, s in zip(losses, sizes)) / sum(sizes))

def save_results_xlsx(results: dict, xlsx_path: str, max_rounds: int = None):
    """
    Create an Excel file with:
      - summary: test loss/ppl, early stopping, best round by val loss,
                 max client MIA across all rounds, max mean MIA across rounds
      - rounds:  per-round weighted val loss + mean MIA across clients
      - mia:     flattened per-(round, client) MIA details
      - rouge:   ROUGE scores (single-row table)
    """
    # -------- collect per-round / per-client MIA + weighted val loss --------
    round_rows = []
    mia_rows = []

    # Identify round keys
    round_keys = [k for k in results.keys() if k.startswith("round_")]
    # Sort by round number
    def _rnum(k): 
        try: 
            return int(k.split("_")[1])
        except Exception:
            return 10**9
    round_keys.sort(key=_rnum)
    
    # Limit to max_rounds if max_rounds is set
    if max_rounds is not None:
        round_keys = [rk for rk in round_keys if _rnum(rk) < max_rounds]

    for rk in round_keys:
        rnum = _rnum(rk)
        rdict = results.get(rk, {})
        w_loss = rdict.get("weighted_val_loss", float("nan"))

        # collect per-client MIA
        client_accs = []
        for ck, cdict in rdict.items():
            if ck.startswith("client_") and isinstance(cdict, dict) and ("mia" in cdict):
                cid = int(ck.split("_")[1])
                mia = cdict.get("mia", {}) or {}
                acc = mia.get("accuracy", float("nan"))
                client_accs.append(acc)
                mia_rows.append({
                    "round": rnum,
                    "client": cid,
                    "accuracy": acc,
                    "k": mia.get("k", None),
                    "thr_in": mia.get("thr_in", None),
                    "thr_out": mia.get("thr_out", None),
                })

        mean_mia = float(np.nanmean(client_accs)) if client_accs else float("nan")
        round_rows.append({
            "round": rnum,
            "weighted_val_loss": w_loss,
            "mean_mia_acc": mean_mia,
            "num_clients": len(client_accs),
        })

    # -------- compute summary stats --------
    test_loss = results.get("test_loss", float("nan"))
    test_ppl  = results.get("test_ppl", float("nan"))
    early_stop_reached = results.get("early_stop_reached", False)
    early_stop_round   = results.get("early_stop_round", None)

    # best round by weighted val loss
    best_round = None
    best_val  = float("nan")
    if round_rows:
        # filter out NaNs
        valid = [(r["round"], r["weighted_val_loss"]) for r in round_rows if not (isinstance(r["weighted_val_loss"], float) and math.isnan(r["weighted_val_loss"]))]
        if valid:
            best_round, best_val = min(valid, key=lambda x: x[1])

    # max client MIA over all rounds/clients
    max_client_mia = float("nan")
    if mia_rows:
        accs = [r["accuracy"] for r in mia_rows if r["accuracy"] is not None and not (isinstance(r["accuracy"], float) and math.isnan(r["accuracy"]))]
        if accs:
            max_client_mia = float(np.max(accs))

    # max mean MIA across rounds
    max_mean_mia = float("nan")
    if round_rows:
        means = [r["mean_mia_acc"] for r in round_rows if r["mean_mia_acc"] is not None and not (isinstance(r["mean_mia_acc"], float) and math.isnan(r["mean_mia_acc"]))]
        if means:
            max_mean_mia = float(np.max(means))

    rouge_scores = results.get("rouge", {}) or {}

    # -------- build dataframes --------
    df_rounds = pd.DataFrame(round_rows).sort_values("round") if round_rows else pd.DataFrame(columns=["round","weighted_val_loss","mean_mia_acc","num_clients"])
    df_mia    = pd.DataFrame(mia_rows).sort_values(["round","client"]) if mia_rows else pd.DataFrame(columns=["round","client","accuracy","k","thr_in","thr_out"])
    df_rouge  = pd.DataFrame([rouge_scores]) if rouge_scores else pd.DataFrame(columns=["rouge1","rouge2","rougeL","rougeLsum"])

    df_summary = pd.DataFrame([{
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "early_stop_reached": early_stop_reached,
        "early_stop_round": early_stop_round,
        "best_round_by_val": best_round,
        "best_weighted_val_loss": best_val,
        "max_client_mia_accuracy": max_client_mia,
        "max_mean_mia_accuracy_across_rounds": max_mean_mia,
    }])

    # -------- write Excel --------
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    try:
        writer = pd.ExcelWriter(xlsx_path, engine="xlsxwriter")
    except Exception:
        writer = pd.ExcelWriter(xlsx_path)  # fallback (openpyxl)

    with writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_rounds.to_excel(writer, sheet_name="rounds", index=False)
        df_mia.to_excel(writer, sheet_name="mia", index=False)
        df_rouge.to_excel(writer, sheet_name="rouge", index=False)  

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
# SIA dataset + collator
# -----------------------------
class SiaTextConcatDataset(Dataset):
    """
    Concatenate per-client HF datasets while keeping track of true owner (client id).
    Assumes items are already tokenized dicts with keys: input_ids, attention_mask, labels.
    """
    def __init__(self, per_client_splits: List, per_client_max: int | None = None, seed: int = 0):
        self.items = []
        rng = np.random.RandomState(seed)
        for cid, ds in enumerate(per_client_splits):
            n = len(ds)
            if n == 0:
                continue
            idxs = np.arange(n)
            rng.shuffle(idxs)
            if per_client_max is not None:
                idxs = idxs[:min(per_client_max, n)]
            for i in idxs:
                ex = ds[int(i)]
                ex = {k: ex[k] for k in ["input_ids", "attention_mask", "labels"] if k in ex}
                ex["owner"] = cid
                self.items.append(ex)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ex = self.items[i]
        return dict(ex)

    @property
    def true_owners(self):
        # Returns a numpy array aligned with dataset order
        return np.array([ex["owner"] for ex in self.items], dtype=int)

@dataclass
class DataCollatorForCausalLMWithOwner:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        owners = [int(f.get("owner", -1)) for f in features]

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            L = len(f["input_ids"])
            pad_len = max_len - L
            input_ids.append(torch.tensor(f["input_ids"] + [pad_id] * pad_len, dtype=torch.long))
            attention_mask.append(torch.tensor(f["attention_mask"] + [0] * pad_len, dtype=torch.long))
            labels.append(torch.tensor(f["labels"] + [-100] * pad_len, dtype=torch.long))

        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
            "owners": torch.tensor(owners, dtype=torch.long),  # keep owners aligned to rows
        }
        return batch

@torch.no_grad()
def _sia_batch_losses_text(model, batch, device) -> np.ndarray:
    """
    Return per-example mean token NLL (ignoring -100) for a batch.
    """
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # Shifted LM loss (ignore -100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce = torch.nn.CrossEntropyLoss(reduction="none")
    vocab = shift_logits.size(-1)
    tok_losses = ce(shift_logits.view(-1, vocab), shift_labels.view(-1)).view(shift_labels.size())
    mask = (shift_labels != -100)
    seq_loss = (tok_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return seq_loss.detach().cpu().numpy()

def run_sia_attack_llm(
    probe_model,                         # a HF CausalLM model object (we'll overwrite weights)
    sia_loader: torch.utils.data.DataLoader,
    client_params: List[Tuple[List[np.ndarray], int]],  # [(weights_as_np, n_samples), ...]
    device: str,
    set_params_fn = set_parameters_to_model,           # your helper
) -> Dict[str, Any]:
    """
    Text SIA: for each client model, compute per-example losses on the joint SIA set;
    predict owner = argmin(losses) per example.
    Returns accuracy and confusion matrix.
    """
    n_clients  = len(client_params)
    n_samples  = len(sia_loader.dataset)
    losses_all = np.empty((n_samples, n_clients), dtype=np.float32)

    # evaluate column-wise (client by client) to reuse memory
    for cid in range(n_clients):
        # Load this client's weights
        weights_np = client_params[cid][0]
        set_params_fn(probe_model, weights_np)
        probe_model.to(device)
        probe_model.eval()

        # Collect losses for all samples in *fixed dataloader order*
        per_ex_losses = []
        for batch in sia_loader:
            per_ex_losses.append(_sia_batch_losses_text(probe_model, batch, device))
        losses_all[:, cid] = np.concatenate(per_ex_losses)

    # Predictions
    pred_cid = losses_all.argmin(axis=1)

    # Ground-truth owners from dataset
    if hasattr(sia_loader.dataset, "true_owners"):
        true_cid = sia_loader.dataset.true_owners
    else:
        # Fallback (shouldn't happen): try to read from last batch's "owners" and rebuild
        owners_all = []
        for batch in sia_loader:
            owners_all.append(batch["owners"].numpy())
        true_cid = np.concatenate(owners_all)

    acc = float((pred_cid == true_cid).mean())

    # Confusion matrix [true, pred]
    cm = np.zeros((n_clients, n_clients), dtype=int)
    for t, p in zip(true_cid.tolist(), pred_cid.tolist()):
        cm[t, p] += 1

    return {
        "accuracy": acc,
        "confusion": cm.tolist(),   # JSON-serializable
    }


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

def plot_training_and_mia(results: dict, pdf_path: str, title: str | None = None, max_rounds: int | None = None):
    """
    Plot training trends for FL rounds:
      - Left: weighted validation loss across rounds
      - Right: mean and max MIA accuracy across rounds (in %), with peak values annotated
    Saves the figure as a vector PDF at `pdf_path`.
    """

    # ---- Extract series from results ----
    def _round_num(k: str) -> int:
        try:
            return int(k.split("_")[1])
        except Exception:
            return 10**9

    round_keys = sorted([k for k in results.keys() if k.startswith("round_")], key=_round_num)
    if not round_keys:
        raise ValueError("No 'round_*' entries found in results.")
    
    # Limit to max_rounds if max_rounds is set
    if max_rounds is not None:
        round_keys = [rk for rk in round_keys if _round_num(rk) < max_rounds]

    rounds, wloss, mia_mean, mia_max = [], [], [], []
    for rk in round_keys:
        r = _round_num(rk)
        rd = results.get(rk, {})
        rounds.append(r)

        # weighted val loss (float or NaN)
        w = rd.get("weighted_val_loss", float("nan"))
        wloss.append(float(w) if w is not None else float("nan"))

        # MIA metrics saved as fractions in your loop
        mia = rd.get("mia", {}) or {}
        mmean = mia.get("avg_accuracy", float("nan"))
        mmax  = mia.get("max_accuracy", float("nan"))
        # convert to %
        mia_mean.append(100.0 * float(mmean) if mmean is not None else float("nan"))
        mia_max.append(100.0 * float(mmax) if mmax is not None else float("nan"))

    rounds = np.array(rounds, dtype=int)
    wloss  = np.array(wloss, dtype=float)
    mia_mean = np.array(mia_mean, dtype=float)
    mia_max  = np.array(mia_max, dtype=float)

    # ---- Figure style tuned for papers ----
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
    })

    # Wide, compact: good for 2-column or 1.5-column widths
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    # ---- Left: weighted validation loss ----
    ax[0].plot(rounds, wloss, marker="o", linewidth=2)
    ax[0].set_xlabel("Federated round")
    ax[0].set_ylabel("Weighted validation loss")
    ax[0].set_xticks(rounds)
    ax[0].set_title("Validation loss by round")

    # ---- Right: MIA accuracy curves ----
    ax[1].plot(rounds, mia_mean, marker="o", linewidth=2, label="Mean MIA acc")
    ax[1].plot(rounds, mia_max,  marker="s", linewidth=2, label="Max MIA acc")
    ax[1].set_xlabel("Federated round")
    ax[1].set_ylabel("MIA accuracy (%)")
    ax[1].set_xticks(rounds)
    ax[1].set_title("MIA accuracy by round")
    ax[1].legend(loc="best", frameon=False)

    # Annotate top points for the accuracy curves (peak values)
    def _annotate_peak(yvals, axis):
        # handle all-NaN safely
        if not np.isfinite(yvals).any():
            return
        idx = int(np.nanargmax(yvals))
        x, y = rounds[idx], yvals[idx]
        axis.scatter([x], [y], s=36)
        axis.annotate(f"{y:.1f}%", xy=(x, y),
                      xytext=(0, 8), textcoords="offset points",
                      ha="center", va="bottom")

    _annotate_peak(mia_mean, ax[1])
    _annotate_peak(mia_max, ax[1])

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    # ---- Save as PDF ----
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)



# -----------------------------
# SoteriaFL compression and aggregation
# -----------------------------
def init_soteria_state(
    model: torch.nn.Module,
    fl_rounds: int,
    k: int | None = None,
    k_frac: float | None = None,
) -> tuple[int, int, float, List[np.ndarray]]:
    """
    Returns:
      d: total params
      k: number of kept coordinates for random-k
      gamma: SoteriaFL gamma
      s: reference vector list (zeros), one np.ndarray per tensor
    """
    d = count_state_params(model)
    if k is None:
        if k_frac is not None and k_frac > 0 and k_frac < 1:
            k = max(1, int(d * k_frac))
        else:
            # paper’s heuristic (your earlier code)
            k = max(1, int(d / max(1.0, math.log2(max(2, fl_rounds)))))

    # gamma formula used in your prior implementation
    w = (d / k) - 1.0
    gamma = math.sqrt((1.0 + 2.0 * w) / (2.0 * (1.0 + w) ** 3))

    s = []
    for _, t in model.state_dict().items():
        s.append(np.zeros_like(t.detach().cpu().numpy()))
    return d, k, float(gamma), s

def soteria_pack_client_update(
    params_in: List[np.ndarray],
    params_out: List[np.ndarray],
    ref_s: List[np.ndarray],
    k: int,
    rng: np.random.Generator,
) -> tuple[List[np.ndarray], dict]:
    """
    Client-side step *after* local training:
      grads = params_out - params_in
      shifted = grads - s
      shifted_sparse = RandomK(shifted)
      send params_in + shifted_sparse
    """
    grads = [p_out - p_in for p_in, p_out in zip(params_in, params_out)]
    shifted = [g - s for g, s in zip(grads, ref_s)]
    shifted_sparse = compress_random_k(shifted, k=k, rng=rng)
    # what we transmit: params_in + shifted_sparse
    sent = [p_in + gsp for p_in, gsp in zip(params_in, shifted_sparse)]
    d = int(sum(p.size for p in grads))
    stats = {"k": int(k), "d": d, "kept_frac": float(k / d)}
    return sent, stats

def soteria_aggregate(
    results: List[Tuple[List[np.ndarray], int]],
    params_in: List[np.ndarray],
    ref_s: List[np.ndarray],
    gamma: float,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Server aggregation for SoteriaFL:
      For each client result (weights), compute (weights - params_in) = shifted_sparse_grad.
      Average them with example-weighting, then:
        sparse_grads' = s + avg(shifted_sparse_grads)
        params' = params_in + sparse_grads'
        s' = s + gamma * avg(shifted_sparse_grads)
    """
    if not results:
        return params_in, ref_s

    num_total = sum(n for _, n in results) or 1
    # per-client shifted-sparse grads (weighted by num_examples)
    weighted = [
        [(p_out - p_in) * n for p_out, p_in in zip(weights, params_in)]
        for weights, n in results
    ]
    # avg shifted-sparse grads
    avg_shifted_sparse = [
        reduce(np.add, layer_updates) / num_total for layer_updates in zip(*weighted)
    ]

    # remove shift
    sparse_grads_prime = [s + g for s, g in zip(ref_s, avg_shifted_sparse)]
    params_prime = [p_in + g for p_in, g in zip(params_in, sparse_grads_prime)]

    # update reference
    ref_next = [s + gamma * g for s, g in zip(ref_s, avg_shifted_sparse)]
    return params_prime, ref_next

def get_model_norm(m, device):
    params = [p.detach().flatten() for p in m.parameters()]
    flat = torch.cat(params) if params else torch.tensor([], device=device)
    return flat.norm().item()

def dp_sgd_train_causallm(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    max_grad_norm: float,         # C
    noise_multiplier: float,      # sigma
    device: str,
    num_epochs: int = 1,
    grad_accum_steps: int = 1,
    use_fp16: bool = False,
):
    """
    DP-SGD for causal LM with -100-masked labels.
    - Per-sample backward, global L2 clip to C, accumulate microbatches.
    - After `grad_accum_steps` microbatches, average over total B, add N(0, (sigma*C/B)^2),
      then optimizer.step().
    """
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16 and torch.cuda.is_available())

    model.train()
    model.to(device)

    # Accumulators across microbatches (reset every optimizer step)
    def _reset_micro_accum():
        for p in model.parameters():
            p.dp_accum = torch.zeros_like(p, device=p.device)

    _reset_micro_accum()
    mb_count = 0
    mb_total_B = 0

    for epoch in range(int(num_epochs)):
        for batch in tqdm(dataloader, desc=f"DP Epoch {epoch+1}", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            B = input_ids.size(0)

            # Per-sample loop
            for p in model.parameters():
                p.dp_batch_sum = torch.zeros_like(p, device=p.device)

            for i in range(B):
                model.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_fp16 and torch.cuda.is_available()):
                    out = model(input_ids=input_ids[i:i+1],
                                attention_mask=attention_mask[i:i+1],
                                labels=labels[i:i+1])
                    loss_i = out.loss

                if scaler.is_enabled():
                    scaler.scale(loss_i).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss_i.backward()

                # global norm over all params for this sample
                grad_sq_sum = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq_sum += p.grad.data.pow(2).sum().item()
                grad_norm = math.sqrt(grad_sq_sum + 1e-12)
                clip = min(1.0, max_grad_norm / (grad_norm + 1e-12))

                for p in model.parameters():
                    if p.grad is not None:
                        p.dp_batch_sum.add_(p.grad.detach() * clip)

            # Average this batch's clipped grads and add to microbatch accumulator (weighted by B)
            for p in model.parameters():
                p.dp_accum.add_(p.dp_batch_sum)  # weighted sum by B

            mb_count += 1
            mb_total_B += B

            # Do an optimizer step every grad_accum_steps microbatches
            if mb_count == grad_accum_steps:
                # Final average across all microbatches in this step
                for p in model.parameters():
                    if p.requires_grad:
                        final_grad = p.dp_accum / max(1, mb_total_B)
                        # Gaussian noise
                        if noise_multiplier > 0.0:
                            noise = torch.normal(
                                mean=0.0,
                                std=(noise_multiplier * max_grad_norm) / max(1, mb_total_B),
                                size=p.shape,
                                device=p.device,
                            )
                            final_grad = final_grad + noise
                        p.grad = final_grad

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # reset microbatch accumulators
                _reset_micro_accum()
                mb_count = 0
                mb_total_B = 0

        # if leftover microbatches at epoch end
        if mb_count > 0:
            for p in model.parameters():
                if p.requires_grad:
                    final_grad = p.dp_accum / max(1, mb_total_B)
                    if noise_multiplier > 0.0:
                        noise = torch.normal(
                            mean=0.0,
                            std=(noise_multiplier * max_grad_norm) / max(1, mb_total_B),
                            size=p.shape,
                            device=p.device,
                        )
                        final_grad = final_grad + noise
                    p.grad = final_grad
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            _reset_micro_accum()
            mb_count = 0
            mb_total_B = 0

def dp_prepare_client_sigma(N_client: int, batch_size: int, local_epochs: int, fl_rounds: int,
                            target_epsilon: float, delta: float) -> float:
    """
    Compute sigma for total training on this client across all rounds:
      total_epochs = local_epochs * fl_rounds
    Uses Opacus RDP accountant if available; else raises if not installed.
    """
    if not _HAS_OPACUS:
        raise RuntimeError("Opacus not available. Install `opacus>=1.4` or set sigma manually.")
    sample_rate = min(1.0, batch_size / max(1, N_client))
    total_epochs = max(1, int(local_epochs * fl_rounds))
    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=total_epochs,
        accountant="rdp",
    )
    return float(sigma)

def dp_report_epsilon(sigma: float, N_client: int, batch_size: int,
                      local_epochs: int, fl_rounds: int, delta: float) -> float:
    """Report achieved epsilon after training using RDP accountant."""
    if not _HAS_OPACUS:
        return float("nan")
    sample_rate = min(1.0, batch_size / max(1, N_client))
    steps_per_epoch = math.ceil(N_client / max(1, batch_size))
    total_steps = int(steps_per_epoch * local_epochs * fl_rounds)
    acc = RDPAccountant()
    for _ in range(total_steps):
        acc.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return float(acc.get_epsilon(delta))



# -----------------------------
# FL helpers: partitioning, params, FedAvg, local trainer, eval, pruning
# -----------------------------
def prune_largest_grads(
    grads: List[np.ndarray], pruning_rate: float = 0.3
) -> tuple[List[np.ndarray], dict]:
    """
    Zero-out the top `pruning_rate` fraction of gradient magnitudes *globally*
    across all tensors, then reshape back per tensor.

    Returns:
      pruned_grads: List[np.ndarray] shaped like `grads`
      stats: {"threshold": float, "kept_frac": float, "pruning_rate": float}
    """
    assert 0.0 < pruning_rate < 1.0, "Pruning rate must be in (0, 1)."
    if not grads:
        return grads, {"threshold": None, "kept_frac": 1.0, "pruning_rate": pruning_rate}

    # Flatten all grads into a single 1D vector
    flat_parts, shapes, dtypes = [], [], []
    for g in grads:
        arr = np.asarray(g)
        flat_parts.append(arr.reshape(-1))
        shapes.append(arr.shape)
        dtypes.append(arr.dtype)
    flat = np.concatenate(flat_parts, axis=0)
    abs_flat = np.abs(flat)

    # Global threshold for top-k pruning
    thr = np.percentile(abs_flat, 100.0 * (1.0 - pruning_rate))

    # Keep only elements <= threshold (largest magnitudes are zeroed)
    mask = (abs_flat <= thr)
    pruned_flat = flat * mask

    # Reconstruct per-tensor arrays
    pruned, start = [], 0
    for shape, dtype in zip(shapes, dtypes):
        n = int(np.prod(shape))  # works for scalars too
        chunk = pruned_flat[start:start + n].astype(dtype, copy=False).reshape(shape)
        pruned.append(chunk)
        start += n

    stats = {
        "threshold": float(thr),
        "kept_frac": float(mask.mean()),
        "pruning_rate": float(pruning_rate),
    }
    return pruned, stats



# -----------------------------
# ERIS compression and aggregation
# -----------------------------
def flatten_params(params):
    shape_list = [p.shape for p in params]
    flattened_params = np.concatenate([p.flatten() for p in params])
    return flattened_params, shape_list

def unflatten_params(flattened_params, shape_list):
    params = []
    start_idx = 0
    for shape in shape_list:
        size = np.prod(shape)
        param = flattened_params[start_idx:start_idx + size].reshape(shape)
        params.append(param)
        start_idx += size
    return params

def create_mask(params, n_splits, seed):
    flat_params, shape_list = flatten_params(params)
    aggregators_ass = np.zeros_like(flat_params)
    n_elements_per_aggr = len(aggregators_ass)//n_splits
    rest = len(aggregators_ass) % n_splits
    i = 0
    for aggr in range(0,n_splits):
        fragment_size = n_elements_per_aggr + (1 if aggr < rest else 0)
        aggregators_ass[i:i+fragment_size] = aggr
        i = i + fragment_size
    
    # Create a random generator with the given seed
    gen = np.random.MT19937(seed=seed)
    rng = np.random.Generator(gen)
    # Randomly shuffle the aggregator assignments
    rng.shuffle(aggregators_ass)
    
    return unflatten_params(aggregators_ass, shape_list)

def count_state_params(model: torch.nn.Module) -> int:
    return int(sum(t.numel() for _, t in model.state_dict().items()))

def init_eris_state(
    model: torch.nn.Module,
    fl_rounds: int,
    k: int | None = None,
    k_frac: float | None = None,
) -> tuple[int, int, float, List[np.ndarray]]:
    """
    Returns:
      d: total params
      k: number of kept coordinates for random-k
      gamma: SoteriaFL gamma
      s: reference vector list (zeros), one np.ndarray per tensor
    """
    d = count_state_params(model)
    if k is None:
        if k_frac is not None and k_frac > 0 and k_frac < 1:
            k = max(1, int(d * k_frac))
        else:
            # paper’s heuristic (your earlier code)
            k = max(1, int(d / max(1.0, math.log2(max(2, fl_rounds)))))

    # gamma formula used in your prior implementation
    w = (d / k) - 1.0
    gamma = math.sqrt((1.0 + 2.0 * w) / (2.0 * (1.0 + w) ** 3))

    s = []
    for _, t in model.state_dict().items():
        s.append(np.zeros_like(t.detach().cpu().numpy()))
    return d, k, float(gamma), s

def compress_random_k(params: List[np.ndarray], k: int, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Random-k compressor with scaling d/k (unbiased).
    Works on a list of arrays (e.g., grads), flattening globally then reshaping back.
    """
    flats, shapes, dtypes = [], [], []
    for p in params:
        arr = np.asarray(p)
        flats.append(arr.reshape(-1))
        shapes.append(arr.shape)
        dtypes.append(arr.dtype)
    flat = np.concatenate(flats, axis=0)
    d = flat.size
    if k >= d:
        return params  # nothing to compress

    idx = rng.choice(d, size=k, replace=False)
    mask = np.zeros(d, dtype=bool); mask[idx] = True
    scale = d / k
    comp = np.zeros_like(flat)
    comp[mask] = flat[mask] * scale

    out, start = [], 0
    for shape, dtype in zip(shapes, dtypes):
        n = int(np.prod(shape))
        out.append(comp[start:start+n].astype(dtype, copy=False).reshape(shape))
        start += n
    return out

def eris_pack_client_update(
    params_in: List[np.ndarray],
    params_out: List[np.ndarray],
    ref_s: List[np.ndarray],
    k: int,
    rng: np.random.Generator,
) -> tuple[List[np.ndarray], dict]:
    """
    Client-side step *after* local training:
      grads = params_out - params_in
      shifted = grads - s
      shifted_sparse = RandomK(shifted)
      send params_in + shifted_sparse
    """
    grads = [p_out - p_in for p_in, p_out in zip(params_in, params_out)]
    shifted = [g - s for g, s in zip(grads, ref_s)]
    shifted_sparse = compress_random_k(shifted, k=k, rng=rng)
    # what we transmit: params_in + shifted_sparse
    sent = [p_in + gsp for p_in, gsp in zip(params_in, shifted_sparse)]
    d = int(sum(p.size for p in grads))
    stats = {"k": int(k), "d": d, "kept_frac": float(k / d)}
    return sent, stats

def eris_aggregate(
    results: List[Tuple[List[np.ndarray], int]],
    params_in: List[np.ndarray],
    ref_s: List[np.ndarray],
    gamma: float,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Server aggregation for SoteriaFL:
      For each client result (weights), compute (weights - params_in) = shifted_sparse_grad.
      Average them with example-weighting, then:
        sparse_grads' = s + avg(shifted_sparse_grads)
        params' = params_in + sparse_grads'
        s' = s + gamma * avg(shifted_sparse_grads)
    """
    if not results:
        return params_in, ref_s

    num_total = sum(n for _, n in results) or 1
    # per-client shifted-sparse grads (weighted by num_examples)
    weighted = [
        [(p_out - p_in) * n for p_out, p_in in zip(weights, params_in)]
        for weights, n in results
    ]
    # avg shifted-sparse grads
    avg_shifted_sparse = [
        reduce(np.add, layer_updates) / num_total for layer_updates in zip(*weighted)
    ]

    # remove shift
    sparse_grads_prime = [s + g for s, g in zip(ref_s, avg_shifted_sparse)]
    params_prime = [p_in + g for p_in, g in zip(params_in, sparse_grads_prime)]

    # update reference
    ref_next = [s + gamma * g for s, g in zip(ref_s, avg_shifted_sparse)]
    return params_prime, ref_next

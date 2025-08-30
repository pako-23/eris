# finetune_cnn_dm_gpt.py
# pip install -U "transformers>=4.42.0" "datasets>=2.19.0" "accelerate>=0.30.0" "evaluate>=0.4.2" sentencepiece
# Optional (for ROUGE): pip install rouge-score

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any
from torch.utils.data import DataLoader
try:
    from opacus.accountants.utils import get_noise_multiplier
    from opacus.accountants.rdp import RDPAccountant
    _HAS_OPACUS = True
except Exception:
    _HAS_OPACUS = False

import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np

from datasets import load_dataset
import evaluate
from datasets import load_dataset, DatasetDict
import time
import re, glob
from collections import OrderedDict
from functools import reduce
from typing import Tuple
import json
import pandas as pd
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


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
    sigma=1000
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

def save_results_xlsx(results: dict, xlsx_path: str):
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
    parser.add_argument("--model_name", type=str, default="gpt2", #default="gpt2-xl",
                        help="gpt2-xl or EleutherAI/gpt-j-6B")
    parser.add_argument("--output_dir", type=str, default="./outputs_gpt_cnn_dm_light")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_source_len", type=int, default=800)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=1024)  # gpt2 context=1024; GPT-J supports 2048
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
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
    parser.add_argument("--tot_samples", type=int, default=100, help="Total samples to use from CNN/DM")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run eval/MIA on a trained checkpoint")
    parser.add_argument("--ckp", type=str, default="", help="Path to a HF checkpoint dir to load (e.g., ./outputs_gpt_cnn_dm/checkpoint-86)")
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--fl_rounds", type=int, default=2)
    parser.add_argument("--local_epochs", type=float, default=1.0)
    parser.add_argument("--partition_seed", type=int, default=123)
    parser.add_argument("--save_global_each_round", action="store_true")
    parser.add_argument("--client_canary_frac", type=float, default=0.2,help="Fraction of each client's train shard used as canary (members)")
    parser.add_argument("--mia_k_frac", type=float, default=1/3,help="Fraction for loss-threshold MIA (lowest/highest)")
    parser.add_argument("--fold", type=int, default=0, help="Experiment fold number (for logging)")
    parser.add_argument("--dp_epsilon", type=float, default=.001)
    parser.add_argument("--dp_delta", type=float, default=None, help="If None, set to 1/N_client_samples")
    parser.add_argument("--dp_max_grad_norm", type=float, default=1.0)   # C
    parser.add_argument("--dp_accountant", type=str, default="rdp", choices=["rdp"])
    args = parser.parse_args()
    
    # update seed
    args.seed = args.seed + args.fold
    args.partition_seed = args.partition_seed + args.fold
    
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
        load_path = args.ckp or find_latest_checkpoint(args.output_dir)
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
    model.config.loss_type = "ForCausalLMLoss"

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

        # ----------------- Client splits -----------------
        n_clients = args.n_clients
        print(f"\nSplitting into {n_clients} clients (train/val)...")
        client_trains = split_even_hf_dataset(train_ds, n_clients, seed=args.partition_seed)
        client_vals   = split_even_hf_dataset(val_ds,   n_clients, seed=args.partition_seed)

        for cid, (tr, va) in enumerate(zip(client_trains, client_vals)):
            print(f"Client {cid}: train={len(tr)}, val={len(va)}")

        # Prepare per-client MIA splits
        # - Canary per client: subset of its own train shard (members)
        # - Non-canary per client: split the global non_canary_ds evenly (non-members)
        print("\nPreparing per-client canary/non-canary splits for MIA...")
        client_canaries = [
            select_fraction_subset(tr, frac=args.client_canary_frac, seed=args.partition_seed + cid)
            for cid, tr in enumerate(client_trains)
        ]
        noncanary_splits = split_even_hf_dataset(non_canary_ds, n_clients, seed=args.partition_seed)
        client_noncanaries = [noncanary_splits[cid] for cid in range(n_clients)]

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
        
        # ----------------- Client splits -----------------
        n_clients = args.n_clients
        print(f"\nSplitting into {n_clients} clients (train/val)...")
        client_trains = split_even_hf_dataset(train_ds, n_clients, seed=args.partition_seed)
        client_vals   = split_even_hf_dataset(val_ds,   n_clients, seed=args.partition_seed)

        for cid, (tr, va) in enumerate(zip(client_trains, client_vals)):
            print(f"Client {cid}: train={len(tr)}, val={len(va)}")

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

        # Prepare per-client MIA splits
        print("\nPreparing per-client canary/non-canary splits for MIA...")
        client_canaries = [
            select_fraction_subset(tr, frac=args.client_canary_frac, seed=args.partition_seed + cid)
            for cid, tr in enumerate(client_trains)
        ]
        noncanary_splits = split_even_hf_dataset(non_canary_ds, n_clients, seed=args.partition_seed)
        client_noncanaries = [noncanary_splits[cid] for cid in range(n_clients)]

        # Save all
        torch.save(train_ds,      dataset_paths["train"])
        torch.save(val_ds,        dataset_paths["validation"])
        torch.save(test_ds,       dataset_paths["test"])
        torch.save(canary_ds,     dataset_paths["canary"])
        torch.save(non_canary_ds, dataset_paths["non_canary"])
        print("Saved tokenized datasets.")

    collator = DataCollatorForCausalLMWithLabelMask(tokenizer, pad_to_multiple_of=8)

    # ----------------- Training -----------------
    print(f"\nTotal train examples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print("\033[93m\nStarting Federated Averaging training...\033[0m")
    # Track best checkpoint directory across rounds
    best_ckpt_dir = None
    if not args.skip_train:
        # Initialize "global" weights from the current model
        global_params = get_parameters_from_model(model)

        best_w_loss = float("inf")
        patience = args.patience
        no_improve = 0
        results = {"early_stop_reached": False, "early_stop_round": None}

        for rnd in range(1, args.fl_rounds + 1):
            print(f"\033[92m\n--> FL ROUND {rnd}/{args.fl_rounds}\033[0m")

            client_results: List[Tuple[List[np.ndarray], int]] = []
            results[f"round_{rnd}"] = {}

            # Sequential local training per client (each starts from the same global weights)
            for cid in range(n_clients):
                print(f"\033[93mClient {cid}: local training on {len(client_trains[cid])} samples\033[0m")
                results[f"round_{rnd}"][f"client_{cid}"] = {}

                # Fresh local model with the right embedding size & config
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_src,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                local_model.resize_token_embeddings(len(tokenizer))
                local_model.config.pad_token_id = pad_id
                local_model.config.use_cache = False
                local_model.config.loss_type = "ForCausalLMLoss"
                if args.gradient_checkpointing:
                    local_model.gradient_checkpointing_enable()
                local_model.to(device)

                # Load global weights
                set_parameters_to_model(local_model, global_params)

                # Train locally
                # Build DP dataloader for this client
                dl = DataLoader(
                    client_trains[cid],
                    batch_size=args.train_batch_size,
                    shuffle=True,
                    collate_fn=collator,
                    num_workers=2,
                    pin_memory=torch.cuda.is_available(),
                )
                N_client = len(client_trains[cid])
                # delta default to 1/N_client_samples
                delta = args.dp_delta if args.dp_delta is not None else 1.0 / max(1, N_client)

                # Compute (or cache) sigma for this client across ALL rounds
                sigma = dp_prepare_client_sigma(
                    N_client=N_client,
                    batch_size=args.train_batch_size,
                    local_epochs=args.local_epochs,
                    fl_rounds=args.fl_rounds,
                    target_epsilon=args.dp_epsilon,
                    delta=delta,
                )

                optimizer_dp = torch.optim.AdamW(local_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scheduler_dp = None  # optional: wire a real scheduler if you like

                print(f"[DP] Client {cid}: sigma={sigma:.4f}, delta={delta:.2e}, C={args.dp_max_grad_norm}")
                # print(f"Model norm before DP-SGD: {get_model_norm(local_model, device):.4f}")
                dp_sgd_train_causallm(
                    model=local_model,
                    dataloader=dl,
                    optimizer=optimizer_dp,
                    scheduler=scheduler_dp,
                    max_grad_norm=args.dp_max_grad_norm,
                    noise_multiplier=sigma,
                    device=device,
                    num_epochs=args.local_epochs,                  # per-round local epochs
                    grad_accum_steps=args.gradient_accumulation_steps,
                    # use_fp16=args.fp16 and not args.bf16,         # DP in fp16 is okay if you keep noise in fp32 (we do)
                )
                # print(f"Model norm after DP-SGD: {get_model_norm(local_model, device):.4f}")

                # (optional) Log achieved epsilon
                eps_hat = dp_report_epsilon(
                    sigma=sigma, N_client=N_client, batch_size=args.train_batch_size,
                    local_epochs=args.local_epochs, fl_rounds=rnd,  # privacy spent up to current round
                    delta=delta
                )
                results[f"round_{rnd}"][f"client_{cid}"]["dp"] = {
                    "sigma": sigma, "delta": delta, "C": args.dp_max_grad_norm, "epsilon_so_far": eps_hat
                }
                
                # ----------------- Membership Inference Attack (MIA) -----------------
                print("\033[94mRunning per-client MIA on local model...\033[0m")
                mia_result = run_simple_mia(
                    model=local_model,
                    device=device,
                    collator=collator,
                    canary_ds=client_canaries[cid],
                    non_canary_ds=client_noncanaries[cid],
                    k_frac=args.mia_k_frac,
                    batch_size=args.eval_batch_size,
                )
                # Save MIA summary for this round/client
                print(f"\033[94mMIA result: {mia_result}\033[0m")
                results[f"round_{rnd}"][f"client_{cid}"]["mia"] = mia_result

                # Collect updated weights and “importance” (=num samples)
                client_params_np = get_parameters_from_model(local_model)
                client_results.append((client_params_np, len(client_trains[cid])))

                # Cleanup GPU RAM
                del local_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # FedAvg aggregation
            print(f"\033[93mAggregating {len(client_results)} client models (FedAvg)\033[0m")
            global_params = fedavg_weighted(client_results)

            # Load aggregated weights into the global model object
            set_parameters_to_model(model, global_params)
            model.to(device)

            # (Optional) evaluate on client validation shards (weighted loss)
            print(f"\033[93mEvaluating aggregated model on client validation sets\033[0m")
            w_val_loss = evaluate_weighted_val_loss(model, client_vals, collator, tokenizer, args, round_idx=rnd)
            results[f"round_{rnd}"]["weighted_val_loss"] = w_val_loss
            print(f"\033[92m✅ Round {rnd}: weighted val_loss={w_val_loss:.4f}\033[0m")

            # Early stopping on weighted val loss (like your example)
            improved = w_val_loss < best_w_loss
            if improved:
                best_w_loss = w_val_loss
                no_improve = 0
                # Save the current best global model (checkpoint-style)
                best_dir = os.path.join(args.output_dir, f"global_round_{rnd}")
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                best_ckpt_dir = best_dir
            else:
                no_improve += 1

            if args.save_global_each_round:
                round_dir = os.path.join(args.output_dir, f"global_round_{rnd}_always")
                os.makedirs(round_dir, exist_ok=True)
                model.save_pretrained(round_dir)
                tokenizer.save_pretrained(round_dir)

            if no_improve >= patience:
                print(f"\033[91mEarly stopping: no improvement for {patience} rounds.\033[0m")
                results["early_stop_reached"] = True
                results["early_stop_round"] = rnd
                break


    # ----------------- Eval: test loss & perplexity -----------------
    print("\033[93m\nLoading best global model from disk...\033[0m")
    # Load best model from training, or provided checkpoint if skipping training
    if args.skip_train:
        # When skipping training, use the already loaded model (from --ckp or base)
        best_model = model
    else:
        if best_ckpt_dir and os.path.isdir(best_ckpt_dir):
            best_model = AutoModelForCausalLM.from_pretrained(best_ckpt_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            best_model.resize_token_embeddings(len(tokenizer))
            best_model.config.pad_token_id = tokenizer.pad_token_id
            best_model.config.use_cache = False
            best_model.config.loss_type = "ForCausalLMLoss"
            if args.gradient_checkpointing:
                best_model.gradient_checkpointing_enable()
            best_model.to(device)
        else:
            # Fallback to current in-memory model if no improvement was saved
            best_model = model
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=best_model, args=eval_args, data_collator=collator, processing_class=tokenizer)
    print("\033[93m\nEvaluating on test set (loss/perplexity)...\033[0m")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    test_loss = test_metrics.get("eval_loss", float("nan"))
    test_ppl = math.exp(test_loss) if test_loss < 20 else float("inf")
    print(f"[Test] loss: {test_loss:.4f} | ppl: {test_ppl:.2f}")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl

    # ----------------- Eval: ROUGE on generated summaries (subset) -----------------
    print("\033[93m\nEvaluating ROUGE (subset)...\033[0m")
    rouge_scores = evaluate_rouge(best_model, tokenizer, test_ds, device, num_samples=args.eval_rouge_samples)
    print("ROUGE:", {k: round(v, 4) for k, v in rouge_scores.items()})
    results["rouge"] = rouge_scores

    # ----------------- Membership Inference Attack (MIA) -----------------
    # print("\033[93m\nRunning loss-based MIA (⅓ lowest vs ⅓ highest)...\033[0m")
    # _ = run_simple_mia(
    #     model=best_model,
    #     device=device,
    #     collator=collator,
    #     canary_ds=canary_ds,
    #     non_canary_ds=non_canary_ds,
    #     k_frac=1/3,
    #     batch_size=args.eval_batch_size,
    # )

    # Save results summary
    results_path = os.path.join(args.output_dir, f"results_summary_F{args.fold}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)  

    # XLSX export with key metrics
    xlsx_path = os.path.join(args.output_dir, f"results_summary_F{args.fold}.xlsx")
    save_results_xlsx(results, xlsx_path)
    print(f"[OK] Wrote {results_path} and {xlsx_path}")
    
    # Cleanup: remove intermediate round checkpoints
    remove_empty_dirs(args.output_dir, ["fl_eval_round_*", "fl_r*_c*"])
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\033[90mTotal training time: {end_time - start_time:.2f} seconds\033[0m")





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

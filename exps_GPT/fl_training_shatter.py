# finetune_cnn_dm_gpt.py
# pip install -U "transformers>=4.42.0" "datasets>=2.19.0" "accelerate>=0.30.0" "evaluate>=0.4.2" sentencepiece
# Optional (for ROUGE): pip install rouge-score

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import random
import gc

import torch
import time
import json
from functools import reduce

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)

from utils import (
    tokenize_dataset,
    DataCollatorForCausalLMWithLabelMask,
    split_even_hf_dataset,
    select_fraction_subset,
    get_parameters_from_model,
    set_parameters_to_model,
    make_local_trainer,
    find_latest_checkpoint,
    save_results_xlsx,
    remove_empty_dirs,
    create_mask,
    params_copy,
    shatter_train_one_round,
    evaluate_clients_rouge,
    evaluate_clients_loss_ppl,
    assignment_to_bool_masks,
)



# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B", #default="gpt2-xl",
                        help="gpt2-xl or EleutherAI/gpt-j-6B")
    parser.add_argument("--output_dir", type=str, default="./outputs_shatter_testgptneo")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_source_len", type=int, default=800)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=1024)  # gpt2 context=1024; GPT-J supports 2048
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5) # try smaller e.g., 1e-5
    parser.add_argument("--warmup_ratio", type=float, default=0.03) # try larger e.g., 0.06
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--eval_rouge_samples", type=int, default=20)
    parser.add_argument("--device_idx", type=int, default=0, help="GPU device index (if using CUDA)")
    parser.add_argument("--tot_samples", type=int, default=100, help="Total samples to use from CNN/DM")
    parser.add_argument("--client_training_samples", type=int, default=8, help="Number of training samples per client.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run eval/MIA on a trained checkpoint")
    parser.add_argument("--ckp", type=str, default="", help="Path to a HF checkpoint dir to load (e.g., ./outputs_gpt_cnn_dm/checkpoint-86)")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--fl_rounds", type=int, default=2)
    parser.add_argument("--local_epochs", type=float, default=1.0)
    parser.add_argument("--partition_seed", type=int, default=123)
    parser.add_argument("--save_global_each_round", action="store_true")
    parser.add_argument("--client_canary_frac", type=float, default=0.2,help="Fraction of each client's train shard used as canary (members)")
    parser.add_argument("--mia_k_frac", type=float, default=1/3,help="Fraction for loss-threshold MIA (lowest/highest)")
    parser.add_argument("--fold", type=int, default=0, help="Experiment fold number (for logging)")
    parser.add_argument("--n_aggregators", type=int, default=2, help="Number of aggregators (for mask generation)")
    parser.add_argument("--run_sia", action="store_true", default=True, help="Run Source Inference Attack each round (text SIA).")
    parser.add_argument("--sia_per_client_max", type=int, default=32, help="Max SIA samples per client (from client val split). Use small for speed.")
    parser.add_argument("--degree_graph", type=int, default=3, help="Peers per split (per client) for SHATTER.")
    args = parser.parse_args()
    
    
    # update seed
    args.seed = args.seed + args.fold
    args.partition_seed = args.partition_seed + args.fold
    
    # Set total number of samples and split proportions
    train_prop = 0.7
    val_prop = 0.15
    if args.client_training_samples > 0: 
        total_samples = int(round(args.client_training_samples * args.n_clients * 1 / train_prop))
    else: 
        total_samples = args.tot_samples
    print(f"Using total {total_samples} samples from CNN/DailyMail")
    
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
    
    # Before rounds: initialize client params (all start from same initial/global)
    global_params = get_parameters_from_model(model)  # you already have this
    client_full_params = [params_copy(global_params) for _ in range(args.n_clients)]

    # Build a single shared mask assignment for all clients (stable across rounds)
    masks_assignment = create_mask(global_params, args.n_aggregators, seed=args.seed)  # per-layer int arrays

    best_ckpt_dir = None
    best_val = float("inf")

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

    train_count = int(round(total_samples * train_prop))
    val_count = int(round(total_samples * val_prop))
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
        del raw_limited, proc_mia, proc

    collator = DataCollatorForCausalLMWithLabelMask(tokenizer, pad_to_multiple_of=8)
    del client_canaries, noncanary_splits, client_noncanaries, canary_ds, non_canary_ds, raw
    client_canaries, noncanary_splits, client_noncanaries = None, None, None

    # ----------------- Training -----------------
    print(f"\nTotal train examples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    del train_ds, val_ds
    print("\033[93m\nStarting Federated Averaging training...\033[0m")
    # Track best checkpoint directory across rounds
    best_ckpt_dir = None
    best_round = None
    if not args.skip_train:
        # Initialize "global" weights from the current model
        global_params = get_parameters_from_model(model)

        best_w_loss = float("inf")
        patience = args.patience
        no_improve = 0
        results = {"early_stop_reached": False, "early_stop_round": None}
        # results["eris"] = {"d": d, "k": eris_k, "gamma": eris_gamma}
        masks_bool = assignment_to_bool_masks(masks_assignment, args.n_aggregators)

        for rnd in range(1, args.fl_rounds + 1):
            print(f"\033[92m\n--> FL ROUND {rnd}/{args.fl_rounds}\033[0m")

            client_results: List[Tuple[List[np.ndarray], int]] = []
            results[f"round_{rnd}"] = {}

            client_full_params, global_params = shatter_train_one_round(
                rnd=rnd,
                model_src=model_src,
                tokenizer=tokenizer,
                collator=collator,
                args=args,
                device=device,
                client_trains=client_trains,
                client_canaries=client_canaries,
                client_noncanaries=client_noncanaries,
                client_full_params=client_full_params,
                masks_assignment=masks_assignment,
                n_splits=args.n_aggregators,           # set to 2 for your case
                degree_graph=args.degree_graph,
                results_sink=results,
                masks_bool=masks_bool,
            )
            
            # Cleanup GPU RAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # ----------------- Source Inference Attack (SIA) across clients -----------------
            results[f"round_{rnd}"]["sia"] = {"accuracy": float("nan"), "confusion": []} # not for the moment
                
            # ---- NEW: per-round validation over client models ----
            print("\033[93m\nEvaluating validation loss/ppl across client models...\033[0m")
            val_eval = evaluate_clients_loss_ppl(
                model_src=model_src,
                tokenizer=tokenizer,
                collator=collator,
                device=device,
                client_params_list=client_full_params,
                datasets_per_client=client_vals,    # each client on its own validation shard
                args=args,
                tag=f"round_{rnd}_val"
            )
            results[f"round_{rnd}"]["val"] = val_eval
            print(f"\033[92m[Round {rnd}] Val mean loss={val_eval['mean_loss']:.4f} | mean ppl={val_eval['mean_ppl']:.2f}\033[0m")
            results[f"round_{rnd}"]["weighted_val_loss"] = val_eval['mean_loss']

            # # Early stopping on weighted val loss (like your example)
            # w_val_loss = val_eval["mean_loss"]
            # improved = w_val_loss < best_w_loss
            # if improved:
            #     best_w_loss = w_val_loss
            #     no_improve = 0
            #     # Save the current best global model (checkpoint-style)
            #     best_dir = os.path.join(args.output_dir, f"global_round_{rnd}")
            #     os.makedirs(best_dir, exist_ok=True)
            #     model.save_pretrained(best_dir)
            #     tokenizer.save_pretrained(best_dir)
            #     best_ckpt_dir = best_dir
            #     best_round = rnd
            # else:
            #     no_improve += 1

            # if args.save_global_each_round:
            #     round_dir = os.path.join(args.output_dir, f"global_round_{rnd}_always")
            #     os.makedirs(round_dir, exist_ok=True)
            #     model.save_pretrained(round_dir)
            #     tokenizer.save_pretrained(round_dir)

            # if no_improve >= patience:
            #     print(f"\033[91mEarly stopping: no improvement for {patience} rounds.\033[0m")
            #     results["early_stop_reached"] = True
            #     results["early_stop_round"] = rnd
            #     break


    # ----------------- Final Eval: client-wise on TEST -----------------
    print("\033[93m\nFinal evaluation on test set (averaged across client models)...\033[0m")

    # Evaluate loss/ppl: each client model on the same global test_ds, then average
    test_eval = evaluate_clients_loss_ppl(
        model_src=model_src,
        tokenizer=tokenizer,
        collator=collator,
        device=device,
        client_params_list=client_full_params,     # final client models after last round
        datasets_per_client=[test_ds] * args.n_clients,
        args=args,
        tag="test"
    )
    print(f"[Test] mean loss: {test_eval['mean_loss']:.4f} | mean ppl: {test_eval['mean_ppl']:.2f}")
    results["test"] = test_eval
    results["test_loss"] = test_eval['mean_loss']
    results["test_ppl"] = test_eval['mean_ppl']

    # ROUGE averaged across clients
    print("\033[93m\nEvaluating ROUGE on test set (averaged across clients)...\033[0m")
    rouge_eval = evaluate_clients_rouge(
        model_src=model_src,
        tokenizer=tokenizer,
        device=device,
        collator=collator,
        client_params_list=client_full_params,
        test_ds=test_ds,
        args=args,
    )
    print("ROUGE (mean across clients):", {k: round(v, 4) for k, v in rouge_eval["mean"].items()})
    results["rouge_mean_across_clients"] = rouge_eval["mean"]
    results["rouge_per_client"] = rouge_eval["per_client"]
    results["rouge"] = rouge_eval["mean"]

    # Save results summary
    results_path = os.path.join(args.output_dir, f"results_summary_F{args.fold}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)  

    # Compute max_rounds from recorded rounds (avoids None issues)
    # max_rounds = 0
    # try:
    #     max_rounds = max(int(k.split("_")[1]) for k in results.keys() if k.startswith("round_"))
    # except ValueError:
    #     max_rounds = 0
    max_rounds = args.fl_rounds + 1

    xlsx_path = os.path.join(args.output_dir, f"results_summary_F{args.fold}.xlsx")
    save_results_xlsx(results, xlsx_path, max_rounds=max_rounds)
    print(f"[OK] Wrote {results_path} and {xlsx_path}")

    # XLSX export with key metrics
    # xlsx_path = os.path.join(args.output_dir, f"results_summary_F{args.fold}.xlsx")
    # save_results_xlsx(results, xlsx_path, max_rounds=best_round+1)
    # print(f"[OK] Wrote {results_path} and {xlsx_path}")

    # save figure
    # fig_path = os.path.join(args.output_dir, f"images/training_mia_trends_ERIS_F{args.fold}.pdf")
    # plot_training_and_mia(results, fig_path, title=None, max_rounds=None) # None so it will plot all rounds
    # print(f"[OK] Saved trends figure to {fig_path}")
    
    # Cleanup: remove intermediate round checkpoints
    remove_empty_dirs(args.output_dir, ["fl_eval_round_*", "fl_r*_c*"])
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\033[90mTotal training time: {end_time - start_time:.2f} seconds\033[0m")




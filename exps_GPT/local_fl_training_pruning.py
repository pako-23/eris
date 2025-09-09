# finetune_cnn_dm_gpt.py
# pip install -U "transformers>=4.42.0" "datasets>=2.19.0" "accelerate>=0.30.0" "evaluate>=0.4.2" sentencepiece
# Optional (for ROUGE): pip install rouge-score

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import math
import argparse
import numpy as np
from typing import List, Tuple
from datasets import load_dataset, DatasetDict

import time
import json
import torch

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
    DataCollatorForCausalLMWithOwner,
    evaluate_rouge,
    SiaTextConcatDataset,
    split_even_hf_dataset,
    select_fraction_subset,
    get_parameters_from_model,
    set_parameters_to_model,
    fedavg_weighted,
    make_local_trainer,
    evaluate_weighted_val_loss,
    run_simple_mia,
    run_sia_attack_llm,
    find_latest_checkpoint,
    save_results_xlsx,
    plot_training_and_mia,
    remove_empty_dirs,
    prune_largest_grads, 
)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2", #default="gpt2-xl", gpt2 for testing
                        help="gpt2-xl or EleutherAI/gpt-j-6B")
    parser.add_argument("--output_dir", type=str, default="./outputs_gpt_cnn_dm_test_sia")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_source_len", type=int, default=800)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=1024)  # gpt2 context=1024; GPT-J supports 2048
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=8)
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
    parser.add_argument("--client_training_samples", type=int, default=64, help="Number of training samples per client.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run eval/MIA on a trained checkpoint")
    parser.add_argument("--ckp", type=str, default="", help="Path to a HF checkpoint dir to load (e.g., ./outputs_gpt_cnn_dm/checkpoint-86)")
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--fl_rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=float, default=1.0)
    parser.add_argument("--partition_seed", type=int, default=123)
    parser.add_argument("--save_global_each_round", action="store_true")
    parser.add_argument("--client_canary_frac", type=float, default=0.2,help="Fraction of each client's train shard used as canary (members)")
    parser.add_argument("--mia_k_frac", type=float, default=1/3,help="Fraction for loss-threshold MIA (lowest/highest)")
    parser.add_argument("--fold", type=int, default=0, help="Experiment fold number (for logging)")
    parser.add_argument("--pruning_rate", type=float, default=0.3, help="Fraction of largest-magnitude gradients to prune (0,1)")
    parser.add_argument("--run_sia", action="store_true", default=True, help="Run Source Inference Attack each round (text SIA).")
    parser.add_argument("--sia_per_client_max", type=int, default=32, help="Max SIA samples per client (from client val split). Use small for speed.")
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

        # ----------------- SIA dataset/loader (use client-specific VALIDATION to avoid mixing with training) -----------------
        print("\nBuilding SIA set from client validation shards...")
        sia_dataset = SiaTextConcatDataset(
            per_client_splits=client_vals,
            per_client_max=args.sia_per_client_max,
            seed=args.partition_seed
        )
        sia_collator = DataCollatorForCausalLMWithOwner(tokenizer, pad_to_multiple_of=8)
        sia_loader   = torch.utils.data.DataLoader(
            sia_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=sia_collator
        )
        print(f"SIA set size: {len(sia_dataset)} examples (≈{args.sia_per_client_max} per client)")

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

        # ----------------- SIA dataset/loader (use client-specific VALIDATION to avoid mixing with training) -----------------
        print("\nBuilding SIA set from client validation shards...")
        sia_dataset = SiaTextConcatDataset(
            per_client_splits=client_vals,
            per_client_max=args.sia_per_client_max,
            seed=args.partition_seed
        )
        sia_collator = DataCollatorForCausalLMWithOwner(tokenizer, pad_to_multiple_of=8)
        sia_loader   = torch.utils.data.DataLoader(
            sia_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=sia_collator
        )
        print(f"SIA set size: {len(sia_dataset)} examples (≈{args.sia_per_client_max} per client)")

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
    best_round = None
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
                print(f"\033[93mClient {cid} - Round {rnd}: local training on {len(client_trains[cid])} samples\033[0m")
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
                local_trainer = make_local_trainer(
                    local_model, client_trains[cid], collator, tokenizer, args, round_idx=rnd, client_id=cid
                )
                local_trainer.train()
                
                # ----------------- Gradient Pruning -----------------
                # Current global weights (the "starting point" on this round)
                params_in = global_params
                # Client's weights after local training
                params_out = get_parameters_from_model(local_model)
                # Compute "gradients" as weight deltas
                grads = [p_out - p_in for p_in, p_out in zip(params_in, params_out)]
                pruned_grads, pstats = prune_largest_grads(grads, pruning_rate=args.pruning_rate)
                # Recompose the pruned client weights to be sent
                params_out_pruned = [p_in + g for p_in, g in zip(params_in, pruned_grads)]
                client_params_np = params_out_pruned
                # Load pruned weights back into local_model (only if you want local model == pruned)
                set_parameters_to_model(local_model, client_params_np)

                # (Optional) log pruning stats
                results[f"round_{rnd}"][f"client_{cid}"]["pruning"] = {"enabled": True, **pstats}
                print(f"[PRUNE] Client {cid}: rate={args.pruning_rate:.2f} "
                    f"thr={pstats['threshold']:.4e} kept≈{pstats['kept_frac']:.2%}")
                
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
                del local_trainer, local_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ----------------- Source Inference Attack (SIA) across clients -----------------
            if args.run_sia and len(sia_dataset) > 0:
                print("\033[93mRunning Source Inference Attack (SIA) on validation-based SIA set\033[0m")
                # Reuse the global 'model' object as probe; we'll overwrite weights per client
                sia_metrics = run_sia_attack_llm(
                    probe_model=model,
                    sia_loader=sia_loader,
                    client_params=client_results,   # list of (weights_np, n_samples)
                    device=device,
                    set_params_fn=set_parameters_to_model,
                )
                print(f"\033[92mSIA accuracy this round: {sia_metrics['accuracy']:.4f}\033[0m")
                results[f"round_{rnd}"]["sia"] = sia_metrics
            else:
                results[f"round_{rnd}"]["sia"] = {"accuracy": float("nan"), "confusion": []}

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
            # aggregate MIA results
            avg_mia_acc = np.mean([results[f"round_{rnd}"][f"client_{cid}"]["mia"]["accuracy"] for cid in range(n_clients)])
            max_mia_acc = np.max([results[f"round_{rnd}"][f"client_{cid}"]["mia"]["accuracy"] for cid in range(n_clients)])
            results[f"round_{rnd}"]["mia"] = {"avg_accuracy": avg_mia_acc, "max_accuracy": max_mia_acc}
            print(f"\033[92m✅ Round {rnd}: MIA avg_acc={avg_mia_acc*100:.2f}% | max_acc={max_mia_acc*100:.2f}%\033[0m")

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
                best_round = rnd
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
    save_results_xlsx(results, xlsx_path, max_rounds=best_round+1)
    print(f"[OK] Wrote {results_path} and {xlsx_path}")

    # save figure
    fig_path = os.path.join(args.output_dir, f"images/training_mia_trends_PriPrune_F{args.fold}.pdf")
    plot_training_and_mia(results, fig_path, title=None, max_rounds=None) # None so it will plot all rounds
    print(f"[OK] Saved trends figure to {fig_path}")
    
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

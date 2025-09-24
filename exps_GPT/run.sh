#!/bin/sh -eu

n_folds=3
n_clients=10
fl_rounds=2  # 2, change to 3 with 16 samples
local_epochs=2
gradient_accumulation_steps=8  #8, change to 4 with 16 samples
model_name="EleutherAI/gpt-neo-1.3B"  # or gpt2-xl, gpt2, 
output_dir="./outputs_gpt_neo_1.3B_DP"
client_training_samples=(32 64 128) # (4 8 16 32 64 128)



# FedAvg
method="FedAvg"
echo -e "\n\033-- FedAvg --\033[0m"
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $client_training_samples, folds: $n_folds\033[0m"

for n_samples in "${client_training_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python fl_training.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir --local_epochs $local_epochs --gradient_accumulation_steps $gradient_accumulation_steps

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir $output_dir --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done




# ERIS
method="ERIS"
echo -e "\n\033-- ERIS --\033[0m"
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $client_training_samples, folds: $n_folds\033[0m"

for n_samples in "${client_training_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python fl_training_eris.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir --local_epochs $local_epochs --gradient_accumulation_steps $gradient_accumulation_steps

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir $output_dir --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done




# FedAvg+DP
method="FedAvg+DP"
echo -e "\n\033-- FedAvg+DP --\033[0m"
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $client_training_samples, folds: $n_folds\033[0m"

for n_samples in "${client_training_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python fl_training_dp.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir --local_epochs $local_epochs --gradient_accumulation_steps $gradient_accumulation_steps

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir $output_dir --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done




# PriPrune
method="PriPrune"
echo -e "\n\033-- PriPrune --\033[0m"
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $client_training_samples, folds: $n_folds\033[0m"

for n_samples in "${client_training_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python fl_training_pruning.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir --local_epochs $local_epochs --gradient_accumulation_steps $gradient_accumulation_steps

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir $output_dir --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done



# SoteriaFL
method="SoteriaFL"
echo -e "\n\033-- SoteriaFL --\033[0m"
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $client_training_samples, folds: $n_folds\033[0m"

for n_samples in "${client_training_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python fl_training_soteriafl.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir --local_epochs $local_epochs --gradient_accumulation_steps $gradient_accumulation_steps

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir $output_dir --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done



# Shatter
method="Shatter"
echo -e "\n\033[1;36m-- Shatter --\033[0m"
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $client_training_samples, folds: $n_folds\033[0m"

for n_samples in "${client_training_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python fl_training_shatter.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir --local_epochs $local_epochs --gradient_accumulation_steps $gradient_accumulation_steps

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir $output_dir --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done

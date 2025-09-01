#!/bin/sh -eu

n_folds=3
n_clients=5
fl_rounds=10
model_name="EleutherAI/gpt-neo-1.3B"  # or gpt2-xl, gpt2, 
output_dir="./outputs_gpt_neo_1.3B"
# client_training_samples=(4 8 16 32 64 128)
client_training_samples=(128)


# FedAvg
method="FedAvg"
echo -e "\n\033-- FedAvg --\033[0m"
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $client_training_samples, folds: $n_folds\033[0m"

for n_samples in "${client_training_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python local_fl_training.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir 

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

        python local_fl_training_eris.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir 

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

        python local_fl_training_dp.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir 

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

        python local_fl_training_pruning.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir 

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

        python local_fl_training_soteriafl.py --model_name $model_name --n_clients $n_clients --client_training_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold --output_dir $output_dir 

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir $output_dir --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done

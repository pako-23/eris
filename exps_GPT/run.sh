#!/bin/sh -eu

n_folds=2
n_clients=3
fl_rounds=2
method="FedAvg"
model_name="gpt2"  # or gpt2-xl
tot_n_samples=(100 500 1000)

# Print the number of clients
echo -e "\n\033[1;36mStart training on GPT with $n_clients clients, method: $method, total number of samples: $tot_n_samples, folds: $n_folds\033[0m"

for n_samples in "${tot_n_samples[@]}"; do
    echo -e "\n\033[1;36mStarting experiment with $n_samples samples\033[0m"

    # Cycle through the folds
    for fold in $(seq 0 $((n_folds - 1))); do
        echo -e "\n\033[1;36mFold $fold - Creating dataset with $n_samples samples\033[0m"

        python local_fl_training.py --model_name "gpt2" --n_clients $n_clients --tot_samples $n_samples --fold $fold --fl_rounds $fl_rounds --fold $fold 

    done

    # Aggregate results
    echo -e "\n\033[1;36mAggregating results from $n_folds folds\033[0m\n"
    python average_results.py --results_dir ./outputs_gpt_cnn_dm_light --n_folds $n_folds --method $method --n_samples $n_samples

    echo -e "\n\033[1;36mFinished training correctly!\033[0m\n"

done




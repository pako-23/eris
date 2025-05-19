#!/bin/sh -eu

# This script is used to run the federated learning experiments for the ERIS framework (with LLMs).
# It starts the server and clients for the federated learning process. The server is started first,
# and then the clients are started. The script also handles cross-validation if specified in the config,
# by averaging the results across multiple folds (different random seeds).
# The script is designed to be run from the command line. It runs across n_exp which determines 
# the number of local training samples for each client.

# Function to extract variable from config
extract_config_var() {
    var_name=$1
    python -c "
import sys
import os

try:
    # Get the current working directory
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    # Import the config module from the public folder
    from public import config as cfg

    # Split the var_name by dots to handle nested keys
    keys = '${var_name}'.split('.')
    value = cfg
    for key in keys:
        if isinstance(value, dict):
            value = value[key]
        else:
            value = getattr(value, key)
    print(value)
except (AttributeError, KeyError) as e:
    print(f'Error: Variable {var_name} not found in config module. ({e})')
    sys.exit(1)
except ImportError:
    print('Error: Unable to import config module from public folder.')
    sys.exit(1)
"
}

# Extract variables using the function
k_folds=$(extract_config_var "k_folds")
dataset_name=$(extract_config_var "dataset_name")
n_clients=$(extract_config_var "experiments.${dataset_name}.clients")
aggregators=$(extract_config_var "experiments.${dataset_name}.splits")
experiments=$(extract_config_var "experiments.${dataset_name}")

# Print the number of clients
echo -e "\n\033[1;36mStart training on $dataset_name with $n_clients clients\033[0m"

# if k_folds > 1, print "Cross validation with k_folds"
if [ "$k_folds" -gt 1 ]; then
    echo -e "\n\033[1;36mCross validation with $k_folds folds\033[0m"
elif [ "$k_folds" -eq 1 ]; then # if k_folds = 1, print "No cross validation"
    echo -e "\n\033[1;36mNo cross validation\033[0m"
fi


for exp_n in $(seq 0 5); do

    # Cycle through the folds
    for fold in $(seq 1 $k_folds); do
        if [ $k_folds -gt 1 ]; then
            echo -e "\n\033[1;36mFold $fold\033[0m"
        fi
        
        # Creating dataset
        cd ../data
        python client_datasets_split.py --n_clients $n_clients --dataset $dataset_name --seed $fold
        cd ../eris_llm
        pkill -9 -f coordinator_llm.py
        pkill -9 -f client_llm.py
        sleep 2

        echo -e "\n\033[1;36mStarting server with model \033[0m\n"

        # Start training
        ./coordinator_llm.py --dataset_name "$dataset_name" --exp_n "$exp_n" &
        # sleep 0.5
        sleep 1

        for i in $(seq 1 "$n_clients"); do
        if [ "$i" -le "$aggregators" ]; then
                ./client_llm.py --aggregator                   \
                --id "$i"                      \
                --dataset "$dataset_name"      \
                --shard "../data/client_datasets/IID_data_client_$i" \
                --fold "$fold" \
                --exp_n "$exp_n" &
        else
            ./client_llm.py --id "$i"                 \
                --dataset "$dataset_name" \
                --shard "../data/client_datasets/IID_data_client_$i" \
                --exp_n "$exp_n" &
        fi
        sleep 0.2
        done

        while pgrep -f ./client_llm.py >/dev/null; do
            sleep 5
        done

        pkill -9 -f coordinator_llm.py
        pkill -9 -f client_llm.py
        sleep 2
    done

    # Aggregate results
    if [ $k_folds -gt 1 ]; then
        echo -e "\n\033[1;36mAveraging results from cross-validation...\033[0m\n"
        cd ../public
        python average_results.py --strategy "eris_llm" --dataset $dataset_name --exp_n "$exp_n"
        sleep 1
    fi

    echo -e "\n\033[1;36mFinished training correctly on $dataset_name with $n_clients clients\033[0m\n"

done

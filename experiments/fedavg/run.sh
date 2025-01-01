#!/bin/sh

# This code is usually called from cross_validation.sh, and it starts the server and clients 
# for the federated learning process. The server is started first, and then the clients are started.

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

# Print the number of clients
echo -e "\n\033[1;36mStart training on $dataset_name with $n_clients clients\033[0m"

# if k_folds > 1, print "Cross validation with k_folds"
if [ $k_folds -gt 1 ]; then
    echo -e "\n\033[1;36mCross validation with $k_folds folds\033[0m"
fi
# if k_folds = 1, print "No cross validation"
if [ $k_folds -eq 1 ]; then
    echo -e "\n\033[1;36mNo cross validation\033[0m"
fi

# Cycle through the folds
for fold in $(seq 1 $k_folds); do
    if [ $k_folds -gt 1 ]; then
        echo -e "\n\033[1;36mFold $fold\033[0m"
    fi
    
    # Creating dataset
    cd ../data
    python client_datasets_split.py --n_clients $n_clients --dataset $dataset_name --seed $fold
    cd ../fedavg

    echo -e "\n\033[1;36mStarting server with model \033[0m\n"

    python server.py --fold $fold --dataset $dataset_name &
    sleep 2  # Sleep for 2s to give the server enough time to start

    for i in $(seq 1 $n_clients); do
        echo "Starting client ID $i"
        python client.py --id "$i" --dataset $dataset_name &
    done

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait

    # Clean up
    echo "Fold completed correctly"
    pkill -9 -f server.py
    pkill -9 -f client.py
done

# Aggregate results
if [ $k_folds -gt 1 ]; then
    echo -e "\n\033[1;36mAveraging results from cross-validation...\033[0m\n"
    cd ../public
    python average_results.py --strategy "fedavg" --dataset $dataset_name
    sleep 1
fi

echo -e "\n\033[1;36mFinished training correctly on $dataset_name with $n_clients clients\033[0m\n"

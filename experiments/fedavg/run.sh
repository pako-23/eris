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

    # Print the requested variable
    print(cfg.${var_name})
except AttributeError:
    print('Error: Variable ${var_name} not found in config module.')
    sys.exit(1)
except ImportError:
    print('Error: Unable to import config module from public folder.')
    sys.exit(1)
"
}

# Extract variables using the function
k_folds=$(extract_config_var "k_folds")
n_clients=$(extract_config_var "client_number")
dataset_name=$(extract_config_var "dataset_name")

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

    python server.py --fold $fold &
    sleep 2  # Sleep for 2s to give the server enough time to start

    for i in $(seq 1 $n_clients); do
        echo "Starting client ID $i"
        python client.py --id "$i" &
    done

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait

    # Clean up
    echo "Fold completed correctly"
    trap - SIGTERM 
    # pkill -u dario -f client.py
    pkill -u dariofenoglio -f client.py
    # pkill -u dario -f server.py
    pkill -u dariofenoglio -f sever.py

done

# Aggregate results
if [ $k_folds -gt 1 ]; then
    cd ../public
    python average_results.py --strategy "fedavg" 
    sleep 1
fi

echo -e "\n\033[1;36mFinished training correctly on $dataset_name with $n_clients clients\033[0m\n"

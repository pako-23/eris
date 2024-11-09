#!/bin/bash

# This code is usually called from cross_validation.sh, and it starts the server and clients 
# for the federated learning process. The server is started first, and then the clients are started.

# import client_number from config.py
k_folds=$(python -c "from config import k_folds; print(k_folds)")
n_clients=$(python -c "from config import client_number; print(client_number)")
dataset_name=$(python -c "from config import dataset_name; print(dataset_name)")

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
    cd ../federated_Learning

    echo -e "\n\033[1;36mStarting server with model \033[0m\n"

    python server_FedAvg.py --fold $fold &
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
    python average_results.py 
    sleep 1
fi

echo -e "\n\033[1;36mFinished training correctly on $dataset_name with $n_clients clients\033[0m\n"




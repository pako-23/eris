#!/bin/sh -eu

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
fold=1
k_folds=$(extract_config_var "k_folds")
n_clients=$(extract_config_var "client_number")
dataset_name=$(extract_config_var "dataset_name")
echo -e "\n\033[1;36mStart training on $dataset_name with $n_clients clients and k-folds $k_folds\033[0m"

echo -e "\033[1;36m\nData Generation\033[0m"
cd ../data
python client_datasets_split.py --n_clients $n_clients --dataset $dataset_name --seed $fold
cd ../eris

./coordinator.py &
sleep 0.5

for i in $(seq 1 $n_clients); do
    ./client.py "$(expr "50051" + "$i")" "$(expr "5555" + "$i")" --id $i &
    sleep 0.2
done
for i in $(seq 1 $n_clients); do
    ./client.py &
done


while pgrep -f ./client.py >/dev/null; do
    sleep 5
done

pkill -9 -f coordinator.py

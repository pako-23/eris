"""
This script loads evaluation metrics from multiple test folds of a federated learning experiment,
computes the mean and standard deviation for each metric (e.g., accuracy, F1-score), and saves
the aggregated results to an Excel file.

It supports dataset-specific renaming and cleans up temporary files.
To run, specify the strategy, dataset, experiment index, and optional scaling flag.
"""

import numpy as np
import os
import pandas as pd
import config as cfg
import argparse

#get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def calculate_mean_std_metrics(metrics, model_name=None):
    # Initialize a dictionary to hold the means of all keys
    mean_std_metrics = {}

    # Extract the keys from the first entry in the metrics list
    keys = metrics[0].keys()

    for key in keys:
        # Compute the mean for the scalar values across all entries for this key
        mean_metrics = np.mean([metric[key] for metric in metrics])
        std_metrics = np.std([metric[key] for metric in metrics])
        
        if model_name == 'LinearModel':
            if key == 'f1_score':
                key = 'mse'
            elif key == 'accuracy':
                key = 'mae'
        
        mean_std_metrics[f'{key}_mean'] = mean_metrics
        mean_std_metrics[f'{key}_std'] = std_metrics
        
    return mean_std_metrics



# get arguments
parser = argparse.ArgumentParser(description='Average results')
parser.add_argument('--strategy', type=str, default='fedavg', help='Strategy to use')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
parser.add_argument("--exp_n", type=int, help="exp number", default=0)
parser.add_argument("--scaling_dp", type=int, help="scaling dp", default=0)
args = parser.parse_args()
config = cfg.experiments[args.dataset]

default_path = f"{config["model_name"]}/{config["dataset"]}"

# Load metrics from all folds
metrics = []
for i in range(cfg.k_folds):
    # Load metrics
    metrics.append(
        np.load(f'../{args.strategy}/test_metrics_fold_{i+1}.npy',
                allow_pickle=True
                ).item()
        )

# Delete files
for i in range(cfg.k_folds):
    os.remove(f'../{args.strategy}/test_metrics_fold_{i+1}.npy')

# Calculate the mean metrics
result = calculate_mean_std_metrics(metrics, model_name=config["model_name"])

# Save the mean metrics to a file
result_pd = pd.DataFrame(result, index=[0])
print(f'../{args.strategy}/results/{default_path}/metrics_{args.strategy}_S{config["client_train_samples"][args.exp_n]}_R{config["rounds"][args.exp_n]}_C{config["clients"]}_A{config["splits"]}.xlsx')
result_pd.to_excel(f'../{args.strategy}/results/{default_path}/metrics_{args.strategy}_S{config["client_train_samples"][args.exp_n]}_R{config["rounds"][args.exp_n]}_C{config["clients"]}_A{config["splits"]}_scaling_dp{args.scaling_dp}.xlsx', index=False)



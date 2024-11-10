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


model_names = {
    "mnist": "LeNet5",
    "cifar10": "ResNet9",
    "fmnist": "LeNet5",
    "breast": "MLP",  
    "diabetes": "MLP",
    "adult": "MLP",
    "airline": "LinearModel",
    "lsst": "TransformerModelFlexible",
}


# get arguments
parser = argparse.ArgumentParser(description='Average results')
parser.add_argument('--strategy', type=str, default='fedavg', help='Strategy to use')
args = parser.parse_args()

default_path = f"{model_names[cfg.dataset_name]}/{cfg.dataset_name}"

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
result = calculate_mean_std_metrics(metrics, model_name=model_names[cfg.dataset_name])

# Save the mean metrics to a file
result_pd = pd.DataFrame(result, index=[0])
result_pd.to_excel(f'../{args.strategy}/results/{default_path}/mean_std_test_metrics.xlsx', index=False)



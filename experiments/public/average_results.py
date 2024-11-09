import numpy as np
import os
import pandas as pd
import config as cfg


def calculate_mean_std_metrics(metrics):
    # Initialize a dictionary to hold the means of all keys
    mean_std_metrics = {}
    n_clients = len(metrics[0]['loss'])
    mean_std_metrics['-'] = [f"Client {i}" for i in range(n_clients)]

    # Extract the keys from the first entry in the metrics list
    keys = metrics[0].keys()

    for key in keys:
        # Check if the value corresponding to the key is a list
        if isinstance(metrics[0][key], list):
            # Compute the mean for each element across all entries for this key
            mean_value = np.mean([metric[key] for metric in metrics], axis=0).tolist()
            std_value = np.std([metric[key] for metric in metrics], axis=0).tolist()
            mean_std_metrics[f'{key}_mean'] = mean_value
            mean_std_metrics[f'{key}_std'] = std_value
        else:
            # Compute the mean for the scalar values across all entries for this key
            mean_value = np.mean([metric[key] for metric in metrics])
            std_value = np.std([metric[key] for metric in metrics])
            mean_std_metrics[f'{key}_mean'] = [mean_value] + [None]*(n_clients-1)
            mean_std_metrics[f'{key}_std'] = [std_value] + [None]*(n_clients-1)
        
    return mean_std_metrics


# Load metrics from all folds
metrics = []
for i in range(cfg.k_folds):
    # Load metrics
    metrics.append(
        np.load(f'{cfg.strategy}/results/{cfg.default_path}/test_metrics_fold_{i}.npy',
                allow_pickle=True
                ).item()
        )

# Delete files
for i in range(cfg.k_folds):
    os.remove(f'{cfg.strategy}/results/{cfg.default_path}/test_metrics_fold_{i}.npy')

# Calculate the mean metrics
result = calculate_mean_std_metrics(metrics)

# Save the mean metrics to a file
result_pd = pd.DataFrame(result)
result_pd.to_excel(f'{cfg.strategy}/results/{cfg.default_path}/mean_std_test_metrics_{cfg.non_iid_type}_{cfg.args}.xlsx', index=False)



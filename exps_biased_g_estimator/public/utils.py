"""
This module provides utility functions and classes to support federated learning experiments.
It includes:
- Device and seed management
- Folder and checkpoint handling
- Training and evaluation plotting
- Local Differential Privacy (LDP) clipping and noise mechanisms
- Metrics aggregation across clients
- Auditing functions for differential privacy (DP) leakage
- Visualization tools (metrics over rounds, gradients, histograms)

Used by both server and client components in the Flower FL framework.
"""

import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import csv
from sklearn.metrics import accuracy_score, f1_score
from flwr.common import NDArrays

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import config as cfg



# create folder if not exists
def create_delede_folders(config):
    dataset_name = config['dataset']
    client_number = config['clients']
    predictor_name = config['model_name']
    # Results
    if not os.path.exists(f"results/{predictor_name}/{dataset_name}"):
        os.makedirs(f"results/{predictor_name}/{dataset_name}")
    # Histories
    if not os.path.exists(f"histories/{predictor_name}/{dataset_name}"):
        os.makedirs(f"histories/{predictor_name}/{dataset_name}")
    else:
        # remove the client folders
        for c in range(client_number):
            os.system(f"rm -r histories/{predictor_name}/{dataset_name}/client_{c+1}")  
    # Checkpoints
    if not os.path.exists(f"checkpoints/{predictor_name}/{dataset_name}"):
        os.makedirs(f"checkpoints/{predictor_name}/{dataset_name}")
    # Images
    if not os.path.exists(f"images/{predictor_name}/{dataset_name}"):
        os.makedirs(f"images/{predictor_name}/{dataset_name}")
    
    
# define device
def check_gpu(seed=0, print_info=True, client_id=1):
    torch.manual_seed(seed)
    if cfg.gpu == -1:
        device = 'cpu'
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
        if print_info:
            print("CUDA is available")
            
        if cfg.gpu == -2: # multiple gpu
            assert client_id >=0, "client_id must be passed to select the respective GPU"
            n_total_gpus = torch.cuda.device_count() 
            device = 'cuda:' + str(int(client_id % n_total_gpus)) 
            
        else:
            device = 'cuda:' + str(cfg.gpu)
    elif torch.backends.mps.is_available():
        if print_info:
            print("MPS is available")
        device = torch.device("mps")
        torch.mps.manual_seed(seed)
    else:
        if print_info:
            print("CUDA is not available")
        device = 'cpu'
    return device


def print_num_parameters(model):
    # Sum the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print in red color
    print(f"\033[91mNumber of trainable parameters: {total_params}\033[0m")

def set_seed(seed):
    # Set seed for torch
    torch.manual_seed(seed)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)


# plot and save plot on server side
def plot_loss_and_accuracy(metrics_distributed, config, exp_n=1, fold=1, show=True, eris=False):
    
    if eris:
        m = {
            'loss' : metrics_distributed['Loss'].values.tolist(),
            'accuracy': metrics_distributed['Accuracy'].values.tolist(),
            'f1_score': metrics_distributed['f1_score'].values.tolist()
        }
        
        if cfg.privacy_audit:            
            m['accuracy_mia'] = metrics_distributed['MIA Accuracy'].values.tolist()
            m['privacy_estimate'] = metrics_distributed['Privacy'].values.tolist()
            m['accumulative_accuracy_mia'] = metrics_distributed['Accumulative MIA Accuracy'].values.tolist()
            m['accumulative_privacy_estimate'] = metrics_distributed['Accumulative Privacy'].values.tolist()
            m['accuracy_mia_mean'] = metrics_distributed['MIA Accuracy Mean'].values.tolist()
            m['privacy_estimate_mean'] = metrics_distributed['Privacy Mean'].values.tolist()
            m['accumulative_accuracy_mia_mean'] = metrics_distributed['Accumulative MIA Accuracy Mean'].values.tolist()
            m['accumulative_privacy_estimate_mean'] = metrics_distributed['Accumulative Privacy Mean'].values.tolist()
        
        metrics_distributed = m
    
    # Read arguments
    rounds = config['rounds'][exp_n]
    samples = config['client_train_samples'][exp_n]
    predictor_name = config['model_name']
    loss = metrics_distributed['loss']
    accuracy = metrics_distributed['accuracy']
    f1_score = metrics_distributed['f1_score']

    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # Create a 2x2 grid of subplots
    if cfg.privacy_audit:
        accuracy_mia = metrics_distributed['accuracy_mia']
        privacy_estimate = metrics_distributed['privacy_estimate']
        acc_accuracy_mia = metrics_distributed['accumulative_accuracy_mia']
        acc_privacy_estimate = metrics_distributed['accumulative_privacy_estimate']
        accuracy_mia_mean = metrics_distributed['accuracy_mia_mean']
        privacy_estimate_mean = metrics_distributed['privacy_estimate_mean']
        acc_accuracy_mia_mean = metrics_distributed['accumulative_accuracy_mia_mean']
        acc_privacy_estimate_mean = metrics_distributed['accumulative_privacy_estimate_mean']
        
        fig, axs = plt.subplots(3, 2, figsize=(21, 10))
        
        # -------------------- Subplot 1: Loss --------------------
        ax = axs[0, 0]
        ax.plot(loss, label='Loss', color='blue')
        
        # Identify and plot the minimum loss point
        min_loss_index = loss.index(min(loss))
        ax.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=150, label='Min Loss')
        
        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Loss')
        ax.set_title('Loss over Rounds')
        ax.legend()
        ax.grid(True)
        
        # -------------------- Subplot 2: Accuracy & F1 Score --------------------
        ax = axs[0, 1]
        ax.plot(accuracy, label='Accuracy', color='orange')
        ax.plot(f1_score, label='F1 Score', color='green')
        
        # Identify and plot the maximum accuracy and F1 score points
        max_accuracy_index = accuracy.index(max(accuracy))
        max_f1score_index = f1_score.index(max(f1_score))
        ax.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=150, label='Max Accuracy')
        ax.scatter(max_f1score_index, f1_score[max_f1score_index], color='green', marker='*', s=150, label='Max F1 Score')
        
        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Score')
        ax.set_title('Accuracy & F1 Score over Rounds')
        ax.legend()
        ax.grid(True)
        
        # -------------------- Subplot 3: MIA Accuracy --------------------
        ax = axs[1, 0]
        ax.plot(accuracy_mia, label='MIA Accuracy', color='red')
        ax.plot(acc_accuracy_mia, label='Acc. MIA Accuracy', color='black')
        
        # Identify and plot the maximum MIA accuracy point
        max_accuracy_mia_index = accuracy_mia.index(max(accuracy_mia))
        max_acc_accuracy_mia_index = acc_accuracy_mia.index(max(acc_accuracy_mia))
        ax.scatter(max_accuracy_mia_index, accuracy_mia[max_accuracy_mia_index], color='red', marker='*', s=150, label='Max MIA Accuracy')
        ax.scatter(max_acc_accuracy_mia_index, acc_accuracy_mia[max_acc_accuracy_mia_index], color='black', marker='*', s=150, label='Max Acc. MIA Accuracy')

        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('MIA Accuracy')
        ax.set_title('MIA Accuracy over Rounds')
        ax.legend()
        ax.grid(True)
        
        # -------------------- Subplot 4: Privacy Estimate --------------------
        ax = axs[1, 1]
        ax.plot(privacy_estimate, label='Privacy Estimate (Epsilon)', color='purple')
        ax.plot(acc_privacy_estimate, label='Acc. Privacy Estimate (Epsilon)', color='blue')
        
        # Identify and plot the maximum privacy estimate point
        max_privacy_index = privacy_estimate.index(max(privacy_estimate))
        max_acc_privacy_index = acc_privacy_estimate.index(max(acc_privacy_estimate))
        ax.scatter(max_privacy_index, privacy_estimate[max_privacy_index], color='purple', marker='*', s=150, label='Max Privacy Leakage')
        ax.scatter(max_acc_privacy_index, acc_privacy_estimate[max_acc_privacy_index], color='blue', marker='*', s=150, label='Max Acc. Privacy Leakage')
        
        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Epsilon')
        ax.set_title('Privacy Estimate over Rounds')
        ax.legend()
        ax.grid(True)

        # -------------------- Subplot 5: MIA Accuracy Mean --------------------
        ax = axs[2, 0]
        ax.plot(accuracy_mia_mean, label='MIA Accuracy Mean', color='red')
        ax.plot(acc_accuracy_mia_mean, label='Acc. MIA Accuracy Mean', color='black')
        
        # Identify and plot the maximum MIA accuracy point
        max_accuracy_mia_index_mean = accuracy_mia_mean.index(max(accuracy_mia_mean))
        max_acc_accuracy_mia_index_mean = acc_accuracy_mia_mean.index(max(acc_accuracy_mia_mean))
        ax.scatter(max_accuracy_mia_index_mean, accuracy_mia_mean[max_accuracy_mia_index_mean], color='red', marker='*', s=150, label='Max MIA Accuracy Mean')
        ax.scatter(max_acc_accuracy_mia_index_mean, acc_accuracy_mia_mean[max_acc_accuracy_mia_index_mean], color='black', marker='*', s=150, label='Max Acc. MIA Accuracy Mean')
        
        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('MIA Accuracy Mean')
        ax.set_title('MIA Accuracy Mean over Rounds')
        ax.legend()
        ax.grid(True)
        
        # -------------------- Subplot 6: Privacy Estimate Mean --------------------
        ax = axs[2, 1]
        ax.plot(privacy_estimate_mean, label='Privacy Estimate (Epsilon)', color='purple')
        ax.plot(acc_privacy_estimate_mean, label='Acc. Privacy Estimate (Epsilon)', color='blue')
        
        # Identify and plot the maximum privacy estimate point
        max_privacy_index_mean = privacy_estimate_mean.index(max(privacy_estimate_mean))
        max_acc_privacy_index_mean = acc_privacy_estimate_mean.index(max(acc_privacy_estimate_mean))
        ax.scatter(max_privacy_index_mean, privacy_estimate_mean[max_privacy_index_mean], color='purple', marker='*', s=150, label='Max Privacy Leakage')
        ax.scatter(max_acc_privacy_index_mean, acc_privacy_estimate_mean[max_acc_privacy_index_mean], color='blue', marker='*', s=150, label='Max Acc. Privacy Leakage')
        
        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Epsilon')
        ax.set_title('Privacy Estimate Mean over Rounds')
        ax.legend()
        ax.grid(True)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Print summary information
        print(
            f"""
            \n\033[1;34mServer Side\033[0m 
            Minimum Loss: round {min_loss_index + 1}, value {loss[min_loss_index]:.3f} 
            Maximum Accuracy: round {max_accuracy_index + 1}, value {accuracy[max_accuracy_index] * 100:.2f}% 
            Maximum F1 Score: round {max_f1score_index + 1}, value {f1_score[max_f1score_index] * 100:.2f}%
            Maximum MIA Accuracy: round {max_accuracy_mia_index + 1}, value {accuracy_mia[max_accuracy_mia_index] * 100:.2f}%
            Maximum Epsilon (Privacy): round {max_privacy_index + 1}, value {privacy_estimate[max_privacy_index]:.3f}
            Maximum Accumulative MIA Accuracy: round {max_acc_accuracy_mia_index + 1}, value {acc_accuracy_mia[max_acc_accuracy_mia_index] * 100:.2f}%
            Maximum Accumulative Epsilon (Privacy): round {max_acc_privacy_index + 1}, value {acc_privacy_estimate[max_acc_privacy_index]:.3f}
            Maximum MIA Accuracy Mean: round {max_accuracy_mia_index_mean + 1}, value {accuracy_mia_mean[max_accuracy_mia_index_mean] * 100:.2f}%
            Maximum Epsilon (Privacy) Mean: round {max_privacy_index_mean + 1}, value {privacy_estimate_mean[max_privacy_index_mean]:.3f}
            Maximum Accumulative MIA Accuracy Mean: round {max_acc_accuracy_mia_index_mean + 1}, value {acc_accuracy_mia_mean[max_acc_accuracy_mia_index_mean] * 100:.2f}%
            Maximum Accumulative Epsilon (Privacy) Mean: round {max_acc_privacy_index_mean + 1}, value {acc_privacy_estimate_mean[max_acc_privacy_index_mean]:.3f} 
            \n
            """
        )
        
        # Save the figure
        plt.savefig(f"images/{predictor_name}/{config['dataset']}/training_R{rounds}_S{samples}_F{fold}.png")
        
        # Show the plot if required
        if show:
            plt.show()
        
        # Close the plot to free memory
        plt.close()
        
        return min_loss_index + 1, max_accuracy_index + 1
    
    else:

        fig, axs = plt.subplots(1, 2, figsize=(14, 10))
        
        # -------------------- Subplot 1: Loss --------------------
        ax = axs[0]
        ax.plot(loss, label='Loss', color='blue')
        
        # Identify and plot the minimum loss point
        min_loss_index = loss.index(min(loss))
        ax.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=150, label='Min Loss')
        
        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Loss')
        ax.set_title('Loss over Rounds')
        ax.legend()
        ax.grid(True)
        
        # -------------------- Subplot 2: Accuracy & F1 Score --------------------
        ax = axs[1]
        ax.plot(accuracy, label='Accuracy', color='orange')
        ax.plot(f1_score, label='F1 Score', color='green')
        
        # Identify and plot the maximum accuracy and F1 score points
        max_accuracy_index = accuracy.index(max(accuracy))
        max_f1score_index = f1_score.index(max(f1_score))
        ax.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=150, label='Max Accuracy')
        ax.scatter(max_f1score_index, f1_score[max_f1score_index], color='green', marker='*', s=150, label='Max F1 Score')
        
        # Set labels and title
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Score')
        ax.set_title('Accuracy & F1 Score over Rounds')
        ax.legend()
        ax.grid(True)
        
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Print summary information
        print(
            f"""
            \n\033[1;34mServer Side\033[0m 
            Minimum Loss: round {min_loss_index + 1}, value {loss[min_loss_index]:.3f} 
            Maximum Accuracy: round {max_accuracy_index + 1}, value {accuracy[max_accuracy_index] * 100:.2f}% 
            Maximum F1 Score: round {max_f1score_index + 1}, value {f1_score[max_f1score_index] * 100:.2f}%
            Maximum MIA Accuracy: round None, value None%
            Maximum Epsilon (Privacy): round None, value None\n
            """
        )
        
        # Save the figure
        plt.savefig(f"images/{predictor_name}/{config['dataset']}/training_{rounds}_rounds.png")
        
        # Show the plot if required
        if show:
            plt.show()
        
        # Close the plot to free memory
        plt.close()
        
        return min_loss_index + 1, max_accuracy_index + 1
    



# save client metrics
def save_client_metrics(round_num, loss, accuracy, f1_score, client_id=1, history_folder="histories/"):

    # Create folder for client
    folder = history_folder + f"client_{client_id}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # file path
    file_path = folder + f'metrics.csv'
    # Check if the file exists; if not, create it and write headers
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Round', 'Loss', 'Accuracy', 'f1_score'])

        # Write the metrics
        writer.writerow([round_num, loss, accuracy, f1_score])


def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    array_norms = [np.linalg.norm(array.flat) for array in input_arrays]
    # pylint: disable=consider-using-generator
    return float(np.sqrt(sum([norm**2 for norm in array_norms])))

        
def my_compute_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> None:
    """Compute model update (param1 - param2) and clip it.
    
    By FlatClip method of the paper: https://arxiv.org/abs/1710.06963

    Then add the clipped value to param1."""
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2)]
    # clip_inputs_inplace(model_update, clipping_norm)
    
    input_norm = get_norm(model_update)
    # print(f"L2 norm BEFORE clipping: {input_norm:.4f}")

    scaling_factor = min(1, clipping_norm / input_norm)
    
    for ii in range(len(model_update)):
        model_update[ii] *= scaling_factor
    
    # input_norm_after = get_norm(model_update)
    # print(f"L2 norm AFTER  clipping: {input_norm_after:.4f}")

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]
    
    return param1


class LocalDpMod:
    """
    Modifier for local differential privacy.

    This mod clips the client model updates and
    adds noise to the params before sending them to the server.
    
    -------------------
    Parameters:
    clipping_norm: float - The clipping norm for the model updates.
    sensitivity: float - The sensitivity of the model.
    epsilon: float - The epsilon value for the local differential privacy.
    delta: float - The delta value for the local differential privacy.    
    
    """

    def __init__(
        self, clipping_norm: float, epsilon: float, delta: float, client_id: int = 1
    ) -> None:
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        # if sensitivity < 0:
        #     raise ValueError("The sensitivity should be a non-negative value.")

        if epsilon < 0:
            raise ValueError("Epsilon should be a non-negative value.")

        if delta < 0:
            raise ValueError("Delta should be a non-negative value.")

        self.clipping_norm = clipping_norm
        self.sensitivity = clipping_norm
        self.epsilon = epsilon
        self.delta = delta
        self.client_id = client_id

    def __call__(
        self, server_to_client_params: NDArrays, client_to_server_params: NDArrays
    ) -> NDArrays:
        """
        Perform local DP on the client model parameters.
        
        -------------------
        Parameters:
        server_to_client_params: NDArrays - The model parameters from the server.
        client_to_server_params: NDArrays - The model parameters from the client.
        """
        
        # Clip the client update
        params_clipped = my_compute_clip_model_update(
            client_to_server_params,
            server_to_client_params,
            self.clipping_norm,
        )

        # Add noise to model params        
        # std_dev = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon 
        scale = self.sensitivity / self.epsilon       
        for ii in range(len(params_clipped)):
            # params_clipped[ii] += np.random.normal(0, std_dev, params_clipped[ii].shape)
            params_clipped[ii] += np.random.laplace(loc=0.0, scale=scale, size=params_clipped[ii].shape)
                
        return params_clipped

# Plotting the metrics
def plot_mean_std_metrics(plot_metrics, name):
    # Initialize dictionaries to store the mean and std of each variable
    mean_metrics = {}
    std_metrics = {}

    # Initialize keys in mean and std dictionaries
    for key in plot_metrics[0]:
        mean_metrics[key] = []
        std_metrics[key] = []

    # Calculate mean and std
    for key in plot_metrics[0]:  # Assuming all dicts have the same keys
        # Gather data from each entry in plot_metrics for the current key
        data = [entry[key] for entry in plot_metrics]
        # Convert list of lists to a numpy array
        data_array = np.array(data)
        # Compute the mean and std along the first axis (across dictionaries)
        mean_metrics[key] = np.mean(data_array, axis=0)
        std_metrics[key] = np.std(data_array, axis=0)

    # Creating a DataFrame to hold all data points
    data = {
        'Iteration': [],
        'Value': [],
        'Variable': []
    }

    # Extract data for plotting
    for key in plot_metrics[0].keys():
        for index, metric in enumerate(plot_metrics):
            for iteration, value in enumerate(metric[key]):
                data['Iteration'].append(iteration)
                data['Value'].append(value)
                data['Variable'].append(key)

    # Convert the dictionary to DataFrame
    df = pd.DataFrame(data)

    # Set up the plotting
    sns.set(style="whitegrid")

    # Set the figure size for the plot
    plt.figure(figsize=(10, 6))

    # Create a line plot with confidence intervals
    g = sns.lineplot(x="Iteration", y="Value", hue="Variable", style="Variable",
                    markers=True, dashes=False, data=df, errorbar='sd', palette='deep')

    # Customizing the plot
    plt.title('Trend of Metrics With Confidence Interval')
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.legend(title='Metric')
    plt.ylim(-0.05, 1.4)

    # Enhance layout
    plt.tight_layout(pad=1.0)  # Adjust the padding if necessary

    # Save the figure with adjusted bounding box
    plt.savefig(name+'.png', dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()
    plt.close()


# p-value of audit hypothesis test
def p_value_DP_audit(m, r, v, eps, delta):
    '''
    The following functions for computing the p-value of the
    audit hypothesis test and the largest lower bound on epsilon
    are taken directly from Appendix D in https://arxiv.org/pdf/2305.08846
    
    Args:
    m: number of examples, each included independently with probability 0.5
    r: number of guesses (i.e. excluding abstentions)
    v: number of correct guesses by auditor
    eps, delta: DP guarantee of null hypothesis

    Returns:
    p-value = probability of >=v correct guesses under null hypothesis
    '''
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        sum = sum + scipy.stats.binom.pmf(v - i, r, q)
        if sum > i * alpha:
            alpha = sum / i
    p = beta + alpha * delta * 2 * m
    return min(p, 1)


def get_eps_audit(m, r, v, delta, p):
    """
    The following functions for computing the p-value of the
    audit hypothesis test and the largest lower bound on epsilon
    are taken directly from Appendix D in https://arxiv.org/pdf/2305.08846
    
    Args:   
    m: number of examples, each included independently with probability 0.5
    r: number of guesses (i.e. excluding abstentions)
    v: number of correct guesses by auditor
    p: 1-confidence e.g. p=0.05 corresponds to 95%
    
    Returns:
    lower bound on eps i.e. algorithm is not (eps,delta)-DP
    """
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 1 # maintain p_value_DP(eps_max) >= p
    while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min


def parameters_to_1d(parameters):
    """
    Transform a list of parameters into a 1D numpy array.
    """
    return np.concatenate([x.flatten() for x in parameters])


# save some auditing metrics
def save_audit_metrics(
    round_num, 
    accuracy, 
    privacy_estimate, 
    acc_accuracy, 
    acc_privacy_estimate,
    accuracy_mean, 
    privacy_estimate_mean, 
    acc_accuracy_mean, 
    acc_privacy_estimate_mean,  
    client_id=1, 
    history_folder="histories/"):

    # Create folder for client
    folder = history_folder + f"client_{client_id}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # file path
    file_path = folder + f'audit.csv'
    # Check if the file exists; if not, create it and write headers
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Round', 'MIA Accuracy', 'Privacy', 'Accumulative MIA Accuracy',
                             'Accumulative Privacy', 'MIA Accuracy Mean', 'Privacy Mean', 
                             'Accumulative MIA Accuracy Mean', 'Accumulative Privacy Mean'])

        # Write the metrics
        writer.writerow([
            round_num, 
            accuracy, 
            privacy_estimate, 
            acc_accuracy, 
            acc_privacy_estimate,
            accuracy_mean, 
            privacy_estimate_mean, 
            acc_accuracy_mean, 
            acc_privacy_estimate_mean, 
            ])


# plot and save plot on client side
def plot_client_metrics(client_id, config, show=True):
    predictor_name = config['model_name']
    dataset_name = config['dataset']

    history_folder = f"histories/{predictor_name}/{dataset_name}/"
    df = pd.read_csv(history_folder + f'client_{client_id}/metrics.csv')
    image_folder = f"images/{predictor_name}/{dataset_name}/client_{client_id}/"

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Extract data from DataFrame
    rounds = df['Round']
    loss = df['Loss']
    accuracy = df['Accuracy']
    f1_score = df['f1_score']
    
    # Set up the plotting
    sns.set(style="whitegrid")

    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, loss, label='Loss')
    plt.plot(rounds, accuracy, label='Accuracy')
    plt.plot(rounds, f1_score, label='f1_score')

    # Find the index (round) of minimum loss and maximum accuracy
    min_loss_round = df.loc[loss.idxmin(), 'Round']
    max_accuracy_round = df.loc[accuracy.idxmax(), 'Round']
    max_f1score_round = df.loc[f1_score.idxmax(), 'Round']

    # Mark these points with a star
    plt.scatter(min_loss_round, loss.min(), color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_round, accuracy.max(), color='orange', marker='*', s=100, label='Max Accuracy')
    plt.scatter(max_f1score_round, f1_score.max(), color='green', marker='*', s=100, label=f'Max {'f1_score'}')

    # Labels and title
    plt.xlabel('Round')
    plt.ylabel('Metrics')
    plt.title(f'Client {client_id} Metrics (Validation Set)')
    plt.legend()
    plt.ylim(-0.05, 1.4)
    plt.savefig(image_folder + f"/training_{rounds.iloc[-1]}_rounds.png")
    if show:
        plt.show()
    plt.close()
    
    max_accuracy_mia = None
    max_accuracy_round_mia = None
    max_privacy = None
    max_privacy_round = None
    
    # privacy audit
    if cfg.privacy_audit:
        df = pd.read_csv(history_folder + f'client_{client_id}/audit.csv')

        # Extract data from DataFrame
        rounds = df['Round']
        accuracy = df['MIA Accuracy']
        privacy_estimate = df['Privacy']
        
        # Set up the plotting style
        sns.set(style="whitegrid")

        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

        # Plot MIA Accuracy on the first subplot
        axes[0].plot(rounds, accuracy, label='MIA Accuracy', color='blue')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Client {client_id} MIA Accuracy')
        
        # Find and mark the max accuracy
        max_accuracy_mia = accuracy.max()
        max_accuracy_round_mia = df.loc[accuracy.idxmax(), 'Round']
        axes[0].scatter(max_accuracy_round_mia, max_accuracy_mia, color='orange', marker='*', s=150, label='Max MIA Accuracy')
        # Define the random guess accuracy value
        random_guess_accuracy = 0.5  # 50%
        # Add the dashed horizontal line
        axes[0].axhline(
            y=random_guess_accuracy,            # Y-axis position
            color='red',                        # Line color
            linestyle='--',                     # Dashed line style
            linewidth=2,                        # Line width for visibility
            label='Random Guess' # Label for the legend
        )
        axes[0].legend()

        # Plot Privacy Estimate on the second subplot
        axes[1].plot(rounds, privacy_estimate, label='Empirical Privacy Leakage Lower Bound (p=0.05)', color='green')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Privacy Leakage (ε)')
        axes[1].set_title(f'Client {client_id} Privacy Leakage')
        
        # Find and mark the max privacy leakage
        max_privacy = privacy_estimate.max()
        max_privacy_round = df.loc[privacy_estimate.idxmax(), 'Round']
        axes[1].scatter(max_privacy_round, max_privacy, color='red', marker='*', s=150, label='Max Privacy Leakage')
        axes[1].legend()

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(image_folder, f"audit_{rounds.iloc[-1]}_rounds.png"))
        if show:
            plt.show()
        plt.close()

    # Print the rounds where min loss and max accuracy occurred
    print(
        f"""
        \n\033[1;33mClient {client_id}\033[0m 
        Minimum Loss: round {min_loss_round}, value {loss.min()} 
        Maximum Accuracy: round {max_accuracy_round}, value {accuracy.max()} 
        Maximum f1_score: round {max_f1score_round}, value {f1_score.max()} 
        Maximum MIA Accuracy: round {max_accuracy_round_mia}, value {max_accuracy_mia} 
        Max Privacy Leakage Lower Bound: round {max_privacy_round}, epsilon {max_privacy}\n
        """
    )


def aggregate_client_data(config):
    """
    Aggregates audit.csv and metrics.csv files from all clients and computes aggregated metrics per round.

    """
    # Define metric categories    
    metrics_average = ['Loss', 'Accuracy', 'f1_score']
    audit_mean_metrics = ['Privacy Mean', 'Accumulative Privacy Mean', 'MIA Accuracy Mean', 'Accumulative MIA Accuracy Mean']
    audit_max_metrics = ['Privacy', 'Accumulative Privacy', 'MIA Accuracy', 'Accumulative MIA Accuracy']

    # Initialize dictionaries to hold metrics per round
    aggregated_audit = {}
    aggregated_audit_mean = {}
    aggregated_metrics = {}
    
    # Iterate over each client
    for client_id in range(1, config['clients'] + 1):
        client_dir = f"histories/{config['model_name']}/{config['dataset']}/client_{client_id}"
        
        metrics_path = os.path.join(client_dir, "metrics.csv")
        
        # Check if metrics.csv exists
        if not os.path.exists(metrics_path):
            print(f"Warning: {metrics_path} does not exist. Skipping metrics for this client.")
            continue
        
        # Read metrics.csv
        metrics_df = pd.read_csv(metrics_path)
        
        if cfg.privacy_audit:
            audit_path = os.path.join(client_dir, "audit.csv")
            
            # Check if audit.csv exists
            if not os.path.exists(audit_path):
                print(f"Warning: {audit_path} does not exist. Skipping this client.")
                raise KeyError
            
            # Read audit.csv
            audit_df = pd.read_csv(audit_path)

            # Ensure both DataFrames have the same number of rounds
            if len(audit_df) != len(metrics_df):
                print(f"Warning: Mismatch in number of rounds for client {client_id}. Skipping this client.")
                continue

        
        # Iterate over each round
        for idx in range(len(metrics_df)):
            round_num = metrics_df.loc[idx, 'Round']
            
            # Initialize dictionaries for this round if not already
            if cfg.privacy_audit:
                if round_num not in aggregated_audit:
                    aggregated_audit[round_num] = {metric: [] for metric in audit_max_metrics}
                if round_num not in aggregated_audit_mean:
                    aggregated_audit_mean[round_num] = {metric: [] for metric in audit_mean_metrics}
            if round_num not in aggregated_metrics:
                aggregated_metrics[round_num] = {metric: [] for metric in metrics_average}
            
            # Extract audit metrics
            if cfg.privacy_audit:
                for metric in audit_max_metrics:
                    value = audit_df.loc[idx, metric]
                    aggregated_audit[round_num][metric].append(value)
                for metric in audit_mean_metrics:
                    value = audit_df.loc[idx, metric]
                    aggregated_audit_mean[round_num][metric].append(value)
            
            # Extract metrics.csv metrics
            for metric in metrics_average:
                value = metrics_df.loc[idx, metric]
                aggregated_metrics[round_num][metric].append(value)
    
    # Now, compute aggregated metrics per round
    aggregated_results = []
    for round_num in sorted(aggregated_metrics.keys()):
        round_data = {'Round': round_num}
        
        # Aggregate audit metrics
        if cfg.privacy_audit:
            for metric in audit_max_metrics:
                values = aggregated_audit[round_num][metric]
                round_data[metric] = max(values) if len(values) > 0 else None
            for metric in audit_mean_metrics:
                values = aggregated_audit_mean[round_num][metric]
                round_data[metric] = sum(values) / len(values) if len(values) > 0 else None
        
        # Aggregate metrics.csv metrics
        for metric in metrics_average:
            values = aggregated_metrics[round_num][metric]
            round_data[metric] = sum(values) / len(values) if len(values) > 0 else None
        
        aggregated_results.append(round_data)
    
    # Convert to DataFrame
    aggregated_df = pd.DataFrame(aggregated_results)
    
    # Define the output directory and ensure it exists
    output_folder = f"histories/{config['model_name']}/{config['dataset']}/"
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the output file path
    output_path = os.path.join(output_folder, "aggregated_metrics.csv")
    
    # Save the aggregated DataFrame to CSV
    aggregated_df.to_csv(output_path, index=False)
    
    print(f"Aggregated results saved to {output_path}")
    
    return aggregated_df


def print_max_metrics(aggregated_metrics: pd.DataFrame):
    """
    Identifies and prints the first occurrence of the maximum values of specified audit metrics.

    Args:
        aggregated_metrics (pd.DataFrame): DataFrame containing aggregated metrics per round.
    """
    # Define the audit max metrics
    audit_max_metrics = ['Privacy', 'Accumulative Privacy', 'MIA Accuracy', 'Accumulative MIA Accuracy']
    
    # Initialize a dictionary to hold max values and corresponding rounds
    max_metrics_info = {}
    print("\n")
    
    for metric in audit_max_metrics:
        if metric in aggregated_metrics.columns:
            max_value = aggregated_metrics[metric].max()
            # Identify the first round where this max value was achieved
            round_num = aggregated_metrics.loc[aggregated_metrics[metric] == max_value, 'Round'].iloc[0]
            max_metrics_info[metric] = {'max_value': max_value, 'round': round_num}
        else:
            print(f"\033[93mWarning: Metric '{metric}' not found in the aggregated metrics.\033[0m")
        
    for metric, info in max_metrics_info.items():
        print(f"\033[93mMax {metric} {info['max_value']} in round {info['round']} \033[0m")
    
    print("\n")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
    }


def plot_histogram(flat_gradients, path):
    # create fold if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
     
    # Compute the magnitude (absolute value) of each gradient
    gradient_magnitudes = np.abs(flat_gradients)

    # Create a histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    n_bins = 50  

    # Plot the histogram with a light color and black edges for clarity
    ax.hist(gradient_magnitudes, bins=n_bins, color='skyblue', edgecolor='black')

    # Label the axes and set the title
    ax.set_xlabel("Gradient Magnitude", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Histogram of Gradient Magnitudes", fontsize=16)

    # Optional: improve layout and styling for publication quality
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

    # Save the figure in a high-quality format (e.g., PDF)
    plt.savefig(f"{path}.pdf", dpi=300)

    # Display the plot
    # plt.show()
    plt.close()
        
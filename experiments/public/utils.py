import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import seaborn as sns
# import config as cfg
import torch
import os
import csv
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.differential_privacy import ( # type: ignore
    add_localdp_gaussian_noise_to_params,
    compute_clip_model_update,
)
from flwr.common import (
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import config as cfg



# create folder if not exists
def create_delede_folders(predictor_name):
    # Results
    if not os.path.exists(f"results/{predictor_name}/{cfg.dataset_name}"):
        os.makedirs(f"results/{predictor_name}/{cfg.dataset_name}")
    else:
        # remove the directory and create a new one
        os.system(f"rm -r results/{predictor_name}/{cfg.dataset_name}")
        os.makedirs(f"results/{predictor_name}/{cfg.dataset_name}")
    # Histories
    if not os.path.exists(f"histories/{predictor_name}/{cfg.dataset_name}"):
        os.makedirs(f"histories/{predictor_name}/{cfg.dataset_name}")
    else:
        # remove the client folders
        for c in range(cfg.client_number):
            os.system(f"rm -r histories/{predictor_name}/{cfg.dataset_name}/client_{c+1}")  
    # Checkpoints
    if not os.path.exists(f"checkpoints/{predictor_name}/{cfg.dataset_name}"):
        os.makedirs(f"checkpoints/{predictor_name}/{cfg.dataset_name}")
    # Images
    if not os.path.exists(f"images/{predictor_name}/{cfg.dataset_name}"):
        os.makedirs(f"images/{predictor_name}/{cfg.dataset_name}")
    
    
# define device
def check_gpu(seed=0, print_info=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if print_info:
            print("CUDA is available")
        device = 'cuda'
        torch.cuda.manual_seed_all(seed) 
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
def plot_loss_and_accuracy(loss, accuracy, f1_score, predictor_name, show=True):
    # read args
    rounds = cfg.n_rounds

    # Set up the plotting
    sns.set(style="whitegrid")
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))

    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')
    plt.plot(f1_score, label='f1_score')
    min_loss_index = loss.index(min(loss))
    max_accuracy_index = accuracy.index(max(accuracy))
    max_f1score_index = f1_score.index(max(f1_score))
    print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]:.3f} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]*100:.2f}% \nMaximum {'f1_score'} occurred at round {max_f1score_index + 1} with a {'f1_score'} value of {f1_score[max_f1score_index]*100:.2f}%\n")
    plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
    plt.scatter(max_f1score_index, f1_score[max_f1score_index], color='green', marker='*', s=100, label=f'Max {'f1_score'}')
    
    # Labels and title
    plt.xlabel('Rounds')
    plt.ylabel('Metrics')
    plt.title('Distributed Metrics (Weighted Average on Test-Set)')
    plt.legend()
    plt.ylim(-0.05, 1.4)
    plt.grid()
    plt.savefig(f"images/{predictor_name}/{cfg.dataset_name}/training_{rounds}_rounds.png")
    if show:
        plt.show()
    return min_loss_index+1, max_accuracy_index+1


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


# plot and save plot on client side
def plot_client_metrics(client_id, predictor_name, dataset_name, show=True):

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

    # Print the rounds where min loss and max accuracy occurred
    print(f"\n\033[1;33mClient {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {loss.min()} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()} \nMax {'f1_score'} occurred at round {max_f1score_round} with a {'f1_score'} value of {f1_score.max()}\n")
    
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
        self, clipping_norm: float, sensitivity: float, epsilon: float, delta: float
    ) -> None:
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if sensitivity < 0:
            raise ValueError("The sensitivity should be a non-negative value.")

        if epsilon < 0:
            raise ValueError("Epsilon should be a non-negative value.")

        if delta < 0:
            raise ValueError("Delta should be a non-negative value.")

        self.clipping_norm = clipping_norm
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta

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
        compute_clip_model_update(
            client_to_server_params,
            server_to_client_params,
            self.clipping_norm,
        )

        params_clipped = ndarrays_to_parameters(client_to_server_params)

        # Add noise to model params
        add_localdp_gaussian_noise_to_params(
            params_clipped, self.sensitivity, self.epsilon, self.delta
        )

        # noise_value_sd = (
        #     self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        # )
        
        return parameters_to_ndarrays(params_clipped)

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
def save_audit_metrics(round_num, accuracy, privacy_estimate, client_id=1, history_folder="histories/"):

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
            writer.writerow(['Round', 'Accuracy', 'Privacy'])

        # Write the metrics
        writer.writerow([round_num, accuracy, privacy_estimate])


# plot and save privacy audit metrics
def plot_audit_metrics(client_id, model_name, dataset_name, show=True):
    history_folder = f"histories/{model_name}/{dataset_name}/"
    df = pd.read_csv(history_folder + f'client_{client_id}/audit.csv')
    image_folder = f"images/{model_name}/{dataset_name}/client_{client_id}/"

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Extract data from DataFrame
    rounds = df['Round']
    accuracy = df['Accuracy']
    privacy_estimate = df['Privacy']
    
    # Set up the plotting
    sns.set(style="whitegrid")

    # Plot loss and privacy
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, accuracy, label='MIA Accuracy')
    plt.plot(rounds, privacy_estimate, label='Empirical privacy leakage lower bound (p=0.05)')

    # Find the index (round) of max accuracy and privacy loss
    max_accuracy_round = df.loc[accuracy.idxmax(), 'Round']
    max_privacy_round = df.loc[privacy_estimate.idxmax(), 'Round']

    # Print the rounds where max accuracy and max privacy occurred
    print(f"\n\033[1;33mClient {client_id}\033[0m \nMaximum MIA accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()} \nMax privacy leakage lower bound occurred at round {max_privacy_round} with an epsilon value of {privacy_estimate.max()}\n")
    
    # Mark these points with a star
    plt.scatter(max_accuracy_round, accuracy.max(), color='orange', marker='*', s=100, label='Max MIA Accuracy')
    plt.scatter(max_privacy_round, privacy_estimate.max(), color='green', marker='*', s=100, label='Max Privacy Leakage lower bound')

    # Labels and title
    plt.xlabel('Round')
    plt.ylabel('Metrics')
    plt.title(f'Client {client_id} privacy metrics')
    plt.legend()
    # plt.ylim(-0.05, 1.4)
    plt.savefig(image_folder + f"/audit_{rounds.iloc[-1]}_rounds.png")
    if show:
        plt.show()

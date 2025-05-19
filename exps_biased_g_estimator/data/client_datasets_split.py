"""
This script splits a dataset into multiple client-specific subsets for federated learning experiments. 
It handles various datasets by downloading and loading them as needed.
"""


# Libraries
# import experiments.data.download_datasets as
import download_datasets
import argparse
import torch
from torch.utils.data import Subset
from datasets import load_from_disk
import os
import warnings
warnings.filterwarnings("ignore")


# Get input arguments
args = argparse.ArgumentParser(description="Split the dataset into N clients")
args.add_argument(
    "--n_clients", type=int, default=5, help="Number of clients to create"
)
args.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    help="Dataset to split",
    choices=[
        "mnist",
        "cifar10",
        "imdb",
        "fmnist",
        "breast",
        "diabetes",
        "airline",
        "adult",
        "lsst",
    ],
)
args.add_argument("--seed", type=int, default=1, help="Random seed")
args = args.parse_args()

print(f"\n\n\033[33mData creation\033[0m")
print(f"Number of clients: {args.n_clients}")
print(f"Dataset: {args.dataset}")
print(f"Random seed: {args.seed}")


#########################################################################################
# Load dataset
#########################################################################################
if args.dataset == "mnist":
    # if not exists, download MNIST dataset
    if not os.path.exists("datasets/mnist_train.pt"):
        download_datasets.download_mnist()

    # Load MNIST dataset
    X_train = torch.load("datasets/mnist_train.pt", weights_only=False)
    X_test = torch.load("datasets/mnist_test.pt", weights_only=False)

elif args.dataset == "cifar10":
    # if not exists, download CIFAR-10 dataset
    if not os.path.exists("datasets/cifar10_train.pt"):
        download_datasets.download_cifar10()

    # Load CIFAR-10 dataset
    X_train = torch.load("datasets/cifar10_train.pt", weights_only=False)
    X_test = torch.load("datasets/cifar10_test.pt", weights_only=False)

elif args.dataset == "fmnist":
    # if not exists, download Fashion MNIST dataset
    if not os.path.exists("datasets/fmnist_train.pt"):
        download_datasets.download_fashion_mnist()

    # Load CIFAR-10 dataset
    X_train = torch.load("datasets/fmnist_train.pt", weights_only=False)
    X_test = torch.load("datasets/fmnist_test.pt", weights_only=False)

elif args.dataset == "breast":
    # if not exists, download breast cancer dataset
    if not os.path.exists("datasets/breast_train.pt"):
        download_datasets.download_breast()

    # Load breast cancer dataset
    X_train = torch.load("datasets/breast_train.pt", weights_only=False)
    X_test = torch.load("datasets/breast_test.pt", weights_only=False)

elif args.dataset == "diabetes":
    # if not exists, download diabetes dataset
    if not os.path.exists("datasets/diabetes_train.pt"):
        download_datasets.download_diabetes()

    # Load diabetes dataset
    X_train = torch.load("datasets/diabetes_train.pt", weights_only=False)
    X_test = torch.load("datasets/diabetes_test.pt", weights_only=False)

elif args.dataset == "airline":
    # if not exists, download airline dataset
    if not os.path.exists("datasets/airline_train.pt"):
        download_datasets.download_airline()

    # Load airline dataset
    X_train = torch.load("datasets/airline_train.pt", weights_only=False)
    X_test = torch.load("datasets/airline_test.pt", weights_only=False)

elif args.dataset == "adult":
    # if not exists, download adult dataset
    if not os.path.exists("datasets/adult_train.pt"):
        download_datasets.download_adult()

    # Load adult dataset
    X_train = torch.load("datasets/adult_train.pt", weights_only=False)
    X_test = torch.load("datasets/adult_test.pt", weights_only=False)

elif args.dataset == "lsst":
    # if not exists, download lsst dataset
    if not os.path.exists("datasets/lsst_train.pt"):
        download_datasets.download_lsst()

    # Load lsst dataset
    X_train = torch.load("datasets/lsst_train.pt", weights_only=False)
    X_test = torch.load("datasets/lsst_test.pt", weights_only=False)

elif args.dataset == "imdb":
    # if not exists, download imdb dataset
    if not os.path.exists("datasets/imdb_train"):
        download_datasets.download_imdb()

    # Load imdb dataset
    X_train = load_from_disk("datasets/imdb_train")
    test_data = load_from_disk("datasets/imdb_test")

else:
    raise ValueError("Invalid dataset name")







#########################################################################################
# Split dataset
#########################################################################################

# -----  1) IID-scenario -----


def IID_split_and_save_torch(dataset, num_parts, save_dir="./client_datasets", seed=1):
    """
    Splits a dataset into num_parts IID sub-datasets and saves each subset to disk.

    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset to split.
    - num_parts (int): The number of parts to split the dataset into.
    - save_dir (str): Directory where the sub-datasets will be saved. Default is './subdatasets'.

    Returns:
    - List of file paths where each subset is saved.
    """

    print(f"\nSplitting the dataset into {num_parts} IID parts...")

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the total number of samples
    total_samples = len(dataset)

    # Calculate the size of each subset
    subset_size = total_samples // num_parts

    # Shuffle the dataset indices
    torch.manual_seed(seed)
    indices = torch.randperm(total_samples)

    for i in range(num_parts):
        # Define the start and end indices for this subset
        start_idx = i * subset_size
        if i == num_parts - 1:
            end_idx = total_samples  # Include the remaining samples in the last subset
        else:
            end_idx = start_idx + subset_size

        # Get the indices for this subset
        subset_indices = indices[start_idx:end_idx]

        # Create the subset using the Subset class
        subset = Subset(dataset, subset_indices)

        # Define the file path to save this subset
        file_path = os.path.join(save_dir, f"IID_data_client_{i+1}.pt")
        print(f"Saving data client {i+1} to {file_path}...")

        # Save the subset
        torch.save(subset, file_path)

def IID_split_and_save(dataset, num_parts, save_dir="./client_datasets", seed=1):
    """
    Splits a dataset into num_parts IID sub-datasets and saves each subset to disk.

    Parameters:
    - dataset (datasets.Dataset): The dataset to split.
    - num_parts (int): The number of parts to split the dataset into.
    - save_dir (str): Directory where the sub-datasets will be saved.
    - seed (int): Random seed for shuffling.

    Returns:
    - List of file paths where each subset is saved.
    """

    print(f"\nSplitting the dataset into {num_parts} IID parts...")

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the total number of samples
    total_samples = len(dataset)

    # Calculate the size of each subset
    subset_size = total_samples // num_parts

    # Shuffle the dataset indices
    torch.manual_seed(seed)
    indices = torch.randperm(total_samples).tolist()  # Convert to a Python list

    for i in range(num_parts):
        # Define the start and end indices for this subset
        start_idx = i * subset_size
        if i == num_parts - 1:
            end_idx = total_samples  # Include the remaining samples in the last subset
        else:
            end_idx = start_idx + subset_size

        # Select subset using Hugging Face `select`
        subset = dataset.select(indices[start_idx:end_idx])

        # Define the file path to save this subset
        subset_dir = os.path.join(save_dir, f"IID_data_client_{i+1}")
        print(f"Saving data client {i+1} to {subset_dir}...")

        # Save the subset
        subset.save_to_disk(subset_dir)

# Split the training dataset into N clients
if args.dataset == "imdb":
    IID_split_and_save(X_train, args.n_clients, seed=args.seed)
else:
    IID_split_and_save_torch(X_train, args.n_clients, seed=args.seed)


#!/usr/bin/env python3

"""
This script splits a dataset into multiple client-specific subsets for federated learning experiments. 
It handles various datasets by downloading and loading them as needed.
"""

# Libraries
import download_datasets
import argparse
import torch
from torch.utils.data import Subset
from datasets import load_from_disk
import os
import numpy as np
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
args.add_argument(
    "--split_type",
    type=str,
    default="iid",
    choices=["iid", "dirichlet"],
    help="Splitting strategy: iid or dirichlet (non‑IID)",
)
args.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Dirichlet concentration parameter for non‑IID splitting",
)
args = args.parse_args()

print(f"\n\n\033[33mData creation\033[0m")
print(f"Number of clients: {args.n_clients}")
print(f"Dataset: {args.dataset}")
print(f"Random seed: {args.seed}")
print(f"Split type       : {args.split_type}")
print(f"Dirichlet alpha  : {args.alpha}")


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
        print(f"  ↳ Saving {file_path}")

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
        # print(f"Saving data client {i+1} to {subset_dir}...")
        print(f"  ↳ Saving {subset_dir}")

        # Save the subset
        subset.save_to_disk(subset_dir)

def _allocate_dirichlet(class_count: int, n_clients: int, alpha: float):
    """Return integer counts for each client drawn from a Dirichlet distribution."""
    proportions = np.random.dirichlet(alpha * np.ones(n_clients))
    counts = (proportions * class_count).astype(int)
    # Adjust so that the total equals class_count
    diff = class_count - counts.sum()
    for _ in range(abs(diff)):
        index = np.argmin(counts) if diff > 0 else np.argmax(counts)
        counts[index] += 1 if diff > 0 else -1
    return counts

# ——— Dirichlet splitting ———
def dirichlet_split_and_save_torch(dataset, n_clients, alpha=0.5, save_dir="./client_datasets", seed=1):
    print(f"\nSplitting into {n_clients} non‑IID parts (Dirichlet α={alpha})…")
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build mapping class → indices
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label.item() if torch.is_tensor(label) else label
        class_indices.setdefault(label, []).append(idx)

    client_indices = [[] for _ in range(n_clients)]
    for lbl, idxs in class_indices.items():
        np.random.shuffle(idxs)
        counts = _allocate_dirichlet(len(idxs), n_clients, alpha)
        start = 0
        for client_id, c in enumerate(counts):
            client_indices[client_id].extend(idxs[start:start + c])
            start += c

    tot = 0
    for i, idxs in enumerate(client_indices):
        subset = Subset(dataset, idxs)
        path = os.path.join(save_dir, f"IID_data_client_{i+1}.pt")
        print(f"  ↳ Saving {path} - n_samples={len(idxs)}")
        tot += len(idxs)
        torch.save(subset, path)
    print(f"Total samples saved: {tot}")

def dirichlet_split_and_save(dataset, n_clients, alpha=0.5, save_dir="./client_datasets", seed=1):
    if seed==4:
        seed=5
    print(f"\nSplitting into {n_clients} non‑IID parts (Dirichlet α={alpha})…")
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    label_col = "label" if "label" in dataset.features else "labels"
    labels = dataset[label_col]
    class_indices = {}
    for idx, lbl in enumerate(labels):
        lbl = int(lbl)
        class_indices.setdefault(lbl, []).append(idx)
        
    client_indices = [[] for _ in range(n_clients)]
    for lbl, idxs in class_indices.items():
        np.random.shuffle(idxs)
        counts = _allocate_dirichlet(len(idxs), n_clients, alpha)
        start = 0
        for client_id, c in enumerate(counts):
            client_indices[client_id].extend(idxs[start:start + c])
            start += c
    
    tot_data = 0
    for i, idxs in enumerate(client_indices):
        tot_data += len(idxs)
        subset = dataset.select(idxs)
        #----
        ## label distribution for a Hugging-Face `Dataset`
        from collections import Counter
        labels = subset["label"]      # this is a Python list of ints (or Arrow Scalars)
        labels = [int(l) for l in labels]
        label_counts = Counter(labels)
        for label, cnt in sorted(label_counts.items()):
            print(f"  Client id {i} Class {label}: {cnt} samples")
        #----
        path = os.path.join(save_dir, f"IID_data_client_{i+1}")
        print(f"  ↳ Saving {path} - n_samples={len(subset)}")
        subset.save_to_disk(path)
    print(f"Total samples saved: {tot_data}")


# ─────────────────────── Main ────────────────────────
# # Split the training dataset into N clients
if args.split_type == "iid":
    if args.dataset == "imdb":
        IID_split_and_save(X_train, args.n_clients, seed=args.seed)
    else:
        IID_split_and_save_torch(X_train, args.n_clients, seed=args.seed)
elif args.split_type == "dirichlet":
    if args.dataset == "imdb":
        dirichlet_split_and_save(X_train, args.n_clients, alpha=args.alpha, seed=args.seed)
    else:
        dirichlet_split_and_save_torch(X_train, args.n_clients, alpha=args.alpha, seed=args.seed)
else:
    raise ValueError("Unknown split_type. Choose from ['iid', 'dirichlet']")

print("\n\033[32mData Partition Done!\033[0m")

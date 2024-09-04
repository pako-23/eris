# Libraries
import download_datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rmd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import argparse
from sklearn.cluster import KMeans
import copy
import torch
from torch.utils.data import Subset, DataLoader
import os
import warnings
warnings.filterwarnings('ignore')


# Get input arguments
args = argparse.ArgumentParser(description='Split the dataset into N clients')
args.add_argument('--n_clients', type=int, default=5, help='Number of clients to create')
args.add_argument('--dataset', type=str, default='mnist', help='Dataset to split', choices=['mnist', 'cifar10', 'airline', 'adult', 'lsst'])
args.add_argument('--seed', type=int, default=1, help='Random seed')
args = args.parse_args()

print(f"\n\n\033[33mData creation\033[0m")
print(f"Number of clients: {args.n_clients}")
print(f"Dataset: {args.dataset}")
print(f"Random seed: {args.seed}")



#########################################################################################
# Load dataset
#########################################################################################
if args.dataset == 'mnist':
    # if not exists, download MNIST dataset
    if not os.path.exists('datasets/mnist_train.pt'):
        download_datasets.download_mnist()

    # Load MNIST dataset
    X_train = torch.load('datasets/mnist_train.pt')
    X_test = torch.load('datasets/mnist_test.pt')

elif args.dataset == 'cifar10':
    # if not exists, download CIFAR-10 dataset
    if not os.path.exists('datasets/cifar10_train.pt'):
        download_datasets.download_cifar10()

    # Load CIFAR-10 dataset
    X_train = torch.load('datasets/cifar10_train.pt')
    X_test = torch.load('datasets/cifar10_test.pt')
    
elif args.dataset == 'airline':
    # if not exists, download airline dataset
    if not os.path.exists('datasets/airline_train.pt'):
        download_datasets.download_airline()

    # Load airline dataset
    X_train = torch.load('datasets/airline_train.pt')
    X_test = torch.load('datasets/airline_test.pt')

elif args.dataset == 'adult':
    # if not exists, download adult dataset
    if not os.path.exists('datasets/adult_train.pt'):
        download_datasets.download_adult()

    # Load adult dataset
    X_train = torch.load('datasets/adult_train.pt')
    X_test = torch.load('datasets/adult_test.pt')

elif args.dataset == 'lsst':
    # if not exists, download lsst dataset
    if not os.path.exists('datasets/lsst_train.pt'):
        download_datasets.download_lsst()

    # Load lsst dataset
    X_train = torch.load('datasets/lsst_train.pt')
    X_test = torch.load('datasets/lsst_test.pt')
    
    
    
#########################################################################################
# Split dataset
######################################################################################### 

# -----  1) IID-scenario -----

def IID_split_and_save_torch(dataset, num_parts, save_dir='./client_datasets', seed=1):
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
        file_path = os.path.join(save_dir, f'IID_data_client_{i+1}.pt')
        print(f"Saving data client {i+1} to {file_path}...")

        # Save the subset
        torch.save(subset, file_path)


# Split the training dataset into N clients
IID_split_and_save_torch(X_train, args.n_clients, seed=args.seed)
    


# -----  2) Non-IID-scenario -----
# so far i dont think we need to implement this, but we can do it later if needed






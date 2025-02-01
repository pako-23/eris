import sys
import os
from transformers import TrainingArguments


# ADD DP TO REACH 0 MIA ACC

# Training settings (for everyone)
dataset_name = "imdb"  # Options: "mnist", "cifar10", "imdb" "fmnist, "breast", "diabetes", "adult", "airline, "lsst"
k_folds = 1  # Set 1 to disable cross validation
local_epochs = 2
lr = 0.01
momentum = 0.9
seed = 1
transform = None
gpu = -2 # select the gpu, -1 use cpu, -2 multiple distributed gpus

# Privacy auditing
privacy_audit = True
canary_frac = 0.5  # Fraction of canary samples per client
score_fn = "blackbox"  # Options: "whitebox", "blackbox"
p_value = 0.05  # Significance level for hypothesis testing
k_plus = 1 / 3  # Fraction of clients with highest scores
k_min = 1 / 3  # Fraction of clients with lowest scores

# Differential Privacy
local_dp = False
clipping_norm = 1.0 # (float) limits the L2 norm of each data point’s contribution, affecting the sensitivity of the function (default: 1.0)
sensitivity = 1.0 # (float) defines the maximum change to the function’s output that any single input can cause (default: 1.0) - generally equal to the clipping norm
epsilon = 10.0 # (float) A smaller epsilon value increases privacy (i.e., more noise) because it reduces the amount of information each output reveals about its inputs (default: 0.1)
delta = 1e-5 # (float) Typically, a smaller delta offers more privacy but is used to account for the probability of the privacy guarantee not holding (default: 1e-5)

# Pruning
pruning = False
pruning_rate = 0.2  # Fraction of weights to prune

# k-sparsification
k_sparsification = False
k_sparsity = 0.01  # Fraction of weights to keep (can be automated)

# shifted k-sparsification
shifted_k_sparsification = False
k_sparsity = 0.01  # Fraction of weights to keep (can be automated)

# Experiments config
experiments = {
    "mnist": {
        "dataset": "mnist",
        "client_train_samples": [8, 16, 32, 64, 128, 256, 512], 
        "rounds": [80, 100, 200, 180, 180, 180, 160],  # Originally 15
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 10,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "LeNet5",
        "model_args": {
            "in_channels": 1,
            "num_classes": 10,
            "input_size": (28, 28),
        },
        "n_classes": 10,
    },
    "cifar10": {
        "dataset": "cifar10",
        "client_train_samples": [8, 16, 32, 64, 128, 256, 512], #[8, 16, 32, 64, 128, 256, 512] #[8, 16, 32, 64, 128, 256, 512],
        "rounds": [60, 80, 100, 140, 100, 100, 100], #[160, 140, 180, 160, 140, 100, 100], # Originally 20 
        "clients": 50,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 50,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "ResNet9",
        "model_args": {
            "in_channels": 3,
            "num_classes": 10,
            "input_size": (32, 32),
        },
        "n_classes": 10,
    },
    "imdb": {
        "dataset": "imdb",
        "client_train_samples": [100, 16, 32, 64, 128, 256, 512], #[8, 16, 32, 64, 128, 256, 512] #[8, 16, 32, 64, 128, 256, 512],
        "rounds": [10, 80, 100, 140, 100, 100, 100], #[160, 140, 180, 160, 140, 100, 100], # Originally 20 
        "clients": 2,
        "splits": 2,
        "model_name": "distilbert-base-uncased",
        "training_args": TrainingArguments(
            output_dir="./distilbert-imdb",
            overwrite_output_dir=True,
            # num_train_epochs=None, # Set desired number of epochs - Commented out to use max_steps
            max_steps=50,  # Set desired number of training steps
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,             
            weight_decay=0.0,               
            # adam_beta1=0.9,                   
            # adam_beta2=0.999,                 
            # adam_epsilon=1e-8,                
            eval_strategy="no", #"epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=100,
            seed=42
        ),
        "n_classes": 2,
    },
    "fmnist": {
        "dataset": "fmnist",
        "client_train_samples": 1000,
        "rounds": 2, # Originally 15
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "LeNet5",
        "model_args": {
            "in_channels": 1,
            "num_classes": 10,
            "input_size": (28, 28),
        },
        "n_classes": 10,
    },
    "breast": {
        "dataset": "breast",
        "client_train_samples": 100,
        "rounds": 2, # Originally 200
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "MLP",
        "model_args": {
            "input_size": 30,
            "num_classes": 2,
            "hidden_dim": 128,
        },
        "n_classes": 2,
    },
    "diabetes": {
        "dataset": "diabetes",
        "client_train_samples": 1000,
        "rounds": 2, # Originally 20
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "MLP",
        "model_args": {
            "input_size": 21,
            "num_classes": 2,
            "hidden_dim": 128,
        },
        "n_classes": 2,
    },
    "adult": {
        "dataset": "adult",
        "client_train_samples": 1000,
        "rounds": 2, # Originally 50
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "MLP",
        "model_args": {
            "input_size": 105,
            "num_classes": 2,
            "hidden_dim": 128,
        },
        "n_classes": 2,
    },
    "airline": {
        "dataset": "airline",
        "client_train_samples": 1000,
        "rounds": 2, # Originally 1000
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "LinearModel",
        "model_args": {
            "input_size": 30,
            "num_classes": 1,
        },
        "n_classes": 1,
    },
    "lsst": {
        "dataset": "lsst",
        "client_train_samples": 1000,
        "rounds": 2, # Originally 100
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "TransformerModelFlexible",
        "model_args": {
            "input_size": 6,
            "sequence_length": 36,
            "num_classes": 12,
            "num_heads": 2,
            "num_layers": 2,
            "hidden_dim": 64,
        },
        "n_classes": 12,
    },
}

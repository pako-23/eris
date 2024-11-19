import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

# Training settings
batch_size = 64
test_batch_size = 64
# n_rounds = 30
local_epochs = 2
lr = 0.01
momentum = 0.9
seed = 1
transform = None

# Privacy auditing
privacy_audit = False
canary_frac = 0.2 # Fraction of canary samples per client
score_fn = 'whitebox' # Options: "whitebox", "blackbox"
p_value = 0.05 # Significance level for hypothesis testing
k_plus = 1 / 3 # Fraction of clients with highest scores
k_min = 1 / 3 # Fraction of clients with lowest scores
delta = 1e-5 # Targeted delta for differential privacy

# dataset settings
dataset_name = "mnist"  # Options: "mnist", "cifar10", "fmnist, "breast", "diabetes", "adult", "airline, "lsst"
client_number = 5

# Cross validation
k_folds = 1  # Set 1 to disable cross validation

# Model settings
n_classes_dict = {
    "mnist": 10, # 15
    "cifar10": 10,
    "fmnist": 10,
    "breast": 2,
    "diabetes": 2,
    "adult": 2,
    "airline": 1,  # regression
    "lsst": 12,
}

n_rounds_dict = {
    "mnist": 3, #15
    "cifar10": 20,
    "fmnist": 15, 
    "breast": 200,
    "diabetes": 20,
    "adult": 50,
    "airline": 1000,
    "lsst": 100,
}


experiments = {
    "mnist": {
        "dataset": "mnist",
        "rounds": 2,  # Originally 15
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.LeNet5,
        "model_args": {
            "in_channels": 1,
            "num_classes": 10,
            "input_size": (28, 28),
        },
        "n_classes": 10,
    },
    "cifar10": {
        "rounds": 2,
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.ResNet9,
        "model_args": {
            "in_channels": 3,
            "num_classes": 10,
            "input_size": (32, 32),
        },
        "n_classes": 10,
    },
    "fmnist": {
        "rounds": 2,
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.LeNet5,
        "model_args": {
            "in_channels": 1,
            "num_classes": 10,
            "input_size": (28, 28),
        },
        "n_classes": 10,
    },
    "breast": {
        "rounds": 1,
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.MLP,
        "model_args": {
            "input_size": 30,
            "num_classes": 2,
            "hidden_dim": 128,
        },
        "n_classes": 2,
    },
    "diabetes": {
        "rounds": 1,
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.MLP,
        "model_args": {
            "input_size": 21,
            "num_classes": 2,
            "hidden_dim": 128,
        },
        "n_classes": 2,
    },
    "adult": {
        "rounds": 2,
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.MLP,
        "model_args": {
            "input_size": 105,
            "num_classes": 2,
            "hidden_dim": 128,
        },
        "n_classes": 2,
    },
    "airline": {
        "rounds": 2,
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.LinearModel,
        "model_args": {
            "input_size": 30,
            "num_classes": 1,
        },
        "n_classes": 1,
    },
    "lsst": {
        "rounds": 2,
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.TransformerModelFlexible,
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


import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

# Training settings (for everyone)
dataset_name = "mnist"  # Options: "mnist", "cifar10", "fmnist, "breast", "diabetes", "adult", "airline, "lsst"
k_folds = 2  # Set 1 to disable cross validation
local_epochs = 2
lr = 0.01
momentum = 0.9
seed = 1
transform = None

# Privacy auditing
privacy_audit = False
canary_frac = 0.2  # Fraction of canary samples per client
score_fn = "whitebox"  # Options: "whitebox", "blackbox"
p_value = 0.05  # Significance level for hypothesis testing
k_plus = 1 / 3  # Fraction of clients with highest scores
k_min = 1 / 3  # Fraction of clients with lowest scores
delta = 1e-5  # Targeted delta for differential privacy

# Experiments config
experiments = {
    "mnist": {
        "dataset": "mnist",
        "rounds": 5,  # Originally 15
        "clients": 10,
        "batch": 64,
        "batch_test": 64,
        "epochs": 2,
        "splits": 5,
        "lr": 0.01,
        "momentum": 0.9,
        "model": models.LeNet5,
        "model_name": "LeNet5",
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
        "model_name": "ResNet9",
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
        "model_name": "LeNet5",
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
        "model_name": "MLP",
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
        "model_name": "MLP",
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
        "model_name": "MLP",
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
        "model_name": "LinearModel",
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

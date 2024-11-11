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


# dataset settings
dataset_name = "mnist"  # Options: "mnist", "cifar10", "fmnist, "breast", "diabetes", "adult", "airline, "lsst"
client_number = 10

# Cross validation
k_folds = 1  # Set 1 to disable cross validation

# Model settings
n_classes_dict = {
    "mnist": 10,
    "cifar10": 10,
    "fmnist": 10,
    "breast": 2,
    "diabetes": 2,
    "adult": 2,
    "airline": 1,  # regression
    "lsst": 12,
}

n_rounds_dict = {
    "mnist": 15,
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
        "rounds": 10,
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
    }
}

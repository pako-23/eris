from models import *


# Training settings
batch_size = 64
test_batch_size = 64
n_rounds = 100
local_epochs = 2
lr = 0.01
momentum = 0.9
seed = 1
transform = None


# dataset settings
dataset_name = "breast" # Options: "mnist", "cifar10", "fmnist, "breast", "diabetes", "adult", "airline, "lsst"
client_number = 10

# Cross validation
k_folds = 2  # Set 1 to disable cross validation

# Model settings
n_classes_dict = {
    "mnist": 10,
    "cifar10": 10,
    "fmnist": 10, 
    "breast": 2,
    "diabetes": 2,
    "adult": 2,
    "airline": 1, # regression 
    "lsst":12,
}

model_dict = {
    "mnist": LeNet5,
    "cifar10": ResNet9,
    "fmnist":LeNet5,
    "breast": MLP,  
    "diabetes": MLP,
    "adult": MLP,
    "airline":LinearModel,
    "lsst": TransformerModelFlexible,
}

model_args = {
    "mnist": 
        {
            "in_channels": 1, 
            "num_classes": 10, 
            "input_size": (28, 28),
        },
    "cifar10": 
        {
            "in_channels": 3, 
            "num_classes": 10, 
            "input_size": (32, 32),    
        },
    "fmnist":
        {
            "in_channels": 1, 
            "num_classes": 10, 
            "input_size": (28, 28),   
        },
    "breast":
        {
            "input_size": 30, 
            "num_classes": 2, 
            "hidden_dim": 128
        },
    "diabetes":
        {
            "input_size": 21, 
            "num_classes": 2, 
            "hidden_dim": 128
        },
    "adult":
        {
            "input_size": 105, 
            "num_classes": 2, 
            "hidden_dim": 128
        },
    "airline":
        {
            "input_size": 30, 
            "num_classes": 1, 
        },
    "lsst":
        {
            "input_dim": 6,
            "sequence_length": 36,
            "num_classes": 12,
            "num_heads": 2,
            "num_layers": 2,
            "hidden_dim": 64,
        }
}




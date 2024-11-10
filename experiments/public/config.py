# Training settings
batch_size = 64
test_batch_size = 64
n_rounds = 10
local_epochs = 2
lr = 0.01
momentum = 0.9
seed = 1
transform = None


# dataset settings
dataset_name = "mnist" # Options: "mnist", "cifar10", "fmnist, "breast", "diabetes", "adult", "airline, "lsst"
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





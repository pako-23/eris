from transformers import TrainingArguments # type: ignore

# Training settings (for everyone)
dataset_name = "imdb"  # Options: "mnist", "cifar10", "imdb" "fmnist, "breast", "diabetes", "adult", "airline, "lsst"
k_folds = 5  # Set 1 to disable cross validation
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
local_dp = True
use_opacus = False
clipping_norm = 1.0 # (float) limits the L2 norm of each data point’s contribution, affecting the sensitivity of the function (default: 1.0)
sensitivity = 1.0 # (float) defines the maximum change to the function’s output that any single input can cause (default: 1.0) - generally equal to the clipping norm
epsilon = 100.0 # (float) A smaller epsilon value increases privacy (i.e., more noise) because it reduces the amount of information each output reveals about its inputs (default: 0.1)
delta = 1e-5 # (float) Typically, a smaller delta offers more privacy but is used to account for the probability of the privacy guarantee not holding (default: 1e-5)

# Pruning
pruning = False
pruning_rate = 0.3  # Fraction of weights to prune

# k-sparsification
k_sparsification = False
k_sparsity = 0.01  # Fraction of weights to keep (can be automated)

# shifted k-sparsification
shifted_k_sparsification = False
# k_sparsity = 0.01  # Fraction of weights to keep (can be automated)

# Experiments config
experiments = {
    "mnist": {
        "dataset": "mnist",
        "client_train_samples": [8, 16, 32, 64, 128, 256], 
        "rounds": [120, 140, 200, 250, 250, 250],  # Originally 15
        "clients": 50,
        "batch_test": 64,
        "epochs": 1,
        "splits": 50,
        "lr": 0.01,
        "momentum": 0.9,
        "model_name": "LeNet5",
        "model_args": {
            "in_channels": 1,
            "num_classes": 10,
            "input_size": (28, 28),
        },
        "n_classes": 10,
        "pruning_rate": [0.01],   #[0.0005, 0.001, 0.005, 0.02, 0.03, 0.04]
    },
    "cifar10": {
        "dataset": "cifar10",
        "client_train_samples": [8, 16, 32, 64, 128, 256], 
        "rounds": [80, 140, 140, 140, 140, 140], 
        "clients": 50,
        "batch_test": 64,
        "epochs": 1,
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
        # "sensitivity": [100, 100, 100, 100, 100, 100, 100, 100, 10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        # "sigma": [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.6, 1.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.6, 1.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.6, 1.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.6, 1.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.6, 1.0], 
        "sensitivity": [10, 5, 2, 1, 1, 1],
        "sigma": [0.001, 0.01, 0.1, 0.3, 0.6, 1.0],       
        "pruning_rate": [0.01],   #[0.0005, 0.001, 0.005, 0.02, 0.03, 0.04]
           
    },
    "imdb": {
        "dataset": "imdb",
        "client_train_samples": [8, 16, 32, 64, 128, 256],  
        "rounds": [22, 20, 18, 18, 14, 14], 
        "clients": 25,
        "splits": 25,
        "model_name": "distilbert-base-uncased",
        "training_args": TrainingArguments(
            output_dir="./distilbert-imdb",
            overwrite_output_dir=True,
            num_train_epochs=1, # Set desired number of epochs - Commented out to use max_steps
            # max_steps=100,  # Set desired number of training steps
            # per_device_train_batch_size=16, # set automatically during training
            per_device_eval_batch_size=8,
            learning_rate=5e-5,             
            weight_decay=0.0,               
            adam_beta1=0.9,                   
            adam_beta2=0.999,                 
            adam_epsilon=1e-8,                
            eval_strategy="no", #"epoch",
            save_strategy="no",  #"epoch", Disable saving the model during training
            logging_dir="./logs",
            logging_steps=100,
            seed=42
        ),
        "n_classes": 2,
    },
}

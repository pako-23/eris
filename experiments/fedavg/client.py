"""
This code creates a Flower client that can be used to train a model locally and share the updated 
model with the server. When it is started, it connects to the Flower server and waits for instructions.
If the server sends a model, the client trains the model locally and sends back the updated model.
If abilitated, at the end of the training the client evaluates the last model, and plots the 
metrics during the training.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
have this code running.
"""

# Libraies
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch
import utils as utils
import flwr as fl
import argparse
import config as cfg
from torch.utils.data import random_split
import torch.nn as nn
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import models



# Define Flower client 
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, optimizer, num_examples, 
                 client_id, train_fn, evaluate_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_examples = num_examples
        self.client_id = client_id
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.device = device
        self.dataset_name = cfg.dataset_name
        self.predictor_name = cfg.predictor_name
        
        if cfg.local_dp:
            self.local_dp = True
            self.dp_funtion = utils.LocalDpMod(cfg.clipping_norm, cfg.sensitivity, cfg.epsilon, cfg.delta)
            noise_value_sd = (cfg.sensitivity * np.sqrt(2 * np.log(1.25 / cfg.delta)) / cfg.epsilon)
            if client_id == 1:
                print(f"\n\033[94mLocal Differential Privacy with noise_value_sd: {noise_value_sd}\033[0m\n")
        else:
            self.local_dp = False

 
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)  
        for epoch in range(config["local_epochs"]):
            self.train_fn(self.model, self.device, self.train_loader, self.optimizer, epoch, self.client_id)
            
        # Local Differential Privacy
        if self.local_dp:
            # print(f"Client {self.client_id}, before LDP: {self.get_parameters(config)[0][0][:5][0][0]}")
            dp_params = self.dp_funtion(parameters, self.get_parameters(config))
            # print(f"Client {self.client_id}, after LDP: {dp_params[0][0][:5][0][0]}")
            return dp_params, self.num_examples["train"], {}
        else:
            return self.get_parameters(config), self.num_examples["train"], {}
    
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if cfg.only_predictor:
            loss, accuracy, f1_score = self.evaluate_fn(self.model, self.device, self.val_loader, self.client_id)
            
            # save loss and accuracy client
            utils.save_client_metrics(config["current_round"], loss, accuracy, f1_score, client_id=self.client_id,
                                        history_folder=f"histories/{self.predictor_name}/{self.dataset_name}/")
            
            return float(loss), self.num_examples["val"], {
                "accuracy": float(accuracy),
                "f1_score": float(f1_score),
            }
        else:
            loss, accuracy, validity = self.evaluate_fn(self.model, self.device, self.val_loader, self.client_id)
            
            # save loss and accuracy client
            utils.save_client_metrics(config["current_round"], loss, accuracy, validity, client_id=self.client_id,
                                        history_folder=f"histories/{self.predictor_name}/{self.dataset_name}/")
            
            return float(loss), self.num_examples["val"], {
                "accuracy": float(accuracy),
                "validity": float(validity),
            }
                





# main
def main()->None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 101),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # model and history folder
    model = models.predictors[cfg.predictor_name](
        in_channels=cfg.channels, 
        num_classes=cfg.n_classes, 
        input_size=cfg.input_size
        ).to(device)
        
    if not cfg.only_predictor:
        model = models.generators[cfg.cf_generator](
            predictor=model,
            in_channels=cfg.channels, 
            num_classes=cfg.n_classes, 
            input_size=cfg.input_size
            ).to(device)
        
        train_fn = models.train_generators[cfg.cf_generator]
        test_fn = models.test_generators[cfg.cf_generator]
    else:
        train_fn = models.simple_train
        test_fn = models.simple_test

    # Load data
    data = torch.load(f'../data/client_datasets/IID_data_client_{args.id}.pt', weights_only=False)
    
    # Split the dataset
    val_size = int(len(data) * 0.2)  # 20% for validation
    train_size = len(data) - val_size  # 80% for training
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    num_examples = {
        "train": train_size,
        "val": val_size
    }

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    x, y = next(iter(train_loader))
    print(f"Client {args.id}: Train data shape: {x.shape}, Train labels shape: {y.shape}")

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.MSELoss() if cfg.n_classes_dict[cfg.dataset_name]==1 else nn.CrossEntropyLoss()

    # Start Flower client
    client = FlowerClient(model, train_loader, val_loader, optimizer, num_examples, args.id, 
                           train_fn, test_fn, device).to_client()
    fl.client.start_client(server_address="[::]:8098", client=client) # local host
    
    # read saved data and plot
    utils.plot_client_metrics(args.id, cfg.predictor_name, cfg.dataset_name, show=False)





if __name__ == "__main__":
    main()

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
import flwr as fl
import argparse
from torch.utils.data import random_split
import torch.nn.functional as F
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import models
from public import utils
from public import config as cfg



# Define Flower client 
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, num_examples, 
                 client_id, train_fn, evaluate_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_examples = num_examples
        self.client_id = client_id
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.device = device
        self.dataset_name = cfg.dataset_name
        self.predictor_name = model.__class__.__name__
 
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)  
        for epoch in range(config["local_epochs"]):
            self.train_fn(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch, self.client_id)
            
        return self.get_parameters(config), self.num_examples["train"], {}
    
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, f1_score = self.evaluate_fn(self.model, self.device, self.val_loader, self.criterion, self.client_id)
        
        # save loss and accuracy client
        utils.save_client_metrics(config["current_round"], loss, accuracy, f1_score, client_id=self.client_id,
                                    history_folder=f"histories/{self.predictor_name}/{self.dataset_name}/")
        
        return float(loss), self.num_examples["val"], {
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
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
    device = utils.check_gpu(seed=cfg.seed)
    utils.set_seed(cfg.seed)

    # model and history folder
    model = models.model_dict[cfg.dataset_name](
        models.model_args[cfg.dataset_name]).to(device)

    # Load data
    data = torch.load(f'../data/client_datasets/IID_data_client_{args.id}.pt', weights_only=False)
    
    # Split the dataset
    val_size = int(len(data) * 0.2)  # 20% for validation
    train_size = len(data) - val_size  # 80% for training
    torch.manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    num_examples = {
        "train": train_size,
        "val": val_size
    }

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # x, y = next(iter(train_loader))
    # print(f"Client {args.id}: Train data shape: {x.shape}, Train labels shape: {y.shape}")

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = F.mse_loss if cfg.n_classes_dict[cfg.dataset_name]==1 else F.cross_entropy

    # Start Flower client
    client = FlowerClient(model, train_loader, val_loader, optimizer, criterion, num_examples, args.id, 
                           models.simple_train, models.simple_test, device).to_client()
    fl.client.start_client(server_address="[::]:8098", client=client) # local host
    
    # read saved data and plot
    utils.plot_client_metrics(args.id, model.__class__.__name__, cfg.dataset_name, show=False)





if __name__ == "__main__":
    main()

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
import math
import scipy 

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
    def __init__(self, 
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 num_examples: dict, 
                 client_id: int,
                 train_fn: callable,
                 evaluate_fn: callable,
                 device: torch.device,
                 privacy_audit: bool = True,
                 canary_frac: float = 0.2, 
                 score_fn: str = 'whitebox', 
                 p_value: float = 0.05,
                 k_plus: float = 1 / 3, 
                 k_min: float = 1 / 3,
                 config: dict = {'dataset':'mnist', 'batch':64},
                ):
        
        
        # Define the client
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
        self.canary_frac = canary_frac
        self.p_value = p_value
        self.k_plus = k_plus
        self.k_min = k_min
        self.privacy_audit = privacy_audit
        self.privacy_estimate = -1
        self.accuracy_mia = -1
        self.config = config
        
        if score_fn == 'whitebox':
            self.score_fn = self.score_with_pseudograd_batch
        else:
            NotImplementedError(f'score function {score_fn} is not known')
 
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)  
        
        if self.privacy_audit:
            # Privacy auditing (We follow the procedure from https://proceedings.neurips.cc/paper_files/paper/2023/hash/9a6f6e0d6781d1cb8689192408946d73-Abstract-Conference.html)
            # choose canaries from the training data
            canaries, non_canaries = random_split(self.train_loader.dataset, [self.canary_frac, 1 - self.canary_frac])
            n_canaries = len(canaries)

            # subsample canaries & make new dataloader
            true_in_out = torch.distributions.bernoulli.Bernoulli(torch.ones(n_canaries) * 0.5).sample()
            canaries_in_idx = torch.nonzero(true_in_out)
            canaries_out_idx = torch.nonzero(1 - true_in_out)
            subsampled_train_data = torch.utils.data.ConcatDataset([
                non_canaries,
                torch.utils.data.Subset(canaries, canaries_in_idx)
            ])
            self.train_loader = DataLoader(subsampled_train_data, batch_size=self.config['batch'], shuffle=True)

            # train
            for epoch in range(config["local_epochs"]):
                self.train_fn(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch, self.client_id)

            # normalize client update vector
            params_out = self.get_parameters(config)
            client_update = utils.parameters_to_1d(params_out) - utils.parameters_to_1d(parameters)
            client_update /= np.linalg.norm(client_update)

            # compute scores for each canary, used to predict membership
            scores = []
            # self.set_parameters(parameters)
            dataloader = torch.utils.data.DataLoader(canaries, batch_size=self.config['batch'], shuffle=False)
            for samples, targets in dataloader:
                batch_scores = self.score_fn(samples, targets, client_update)
                scores.extend(batch_scores)
            
            # self.set_parameters(params_out)
            
            # guess which canaries were in the training data
            true_in_out = true_in_out.numpy()
            score_indices_sorted = np.argsort(np.asarray(scores))
            classified_in = score_indices_sorted[:int(n_canaries * self.k_plus + 1)]
            classified_out = score_indices_sorted[int(n_canaries * (1 - self.k_min) + 1):]
            abstained = np.setdiff1d(score_indices_sorted, np.concatenate((classified_in, classified_out)))
            classification = np.zeros(n_canaries)
            classification[classified_in] = 1
            classification[abstained] = 2
            W = true_in_out == classification
            self.accuracy_mia = W.sum() / (len(classified_in) + len(classified_out))
            num_correct = W.sum()

            # compute empirical privacy estimate, which should be < epsilon w/ high probability
            self.privacy_estimate = utils.get_eps_audit(
                m=n_canaries,
                r=len(classified_in) + len(classified_out),
                v=num_correct,
                delta=cfg.delta,
                p=0.05)

            # Kairouz privacy estimate from https://proceedings.mlr.press/v37/kairouz15.html
            # privacy_estimate = np.max([np.log(1 - cfg.delta - fpr) - np.log(fnr), 
                                # np.log(1 - cfg.delta - fnr) - np.log(fpr)])
            
            utils.save_audit_metrics(config["current_round"], self.accuracy_mia, self.privacy_estimate, client_id=self.client_id,
                                            history_folder=f"histories/{self.config['model_name']}/{self.config['dataset']}/")

        
        else:
            # train
            for epoch in range(config["local_epochs"]):
                self.train_fn(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch, self.client_id)
            
            params_out = self.get_parameters(config)
                    
        return params_out, self.num_examples["train"], {}
    
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, f1_score = self.evaluate_fn(self.model, self.device, self.val_loader, self.criterion, self.client_id)
        
        # save loss and accuracy client
        utils.save_client_metrics(config["current_round"], loss, accuracy, f1_score, client_id=self.client_id,
                                    history_folder=f"histories/{self.config['model_name']}/{self.config['dataset']}/")
        
        return float(loss), self.num_examples["val"], {
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
            "privacy_estimate": self.privacy_estimate,
            "accuracy_mia": self.accuracy_mia
        }


    def score_with_pseudograd_batch(self, samples, targets, client_update):
        '''
        Computes membership inference attack scores for a batch by 
        computing the inner product between the 'pseudogradient'
        represented by client update and the true gradients
        for each sample in the batch.
        '''
        self.model.to(self.device)  # Ensure model is on the correct device
        samples = samples.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        predictions = self.model(samples)
        losses = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
        
        scores = []
        for loss in losses:
            # Compute gradients for each sample
            audit_grad = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            audit_grad = np.concatenate([x.cpu().flatten().numpy() for x in audit_grad])
            score = np.dot(client_update, -audit_grad)
            scores.append(score)
        
        return scores







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
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="mnist",
        choices=list(cfg.experiments.keys()),
    )
    args = parser.parse_args()

    # check gpu and set manual seed
    device = utils.check_gpu(seed=cfg.seed)
    utils.set_seed(cfg.seed)
    config = cfg.experiments[args.dataset]

    # model and history folder
    model = config["model"](config["model_args"]).to(device)


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
    train_loader = DataLoader(train_dataset, batch_size=config["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_test"], shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = F.mse_loss if config["n_classes"] == 1 else F.cross_entropy
    
    # Start Flower client
    client = FlowerClient(
                        model, 
                        train_loader, 
                        val_loader, 
                        optimizer, 
                        criterion, 
                        num_examples, 
                        args.id, 
                        models.simple_train, 
                        models.simple_test, 
                        device,
                        privacy_audit=cfg.privacy_audit,
                        canary_frac=cfg.canary_frac,
                        score_fn=cfg.score_fn,
                        p_value=cfg.p_value,
                        k_plus=cfg.k_plus,
                        k_min=cfg.k_min,
                        config=config,                         
                          ).to_client()
    fl.client.start_client(server_address="[::]:8098", client=client) # local host
    
    # read saved data and plot
    utils.plot_client_metrics(args.id, config, show=False)
    






if __name__ == "__main__":
    main()

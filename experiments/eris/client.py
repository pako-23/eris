#!/usr/bin/env python3

import numpy as np
from eris import ErisClient
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import models
from public import utils
from public import config as cfg


class ExampleClient(ErisClient):
    def __init__(
        self,
        router_address: str,
        subscribe_address: str,
        model=None,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        criterion=None,
        num_examples=None,
        client_id=None,
        train_fn=None,
        evaluate_fn=None,
        device=None,
        config=None,
    ):
        # Initialize the superclass with only the required positional arguments
        super().__init__(router_address, subscribe_address)

        # Initialize additional attributes
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
        self.dataset_name = config["dataset"]
        self.predictor_name = model.__class__.__name__ if model else None
        self.config = config
        self.current_round = 0

    def get_parameters(self):
        self.model.to("cpu")
        return [param.data.numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(parameters[i])

    def fit(self):
        self.model.train(True)
        self.model.to(self.device)
        self.current_round += 1
        
        print("Training!!", flush=True)

        for epoch in range(self.config["epochs"]):
            self.train_fn(
                self.model,
                self.device,
                self.train_loader,
                self.optimizer,
                self.criterion,
                epoch,
                self.client_id,
            )

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)

        print("Evaluation!!!", flush=True)
        loss, accuracy, f1_score = self.evaluate_fn(
            self.model, self.device, self.val_loader, self.criterion, self.client_id
        )

        # save loss and accuracy client
        utils.save_client_metrics(
            self.current_round,
            loss,
            accuracy,
            f1_score,
            client_id=self.client_id,
            history_folder=f"histories/{self.predictor_name}/{self.dataset_name}/",
        )


def start_node(
    aggr_rpc_port=None,
    aggr_publish_port=None,
    model=None,
    train_loader=None,
    val_loader=None,
    optimizer=None,
    criterion=None,
    num_examples=None,
    client_id=None,
    train_fn=None,
    evaluate_fn=None,
    device=None,
    config=None,
):
    # Initialize the ExampleClient with positional and keyword arguments
    client = ExampleClient(
        "tcp://127.0.0.1:50051",
        "tcp://127.0.0.1:5555",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_examples=num_examples,
        client_id=client_id,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        device=device,
        config=config,
    )

    # Configure aggregator if ports are provided
    if aggr_rpc_port is not None and aggr_publish_port is not None:
        client.set_aggregator_config("127.0.0.1", aggr_rpc_port, aggr_publish_port)

    # Start training
    if client.train():
        print("Client finished the training successfully")
        return 0

    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Start an Eris client configured with the options for the given experiment"
    )
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 101),
        required=False,
        default=1,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--submit-port",
        type=int,
        help="Aggregator submit port",
    )
    parser.add_argument(
        "--publish-port",
        type=int,
        help="Aggregator publish port",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="mnist",
        choices=list(cfg.experiments.keys()),
    )
    parser.add_argument(
        "shard",
        type=str,
        help="Path to the dataset portion",
    )
    args = parser.parse_args()

    # Check GPU and set manual seed
    device = utils.check_gpu(seed=cfg.seed)
    utils.set_seed(cfg.seed)
    config = cfg.experiments[args.dataset]

    # Initialize model
    model = config["model"](config["model_args"]).to(device)

    # Load data
    data = torch.load(args.shard, weights_only=False)

    # Split the dataset
    val_size = int(len(data) * 0.2)  # 20% for validation
    train_size = len(data) - val_size  # 80% for training
    torch.manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    num_examples = {"train": train_size, "val": val_size}

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_test"], shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = F.mse_loss if config["n_classes"] == 1 else F.cross_entropy

    if args.submit_port and args.publish_port:
        return start_node(
            aggr_rpc_port=args.submit_port,
            aggr_publish_port=args.publish_port,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_examples=num_examples,
            client_id=args.id,
            train_fn=models.simple_train,
            evaluate_fn=models.simple_test,
            device=device,
            config=config,
        )
    else:
        return start_node(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_examples=num_examples,
            client_id=args.id,
            train_fn=models.simple_train,
            evaluate_fn=models.simple_test,
            device=device,
            config=config,
        )


if __name__ == "__main__":
    sys.exit(main())

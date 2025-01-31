#!/usr/bin/env python3

import numpy as np
from eris import ErisClient
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import time


def generate_points(coeffs, z_range, sample_size, sigma):
    x = []

    for i in range(len(coeffs)):
        x.append(np.linspace(*z_range, sample_size) ** i)

    x = np.array(x).T
    y = (x @ coeffs) + np.random.normal(0, sigma, sample_size)

    return x, y


def generate_dataset():
    w = np.array([0, -10, 1, -1, 1 / 100])
    z_range = (-3, 3)
    x_train, y_train = generate_points(w, z_range, 500, 0.5)
    x_val, y_val = generate_points(w, z_range, 500, 0.5)
    training_dataset = TensorDataset(
        torch.from_numpy(x_train[:, 1:]).float(),
        torch.from_numpy(y_train[:, None]).float(),
    )
    training_dataset, _ = torch.utils.data.random_split(training_dataset, [100, 400])
    validation_dataset = TensorDataset(
        torch.from_numpy(x_val[:, 1:]).float(),
        torch.from_numpy(y_val[:, None]).float(),
    )
    validation_dataset, _ = torch.utils.data.random_split(
        validation_dataset, [150, 350]
    )
    training_data = DataLoader(training_dataset, batch_size=100, shuffle=True)
    validation_data = DataLoader(validation_dataset, batch_size=150, shuffle=False)
    return training_data, validation_data


class LinearRegression(nn.Module):
    def __init__(
        self,
        inputs=4,
        bias=True,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        lr=0.001,
        epochs=10,
    ):
        super().__init__()
        self.validation_data = None
        self.training_data = None
        self.linear = nn.Linear(inputs, 1, bias=bias)
        self.criterion = criterion()
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.epochs = epochs

    def forward(self, x):
        return self.linear(x)


class ExampleClient(ErisClient):
    def __init__(self, router_address, subscribe_address):
        super().__init__(router_address, subscribe_address)
        self.model = LinearRegression()
        self.train_data, _ = generate_dataset()

    def get_parameters(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(parameters[i])

    def fit(self):
        self.model.train(True)

        for data in self.train_data:
            x, y = data
            self.model.optimizer.zero_grad()
            pred = self.model.forward(x)
            loss = self.model.criterion(pred, y)
            loss.backward()
            self.model.optimizer.step()

        return self.get_parameters(), len(self.train_data)

    def evaluate(self):
        pass


def start_node(aggregator=False):
    client = ExampleClient("tcp://127.0.0.1:50051", "tcp://127.0.0.1:5555")

    if aggregator:
        client.set_aggregator_config("127.0.0.1")

    if not client.join():
        print("Client failed to join the training")
        return 0

    if client.train():
        print("Client finished the training successfully")
        return 0

    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Start an Eris client configured with the options"
    )
    parser.add_argument(
        "--aggregator",
        action=argparse.BooleanOptionalAction,
        help="Start node as aggregator",
    )
    args = parser.parse_args()
    start_node(aggregator=args.aggregator)


if __name__ == "__main__":
    sys.exit(main())

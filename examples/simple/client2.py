#!/usr/bin/env python3

import eris
import torch
from model import SimplestModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def generate_data(num_samples, num_features, batch_size=64):
    # Generate random data
    torch.manual_seed(0)  # For reproducibility
    inputs = torch.randn(num_samples, num_features)
    weights = torch.randn(num_features, 1)  # Random weights for our linear model
    bias = torch.randn(1)  # Random bias for our linear model
    # Create linear relationship y = Xw + b + noise
    noise = (
        torch.randn(num_samples, 1) * 0.1
    )  # Noise to make the data a bit more realistic
    labels = inputs @ weights + bias + noise
    # Split the dataset into training, validation and test sets
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
    val_dataset = TensorDataset(
        inputs[train_size : train_size + val_size],
        labels[train_size : train_size + val_size],
    )
    test_dataset = TensorDataset(
        inputs[train_size + val_size :], labels[train_size + val_size :]
    )
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class Client(eris.ErisClient):
    def __init__(self, model, train_loader, val_loader, *args, **kwargs):
        builder = eris.ErisAggregatorBuilder()
        builder.add_publish_port(5557)
        builder.add_rpc_port(5053)
        super().__init__(builder)
        self.local_epochs = 200
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # self.shapes = [param.data.numpy().shape for param in self.model.parameters()]

    def get_parameters(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_parameters(self):
        for i, param in enumerate(self.model.parameters()):
            # Ensure the new weights have the correct shape
            reshaped_weight = weights[i].reshape(self.shapes[i])
            param.data = torch.from_numpy(reshaped_weight)

    def fit(self):
        # Training loop
        for epoch in range(self.local_epochs):
            self.model.train()  # Set model to training mode
            total_loss = 0
            for batch, (x, y) in enumerate(self.train_loader):
                # Zero the gradients
                self.optimizer.zero_grad()
                # Perform forward pass
                outputs = self.model(x.float())  # Compute loss
                loss = self.criterion(outputs, y.float())
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            # Print average loss for the epoch
            # print(f'Epoch {epoch+1}/{self.local_epochs}, Loss: {total_loss / (batch + 1)}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            for x, y in self.val_loader:
                outputs = self.model(x.float())
                loss = self.criterion(outputs, y.float())
                test_loss += loss.item()
            print(f"Val Loss: {test_loss / len(self.val_loader)}")


def main():
    # create dataloaders
    train_loader, val_loader, _ = generate_data(1000, 4)
    # create model
    model = SimplestModel()
    # create client and run it
    client = Client(model, train_loader, val_loader)
    print(client.get_parameters())
    # print([p.shape for p in client.get_parameters()])
    client.start("127.0.0.1:5051")


if __name__ == "__main__":
    main()

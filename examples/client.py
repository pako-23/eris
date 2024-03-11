#!/usr/bin/env python3

import eris
import torch
from model import SimplestModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


## Functions to split and reconstruct the weights of the model (eris package)
def concatenate_weights(weights_list):
    flattened_weights = []
    shapes = []
    for layer in weights_list:
        for weight_matrix in layer:
            # Flatten and store the weight matrix
            flattened_weights.append(weight_matrix.flatten())
            # Store the shape for reconstruction
            if len(weight_matrix.shape) > 0:
                shapes.append(weight_matrix.shape)
            else:
                shapes.append((1,))
    # Concatenate all flattened weights into a single vector
    concatenated_weights = np.concatenate(flattened_weights)
    return concatenated_weights, shapes


# def deconcatenate_weights(flat_vector, shapes):
#     reconstructed_weights = []
#     idx = 0
#     for shape in shapes:
#         size = np.prod(shape)
#         weight_matrix = flat_vector[idx:idx + size].reshape(shape)
#         reconstructed_weights.append(weight_matrix)
#         idx += size
#     return reconstructed_weights


class NamedArray:
    def __init__(self, array, name=None):
        self.array = np.array(array)
        self.name = name


def split_weights(weights_list, n_splits, seed=42):
    # Flatten the matrix
    flat_matrix, shapes = concatenate_weights(weights_list)
    # Shuffle both using the same seed
    np.random.seed(seed)
    np.random.shuffle(flat_matrix)
    # Split the matrix into N parts and assign names
    splitted_parts = np.array_split(flat_matrix, n_splits)
    named_parts = [
        NamedArray(part, name=str(i)) for i, part in enumerate(splitted_parts)
    ]
    return named_parts, shapes


# def reconstruct_weights(splitted_named_matrices, shapes, seed=42):
#     # Sort the parts based on their names
#     sorted_parts = sorted(splitted_named_matrices, key=lambda x: int(x.name))
#     # Concatenate the parts
#     concatenated = np.concatenate([part.array for part in sorted_parts])
#     # Shuffle the concatenated array to reconstruct the original order
#     ordered_vector = np.arange(len(concatenated))
#     np.random.seed(seed)
#     np.random.shuffle(ordered_vector)
#     # Reconstruct the original matrix
#     reconstructed = np.empty_like(concatenated)
#     for i, index in enumerate(ordered_vector):
#         reconstructed[index] = concatenated[i]
#     return deconcatenate_weights(reconstructed, shapes)


# function to generate data
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
        builder.add_publish_port(5556)
        builder.add_rpc_port(5052)
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

    # randomly split the weights of the model (fixed)
    def split(self, n_splits, seed=42):
        weights = self.get_parameters()
        # splitted_matrices, self.shapes = split_weights(weights, n_splits, seed)
        splitted_matrices, _ = split_weights(weights, n_splits, seed)
        return splitted_matrices

    # reconstruct the weights of the model (fixed)
    def reconstruct(self, splitted_matrices, seed=42):
        weights = reconstruct_weights(splitted_matrices, self.shapes, seed)
        self.set_parameters(weights)
        return weights


def main():
    # create dataloaders
    train_loader, val_loader, _ = generate_data(1000, 4)
    # create model
    model = SimplestModel()
    # create client and run it
    client = Client(model, train_loader, val_loader)
    client.start("127.0.0.1:5051")


if __name__ == "__main__":
    main()

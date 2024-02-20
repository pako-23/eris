import numpy as np
import torch
from torch import nn
from torch import optim


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

def deconcatenate_weights(flat_vector, shapes):
    reconstructed_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        weight_matrix = flat_vector[idx:idx + size].reshape(shape)
        reconstructed_weights.append(weight_matrix)
        idx += size
    return reconstructed_weights

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
    named_parts = [NamedArray(part, name=str(i)) for i, part in enumerate(splitted_parts)]
    return named_parts, shapes

def reconstruct_weights(splitted_named_matrices, shapes, seed=42):
    # Sort the parts based on their names
    sorted_parts = sorted(splitted_named_matrices, key=lambda x: int(x.name))
    # Concatenate the parts
    concatenated = np.concatenate([part.array for part in sorted_parts])
    # Shuffle the concatenated array to reconstruct the original order
    ordered_vector = np.arange(len(concatenated))
    np.random.seed(seed)
    np.random.shuffle(ordered_vector)
    # Reconstruct the original matrix
    reconstructed = np.empty_like(concatenated)
    for i, index in enumerate(ordered_vector):
        reconstructed[index] = concatenated[i]
    return deconcatenate_weights(reconstructed, shapes)



# Model class
class SimplestModel(nn.Module):
    def __init__(
        self,
        inputs=4,
        bias=True,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        lr=0.001,
        local_epochs=200,
    ):
        super().__init__()
        # Network architecture
        self.linear = nn.Linear(inputs, 1, bias=bias)
        self.criterion = criterion()
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.local_epochs = local_epochs
        self.shapes = None

    # forward pass
    def forward(self, x):
        pred = self.linear(x)
        return pred

    # local training
    def fit(self, train_loader):
        # Training loop
        for epoch in range(self.local_epochs):
            self.train()  # Set model to training mode
            total_loss = 0
            for batch, (x, y) in enumerate(train_loader):
                # Zero the gradients
                self.optimizer.zero_grad()
                # Perform forward pass
                outputs = self(x.float())  
                # Compute loss
                loss = self.criterion(outputs, y.float())  
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            # Print average loss for the epoch
            print(f'Epoch {epoch+1}/{self.local_epochs}, Loss: {total_loss / (batch + 1)}')

    # evaluation on the entire validation set after each local training    
    def evaluate(self, val_loader):
        self.eval()
        with torch.no_grad():
            test_loss = 0
            for x, y in val_loader:
                outputs = self(x.float())
                loss = self.criterion(outputs, y.float())
                test_loss += loss.item()
            print(f'Val Loss: {test_loss / len(val_loader)}')
    
    # get the weights of the model (to be customized)
    def get_weights(self):
        weights = [param.data.numpy() for param in self.parameters()]
        self.shapes = [w.shape for w in weights]  # Store the original shapes
        return weights
    
    # set the weights of the model (to be customized)
    def set_weights(self, weights):
        for i, param in enumerate(self.parameters()):
            # Ensure the new weights have the correct shape
            reshaped_weight = weights[i].reshape(self.shapes[i])
            param.data = torch.from_numpy(reshaped_weight)
    
    # randomly split the weights of the model (fixed)
    def split_weights(self, n_splits, seed=42):
        weights = self.get_weights()
        #splitted_matrices, self.shapes = split_weights(weights, n_splits, seed)
        splitted_matrices, _ = split_weights(weights, n_splits, seed)
        return splitted_matrices
    
    # reconstruct the weights of the model (fixed)
    def reconstruct_weights(self, splitted_matrices, seed=42):
        weights = reconstruct_weights(splitted_matrices, self.shapes, seed)
        self.set_weights(weights)
        return weights



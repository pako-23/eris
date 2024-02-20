import numpy as np
import torch
from torch import FloatTensor
from torch import nn
from torch import optim


class TestModel(nn.Module):
    def __init__(
        self,
        inputs=4,
        bias=True,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        lr=0.001,
        epochs=100,
    ):
        super().__init__()
        # Network architecture
        self.validation_data = None
        self.training_data = None
        self.linear = nn.Linear(inputs, 1, bias=bias)
        self.criterion = criterion()
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.epochs = epochs

    def forward(self, x):
        pred = self.linear(x)
        return pred

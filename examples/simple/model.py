import numpy as np
from torch import nn


# Model class
class SimplestModel(nn.Module):
    def __init__(
        self,
        inputs=4,
        bias=True,
    ):
        super().__init__()
        # Network architecture
        self.linear = nn.Linear(inputs, 1, bias=bias)

    # forward pass
    def forward(self, x):
        pred = self.linear(x)
        return pred

"""
This will be a class of architectures useful for reconstructing markovian proceses.
"""
import torch
import torch.nn.functional as F
from torch import nn


class BCEReconstructor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(BCEReconstructor, self).__init__()
        # Create a 2-layer neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        Note we are not using sigmoid therefore you must have your
        loss funciton be BCEWithLogitsLoss
        """
        return self.model(x)

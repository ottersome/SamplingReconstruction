import torch
from torch import nn


class ValueFunc(nn.Module):
    def __init__(
        self,
        state_dim,  # Decimated
    ):
        super().__init__()
        # Define State
        # self.model = nn.Sequential(
        # nn.Linear(state_dim, 128),
        # nn.ReLU(),
        # nn.Linear(128, 256),
        # nn.ReLU(),
        # nn.Linear(256, 1),
        # )
        self.model = nn.LSTM(state_dim + 1, state_dim + 1, batch_first=True)
        self.value = nn.Linear(state_dim + 1, 1)

    def forward(self, x):
        out,hidden = self.model(x)
        y = self.value(out[:,-1,:])
        return y

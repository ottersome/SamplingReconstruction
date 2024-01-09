from abc import ABC, abstractmethod

import torch
from torch import nn


class ValueEstimator(ABC):
    @abstractmethod
    def estimate(self, state) -> torch.Tensor:
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        pass


class ValueFunc(ValueEstimator, nn.Module):
    def __init__(self, state_dim: int, action_dim: int):  # Decimated
        super().__init__()
        # Define State
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.output_relu = nn.ReLU()
        # self.model = nn.LSTM(state_dim + 1, state_dim + 1, batch_first=True)
        # self.value = nn.Linear(state_dim + 1, 1)

    def forward(self, x):
        # out, hidden = self.model(x)
        # y = self.value(out[:, -1, :])
        y = self.output_relu(self.model(x))
        return y

    def estimate(self, state) -> torch.Tensor:
        return self.__call__(state)

    def get_action_dim(self) -> int:
        return self.action_dim

class SequenceValue(ValueEstimator, nn.Module):

    def __init__(self, state_dim: int, action_dim: int):  # Decimated
        super().__init__()
        self.action_dim = action_dim

        self.lstm = nn.LSTM(state_dim, state_dim, batch_first=True)
        self.linear = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        out, hidden = self.lstm(x)
        y = self.linear(out[:, -1, :])
        # TODO: place a relu here maybe
        return y

    def estimate(self, state) -> torch.Tensor:
        return self.__call__(state)

    def get_action_dim(self) -> int:
        return self.action_dim
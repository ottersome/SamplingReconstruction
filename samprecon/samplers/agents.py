"""
Defines your inteface for everything agent.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class Agent(ABC):
    @abstractmethod
    def act(self, observation: torch.Tensor):
        """
        Decides on an action based on an observed state.

        Args:
            observation (array-like): current environment state
        Returns:
            int: action(continuous or otherwise), specifically new decimation rate
        """
        pass


class SimpleAgent(Agent, nn.Module):
    def __init__(self, state_size: int):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Keep it as regression for now
        )

    def forward(self, state):
        # We want regressiont to be capt between 1 -> action_size:
        cap_regression = (
            nn.Sigmoid(self.sequence(state)) * self.action_size
        )  # CHECK: Doign this usually does not work
        return cap_regression

    def act(self, observation: torch.Tensor):
        return self.forward(observation)

"""
Defines your inteface for everything agent.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
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


class SoftmaxAgent(Agent, nn.Module):
    def __init__(self, sampling_budget: int, dec_range):
        super().__init__()
        self.dec_range = dec_range
        self.model = nn.Sequential(
            nn.Linear(sampling_budget, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, dec_range),  # Keep it as regression for now
        )

    def initialize_grad_hooks(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_backward_hook(print_grad_hook)

    def forward(self, state):
        # We want regressiont to be capt between 1 -> action_size:
        y = self.model(state)
        return F.softmax(y, dim=-1)

    def act(self, observation: torch.Tensor):
        return self.forward(observation)


class SimpleAgent(Agent, nn.Module):
    """
    Mostly used for continuous actions within a range
    """

    def __init__(self, sampling_budget: int, dec_range):
        super().__init__()
        self.dec_range = dec_range
        self.model = nn.Sequential(
            nn.Linear(sampling_budget, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Keep it as regression for now
            nn.Sigmoid(),
        )

    def initialize_grad_hooks(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_backward_hook(print_grad_hook)

    def forward(self, state):
        # We want regressiont to be capt between 1 -> action_size:
        y = self.model(state)
        cap_regression = y * self.dec_range
        return cap_regression

    def act(self, observation: torch.Tensor):
        return self.forward(observation)


def print_grad_hook(module, grad_input, grad_output):
    print("grad_input", grad_input)
    print("grad_output", grad_output)

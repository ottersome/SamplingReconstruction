"""
Defines your inteface for everything agent.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from samprecon.estimators.value_estimators import ValueEstimator


class Agent(ABC):
    @abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation (array-like): current environment state
        Returns:
            torch.Tensor: batch of action(continuous or otherwise), specifically new decimation rate
        """


class SoftmaxAgent(Agent, nn.Module):
    """
    This one will take a whole range of decimation rates it can freely choose from
    """

    def __init__(self, sampling_budget: int, dec_range):
        super().__init__()
        self.dec_range = dec_range
        self.model = nn.Sequential(
            nn.Linear(sampling_budget, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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


class EpsilonGreedyAgent(Agent):
    def __init__(self, value_estimator: ValueEstimator, epsilon, batch_size):
        self.value_estimator = value_estimator
        self.epsilon = epsilon

    def act(self, observation: torch.Tensor):
        batch_size = observation.shape[0]
        action_dim = self.value_estimator.get_action_dim()

        # Make a random binary choice based on epsilon
        choice = torch.rand(1) < self.epsilon
        if choice == 1:
            # Select a random action
            actions = torch.randint(action_dim, (batch_size,)).view(batch_size, -1)
        else:
            # Select maximal action
            action_values = self.value_estimator.estimate(observation)
            actions = (
                torch.argmax(action_values, dim=-1).view(batch_size, -1).to(torch.long)
            )
        return actions


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

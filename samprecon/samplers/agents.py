"""
Defines your inteface for everything agent.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from samprecon.estimators.value_estimators import ValueEstimator
from samprecon.utils.utils import setup_logger


class Agent(ABC):
    @abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation (array-like): current environment state
        Returns:
            torch.Tensor: batch of action(continuous or otherwise), specifically new decimation rate
        """

    @abstractmethod
    def change_property(self):
        pass

    @abstractmethod
    def evaluate(self, observation: torch.Tensor):
        """
        For those policies whose inference and training work a bit differently
        """
        pass


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

    def act(self, observation: torch.Tensor):
        return self.forward(observation)


class EpsilonGreedyAgent(Agent):
    def __init__(self, value_estimator: ValueEstimator, epsilon):
        self.value_estimator = value_estimator
        self.epsilon = epsilon
        self.logger = setup_logger("EpsilonGreedyAgent")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, observation: torch.Tensor):
        batch_size = observation.shape[0]
        action_dim = self.value_estimator.get_action_dim()
        actions = torch.zeros((batch_size), dtype=torch.long).to(self.device)

        rand_mask = torch.rand(batch_size).to(self.device) < self.epsilon
        actions[rand_mask] = torch.randint(
            action_dim, (torch.sum(rand_mask).tolist(),)
        ).to(self.device)
        actions[~rand_mask] = torch.argmin(
            self.value_estimator.estimate(observation[~rand_mask]), dim=-1
        )

        return actions.view(batch_size, -1), rand_mask

    def evaluate(self, observation: torch.Tensor):
        batch_size = observation.shape[0]

        actions = torch.argmin(self.value_estimator.estimate(observation), dim=-1)

        return (
            actions.view(batch_size, -1),
            None,
        )  # The none is there for when we use eval and act in the same place

    def change_property(self, **kwargs):
        # Change self property as per kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.logger.debug(f"Setting up {k} to {v}")


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

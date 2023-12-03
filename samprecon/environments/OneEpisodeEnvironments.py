from logging import INFO

import torch
import torch.nn as nn
import torch.optim as optim

from samprecon.generators.bandlimited import BandlimitedGenerator
from samprecon.reconstructors.NNReconstructors import NNReconstructor
from samprecon.reconstructors.reconstruct_intf import Reconstructor
from samprecon.samplers.agents import Agent
from samprecon.samplers.spatial_transformers import (
    LocalizationNework,
    differentiable_uniform_sampler,
)
from samprecon.utils.utils import setup_logger
from sp_sims.simulators.stochasticprocesses import BDStates

"""
No concept of Markovian chains here. 
Only one step
"""


class Model(nn.Module):
    def __init__(self):
        self.reconstructor = reconstructor
        self.sampling_optimizer = sampling_optimizer

    def forward(self, x):
        return self.reconstructor(self.sampling_optimizer.act(x))


class MarkovianUniformEnvironment:
    def __init__(
        self,
        # Modules
        state_generator: BDStates,
        sampling_arbiter: nn.Module,
        reconstructor: Reconstructor,
        # Some othe random varsj
        starting_decrate: int,
        sampling_budget: int = 4,  # This stays fixed
    ):
        self.state_generator = state_generator
        self.cur_decimation_rate = starting_decrate
        self.reconstructor = reconstructor
        self.sampling_arbiter = sampling_arbiter
        self.sampling_budget = sampling_budget
        self.optimizer = optim.Adam(list(self.sampling_arbiter.parameters()))

        self.prev_state = [0]
        self.criterion = nn.MSELoss()

        self.logger = setup_logger("MarkovianUniformEnvironment", INFO)

    def step(self):
        """
        Args:
        Returns:
            Stats to display
        """
        self.optimizer.zero_grad()

        action = self.sampling_arbiter.sample(self.prev_state)
        mask = differentiable_uniform_sampler(self.prev_state, action)

        # Generate Decimated Path
        new_state = torch.Tensor(
            self.state_generator.sample(action, self.sampling_budget)
        )

        dec_state = new_state * mask
        reconstruction = self.reconstructor.reconstruct(dec_state, action)
        loss = self.criterion(new_state, reconstruction)

        self.optimizer.step()
        # Evaluation
        self.prev_state = new_state

        return loss.item()


class OneEpisodeWeightedEnvironment:
    def __init__(
        self,
        signal_length,
        reconstructor: Reconstructor,
        criterion=nn.MSELoss(),
    ):
        self.N = signal_length
        self.localization_network = LocalizationNework(self.N, 1)
        self.reconstructor = reconstructor
        self.criterion = criterion
        self.optimizer = optim.Adam(
            list(self.localization_network.parameters())
            + list(self.reconstructor(self.reconstructor.parameters())),
        )

    def step(self):
        original_signals = self.signal_generator.generate_signals()

        # Get decimation rate
        decimation_intervals = self.localization_network(original_signals)

        # Sample with decimation rate
        sampled_signals = differentiable_uniform_sampler(
            original_signals, decimation_intervals
        )

        # Reconstructor
        estimated_signals = self.reconstructor(sampled_signals)

        # Get Loss
        loss = self.criterion(estimated_signals, original_signals)
        loss.backward()
        self.optimizer.step()

        return loss.item()

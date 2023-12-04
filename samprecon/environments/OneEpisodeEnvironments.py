from logging import INFO

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        reconstructor: nn.Module,
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

        self.last_action = starting_decrate
        # TODO: We will have to find a heuristic to initialize this
        length = int(starting_decrate * self.sampling_budget)
        self.prev_state = (
            torch.Tensor(
                self.state_generator.sample(starting_decrate, self.sampling_budget)
            )
            .to(torch.float)
            .view(length)
        )
        self.criterion = nn.MSELoss()

        self.logger = setup_logger("MarkovianUniformEnvironment", INFO)

    def step(self):
        """
        Args:
        Returns:
            Stats to display
        """
        self.optimizer.zero_grad()

        # TODO: We may be able to change this into a cumulative gradient
        action: torch.Tensor = (
            self.sampling_arbiter(self.prev_state[:: self.last_action])
            .view(1, -1)
            .to(torch.int)
        )

        # New State
        new_state = (
            torch.Tensor(self.state_generator.sample(action, self.sampling_budget))
            #.view(1, -1)
            #.to(torch.float)
        )
        new_state_oh = F.one_hot(
            new_state.view(1,-1).to(torch.long), num_classes=self.state_generator.max_state+1
        ).float()
        dec_state = differentiable_uniform_sampler(new_state_oh, action)


        reconstruction = self.reconstructor(
            dec_state, action, action.squeeze() * self.sampling_budget
        )

        loss = self.criterion(new_state_oh, reconstruction)
        self.optimizer.step()

        self.prev_state = new_state
        self.last_action = action.item()

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

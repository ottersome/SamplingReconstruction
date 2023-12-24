import copy
from abc import ABC, abstractmethod
from logging import INFO
from math import ceil
from typing import Any, Dict, List, Tuple

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
from sp_sims.detectors.pearsonneyman import take_guesses
from sp_sims.simulators.stochasticprocesses import BDStates

# TOREM : I dont think this is beign used at all
# class Model(nn.Module):
#     def __init__(self):
#         self.reconstructor = reconstructor
#         self.sampling_optimizer = sampling_optimizer
#
#     def forward(self, x):
#         return self.reconstructor(self.sampling_optimizer.act(x))


class Environment(ABC):
    def __init__(self, state_shape):
        pass

    @abstractmethod
    def step(self, cur_state, action) -> Tuple[float, torch.Tensor]:
        pass

    @abstractmethod
    def reset(self):
        pass

    # Define abstract property
    @property
    def state_shape(self) -> torch.Tensor:  # type : ignore
        pass


class MarkovianDualCumulativeEnvironment(Environment):
    def __init__(
        self,
        hyp0_rates: Dict[str, float],
        hyp1_rates: Dict[str, float],
        high_res_frequency: float,
        sampling_budget: int,
        highest_frequency: float,
        num_states: int,
        decimation_ranges: List[int],
        selection_probabilities: List[float],
        batch_size: int,
        episode_length: int,
    ):
        self.hyp0_rates = hyp0_rates
        self.hyp1_rates = hyp1_rates
        self.high_res_frequency = high_res_frequency
        self.episode_length = episode_length
        self.num_states = num_states
        self.sampling_budget = sampling_budget
        self.batch_size = batch_size
        self.selection_probabilities = torch.Tensor(selection_probabilities)
        self.decimation_ranges = decimation_ranges
        self.hypgens = [
            BDStates(self.hyp0_rates, highest_frequency, num_states, init_state=0),
            BDStates(self.hyp1_rates, highest_frequency, num_states, init_state=0),
        ]
        self.logger = setup_logger("MarkovianDualCumulativeEnvironment", INFO)

        # Blacnk slate
        self.cur_step = None

    def reset(self):
        """
        returns
        -------
            - decimated paths: Decimated paths after undergoing decimation
            - chosen initial decimation rates
            - chosen initial hypothesis at random
        """
        assert (
            self.cur_step == None
        ), "Make sure you reset environment after it finishes"

        # TODO: for now we are using initial state of 0
        hypothesis_selection = torch.multinomial(
            self.selection_probabilities, self.batch_size, replacement=True
        ).view(-1, 1)
        rdr = self._gen_random_decimation_rate()
        # Based on decimation rate we estimate length
        # lengths = 1 + (self.sampling_budget - 1) * rdr
        # max_len
        init_states = torch.zeros((self.batch_size, 1)).to(torch.long)  # type: ignore

        new_state = self._generate_decimated_observation(
            rdr, init_states, hypothesis_selection
        ).to(torch.long)
        self.cur_step = 0

        return new_state, rdr.view(self.batch_size, -1), hypothesis_selection

    def step(self, cur_state, actions):
        """
        Arguments
        ~~~~~~~~~
            cur_state:
                Rows will be for batch samples.
                Every row will contain: (hypothesis_selection + cur_rate + decimated_path)
                Where we only present the agent cur_rate + decimated_path and use hypothesis_selection for management
        """
        assert (
            len(cur_state.shape) == 2
            and cur_state.shape[1] == 2 + self.sampling_budget
            and self.cur_step != None
        ), "Incorrect input cur_state"

        cur_hyp = cur_state[:, 0]
        cur_periods = cur_state[:, 1]
        obs_state = cur_state[:, 1:]

        new_periods = (cur_periods + actions).clamp(1, self.decimation_ranges[-1])

        new_dec_path = self._generate_decimated_observation(
            new_periods, obs_state, cur_hyp
        )
        new_state = torch.cat((new_periods.unsqueeze(-1), new_dec_path), dim=-1)

        # Calculate Regret
        regret = self._calculate_regret(new_dec_path, cur_hyp)

        self.cur_step += 1

        if self.cur_step == self.episode_length:
            self._blank_slate()  # CHECK: for correctness

        return regret, new_state

    def _blank_slate(self):
        # self.hypothesis_selection = None
        self.total_path = None

    def _calculate_regret(self, new_state, cur_hyp):
        # First get the corresponding probabilities
        # probabilities = [hg.P for hg in self.hypgens]
        # In comes (batch_size) x () x (sampling_budget)

        # TODO: not like above, get probabilities under decimation rate.
        # New State containas the
        joint_probs = torch.Tensor((self.hypgens[0].P, self.hypgens[1].P))
        prev_steps = new_state[:-1]
        next_steps = new_state[1:]
        selection = joint_probs[cur_hyp, prev_steps, next_steps]

        # TODO: consider power of the test and all other nuances that we are ignoring for now
        # gueses = take_guesses()

    def _generate_decimated_observation(
        self,
        rates: torch.Tensor,  # (self.batch_size) x (1)
        cur_state: torch.Tensor,  # (self.batch_size) x (self.sampling_budget)
        true_hypotheses: torch.Tensor,  # (self.batch_size)  x (1)
    ) -> torch.Tensor:
        lengths = (self.sampling_budget - 1) * rates + 1
        max_len = torch.max(lengths)

        # TODO: See if just using an adjusted \theta * \Delta can make this faster and still be equivalent

        # We first generate the paths. One step at a time
        paths = torch.empty((self.batch_size, int(max_len)), dtype=torch.long)
        paths[:, 0] = cur_state[:, -1]
        # CHECK: For correctness
        for step in range(1, paths.shape[1]):
            paths[:, step] = self._single_state_step(
                paths[:, step - 1], true_hypotheses
            ).squeeze()

        # Then we decimate the paths
        decimated_paths = torch.Tensor(
            [
                paths[i, ::r][: self.sampling_budget].tolist()
                for i, r in enumerate(rates.squeeze())
            ]
        )

        return decimated_paths

    def _single_state_step(
        self, last_states: torch.Tensor, hypothesis_selection: torch.Tensor
    ):
        possible_probabilities = torch.Tensor((self.hypgens[0].P, self.hypgens[1].P))
        selection_probabilities = possible_probabilities[
            hypothesis_selection.squeeze(), last_states.squeeze(), :
        ]
        next_states = torch.multinomial(selection_probabilities, 1)
        return next_states

    def _gen_random_decimation_rate(self) -> torch.Tensor:
        mid_point = (self.decimation_ranges[1] + self.decimation_ranges[0]) // 2
        length = self.decimation_ranges[1] - self.decimation_ranges[0] + 1
        low = mid_point - length // 10
        high = mid_point + length // 10
        decimation_rates = torch.randint(low, high, (1, self.batch_size))
        return decimation_rates


class MarkovianUniformCumulativeEnvironment:
    def __init__(
        self,
        # Modules
        state_generator: BDStates,
        reconstructor: nn.Module,
        # Some othe random varsj
        starting_decrate: int,
        sampling_budget: int = 4,  # This stays fixed
    ):
        # self.device = torch.device("cuda")
        self.state_generator = state_generator
        self.num_states = state_generator.max_state + 1
        self.cur_decimation_rate = starting_decrate
        self.sampling_budget = sampling_budget
        # Modules
        self.reconstructor = reconstructor  # .to(self.device)

        # self.reconstructor_last_weights = list(self.reconstructor.state_dict().values())

        # TODO: We will have to find a heuristic to initialize this

        self.logger = setup_logger("MarkovianUniformEnvironment", INFO)
        self.done = False
        self.criterion = nn.NLLLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self, default_dec_rate):
        initial_state = (
            torch.Tensor(
                self.state_generator.sample(default_dec_rate, self.sampling_budget)
            ).to(torch.float)[:: int(default_dec_rate)]
        )[: self.sampling_budget]
        return initial_state

    def step(self, action: torch.Tensor) -> Dict[str, Any]:
        # TODO: We may be able to change this into a cumulative gradient
        # New State
        new_state = torch.Tensor(
            self.state_generator.sample(action, self.sampling_budget)
        ).to(self.device)

        new_state_oh = F.one_hot(
            new_state.view(1, -1).to(torch.long),
            num_classes=self.state_generator.max_state + 1,
        ).float()

        dec_state = differentiable_uniform_sampler(new_state_oh, action)

        reconstruction = self.reconstructor(
            dec_state,
            action,
            # 1 + torch.ceil(action.squeeze() * (self.sampling_budget - 1)),
        ).squeeze(0)

        logsoft_recon = F.log_softmax(reconstruction, dim=-1)
        regret = self.criterion(logsoft_recon, new_state.to(torch.long))

        # self.prev_state = new_state.to(torch.float)

        return (
            new_state[:: int(action)][: self.sampling_budget].view(1, -1),
            regret,
            self.done,
        )


# %%
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
        lr=0.01,
    ):
        # self.device = torch.device("cuda")
        self.state_generator = state_generator
        self.cur_decimation_rate = starting_decrate
        self.sampling_budget = sampling_budget
        # Modules
        self.reconstructor = reconstructor  # .to(self.device)
        self.sampling_arbiter = sampling_arbiter  # .to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.sampling_arbiter.parameters())
            + list(self.reconstructor.parameters()),
            lr=lr,
        )

        self.sampling_arbiter_last_weights = list(
            self.sampling_arbiter.state_dict().values()
        )
        self.reconstructor_last_weights = list(self.reconstructor.state_dict().values())

        self.last_action = starting_decrate
        # TODO: We will have to find a heuristic to initialize this
        length = 1 + ceil(
            self.last_action * (self.sampling_budget - 1)
        )  # One is given for free with inital state.
        self.prev_state = (
            torch.Tensor(
                self.state_generator.sample(self.last_action, self.sampling_budget)
            ).to(torch.float)
            # .view(length)
        )
        self.criterion = nn.NLLLoss()

        self.logger = setup_logger("MarkovianUniformEnvironment", INFO)

    def step(self):
        """
        Args:
        Returns:
            Stats to display
        """
        self.optimizer.zero_grad()

        # TODO: We may be able to change this into a cumulative gradient
        action: torch.Tensor = self.sampling_arbiter(
            self.prev_state[:: int(self.last_action)][: self.sampling_budget]
        ).view(1, -1)

        # New State
        new_state = torch.Tensor(
            self.state_generator.sample(action, self.sampling_budget)
        ).to(torch.long)

        new_state_oh = F.one_hot(
            new_state.view(1, -1),  # .to(torch.long),
            num_classes=self.state_generator.max_state + 1,
        ).float()

        dec_state = differentiable_uniform_sampler(new_state_oh, action)

        reconstruction = self.reconstructor(
            dec_state,
            action,
            # 1 + torch.ceil(action.squeeze() * (self.sampling_budget - 1)),
        )

        loss = self.criterion(reconstruction.squeeze(), new_state)
        loss.backward()

        self.optimizer.step()

        # DEBUG:
        # For sampling_arbiter
        differences = []
        for i, v in enumerate(self.sampling_arbiter.state_dict().values()):
            differences.append(
                torch.sum(torch.abs(v - self.sampling_arbiter_last_weights[i]))
            )
        differences_arbitrer = torch.sum(torch.tensor(differences))
        # self.logger.info(f"sum of weight difference arbiterer{differences:.8f}")
        # hard copy last weights
        self.sampling_arbiter_last_weights = [
            copy.deepcopy(v) for v in self.sampling_arbiter.state_dict().values()
        ]

        differences = []
        for i, v in enumerate(self.reconstructor.state_dict().values()):
            differences.append(
                torch.sum(torch.abs(v - self.reconstructor_last_weights[i]))
            )
        differences_recon = torch.sum(torch.Tensor(differences))
        # self.logger.info(f"Sum of weight difference  reconstructor{differences:.8f}")
        self.reconstructor_last_weights = [
            copy.deepcopy(v) for v in self.reconstructor.state_dict().values()
        ]

        self.prev_state = new_state.to(torch.float)
        self.last_action = action.item()

        tqdm_bar_info = f"Arbitrer {differences_arbitrer}, Recon {differences_recon}. Loss {loss.item()}"

        return loss.item(), tqdm_bar_info


# %%
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

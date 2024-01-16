import copy
from abc import ABC, abstractmethod
from logging import INFO
from math import ceil
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import expm

from samprecon.feedbacksigs.feedbacks import Feedbacks, LogEstimator, Reconstructor
from samprecon.reconstructors.reconstruct_intf import Reconstructor
from samprecon.samplers.spatial_transformers import (
    LocalizationNework,
    differentiable_uniform_sampler,
)
from samprecon.utils.utils import dec_rep_to_batched_rep, setup_logger
from sp_sims.detectors.pearsonneyman import take_guesses
from sp_sims.simulators.stochasticprocesses import BDStates
from sp_sims.utils.utils import get_q_mat

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
    def step(self, cur_state, action) -> Tuple[torch.Tensor, torch.LongTensor]:
        pass

    @abstractmethod
    def reset(self, batch_size):
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
        episode_length: int,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyp0_rates = hyp0_rates
        self.hyp1_rates = hyp1_rates
        self.rates = [hyp0_rates, hyp1_rates]
        self.high_res_frequency = high_res_frequency
        self.episode_length = episode_length
        self.num_states = num_states
        self.sampling_budget = sampling_budget
        self.selection_probabilities = torch.Tensor(selection_probabilities)
        self.decimation_ranges = decimation_ranges
        self.hypgens = [
            BDStates(self.hyp0_rates, highest_frequency, num_states, init_state=0),
            BDStates(self.hyp1_rates, highest_frequency, num_states, init_state=0),
        ]
        self.logger = setup_logger("MarkovianDualCumulativeEnvironment", INFO)

        # Regret criterion
        self.criterion = nn.BCELoss(reduction="none")

        # Blacnk slate
        self.cur_step = None
        self.batch_size = None  # Determined on reset

        # Learnable scalar for log likelihood
        self.log_likelihood_scalar = torch.tensor(
            [0.1], requires_grad=True, device=self.device
        )

    @property
    def learnable_pararms(self):
        return [self.log_likelihood_scalar]

    def reset(self, batch_size):
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

        self.batch_size = batch_size
        # TODO: for now we are using initial state of 0
        hypothesis_selection = (
            torch.multinomial(
                self.selection_probabilities, self.batch_size, replacement=True
            )
            .view(-1, 1)
            .to(self.device)
        )
        rdr = self._gen_random_decimation_rate().to(self.device)
        # Based on decimation rate we estimate length
        # lengths = 1 + (self.sampling_budget - 1) * rdr
        # max_len
        init_states = torch.zeros((self.batch_size, 1)).to(torch.long)  # type: ignore

        new_state = (
            self._generate_decimated_observation(rdr, init_states, hypothesis_selection)
            .to(torch.long)
            .to(self.device)
        )
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
        Returns
        ~~~~~~~
            new_state: [:,0] will be new current rates, [:,1:] will be observed decimated states
        """
        assert (
            len(cur_state.shape) == 2
            and cur_state.shape[1] == 2 + self.sampling_budget
            and self.cur_step != None
        ), "Incorrect input cur_state"

        cur_hyp = cur_state[:, 0]
        cur_periods = cur_state[:, 1]
        obs_state = cur_state[:, 1:]
        actions = actions.to(self.device)

        new_periods = (cur_periods + actions).clamp(1, self.decimation_ranges[-1])

        new_dec_path = (
            self._generate_decimated_observation(new_periods, obs_state, cur_hyp)
            .to(torch.long)
            .to(self.device)
        )
        new_state = torch.cat((new_periods.unsqueeze(-1), new_dec_path), dim=-1).to(
            torch.long
        )

        # OPTIM: Make this faster
        # Calculate new probabilities according to how fast it is going:
        probabilities_per_samples = []
        for i, p in enumerate(new_periods):
            q0 = get_q_mat(self.rates[0], self.num_states)
            q1 = get_q_mat(self.rates[1], self.num_states)
            p0 = expm(
                q0
                * (
                    self.high_res_frequency * p.item()
                )  # CHECK THIS MULTIPLCIAITON IS CORRECT
            )  # CHECK: There might not exist a unique exponential, or a solution at all
            p1 = expm(q1 * p.item())
            probabilities_per_samples.append([p0, p1])

        # Calculate Regret
        regret = self._calculate_regret(
            new_dec_path,
            cur_hyp,
            torch.Tensor(probabilities_per_samples).to(self.device),
        )

        self.cur_step += 1

        if self.cur_step == self.episode_length:
            self._blank_slate()  # CHECK: for correctness

        return regret, new_state

    def _blank_slate(self):
        # self.hypothesis_selection = None
        self.total_path = None
        self.cur_step = None
        self.batch_size = None

    def _calculate_regret(self, new_state, cur_hyp, probabilities):
        # First get the corresponding probabilities
        # probabilities = [hg.P for hg in self.hypgens]
        # In comes (batch_size) x () x (sampling_budget)

        # TODO: not like above, get probabilities under decimation rate.
        # New State containas the
        # joint_probs = torch.Tensor((self.hypgens[0].P, self.hypgens[1].P))

        prev_steps = new_state[:, :-1]
        next_steps = new_state[:, 1:]

        # OPTIM: make this nicer
        ones = torch.ones_like(prev_steps).to(self.device)
        zeros = torch.zeros_like(prev_steps).to(self.device)
        indexer = (
            torch.arange(probabilities.shape[0])
            .view(-1, 1)
            .repeat_interleave(next_steps.shape[1], dim=1)
        )
        selection_l0 = probabilities[indexer, zeros, prev_steps, next_steps]
        selection_l1 = probabilities[indexer, ones, prev_steps, next_steps]

        # LIkelihood ratio calculation
        log_sum_l0 = torch.sum(torch.log(selection_l0), dim=-1)
        log_sum_l1 = torch.sum(torch.log(selection_l1), dim=-1)

        ratio_matrix = log_sum_l0 - log_sum_l1 + self.log_likelihood_scalar

        # Argmax this boi
        decisions = 1 - F.softmax(ratio_matrix)

        # Cross entropy this boi
        regrets = self.criterion(decisions, cur_hyp.to(torch.float))

        # CHECK: This regret feels very "discrete".I fear it might get in the way of optimization

        return torch.Tensor(regrets)  # Remove last zero

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
        possible_probabilities = torch.Tensor(
            (self.hypgens[0].P, self.hypgens[1].P)
        ).to(self.device)
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
        # reconstructor: nn.Module,
        feedback: Feedbacks,
        # Some othe random varsj
        starting_decrate: int,
        max_decimation: int,
        sampling_budget: int,
    ):
        # self.device = torch.device("cuda")
        self.state_generator = state_generator
        self.num_states = state_generator.max_state
        self.cur_decimation_rate = starting_decrate
        self.sampling_budget = sampling_budget
        self.max_decimation = max_decimation
        # Modules
        # self.reconstructor = reconstructor  # .to(self.device)
        self.feedback = feedback

        # self.reconstructor_last_weights = list(self.reconstructor.state_dict().values())

        # TODO: We will have to find a heuristic to initialize this

        self.logger = setup_logger("MarkovianUniformEnvironment", INFO)
        self.done = False
        self.criterion = nn.NLLLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self, dec_rates, init_states):
        """
        Samples *a budgeted chain*, not a single state
        """
        assert dec_rates.shape[0] == init_states.shape[0]
        self.batch_size = dec_rates.shape[0]

        # initial_tape = (
        #     torch.Tensor(
        #         self.state_generator.sample(default_dec_rates, self.sampling_budget)
        #     ).to(torch.float)[:: int(default_dec_rates)]
        # )[: self.sampling_budget]
        sampled_tape, fullres_tape = self.state_generator.sample(
            dec_rates, self.sampling_budget, init_states
        )

        return sampled_tape, fullres_tape

    def step(
        self, cur_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[Any, Any, Any]:
        """
        Params:
        ~~~~~~~
            cur_state: any 2d tensor where the last column denotes the last states seen
            new_dec_rates: current decimation rate
        """
        # New State
        next_init_states = cur_state[:, -1]
        cur_decimation_period = cur_state[:, 0].view(-1,1)
        new_dec_factor = torch.clamp(
            cur_decimation_period + actions, 1, int(self.max_decimation)
        )

        # Perform actions 
        sampled_chain, fullres_chain = self.state_generator.sample(
            new_dec_factor, self.sampling_budget, next_init_states
        )
        sampled_chain = sampled_chain.to(self.device).to(torch.long)

        # dec_state = differentiable_uniform_sampler(new_state_oh, new_dec_rates) # TOREM:

        oh_fullres_sig = dec_rep_to_batched_rep(
            sampled_chain,
            new_dec_factor,  # CHECK: If first column contains periods
            self.sampling_budget,
            self.num_states + 1,  # For Padding
            add_position=False,  # TODO: See 'true' helps
        )
        regret = self.feedback(oh_fullres_sig, actions, fullres_chain)

        # actual_categories = torch.argmax(F.softmax(reconstruction, dim=-1), dim=-1)
        # self.logger.debug(f"Reconstruction sum {actual_categories}")
        # self.prev_state = new_state.to(torch.float)
        new_state = torch.cat((new_dec_factor, sampled_chain), dim=-1)

        return (  # CHECK: This probably unnecessary ::actions decimation
            # sampled_chain[:: int(actions)][: self.sampling_budget].view(1, -1),
            new_state,
            regret,
            self.done,
        )

    def _single_state_step(
        self, last_states: torch.Tensor, hypothesis_selection: torch.Tensor
    ):
        possible_probabilities = torch.Tensor(
            (self.hypgens[0].P, self.hypgens[1].P)
        ).to(self.device)
        selection_probabilities = possible_probabilities[
            hypothesis_selection.squeeze(), last_states.squeeze(), :
        ]
        next_states = torch.multinomial(selection_probabilities, 1)
        return next_states


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

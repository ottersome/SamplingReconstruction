"""
Just a class that will be useful for RL replay memory 
"""

import random
from collections import deque, namedtuple
from math import ceil
from typing import List

import torch
import torch.nn.functional as F

from samprecon.environments.OneEpisodeEnvironments import (
    Environment,
    MarkovianDualCumulativeEnvironment,
)

Transition = namedtuple("Transition", ("state", "action", "next_state", "regret"))


class ReplayBuffer:
    """
    Samples stored herein will not be dependent on the policy.
    As it is *paths*, that exhibit this dependency, not individual (s,a,s',r).
    Those are entirely dependent on the environment.
    """

    def __init__(
        self,
        sampbudget: int,
        path_length: int,
        environment: Environment,
        sampling_controls,
        decimation_ranges,
        buffer_size=None,
        batch_size=4,
    ):
        self.decimation_ranges = decimation_ranges
        self.sampling_controls = sampling_controls
        self.sampbudget = sampbudget
        self.path_length = path_length
        self.memory = deque(maxlen=buffer_size)
        self.environment = environment
        self.state_shape = environment.state_shape

        # Mostly for memory control
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate Furthest Sample

        # self.experience = namedtuple("Experience", field_names=["state", "samp_rate", "errors" ])
        # self.experience = namedtuple("Experience", field_names=["state", "samp_rate", "errors" ])

    def add(self, state, samp_rate, errors):
        # e = self.experience(state, samp_rate, errors)
        self.memory.append((state, samp_rate, errors))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.empty(
            (batch_size, 2**2)
        )  # TODO Remove hard code to the number of entries in generator matrix

        # The states will be encoded as tensors
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        # actions = torch.from_numpy(np.vstack([e.samp_rate for e in experiences if e is not None])).float()
        # errors = torch.from_numpy(np.vstack([e.errors for e in experiences if e is not None])).float()
        states = torch.from_numpy(
            np.vstack([e[0] for e in experiences if e is not None])
        ).float()
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])
        ).float()
        errors = torch.from_numpy(
            np.vstack([e[2] for e in experiences if e is not None])
        ).float()

        return states, actions, errors

    def __len__(self):
        return len(self.memory)

    def populate_replay_buffer(
        self,
        policy: torch.nn.Module,
        num_samples: int,
    ):
        # Number of batches
        num_batches = ceil(num_samples / self.batch_size)

        # Generated and to be saved for posterity
        gen_regrets = torch.zeros((num_samples, self.path_length))
        for batch_num in range(num_batches):
            # Generate the initial_states
            batch_offset = batch_num * self.batch_size
            dis_batch_size = (
                self.batch_size
                if batch_num != num_batches - 1
                else num_samples % self.batch_size
            )
            cur_state, cur_periods, hyp_selct = self.environment.reset()

            # Repeat hyp_selct across dimension 2
            # rdr_rpt = rdr.repeat_interleave(
            #     self.sampbudget,
            #     dim=1,
            # ).unsqueeze(-1)
            # cur_state_oh = F.one_hot(cur_state, num_classes=self.environment.num_states)  # type: ignore

            useful_state = torch.cat((hyp_selct, cur_periods, cur_state), dim=-1).to(
                torch.float
            )
            # Start the loop
            for step in range(self.path_length):
                # Decide on sampling rate
                action_probs = policy(useful_state[:, 1:])
                dist = torch.distributions.Categorical(action_probs)
                sampled_action = (dist.sample()).to(self.device)
                new_periods = torch.Tensor(
                    [self.sampling_controls[a] for a in sampled_action.squeeze()]
                ).to(torch.long)
                # cur_periods += (
                #     (
                #         torch.Tensor(
                #             [
                #                 self.sampling_controls[a]
                #                 for a in sampled_action.squeeze()
                #             ]
                #         )
                #         .to(torch.long)
                #         .to(self.device)
                #     )
                #     .clamp(min=1, max=self.decimation_ranges[-1])
                #     .view(self.batch_size, -1)
                # )
                # useful_state = torch.cat(
                #     (hyp_selct, cur_periods, cur_state), dim=-1
                # ).to(torch.float)

                # Obseve new state
                regrets, new_states = self.environment.step(
                    useful_state.to(torch.long), new_periods
                )
                useful_state = torch.cat((hyp_selct.unsqueeze(-1), new_states), dim=-1)
                cur_periods = new_periods

                # gen_regrets[batch_offset : batch_offset + dis_batch_size, :] = regrets

    def _populate_replay_buffer(
        self,
        policy: torch.nn.Module,
        num_of_paths: int,
        # guesses_per_rate=1000, # This is to be phased out
    ):
        # TODO: we might want to look ta  this actor-crtioc
        """
        We would ideally like to run this every time we get a significant change in our policy.
        Otherwise the samples added will be similar.
        """
        # Get Errors  # ACTIONS
        chose_hypothesis = torch.randint(2, (num_of_paths, 1))
        regrets = [0] * len(self.sampbudget)  # Amount of errors per action

        # TODO: PArallelize this through threads
        for i in range(num_of_paths):  # For Every State-Action
            # Generate Initial States
            cur_paths, cur_dec_rates, hyps = self.environment.reset()

            # Travel the Path with Current Policy.
            for step_no in range(self.path_length):
                # Take an action after observing the environment
                cur_states = torch.cat((cur_paths, cur_dec_rates), dim=-1)
                cur_actions = policy(cur_states)

                cur_states, dec_rates = self.environment()

            # Do Guesses
            for j in range(guesses_per_rate):  # Sample a bunch of paths
                # Get the corresponding Losses
                # TODO: Maybe scale down the errors /kj
                regrets[i] += (
                    multiplicity_guess(tmpSampTape, replicas, p0, p1) != true_hyps[j]
                )

        regrets = np.array(regrets) / guesses_per_rate
        # errors = np.array(errors)

        for i in range(num_of_paths):
            self.add(list(rates0[i]) + list(rates1[i]), smp_rates[i], regrets[i])

    def add_actions_to_buffer(self, states, actions, guesses_per_rate=1000):
        errors = [0] * len(actions)  # Amount of errors per action

        # TODO this could be parallelized into threads
        for i in range(actions.shape[0]):  # For Every State-Action
            true_hyps = np.random.choice(2, guesses_per_rate)

            p0 = expm(
                (1 / actions[i])
                * np.array(
                    [[-states[i, 0], states[i, 0]], [states[i, 1], -states[i, 1]]]
                )
            )  # 0=lam, 1=mu
            p1 = expm(
                (1 / actions[i])
                * np.array(
                    [[-states[i, 2], states[i, 2]], [states[i, 3], -states[i, 3]]]
                )
            )

            for j in range(guesses_per_rate):  # Sample a bunch of paths
                offset = true_hyps[j] * 1

                # roe = RaceOfExponentials(args2.length,states[i,offset:offset+2].detach(),state_limit=args2.state_limit)
                roe = RaceOfExponentials(
                    guesses_per_rate * 1 / (actions[i]),
                    states[i, offset : offset + 2].detach(),
                )
                holdTimes_tape, state_tape = roe.generate_history(args2.init_state)
                # Action Performance(Sampling At Rate)
                tape = quick_sample_budget(
                    actions[i], state_tape, holdTimes_tape, self.sampbudget
                )
                # tape = quick_sample(actions[i],state_tape,holdTimes_tape,args2.num_samples)# Could be that this was plain wrong.

                # Get the corresponding Losses
                # errors[i] += take_a_guess(tape, p0,p1)  != true_hyps[j]
                errors[i] += take_guess_multiplicity(tape, p0, p1) != true_hyps[j]

        errors = np.array(errors) / guesses_per_rate

        for i in range(states.shape[0]):
            self.add(list(states[i, :]), actions[i], errors[i])

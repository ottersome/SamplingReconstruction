"""
Just a class that will be useful for RL replay memory 
"""

import random
from collections import deque, namedtuple
from logging import INFO
from math import ceil
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from samprecon.environments.Environments import (
    Environment,
    MarkovianDualCumulativeEnvironment,
)
from samprecon.samplers.agents import Agent
from samprecon.utils.utils import setup_logger

Transition = namedtuple("Transition", ("state", "action", "next_state", "returns"))


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
        bundle_size=32,
        return_gamma: float = 0.9,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_gamma = return_gamma
        self.decimation_ranges = decimation_ranges
        self.sampling_controls = sampling_controls
        self.sampbudget = sampbudget
        self.path_length = path_length
        self.memory = deque(maxlen=buffer_size)
        self.environment = environment
        self.state_shape = environment.state_shape

        # TODO: add to arguments
        self.init_samples_to_take = 3

        # Mostly for memory control
        self.bundle_size = bundle_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = setup_logger("ReplayBuffer")

        # Calculate Furthest Sample

    def add(self, state, samp_rate, errors):
        # e = self.experience(state, samp_rate, errors)
        self.memory.append((state, samp_rate, errors))

    def sample(self, amount):
        experiences = Transition(*zip(*random.sample(self.memory, k=amount)))

        # states = torch.cat(experiences.state)
        # actions = torch.cat(experiences.action)
        # returns = torch.cat(experiences.returns)
        # next_state = torch.cat(experiences.next_state)

        states = torch.tensor(experiences.state, device=self.device)
        actions = torch.tensor(experiences.action, device=self.device)
        returns = torch.tensor(experiences.returns, device=self.device)
        next_state = torch.tensor(experiences.next_state, device=self.device)

        return states, actions, returns, next_state

    def __len__(self):
        return len(self.memory)

    def populate_replay_buffer(
        self,
        policy: Agent,
        num_samples: int,
    ):
        self.logger.debug(
            f"Populating Replay Buffer (of length {len(self.memory)}) with {num_samples} with bundle size {self.bundle_size}"
        )
        # Number of batches
        num_bundles = ceil(num_samples / self.bundle_size)  # type: ignore

        for bun in range(num_bundles):
            amount = min(self.bundle_size, num_samples - bun * self.bundle_size)
            # Generate the initial_states
            cur_states, cur_periods, true_hyps = self.environment.reset(
                amount
            )  # type:ignore
            observed_states, actions, regrets = self._batch_loop(
                policy, cur_states, cur_periods, true_hyps, amount
            )
            # returns = self._batch_returns(regrets)
            returns = regrets

            self.logger.debug(
                f"Adding new samples to replay memory with average regret {torch.mean(returns)}"
            )

            # Now we only take the first `self.init_samples_to_take`
            # for i in range(self.init_samples_to_take):
            for b in range(cur_states.shape[0]):
                for j in range(cur_states.shape[1] - 1):
                    self.memory.append(
                        Transition(
                            observed_states[b, j].tolist(),
                            actions[b, j].item(),
                            observed_states[b, j + 1].tolist(),
                            returns[b, j].item(),
                        )
                    )

        self.logger.debug("Replay Buffer populated")

    # def evaluate(self, observations: torch.Tensor, policy: Agent):
    #     for i in range(observations.shape[1]):
    #         cur_paths, cur_dec_rates, hyps = self.environment.reset()
    #         for step_no in range(self.path_length):
    #             cur_states = torch.cat((cur_dec_rates, cur_paths))
    #             cur_actions = policy.evaluate(cur_states)
    #     returns

    def evaluate_batch(self, policy, num_eval):
        self.logger.debug(f"Evaluating poliy")
        num_bundles = ceil(num_eval / self.bundle_size)
        l_obs_states = []
        l_actions = []
        l_regrets = []
        for bun in range(num_bundles):
            amount = min(self.bundle_size, num_eval - bun * self.bundle_size)
            cur_states, cur_periods, true_hyps = self.environment.reset(amount)
            obs_states, actions, regrets = self._batch_loop(
                policy, cur_states, cur_periods, true_hyps, amount, eval=True
            )
            l_obs_states.append(obs_states)
            l_actions.append(actions)
            l_regrets.append(regrets)
        ostates = torch.cat(l_obs_states)
        actions = torch.cat(l_actions)
        regrets = torch.cat(l_regrets)
        return ostates, actions, regrets

    def _batch_returns(self, regrets: torch.Tensor):
        returns = torch.zeros_like(regrets)
        returns[:, -1] = regrets[:, -1]
        for i in reversed(range(regrets.shape[1] - 1)):
            returns[:, i] = regrets[:, i] + self.return_gamma * returns[:, i + 1]

        return returns

    def _batch_loop(
        self,
        policy: Agent,
        cur_decimation: torch.Tensor,
        cur_periods: torch.Tensor,
        true_hyps: torch.Tensor,
        amount: int,
        eval=False,
    ):
        meta_state = torch.cat((true_hyps, cur_periods, cur_decimation), dim=-1).to(
            self.device
        )
        generated_regrets = torch.zeros((amount, self.path_length)).to(self.device)
        actions = torch.zeros((amount, self.path_length)).to(self.device)
        observed_states = torch.zeros(
            (amount, self.path_length, 1 + self.sampbudget)
        ).to(self.device)

        # Start the loop
        for step in range(self.path_length):
            # Decide on sampling rate
            sampled_action, mask = (
                policy.act(meta_state[:, 1:].to(torch.float))
                if eval == False
                else policy.evaluate(meta_state[:, 1:].to(torch.float))
            )  # type: ignore
            period_delta = torch.Tensor(
                [self.sampling_controls[a] for a in sampled_action.squeeze()]
            ).to(torch.long)

            observed_states[:, step] = meta_state[:, 1:]
            actions[:, step] = period_delta

            # Obseve new state
            regret, new_states = self.environment.step(
                meta_state.to(torch.long), period_delta
            )

            # Post-sim upadte
            meta_state = torch.cat((true_hyps, new_states), dim=-1)
            cur_periods = new_states[:, 0]
            generated_regrets[:, step] = regret

        return observed_states, actions, generated_regrets

    # def _populate_replay_buffer(
    #     self,
    #     policy: Agent,
    #     num_of_paths: int,
    #     # guesses_per_rate=1000, # This is to be phased out
    # ):
    #     # TODO: we might want to look ta  this actor-crtioc
    #     """
    #     We would ideally like to run this every time we get a significant change in our policy.
    #     Otherwise the samples added will be similar.
    #     """
    #     # Get Errors  # ACTIONS
    #     chose_hypothesis = torch.randint(2, (num_of_paths, 1))
    #     regrets = [0] * len(self.sampbudget)  # Amount of errors per action
    #
    #     # TODO: PArallelize this through threads
    #     for i in range(num_of_paths):  # For Every State-Action
    #         # Generate Initial States
    #         cur_paths, cur_dec_rates, hyps = self.environment.reset()
    #
    #         # Travel the Path with Current Policy.
    #         for step_no in range(self.path_length):
    #             # Take an action after observing the environment
    #             cur_states = torch.cat(
    #                 (cur_paths, cur_dec_rates), dim=-1
    #             )  # CHECK: this seems wrong cur_dec_rates should go at beggining
    #             cur_actions = policy.act(cur_states)
    #
    #             cur_states, dec_rates = self.environment()
    #
    #         # Do Guesses
    #         for j in range(guesses_per_rate):  # Sample a bunch of paths
    #             # Get the corresponding Losses
    #             # TODO: Maybe scale down the errors /kj
    #             regrets[i] += (
    #                 multiplicity_guess(tmpSampTape, replicas, p0, p1) != true_hyps[j]
    #             )
    #
    #     regrets = np.array(regrets) / guesses_per_rate
    #     # errors = np.array(errors)
    #
    #     for i in range(num_of_paths):
    #         self.add(list(rates0[i]) + list(rates1[i]), smp_rates[i], regrets[i])

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

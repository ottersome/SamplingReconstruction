# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %% [markdown]
# # Introduction
# This experiment will be simply of detection. It will attempt to learn a single environment
# This will be done by:
# - Using Likelihood Ratio as Optimal Test for feedback
# - Using the differentiabel algorithm we have already developed before
# - Probably using some notion of memory

# %% [python]
# Get all imports

from math import ceil

import numpy as np
import torch
import torch.optim as optim

from samprecon.environments.OneEpisodeEnvironments import (
    MarkovianDualCumulativeEnvironment,
)
from samprecon.estimators.value_estimators import ValueFunc
from samprecon.memory.replaymemory import ReplayBuffer
from samprecon.samplers.agents import EpsilonGreedyAgent
from samprecon.utils.utils import setup_logger

# %% [markdown]
# ## Setup all Constants

# %% [python]
hyp0_baseline_rates = {"lam": 1 / 10, "mu": 4 / 10}
hyp1_baseline_rates = {"lam": 4 / 10, "mu": 4 / 10}
logger = setup_logger("Main")

# Steering Wheel
sampling_controls = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
actions_to_idx = {v: i for i, v in enumerate(sampling_controls)}
action_space_size = len(sampling_controls)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_ACTOR = 1e-4
LR_CRITIC = 1e-4

sampling_budget = 10
highest_frequency = 1e-0
num_states = 4
avg_timespan = torch.mean(
    1
    / torch.tensor(
        list(hyp0_baseline_rates.values()) + list(hyp1_baseline_rates.values())
    )
)
decimation_ranges = [1, int(avg_timespan // highest_frequency * 4)]
episode_length = 12
init_policy_sampling = 32
batch_size = 4
target_net_update_epochs = 2

# %% [markdown]
# ## Setup Models

# %% [python]

# Setup the parameters and optimizers
# sampling_agent = SoftmaxAgent(sampling_budget + 1, len(sampling_controls)).to(device)
policy_net = ValueFunc(sampling_budget + 1, action_space_size)
target_net = ValueFunc(sampling_budget + 1, action_space_size)
target_net.eval()  # CHECK: if you have to load critic_new weights

# actor_optimizer = optim.Adam(sampling_agent.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(policy_net.parameters(), lr=LR_CRITIC)


# %% [markdown]
# ## Setup Environments
epsilon = 0.3
sampling_agent = EpsilonGreedyAgent(
    policy_net, epsilon, batch_size=batch_size
)  # CHECK: shoudl I use greedy new?

# Setup The Environment
dual_env = MarkovianDualCumulativeEnvironment(
    high_res_frequency=highest_frequency,
    episode_length=episode_length,
    hyp0_rates=hyp0_baseline_rates,
    hyp1_rates=hyp1_baseline_rates,
    sampling_budget=10,
    highest_frequency=highest_frequency,
    num_states=num_states,
    decimation_ranges=decimation_ranges,
    selection_probabilities=[0.5, 0.5],
    batch_size=batch_size,
)

# TODO: create a Q-Function Model

replay_buffer = ReplayBuffer(
    sampbudget=sampling_budget,
    path_length=episode_length,
    environment=dual_env,
    decimation_ranges=decimation_ranges,
    sampling_controls=sampling_controls,
    buffer_size=128,
    bundle_size=batch_size,
)

# %% [markdown]
# # Executions

# %% [python]

# Constants
epochs = 10
epsilon = 0.4

# %% [python]

for i in range(epochs):
    # Create some initial data using the initial policy
    replay_buffer.populate_replay_buffer(policy=sampling_agent, num_samples=32)

    # Sample from our history
    samples = replay_buffer.sample(batch_size=batch_size)

    # Learn from said samples
    buffer_len = len(replay_buffer)
    num_batches = ceil(buffer_len / batch_size)

    for bn in range(num_batches):
        # Sample Uniformly
        # TODO: create an exhaustive way of sampling from the buffer.
        states, actions, returns, next_states = replay_buffer.sample(
            batch_size=batch_size
        )
        actions_become_idx = torch.tensor(
            [actions_to_idx[a.item()] for a in actions], dtype=torch.long
        )

        policy_estimations = policy_net(states)

        policy_estimations_gathered = policy_estimations.gather(
            1, actions_become_idx.to(torch.long).view(-1, 1)
        )

        target_estimations = returns + target_net(next_states).max(1)[0].detach()

        # Loss
        loss = torch.nn.functional.mse_loss(
            policy_estimations_gathered, target_estimations
        )
        logger.info(f"Loss: {loss}")

        # Optimize the model
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()

    # See if you can update the target network
    if i % target_net_update_epochs == 0:
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()  # To be safe

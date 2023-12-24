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

import numpy as np
import torch

from samprecon.environments.OneEpisodeEnvironments import (
    MarkovianDualCumulativeEnvironment,
)
from samprecon.memory.replaymemory import ReplayBuffer
from samprecon.samplers.agents import SoftmaxAgent

# %% [markdown]
# ## Setup all Constants

# %% [python]
hyp0_baseline_rates = {"lam": 1 / 10, "mu": 4 / 10}
hyp1_baseline_rates = {"lam": 4 / 10, "mu": 4 / 10}

# Steering Wheel
sampling_controls = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# %% [markdown]
# ## Setup Environments

# %% [python]

# Setup the Agent
sampling_agent = SoftmaxAgent(sampling_budget + 1, len(sampling_controls)).to(device)

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
    batch_size=batch_size,
)

# %% [markdown]
# # Executions

# %% [python]

# Create some initial data using the initial policy
replay_buffer.populate_replay_buffer(policy=sampling_agent, num_samples=32)

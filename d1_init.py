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

# %% [markdown]
# ## Setup all Constants

# %% [python]
hyp0_baseline_rates = {"lam": 1 / 10, "mu": 4 / 10}
hyp1_baseline_rates = {"lam": 4 / 10, "mu": 4 / 10}
sampling_budget = 10
highest_frequency = 1e-0
num_states = 4
avg_timespan = torch.mean(
    1
    / torch.tensor(
        list(hyp0_baseline_rates.values()) + list(hyp1_baseline_rates.values())
    )
)
decimation_ranges = [1, avg_timespan // highest_frequency * 4]

# %% [markdown]
# ## Setup Environments

# %% [python]

dual_env = MarkovianDualCumulativeEnvironment(
    hyp0_rates=hyp0_baseline_rates,
    hyp1_rates=hyp1_baseline_rates,
    sampling_budget=10,
    highest_frequency=highest_frequency,
    num_states=num_states,
    decimation_ranges=decimation_ranges,
    selection_probabilities=[0.5, 0.5],
    parallel_paths=4,
)

# %% [markdown]
# # Executions

# %% [python]

dual_env.reset()

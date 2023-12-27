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
from tqdm import tqdm

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

# Set random seeds
np.random.seed(0)
torch.manual_seed(0)


# Steering Wheel
sampling_controls = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
actions_to_idx = {v: i for i, v in enumerate(sampling_controls)}
action_space_size = len(sampling_controls)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_ACTOR = 1e-3
LR_CRITIC = 1e-3

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
batch_size = 32
target_net_update_epochs = 5


epochs = 100
epsilon_start = 1.0
epsilon_end = 0.01

return_gamma = 0.99

epsilon_decay = epochs // 2
evaluation_frequency = epochs // 10

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
    policy_net, epsilon_start, batch_size=batch_size
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
    buffer_size=1024,
    bundle_size=batch_size,
    return_gamma=return_gamma,
)

# %% [markdown]
# # Executions

# %% [python]

# Constants

cur_epsilon = lambda frame: epsilon_end + (epsilon_start - epsilon_end) * np.exp(
    -1.0 * frame / epsilon_decay
)


def evaluate_performance(replay_buffer: ReplayBuffer):
    if len(replay_buffer) < 100:
        logger.debug(f"Not adding evaluation. Currently at {len(replay_buffer)}/100")
    eval_regrets = replay_buffer.evaluate_batch(sampling_agent)
    return eval_regrets


# %% [python]

evaluations = []
q_estimation_losses = []
e_bar = tqdm(range(epochs), desc="Epochs", leave=True, position=0)

for i in range(epochs):
    # Create some initial data using the initial policy
    e_bar.set_description("Populating replay buffer")
    addition = len(replay_buffer) // 10 if i > 0 else batch_size
    replay_buffer.populate_replay_buffer(
        policy=sampling_agent,
        num_samples=addition,
    )
    sampling_agent.change_property(epsilon=cur_epsilon(i))

    # Sample from our history
    samples = replay_buffer.sample(batch_size=batch_size)

    # Learn from said samples
    buffer_len = len(replay_buffer)
    num_batches = ceil(buffer_len / batch_size)

    e_bar.set_description("Batch optimization")
    b_bar = tqdm(range(num_batches), desc="Batches", leave=False, position=1)
    batch_qestimation_loss = []
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

        target_estimations = (
            returns + return_gamma * target_net(next_states).max(1)[0].detach()
        )

        # Loss
        loss = torch.nn.functional.mse_loss(
            policy_estimations_gathered, target_estimations
        )
        batch_qestimation_loss.append(loss.item())

        # Optimize the model
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
        b_bar.set_description(f"Loss: {loss.item():.3f}")
        b_bar.update(1)

    q_estimation_losses.append(torch.mean(torch.Tensor(batch_qestimation_loss)).item())

    # See if you can update the target network
    if i % target_net_update_epochs == 0:
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()  # To be safe
    if i % evaluation_frequency == 0:
        evaluations.append(evaluate_performance(replay_buffer).item())
    e_bar.update(1)

# %% [markdown]

# Evaluation

# %% [python]

import matplotlib.pyplot as plt

# Plot the evaluation regret
fig, axs = plt.subplots(2, 1, figsize=(14, 5))
axs[0].plot(
    np.arange(0, epochs, evaluation_frequency),
    evaluations,
)
axs[0].set_title("Evaluation of Performance")

axs[1].plot(range(epochs), q_estimation_losses, label="Loss Estimation")
axs[1].set_title("Q-Estimation Losses")


plt.tight_layout()
plt.show()

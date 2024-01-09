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

import copy
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
hyp1_baseline_rates = {"lam": 1 / 1, "mu": 4 / 1}
logger = setup_logger("Main")

# Set random seeds
np.random.seed(1)
torch.manual_seed(1)


# Steering Wheel
sampling_controls = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
actions_to_idx = {v: i for i, v in enumerate(sampling_controls)}
action_space_size = len(sampling_controls)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_ACTOR = 1e-3
LR_CRITIC = 1e-2

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
train_batch_size = 64
target_net_update_epochs = 2


epochs = 100
epsilon_start = 1.0
epsilon_end = 0.01

return_gamma = 0.90

epsilon_decay = 10
evaluation_frequency = 5


# %% [markdown]
# ## Setup Models

# %% [python]

# Setup the parameters and optimizers
# sampling_agent = SoftmaxAgent(sampling_budget + 1, len(sampling_controls)).to(device)
policy_net = ValueFunc(sampling_budget + 1, action_space_size).to(device)
target_net = ValueFunc(sampling_budget + 1, action_space_size).to(device)
target_net.eval()  # CHECK: if you have to load critic_new weights

# actor_optimizer = optim.Adam(sampling_agent.parameters(), lr=LR_ACTOR)


# %% [markdown]
# ## Setup Environments
sampling_agent = EpsilonGreedyAgent(policy_net, epsilon_start)

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
)

# TODO: create a Q-Function Model

replay_buffer = ReplayBuffer(
    sampbudget=sampling_budget,
    path_length=episode_length,
    environment=dual_env,
    decimation_ranges=decimation_ranges,
    sampling_controls=sampling_controls,
    buffer_size=4028,
    return_gamma=return_gamma,
)

critic_optimizer = optim.Adam(
    list(policy_net.parameters()) + dual_env.learnable_pararms, lr=LR_CRITIC
)

# %% [markdown]
# # Executions

# %% [python]

# # Test best sampling rate
# cur_states, cur_periods, true_hyps = self.environment.reset()  # type:ignore
# # Just take the first one
# state = cur_states[0]
# period = cur_periods[0]
# hyp = true_hyps[0]
# tries = 100
# for r in range(decimation_ranges[0], decimation_ranges[1]):
#     # Let multiple of the same
#     regrets = []
#     for t in range(tries):
#         meta_state = torch.cat((true_hyps, cur_periods, cur_decimation), dim=-1)
#
#         returns, new_states = dual_env.step(meta_state.to(torch.long), r)
#
#     avg_regrets = torch.mean(torch.Tensor(regrets)).item()
#

# %% [python]

# Constants

cur_epsilon = lambda frame: epsilon_end + (epsilon_start - epsilon_end) * np.exp(
    -1.0 * frame / epsilon_decay
)
cur_weights = [copy.deepcopy(v) for v in policy_net.state_dict().values()]


def evaluate_performance(replay_buffer: ReplayBuffer):
    if len(replay_buffer) < 100:
        logger.debug(f"Not adding evaluation. Currently at {len(replay_buffer)}/100")

    obs_states, actions, eval_returns = replay_buffer.evaluate_batch(
        sampling_agent, 100
    )

    # Estimated Returns
    # For now only using the first states
    first_states = obs_states[:, 0, :].squeeze()
    actions_become_idx = torch.tensor(
        [actions_to_idx[int(a.item())] for a in actions[:, 0]],
        dtype=torch.long,
        device=device,
    ).view(-1, 1)
    periods = obs_states[:, :, 0].squeeze()
    mean_period = torch.mean(periods).item()
    var_period = torch.var(periods).item()

    second = obs_states[:, 1, :].squeeze()
    estimation = policy_net(first_states).gather(dim=1, index=actions_become_idx)
    actual_value = eval_returns[:, 0].squeeze() + target_net(second).min(1)[0].detach()

    return (
        torch.mean(estimation).item(),
        torch.mean(actual_value).item(),
        mean_period,
        var_period,
    )


# %% [python]

evaluations_est = []
evaluations_actual = []
q_estimation_losses = []
mean_periods = []
var_periods = []
e_bar = tqdm(range(epochs), desc="Epochs", leave=True, position=0)
estimated_regrets = []
actual_regrets = []

for i in range(epochs):
    ebar_msg = ""

    # Performance so far
    eval_act_moving_avg = (
        np.mean(evaluations_actual[-10:]) if len(evaluations_actual) > 0 else 0
    )
    ebar_msg += f"Eval Actual: {eval_act_moving_avg:.3f} | "

    # Create some initial data using the initial policy
    e_bar.set_description(ebar_msg + " Populating replay buffer")
    addition = len(replay_buffer) // 10 if i > 0 else train_batch_size
    replay_buffer.populate_replay_buffer(
        policy=sampling_agent,
        num_samples=addition,
    )
    sampling_agent.change_property(epsilon=cur_epsilon(i))

    # Learn from said samples
    buffer_len = len(replay_buffer)
    num_batches = ceil(buffer_len / train_batch_size)
    # num_batches = 1

    e_bar.set_description(ebar_msg + " Batch optimization")
    b_bar = tqdm(range(num_batches), desc="Batches", leave=False, position=1)
    batch_qestimation_loss = []
    for bn in range(num_batches):
        # Sample Uniformly
        # TODO: create an exhaustive way of sampling from the buffer.
        states, actions, returns, next_states = replay_buffer.sample(
            amount=train_batch_size
        )
        actions_become_idx = torch.tensor(
            [actions_to_idx[a.item()] for a in actions],
            dtype=torch.long,  # type:ignore
            device=device,
        )

        policy_estimations = policy_net(states)

        policy_estimations_gathered = policy_estimations.gather(
            1, actions_become_idx.to(torch.long).view(-1, 1)
        )

        target_estimations = (
            returns + return_gamma * target_net(next_states).min(1)[0].detach()
        )

        # Loss
        loss = torch.nn.functional.mse_loss(
            policy_estimations_gathered, target_estimations
        )
        # loss = torch.nn.functional.smooth_l1_loss(
        #     policy_estimations_gathered, target_estimations
        # )
        batch_qestimation_loss.append(loss.item())

        # Optimize the model
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
        b_bar.set_description(f"Loss: {loss.item():.3f}")
        b_bar.update(1)

        # 🐛
        weight_difference = [
            torch.abs(cur_weights[i] - v).sum()
            for i, v in enumerate(policy_net.state_dict().values())
        ]
        cur_weights = [copy.deepcopy(v) for v in policy_net.state_dict().values()]
        total_sum = torch.sum(torch.Tensor(weight_difference))
        logger.debug(f" Difference of weights: {total_sum}")

    q_estimation_losses.append(torch.mean(torch.Tensor(batch_qestimation_loss)).item())

    # See if you can update the target network
    if i % target_net_update_epochs == 0:
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()  # To be safe
    if i % evaluation_frequency == 0:
        est, actual, mperiod, vperiod = evaluate_performance(replay_buffer)
        evaluations_est.append(est)
        evaluations_actual.append(actual)
        mean_periods.append(mperiod)
        var_periods.append(vperiod)
    e_bar.update(1)

# %% [markdown]

# Evaluation

# %% [python]

import matplotlib.pyplot as plt

# Plot the evaluation regret
# fig, axs = pl(t.subplots(2, 2, figsize=(14, 14))
plt.figure(figsize=(14, 14))

plt.subplot(2, 1, 1)

# Axes below merge into one
plt.plot(np.arange(0, epochs, evaluation_frequency), mean_periods, label="Mean Actions")
var_pos = np.array(mean_periods) + np.array(var_periods)
var_neg = np.array(mean_periods) - np.array(var_periods)
plt.fill_between(
    np.arange(0, epochs, evaluation_frequency),
    var_pos,
    var_neg,
    alpha=0.2,
    label="Variance",
)

# Now  like above but a graph that displays means and averages

plt.legend()
plt.subplot(2, 2, 3)
plt.plot(
    np.arange(0, epochs, evaluation_frequency),
    evaluations_est,
    label="Avg Estimated Evaluation",
)
plt.plot(
    np.arange(0, epochs, evaluation_frequency),
    evaluations_actual,
    label="Avg Actual Evaluation",
)
plt.title("Regret")
plt.legend()

plt.subplot(2, 2, 4)

plt.plot(range(epochs), q_estimation_losses, label="Loss Estimation")
plt.title("Q-Estimation Losses")


plt.tight_layout()
plt.show()
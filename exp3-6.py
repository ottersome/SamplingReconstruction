# %% [markdown]
# # Introduction
#
# This notebook will change from the previous in that we will not have all possible action decimation rates available but lets say something along the lines of multiples. i.e. 1,2,4,8.
#
# We will also include the action in the state tape. So that it knows how fast it is going.
#
# Will probably also add replay memory

from time import time

import matplotlib.pyplot as plt

# %%
import numpy as np
import numpy.random as rnd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from samprecon.environments.OneEpisodeEnvironments import (
    MarkovianUniformCumulativeEnvironment,
)
from samprecon.estimators.value_estimators import ValueFunc
from samprecon.reconstructors.NNReconstructors import RNNReconstructor
from samprecon.samplers.agents import SoftmaxAgent
from samprecon.utils.rl_utils import calculate_returns
from sp_sims.simulators.stochasticprocesses import BDStates

plt.style.use("rose-pine-dawn")
rnd.seed(int(time()))

# %%
# Generate Environments on which to learn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
high_res_delta = 1e-0  # For generating the dataset and later sample
baseline_rates = {"lam": 1 / 10, "mu": 4 / 10}
epochs = 300
lenth_of_episode = 15
step_path_length = 1
sampling_budget = 32
used_path_length = 64  # So that we can let the process reach stationarity and take samples from stationary distribution
num_states = 4
avg_span = np.mean(1 / np.array(list(baseline_rates.values())))
max_decimation = (
    avg_span / high_res_delta
) * 4  # Max decimation factor #CHECK: Maybe not divide by 2
decimation_steps = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
current_decimation_factor = (
    torch.Tensor([int(avg_span // high_res_delta)])  # We can start somewhere in between
    .to(torch.long)
    .to(device)
    .view(-1, 1)
)  # View is because we might want to batch it
print(f"Decimation factor is {current_decimation_factor}")
# Set random seed with time for randomnessj

# %% [markdown]
# # Declarations

# %%
# Initialize context first
state_generator = BDStates(baseline_rates, high_res_delta, num_states)
# sampling_arbiter.initialize_grad_hooks()
reconstructor = RNNReconstructor(
    amnt_states=num_states, max_decimation_rate=max_decimation
).to(device)
# reconstructor.initialize_grad_hooks()
valueEst = ValueFunc(num_states).to(device)
gamma = 0.9

# %% [markdown]
# # RL Loop

# %%
import copy

env = MarkovianUniformCumulativeEnvironment(
    state_generator=state_generator,
    reconstructor=reconstructor,
    starting_decrate=current_decimation_factor,
    sampling_budget=sampling_budget,
)
ebar = tqdm(range(epochs), desc="Epochs", position=0)
sampling_agent = SoftmaxAgent(sampling_budget + 1, len(decimation_steps)).to(
    device
)  # +1 for the decimation factor
# sampling_agent.initialize_grad_hooks()
optimizer = torch.optim.Adam(
    list(reconstructor.parameters())
    + list(sampling_agent.parameters())
    + list(valueEst.parameters()),
    lr=1e-2,
)
# val_opt = torch.optim.Adam(valueEst.parameters(), lr=1e-2)
# Scheduler with warmpu
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
e_returns = []
reconstructor_last_weights = list(reconstructor.state_dict().values())
sampler_last_weights = list(sampling_agent.state_dict().values())
for epoch in range(epochs):
    # We generate a single step from the generator process
    leave = epoch == epochs - 1

    sbar = tqdm(range(lenth_of_episode), desc="Steps", leave=leave, position=1)
    rewards = []
    log_probs = []
    val_ests = []
    states = [
        torch.cat(
            (
                current_decimation_factor,
                env.reset(current_decimation_factor).view(1, -1).to(device),
            ),
            dim=-1,
        )
    ]

    for step in range(lenth_of_episode):
        # with torch.autograd.set_detect_anomaly(True):
        cur_state = states[-1]
        action_probs = sampling_agent(cur_state[: sampling_budget + 2]).to(device)
        dist = torch.distributions.Categorical(action_probs)

        sampled_action = (dist.sample()).to(
            device
        )  # So as to not sample 0 (and end up dividing by zero)

        current_decimation_factor += (
            torch.Tensor([decimation_steps[sampled_action.item()]])
            .to(torch.long)
            .to(device)
        )
        # current_decimation_factor = max(1,min(current_decimation_factor, max_decimation))
        current_decimation_factor = torch.clamp(
            current_decimation_factor, min=1, max=max_decimation
        )

        # TODO: Make batch friendly
        dec_steps = (
            torch.arange(
                0,
                current_decimation_factor.squeeze() * sampling_budget,
                current_decimation_factor.squeeze(),
            )
            .view(-1, 1)
            .to(device)
        )
        one_hot_cur_state = F.one_hot(
            cur_state[0, 1 : sampling_budget + 1].to(torch.long), num_classes=num_states
        ).to(device)
        non_amb_state = (
            torch.cat((one_hot_cur_state, dec_steps), dim=-1).to(torch.float).to(device)
        ).view(1, sampling_budget, -1)
        val_ests.append(valueEst(non_amb_state).to(device))

        new_state, regret, done = env.step(current_decimation_factor)

        new_state_w_rate = torch.cat(
            (current_decimation_factor, new_state),
            dim=-1,
        )

        states.append(new_state_w_rate.to(device))
        rewards.append(regret)
        log_probs.append(dist.log_prob(sampled_action))

        sbar.set_description(f"At step {step}, Regret: {regret}")
        sbar.update(1)

    returns = calculate_returns(rewards, gamma)
    e_returns.append(returns[0].item())
    policy_regrets = []
    value_loss = []

    for lp, val_est, r in zip(log_probs[:3], val_ests[:3], returns[:3]):
        disadvantage = r - val_est.item()
        policy_regrets.append(
            -lp * disadvantage
        )  # TODO: this might require a negative sign
        value_loss.append(
            F.mse_loss(val_est, torch.Tensor([r.item()]).view(1, -1).to(device))
        )
    # We update the whole thingko
    policy_loss = torch.stack(policy_regrets).sum()
    value_loss = torch.stack(value_loss).mean()

    # optimze:
    optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    optimizer.step()
    scheduler.step()

    # 🐛 Debugging
    differences = []
    for i, v in enumerate(sampling_agent.state_dict().values()):
        differences.append(torch.sum(torch.abs(v - sampler_last_weights[i])))
    differences_arbitrer = torch.sum(torch.tensor(differences))
    # hard copy last weights
    sampler_last_weights = [
        copy.deepcopy(v) for v in sampling_agent.state_dict().values()
    ]
    differences = []
    for i, v in enumerate(reconstructor.state_dict().values()):
        differences.append(torch.sum(torch.abs(v - reconstructor_last_weights[i])))
    differences_recon = torch.sum(torch.Tensor(differences))
    reconstructor_last_weights = [
        copy.deepcopy(v) for v in reconstructor.state_dict().values()
    ]
    print(
        f"Differences are : Sampler: {differences_arbitrer}, Reconstrctor: {differences_recon}"
    )

    # 🐛 End Debuggin

    moving_avg_loss = np.mean(e_returns[-3:]) if epoch > 3 else np.mean(e_returns)
    ebar.set_description(f"Epoch Mean Regret: {moving_avg_loss}")
    ebar.update(1)
    # We get reward based on how close we got to maximum information
# Show Losses
plt.figure(figsize=(10, 5))
plt.title("Regret")
plt.xlabel("Epochs")
plt.ylabel("Loss (NLL)")
plt.plot(e_returns)
plt.show()

# %%
# Get time in nice format
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
# SaveModels
torch.save(reconstructor.state_dict(), f"models/reconstructor_{date_time}.pt")
torch.save(sampling_agent.state_dict(), f"models/sampling_agent_{date_time}.pt")


# %%
def hard_decimation_of_state(
    high_freq_signal: torch.Tensor, rate: int, sampling_budget: int, num_classes: int
):
    blank_slate = torch.zeros_like(high_freq_signal)
    seq_len = len(blank_slate)
    samples = high_freq_signal[:: rate.squeeze()][:sampling_budget]
    for i, sample in enumerate(samples):
        blank_slate[i * rate] = sample
    # turn blank_slate into one hot
    one_hot = F.one_hot(blank_slate.to(torch.long), num_classes=num_classes).view(
        1, -1, num_classes
    )
    return one_hot


sampling_agent.eval()
reconstructor.eval()

chosen_actions = []

# Visually confirm proper reconstruction.
num_examples = 3

fig, axs = plt.subplots(num_examples, 1, figsize=(10, 15))
# Start with some previous state.

current_decimation_factor = (
    torch.Tensor([int(avg_span // high_res_delta)])  # We can start somewhere in between
    .to(torch.long)
    .to(device)
    .view(-1, 1)
)  # View is because we might want to batch it
states = [
    torch.cat(
        [
            current_decimation_factor,
            env.reset(current_decimation_factor).view(1, -1).to(device),
        ],dim=-1
    )
]


for ne in range(num_examples):
    cur_state = states[-1]
    # Maybe do argmax instead of sampling
    action_probs = sampling_agent(cur_state[: 1 + sampling_budget])
    # dist = torch.distributions.Categorical(action_probs)
    # sampled_action = dist.sample() + 1 # So as to not sample 0 (and end up dividing by zero)
    max_action = (torch.argmax(action_probs) + 1).view(1, 1)
    chosen_actions.append(max_action)

    new_state = torch.Tensor(state_generator.sample(max_action, sampling_budget)).to(
        device
    )

    new_state_oh = F.one_hot(
        new_state.view(1, -1).to(torch.long),
        num_classes=state_generator.max_state + 1,
    ).float()

    dec_state = hard_decimation_of_state(
        new_state, max_action, sampling_budget, num_states
    )

    reconstruction_probs = F.softmax(
        reconstructor(
            dec_state.to(torch.float),
            max_action.to(torch.float),
        ),
        dim=-1,
    )
    reconstruction_states = (
        torch.argmax(reconstruction_probs, dim=-1).cpu().detach().numpy().squeeze()
    )

    states.append(torch.cat([max_action, new_state[:sampling_budget].view(1,-1)],dim=-1))
    new_state = new_state.cpu().detach().numpy()
    max_action = max_action.cpu().detach().numpy()

    # Do plotting here
    axs[ne].plot(
        np.arange(len(new_state)),
        new_state,
        drawstyle="steps-post",
        label="Full resolution",
    )  # , marker="^",markersize=3)
    # Plot Samples
    dec_x = np.arange(sampling_budget) * (int(max_action))
    axs[ne].scatter(
        dec_x,
        new_state[:: int(max_action)][:sampling_budget],
        label="Decimated",
        marker="o",
        color="r",
        s=30,
    )
    axs[ne].set_title(f"Results for Experiment {ne}")

    # Plot Reconstrunction
    axs[ne].plot(
        np.arange(len(reconstruction_states)),
        reconstruction_states,
        label="Reconstruction",
    )  # , marker="x",markersize=3)
    axs[ne].legend()
plt.tight_layout()
plt.show()

print(f"Choices of actions were {chosen_actions}")

# %%

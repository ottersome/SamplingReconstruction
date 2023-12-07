from re import T

import numpy.typing as npt
import torch
import torch.nn as nn
from torch.nn.modules import distance


class LocalizationNework(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )
        return self.model

    def forward(self, state):
        return self.model(state)


"""
ToThink: How to even feed this represntation to the network. 
We might need to tell it \Delta t between samples. 
Maybe: We might phase this one out
"""


def differentiable_sampler(input_signal, sampling_weights, desired_length):
    # Get Indices of heightest Weights
    top_indices = torch.topk(sampling_weights, desired_length).indices

    # Sort of Indices to maintain temporal order
    top_indices_sorted = torch.sort(top_indices).values

    return input_signal[top_indices_sorted]


def generate_sigmoid_mask(
    batch_size: int,
    signal_length: int,
    decimation_intervals: npt.ArrayLike,
    sharpness=1,
):
    masks = torch.zeros(batch_size, signal_length)
    for i in range(batch_size):
        for j in range(signal_length):
            distance_to_nearest_sample = torch.min(
                torch.fmod(torch.tensor([j]), decimation_intervals[i]),
                # decimation_intervals[i]
                torch.tensor(decimation_intervals[i])
                - torch.fmod(torch.tensor([j]), decimation_intervals[i]),
            )
            distance = torch.Tensor((distance_to_nearest_sample, 10))
            distance_m = torch.min(distance)
            exp = torch.exp(distance_m)
            divisor = torch.min(torch.Tensor((exp, 1000)))
            if divisor == 0:
                print("What the fuck")
            masks[i, j] = 1 / divisor

    return masks


def differentiable_uniform_sampler(
    input_signals: torch.Tensor, decimation_interval: torch.Tensor
):
    # batch_size, signal_length = input_signals.shape
    signal_length = input_signals.shape[1]
    # Esignal_length = len(input_signals)
    batch_size = 1  # for now
    mask = generate_sigmoid_mask(batch_size, signal_length, decimation_interval).view(
        1, -1, 1
    )
    mask = mask.repeat_interleave(input_signals.shape[2], dim=2)
    if input_signals.is_cuda:
        mask = mask.cuda()

    masked_signals = input_signals * mask
    return masked_signals
    # Derive a mask from the decimation internval, to covolve a sample, to covolve a sample

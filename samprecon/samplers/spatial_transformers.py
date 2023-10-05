from re import T
import torch.nn as nn
import torch
from torch.nn.modules import distance
import numpy.typing as npt

class LocalizationNework(nn.Module):
    def __init__(self, in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,out_dim),
            )
        return self.model

    def forward(self, state):
        return self.model(state)


"""
ToThink: How to even feed this represntation to the network. 
We might need to tell it \Delta t between samples. 
Maybe: We might phase this one out
"""
def differentiable_sampler(input_signal, sampling_weights,desired_length):
    # Get Indices of heightest Weights
    top_indices = torch.topk(sampling_weights, desired_length).indices
    
    # Sort of Indices to maintain temporal order
    top_indices_sorted = torch.sort(top_indices).values

    return input_signal[top_indices_sorted]


def generate_sigmoid_mask(
        batch_size: int,
        signal_length: int,
        decimation_intervals: npt.ArrayLike,
        sharpness=10):
    masks = torch.zeros(batch_size, signal_length)
    for i in range(batch_size):
        for j in range(signal_length):
            distance_to_nearest_sample = torch.min(torch.fmod(torch.tensor([j]),decimation_intervals[i]), 
                                                   torch.tensor(decimation_intervals[i]) - torch.fmod(torch.tensor([j]),decimation_intervals[i]))
            masks[i,j] = 1/(1+torch.exp(-sharpness * (1-distance_to_nearest_sample)))

    return masks

def differentiable_uniform_sampler(input_signals, decimation_interval):
    batch_size, signal_length = input_signal.shape
    mask = generate_sigmoid_mask(batch_size,signal_length, decimation_interval)
    if input_signal.is_cuda():
        mask = mask.cuda()

    masked_signals = input_signals * mask
    return masked_signals
    # Derive a mask from the decimation internval, to covolve a sample, to covolve a sample

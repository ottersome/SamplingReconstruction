"""
Any utils related to Reinforcement Learning Computations
"""

import torch


def calculate_returns(rewards, gamma):
    R = torch.zeros((rewards.shape[0]))
    n = len(rewards)
    returns = torch.zeros_like(rewards)

    for i in range(returns.shape[1]-1,-1,-1):
        returns[:,i] = rewards[:,i] + gamma * R

    return returns

"""
Any utils related to Reinforcement Learning Computations
"""


def calculate_returns(rewards, gamma):
    returns = []
    R = 0
    n = len(rewards)
    for i, r in enumerate(reversed(rewards)):
        R = r + gamma * R
        returns.insert(0, R)

    return returns

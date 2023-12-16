"""
Just a class that will be useful for RL replay memory 
"""

import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """
    Replay memory class for RL
    """

    def __init__(self, capacity):
        """
        Initialize the memory with a capacity
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """
        Push a transition into the memory
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Get a random sample from the memory
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Get the length of the memory
        """
        return len(self.memory)

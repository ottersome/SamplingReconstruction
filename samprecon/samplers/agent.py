"""
Defines your inteface for everything agent.
"""

from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, observation):
        """
        Decides on an action based on an observed state.

        Args:
            observation (array-like): current environment state
        Returns:
            int: action(continuous or otherwise)
        """
        pass


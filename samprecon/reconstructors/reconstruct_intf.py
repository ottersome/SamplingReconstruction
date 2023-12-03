from abc import ABC, abstractmethod
from typing import List


class Reconstructor(ABC):
    @abstractmethod
    def __init__(self, decimation_rate, **kwargs):
        """
        Args:
            decimation_rate: How many samples of original signal we represent with single one.
        """
        pass

    @abstractmethod
    def reconstruct(self, subsampled_signal: List, dec_rate: int):
        """
        Reconstruct a signal given a subsampled version.

        Args:
            subsampled_signal (torch.Tensor): The subsampled signal
        """
        pass

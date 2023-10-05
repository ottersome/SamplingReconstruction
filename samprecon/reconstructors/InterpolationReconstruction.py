import torch
import numpy as np
from .reconstruct_intf import Reconstructor

class InterpolationReconstructor(Reconstructor):
    def __init__(self, decimation_rate):
        self.decimation_rate = decimation_rate

    def reconstruct(self, subsampled_signal):
        subsampled_signal_np = subsampled_signal.numpy()
        subsampled_time = np.arange(0,len(subsampled_signal_np)*self.decimation_rate,
                                    self.decimation_rate)
        full_time = np.arange(0,len(subsampled_signal_np)*self.decimation_rate)

        reconstructed_signal = np.interp(full_time, subsampled_time, subsampled_signal)
        return torch.tensor(reconstructed_signal)

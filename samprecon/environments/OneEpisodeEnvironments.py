import torch.nn as nn
import torch
import torch.optim as optim
from samprecon.generators.bandlimited import BandlimitedGenerator
from samprecon.reconstructors.reconstruct_intf import Reconstructor
from samprecon.samplers.spatial_transformers import LocalizationNework, differentiable_uniform_sampler

"""
No concept of Markovian chains here. 
Only one step
"""
class OneEpisodeUnifiormEnvironment():
    def __init__(self,
            signal_generator:BandlimitedGenerator,
            reconstructor: Reconstructor):

        self.signal_generator =  signal_generator
        self.reconstructor = reconstructor


    def step(self):
        """
        Args:
        Returns:
            Stats to display
        """
        original_signals =  self.signal_generator.generate_signals()

        # Get subsampling Rate


        # subsampled_signal = 
        # reconstructed_signal = 



class OneEpisodeWeightedEnvironment():
    def __init__(self, signal_length,
                 reconstructor:Reconstructor,
                 criterion = nn.MSELoss(),
                 ):
        self.N = signal_length
        self.localization_network = LocalizationNework(self.N,1)
        self.reconstructor = reconstructor
        self.criterion = criterion
        self.optimizer = optim.Adam(list(self.localization_network.parameters())
                                    + list(self.reconstructor(self.reconstructor.parameters())),
                                    )

    def step(self):
        original_signals = self.signal_generator.generate_signals()

        # Get decimation rate
        decimation_intervals =  self.localization_network(original_signals)

        # Sample with decimation rate
        sampled_signals  =  differentiable_uniform_sampler(original_signals, decimation_intervals)

        # Reconstructor
        estimated_signals = self.reconstructor(sampled_signals)

        # Get Loss
        loss = self.criterion(estimated_signals,original_signals)
        loss.backward()
        self.optimizer.step()

        return loss.item()






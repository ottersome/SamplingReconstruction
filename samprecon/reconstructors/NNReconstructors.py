import torch.nn as nn
from .reconstruct_intf import Reconstructor

class NNReconstructor(Reconstructor):
    def __init__(self,decimation_rate, **kwargs):
        """
        Args:
            others: same as interface
            **kwargs:
                sub_length (int): Size of input. i.e. size of subsampled signal
                full_length (int): Size of final reconstructed signal.
        """
        self.decimation_rate = decimation_rate # Unused basically
        self.sub_length = kwargs['sub_length']
        self.full_length = kwargs['full_length']
        ## Build Model
        self.model  = nn.Sequential(
            nn.Linear(self.sub_length,128),
            nn.ReLU(),
            nn.Linear(self.sub_length,256),
            nn.ReLU(),
            nn.Linear(256,self.full_length),
            )
        return self.model

    def reconstruct(self, subsampled_signal):
        return self.model(subsampled_signal)
class WideNN(nn.Module):

    def __init__(self, initial_res, final_res):
        super(WideNN,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(initial_res, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, final_res)
        )

    def forward(self, x):
        return self.fc(x)
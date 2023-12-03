from typing import List

import torch
import torch.nn as nn

from .reconstruct_intf import Reconstructor


class RNNResconstructor(Reconstructor, nn.Module):
    def __init__(
        self,
        sample_budget: int,
        hidden_size: int,
    ):
        """
        Args:
            others: same as interface
            **kwargs:
                sub_length (int): Size of input. i.e. size of subsampled signal
                full_length (int): Size of final reconstructed signal.
        """
        super(nn.Module).__init__()
        # Take a variable length scalar input signals
        self.hidden_size = hidden_size

        ## Build Model
        self.model = nn.Sequential(
            nn.Linear(self.sample_buget, 128),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.full_length),
        )
        return self.model

    def reconstruct(self, subsampled_signal: List, rate):
        # For now vstack the rate into a new vertical axis
        x = torch.vstack(
            [torch.Tensor(subsampled_signal), torch.full_like(subsampled_signal, rate)],  # type: ignore
        )
        x = self.rnn(x)
        y = self.model(x)

        return y


class NNReconstructor(Reconstructor, nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            others: same as interface
            **kwargs:
                sub_length (int): Size of input. i.e. size of subsampled signal
                full_length (int): Size of final reconstructed signal.
        """
        super(nn.Module).__init__()
        self.sub_length = kwargs["sub_length"]
        self.full_length = kwargs["full_length"]
        ## Build Model
        self.model = nn.Sequential(
            nn.Linear(self.sub_length, 128),
            nn.ReLU(),
            nn.Linear(self.sub_length, 256),
            nn.ReLU(),
            nn.Linear(256, self.full_length),
        )
        return self.model

    def reconstruct(self, subsampled_signal: List, rate):
        # Concatenate signal and rate into a single torch batch
        x = torch.Tensor(subsampled_signal.append(rate))

        return self.model(x)


class WideNN(nn.Module):
    def __init__(self, initial_res, final_res):
        super(WideNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(initial_res, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, final_res),
        )

    def forward(self, x):
        return self.fc(x)

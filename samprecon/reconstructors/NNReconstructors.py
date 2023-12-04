from typing import List

import torch
import torch.nn as nn

from .reconstruct_intf import Reconstructor


class RNNReconstructor(nn.Module):
    def __init__(
        self,
        subsampled_signal_length: int,
    ):
        """
        Args:
            others: same as interface
            **kwargs:
                sub_length (int): Size of input. i.e. size of subsampled signal
                full_length (int): Size of final reconstructed signal.
        """
        super().__init__()
        # Take a variable length scalar input signals
        self.hidden_size = (
            subsampled_signal_length + 1 + 1
        )  # One for rate and one for count
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, subsampled_signal: List, rate, reconstruct_length: int):
        # Append subsampled_signal, rate, reconstruct_length
        x = torch.Tensor(subsampled_signal.append(rate)).tile((reconstruct_length, 1))
        x_count = torch.arange(reconstruct_length)[::-1].view(reconstruct_length, -1)
        x = torch.hstack([x, x_count])
        x = x.view(1, reconstruct_length, -1)

        y = self.rnn(x)
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

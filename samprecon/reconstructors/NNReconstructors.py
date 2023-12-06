from typing import List

import torch
import torch.nn as nn

from .reconstruct_intf import Reconstructor


class RNNReconstructor(nn.Module):
    def __init__(
        self,
        amnt_states: int,
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
        self.hidden_size = amnt_states + 1 + 1  # One for rate and one for count
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, amnt_states)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        subsampled_signal: torch.Tensor,
        rate: torch.Tensor,
        reconstruct_length: int,
    ):
        # Append subsampled_signal, rate, reconstruct_length
        rate_interleaved = rate.repeat_interleave(reconstruct_length, dim=1).unsqueeze(-1)
        x = torch.cat((subsampled_signal, rate_interleaved), dim=-1)
        x_count = torch.flip(torch.arange(reconstruct_length), [0]).view(
            x.shape[0], -1, 1
        )
        x = torch.cat((x, x_count), dim=-1)

        out,hiddn = self.rnn(x) # Check what is the difference between, out and hiddn
        y = self.classifier(out)
        y = self.sm(y)
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

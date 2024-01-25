from typing import List

import torch
import torch.nn as nn

from .reconstruct_intf import Reconstructor


class RNNReconstructor(nn.Module):
    def __init__(self, amnt_states: int, sampling_budget, bidirectional=True):
        """
        Args:
            others: same as interface
            **kwargs:
                sub_length (int): Size of input. i.e. size of subsampled signal
                full_length (int): Size of final reconstructed signal.
        """
        super().__init__()
        # Take a variable length scalar input signals
        self.sampling_budget = sampling_budget
        self.amnt_states = amnt_states

        self.input_size = amnt_states + 1  # + 1  # One for rate and one for count
        self.hidden_size = self.sampling_budget
        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        classifier_input = (
            self.hidden_size * 2 if bidirectional == True else self.hidden_size
        )
        self.classifier = nn.Linear(classifier_input, amnt_states + 1)  # For padding
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        subsampled_signal_oh: torch.Tensor,
        cell_state: torch.Tensor,
    ):
        """
        Parameters
        ~~~~~~~~~~
            subsampled_signal_oh: (batch_size) x (longest_seq_len) (Padded ofc)
        """

        # Zero initalization
        batch_size = subsampled_signal_oh.shape[0]
        hidden_state = torch.zeros((2, batch_size, self.hidden_size)).to(
            subsampled_signal_oh.device
        )

        twice_cell = torch.stack((cell_state, cell_state)).to(torch.float)

        out, hiddn = self.rnn(
            subsampled_signal_oh, (hidden_state, twice_cell)
        )  # Check what is the difference between, out and hiddn
        y = self.classifier(out)
        # y = self.sm(y)

        # TODO: return the feedback
        return y

    def initialize_grad_hooks(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_backward_hook(print_grad_hook)


def print_grad_hook(module, grad_input, grad_output):
    print("grad_input", grad_input)
    print("grad_output", grad_output)


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

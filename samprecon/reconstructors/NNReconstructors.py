from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn
import torch.nn as nn
from scipy.linalg import expm

from samprecon.utils.utils import dec_rep_to_batched_rep, setup_logger
from sp_sims.estimators.algos import sampling_viterbi


class Reconstructor(ABC):
    def __call__(self, subsampled_signal, new_dec_period, **kwargs):
        return self.reconstruct(subsampled_signal, new_dec_period, **kwargs)

    @abstractmethod
    def reconstruct(self, subsampled_signal, new_dec_period, **kwargs):
        pass


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
        subsampled_signal: torch.Tensor,
        new_dec_period: torch.Tensor,
    ):
        """
        Parameters
        ~~~~~~~~~~
            subsampled_signal_oh: (batch_size) x (longest_seq_len) (Padded ofc)
        """
        subsampled_signal_oh = dec_rep_to_batched_rep(
            subsampled_signal,
            new_dec_period,  # CHECK: If first column contains periods
            self.sampling_budget,
            self.amnt_states + 1,  # For Padding
            add_position=False,  # TODO: See 'true' helps
        )
        cell_state = subsampled_signal  # Yup, straight up

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

    def reconstruct(
        self, subsampled_signal: torch.Tensor, new_dec_period: torch.Tensor, **kwargs
    ):
        return self.forward(subsampled_signal, new_dec_period)

    def initialize_grad_hooks(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_backward_hook(print_grad_hook)


class MLEReconstructor(Reconstructor):
    def __init__(
        self,
        queue_matrix: torch.Tensor,
        padding_value: int,
        sampling_budget,
        highres_delta: float,
    ):
        self.Q = queue_matrix
        self.logger = setup_logger(__class__.__name__)
        self.highres_delta = highres_delta
        self.padding_value = padding_value
        self.samp_budget = sampling_budget

    def reconstruct(
        self,
        sampled_tape: torch.Tensor,
        dec_prop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments
        ~~~~~~~~~
        dec_prop: How many states per sample we must recover
        """
        assert self.samp_budget == sampled_tape.shape[1]
        batch_size = sampled_tape.shape[0]

        max_length = int(torch.max((self.samp_budget - 1) * dec_prop + 1).item())

        reconstruction = torch.full((batch_size, max_length), self.padding_value)

        losses = torch.zeros((sampled_tape.shape[0], 1), dtype=torch.float32)

        # OPTIM: Lots to optimizing to do here

        for b in range(batch_size):  # Batch
            cur_dec = dec_prop[b, 0].item()
            P = torch.from_numpy(expm(cur_dec * self.highres_delta * self.Q))

            if cur_dec == 1:
                reconstruction[b, :self.samp_budget] = sampled_tape[b, :]
                continue
            
            for s in range(self.samp_budget - 1):
                viterbi_recon = torch.Tensor(
                    sampling_viterbi(  # TODO: deal with cur_dec = 1
                        cur_dec - 1,
                        sampled_tape[b, s].item(),
                        sampled_tape[b, s + 1].item(),
                        P,
                    )
                )
                if s != self.samp_budget - 2:
                    reconstruction[b, s * cur_dec : (s + 1) * cur_dec ] = viterbi_recon[:-1]
                else:
                    reconstruction[b, s * cur_dec : (s+1)* cur_dec + 1] = viterbi_recon


        return reconstruction


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

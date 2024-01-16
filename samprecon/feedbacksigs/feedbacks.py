from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from samprecon.samplers.spatial_transformers import differentiable_uniform_sampler
from sp_sims.estimators.algos import frequency_matrix, power_series_log


class Feedbacks(ABC):
    @abstractmethod
    def get_feedback(self, state, action, truth):
        pass

    def __call__(self, state, action, truth) -> torch.Tensor:
        return self.get_feedback(state, action, truth)


class Reconstructor(Feedbacks):
    # TODO: do this from exp3.6
    def __init__(self, num_states: int, reconstructor: nn.Module, criterion: nn.Module):
        self.criterion = criterion
        self.num_states = num_states
        self.reconstructor = reconstructor

    def get_feedback(self, state, action, truth):
        """
        Parameters:
        ~~~~~~~~~~~
            state: oh-encoded chain of full res chains
        """

        # new_oh = F.one_hot(
        #     state.view(1, -1).to(torch.long),
        #     num_classes=self.num_states,
        # ).float()
        # dec_state = differentiable_uniform_sampler(new_oh, action)

        reconstruction = self.reconstructor(
            state,
            # action,
            # 1 + torch.ceil(action.squeeze() * (self.sampling_budget - 1)),
        ).squeeze(0)

        logsoft_recon = F.log_softmax(reconstruction, dim=-1).view(
            -1, reconstruction.shape[-1]
        )

        regret = (
            self.criterion(logsoft_recon, truth.to(torch.long).view(-1))
            .view(reconstruction.shape[0], -1)
            .mean(dim=-1)
        )
        return regret


class LogEstimator(Feedbacks):
    def __init__(self, trueQ, power: int, num_states: int, criterion: nn.Module):
        self.Q = trueQ
        self.power = 10
        self.criterion = criterion
        self.num_states = num_states

    def get_feedback(self, state, action, truth):  # TODO: implement action and truth
        # Will estimate Q via P
        p_est = frequency_matrix(state, self.num_states)
        q_est = power_series_log(p_est, self.power)
        loss = self.criterion(q_est, self.Q)
        loss_tensor = torch.tensor(loss)

        return loss_tensor

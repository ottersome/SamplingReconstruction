import time
from abc import ABC, abstractmethod
from logging import DEBUG
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from samprecon.utils.utils import dec_rep_to_batched_rep, setup_logger
from sp_sims.estimators.algos import frequency_matrix, power_series_log


class Feedbacks(ABC):
    @abstractmethod
    def get_feedback(self, state, action, truth, **kwargs) -> Dict[str, Any]:
        pass

    def __call__(self, state, action, truth, **kwargs) -> Dict[str, Any]:
        return self.get_feedback(state, action, truth, **kwargs)


class Reconstructor(Feedbacks):
    # TODO: do this from exp3.6
    def __init__(self, num_states: int, reconstructor: nn.Module, criterion: nn.Module):
        self.criterion = criterion
        self.num_states = num_states
        self.reconstructor = reconstructor
        self.logger = setup_logger("Reconstructor")

    def get_feedback(self, sampled_chain, action, truth, **kwargs) -> Dict[str, Any]:
        """
        Parameters:
        ~~~~~~~~~~~
            sampled: non-OH sampled chain
        """

        new_dec_period = kwargs["new_decimation_period"]
        sampling_budget = kwargs["sampling_budget"]
        # new_oh = F.one_hot(
        #     state.view(1, -1).to(torch.long),
        #     num_classes=self.num_states,
        # ).float()
        # dec_state = differentiable_uniform_sampler(new_oh, action)

        reconstruction = self.reconstructor(
            sampled_chain,
            new_dec_period,
        ).squeeze(0).to(truth.device)

        #logsoft_recon = F.log_softmax(reconstruction, dim=-1).view(
        #    -1, reconstruction.shape[-1]
        #)
        #self.logger.debug(
        #    f"Reconstruction looks like : {F.softmax(reconstruction, dim=-1)}"
        #)

        regret = (
            #self.criterion(logsoft_recon, truth.to(torch.long).view(-1))
            self.criterion(reconstruction.to(torch.float32), truth.to(torch.float32))
            .view(reconstruction.shape[0], -1)
            .mean(dim=-1)
        )

        return {"batch_loss": regret}


class LogEstimator(Feedbacks):
    def __init__(self, trueQ, power: int, num_states: int, criterion: nn.Module):
        self.Q = trueQ.to("cuda")
        self.power = 10
        self.criterion = criterion
        self.num_states = num_states

    def get_feedback(
        self, new_state, action, truth, **kwargs
    ) -> Dict[str, Any]:  # TODO: implement action and truth
        # Will estimate Q via P
        new_dec_periods = kwargs["new_decimation_period"]
        p_est = frequency_matrix(new_state, self.num_states)
        logp = power_series_log(p_est, self.power)
        q_est = logp / new_dec_periods.unsqueeze(-1)
        repeated_q = self.Q.unsqueeze(0).repeat(q_est.shape[0], 1, 1)
        # TODO we could regularize here to make sure that Q_est is a valid generator matrix
        # regularize_loss =
        loss = self.criterion(q_est, self.Q)
        batch_loss = loss.mean(dim=(-2, -1))
        # loss_tensor = torch.tensor(loss)

        return_dict = {
            "batch_loss": batch_loss,
            "avg_estimated_Q": q_est.mean(dim=0),
        }
        return return_dict


class MaxLikelihood(Feedbacks):
    """
    This depends on P beloning to the transition matrices
    obtainable through exp(Q) where Q is a valid generator matrix
    """

    def __init__(self, q_params: nn.Parameter):
        self.q_params = q_params
        pass

    def get_feedback(self, state, action, truth, **kwargs):
        # Here we just do maximum likelihood
        p_mat = power_series_log(self.q_params)
        pass

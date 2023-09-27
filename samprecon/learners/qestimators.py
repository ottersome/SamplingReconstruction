import torch
from torch import nn

class SimpleQEstimator(nn.Module):
    def __init__(self, state_size, action_size,hidden_state_size):
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size,hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size,hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size,hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size,1),
            nn.ReLU(),
                )
    def forward(self,state,action):
        input_vector = torch.concat(state,action)
        return self.model(input_vector)


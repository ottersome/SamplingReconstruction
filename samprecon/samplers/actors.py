import torch
from torch import nn

class SimpleActor(nn.Module):

    def __init__(self, state_size, hidden_state_size):
        self.model = nn.Sequential(
            nn.Linear(state_size,hidden_state_size), 
            nn.ReLU(),
            nn.Linear(hidden_state_size,hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size,hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size,1)
        )

    def forward(self,sample):
        return self.model(sample)


import torch.nn as nn
import torch.nn.functional as F


"""
Localized network simply inferences parameters for the next step. 
Thus one may consider multiple possible network structures
"""
class LocalizationNetwork(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.fc1 == nn.Linear(input_size,128)
        self.fc2 == nn.Linear(output_size,128)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


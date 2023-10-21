import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class InfiniteWidthNet(nn.Module):
    def __init__(self,how_big=100):
        super(InfiniteWidthNet, self).__init__()
        self.fc0 = spectral_norm(nn.Linear(1,how_big))
        self.fc1 = nn.Linear(how_big,1)

    def forward(self, t):
        """
        Parameters:
        t = time inside the function to evaluate

        """
        r = nn.ReLU()
        x = r(self.fc0(t))
        x = self.fc1(x)
        return x
        
    def get_norms(self):
        norms = []
        for name,layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                norms.append(layer.weight.norm().item())
        return norms
            
    def get_spectral_norms(self):
        meep = [] 
        for name,layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                meep.append(self.get_layers_spectral_norm(layer))
    def get_layers_spectral_norm(self,layer):
        W = getattr(layer, 'weight_orig')
        u = getattr(layer, 'weight_u')
        
        # Compute spectral norm
        W_mat = W.view(W.shape[0], -1)
        with torch.no_grad():
            v = W_mat.t() @ u
            v = v / v.norm()
            u = W_mat @ v
            sigma = u @ W_mat @ v
        return sigma.item()

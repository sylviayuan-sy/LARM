import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, 
        device
        ):
        super().__init__()

        self.device = device
    
    def forward(self, Tpx, py, eps=1e-10):
        loss = torch.sqrt((Tpx - py)**2 + eps).sum(dim=-1).mean()   
        return loss
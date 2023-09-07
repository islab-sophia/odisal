import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusionLayer(nn.Module):
    def __init__(self, channels=3):
        super(FusionLayer, self).__init__()
        self.f = nn.Linear(channels, channels)
        
        
    def forward(self, input_block):
        # print(input_block.size())
        z = nn.AvgPool2d(input_block.size(2))(input_block) # input_block = (bath_size,C,H,W)
        # print(z.size())
        z = z.squeeze(2).squeeze(2)
        x = torch.sigmoid(self.f(z)) # x = (bath_size,channels) 
        x = x.unsqueeze(-1).unsqueeze(-1).expand(input_block.size()) # x = (bath_size,channels,1,1)
        # print(x.size())
        output = torch.sum(torch.mul(input_block, x), dim=1, keepdim=True) 
        return output

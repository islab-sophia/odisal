import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterBiasLayer(nn.Module):
    def __init__(self, size=(20,20), bias=False):
        super(CenterBiasLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,1,size[0],size[1]).cuda())
        if bias:
            self.bias = nn.Parameter(torch.zeros(1,1,size[0],size[1]).cuda())
        else:
            self.bias = None

    def forward(self, x):
        weight = F.upsample(self.weight, size=(x.size(2), x.size(3)), mode='bilinear') # adjust height and width
        weight = weight.expand_as(x) #adjust batch size
        out = x * weight

        if self.bias is not None:
            bias = F.upsample(self.bias, size=(x.size(2), x.size(3)), mode='bilinear') # adjust height and width
            out += bias.expand_as(out) #adjust batch size and add bias
        return F.relu(out)

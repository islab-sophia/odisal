import torch.nn as nn
from torchvision.models import alexnet

from .base_model import BaseModel
from .readout_nets import upsampling_modules

__all__ = ['alexsal']

def alexsal(pretrained=True):
    main_net = alexnet(pretrained).features
    readout_net = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

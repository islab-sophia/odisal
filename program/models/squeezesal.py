import torch.nn as nn
from torchvision.models import squeezenet1_0, squeezenet1_1

from .base_model import BaseModel
from .readout_nets import upsampling_modules

__all__ = ['squeezesal1_0', 'squeezesal1_1']

def squeezesal1_0(pretrained=True):
    main_net = squeezenet1_0(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def squeezesal1_1(pretrained=True):
    main_net = squeezenet1_1(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

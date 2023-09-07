import torch.nn as nn
from torchvision.models import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn

from .base_model import BaseModel
from .readout_nets import upsampling_modules

__all__ = ['vggsal11', 'vggsal11_bn', 'vggsal13', 'vggsal13_bn', 'vggsal16', 'vggsal16_bn', 'vggsal19', 'vggsal19_bn']

def vggsal11(pretrained=True):
    main_net = vgg11(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def vggsal11_bn(pretrained=True):
    main_net = vgg11_bn(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def vggsal13(pretrained=True):
    main_net = vgg13(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def vggsal13_bn(pretrained=True):
    main_net = vgg13_bn(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def vggsal16(pretrained=True):
    main_net = vgg16(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def vggsal16_bn(pretrained=True):
    main_net = vgg16_bn(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def vggsal19(pretrained=True):
    main_net = vgg19(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def vggsal19_bn(pretrained=True):
    main_net = vgg19_bn(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

import torch.nn as nn

from .base_model import BaseModel
from .readout_nets import upsampling_modules
from .pytorch_dpn_pretrained.model_factory import create_model

__all__ = ['dpnsal68', 'dpnsal68b', 'dpnsal92', 'dpnsal98', 'dpnsal131','dpnsal107', 'dpnsal131_coarse',
    'dpnsal131_dilation', 'dpnsal131_dilation_coarse', 'dpnsal131_multipath', 'dpnsal131_dilation_multipath', 
    'dpnsal131_dilation_multipath_NIx1', 'dpnsal131_dilation_multipath_NIx2','dpnsal131_dilation_multipath_NIx3',
    'dpnsal131_dilation_multipath_BIx1', 'dpnsal131_dilation_multipath_BIx2','dpnsal131_dilation_multipath_BIx3',
    'dpnsal131_dilation_multipath_DCx1', 'dpnsal131_dilation_multipath_DCx2','dpnsal131_dilation_multipath_DCx3',
    'dpnsal131_dilation_multipath_SPCx1', 'dpnsal131_dilation_multipath_SPCx2','dpnsal131_dilation_multipath_SPCx3',
    'dpnsal131_dilation_multipath_Convx3',]


def _dilation(net):
    net.conv5_1.c1x1_w_s2.conv.stride = (1, 1)
    net.conv5_1.c3x3_b.conv.stride = (1, 1)
    net.conv5_2.c3x3_b.conv.dilation = (2, 2)
    net.conv5_2.c3x3_b.conv.padding = (2, 2)
    net.conv5_3.c3x3_b.conv.dilation = (2, 2)
    net.conv5_3.c3x3_b.conv.padding = (2, 2)

def dpnsal68(pretrained=True):
    main_net = create_model('dpn68', pretrained=pretrained).features
    readout_net = nn.Conv2d(832, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def dpnsal68b(pretrained=True):
    main_net = create_model('dpn68b', pretrained=pretrained).features
    readout_net = nn.Conv2d(832, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def dpnsal92(pretrained=True):
    main_net = create_model('dpn92', pretrained=pretrained).features
    readout_net = nn.Conv2d(2688, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def dpnsal98(pretrained=True):
    main_net = create_model('dpn98', pretrained=pretrained).features
    readout_net = nn.Conv2d(2688, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def dpnsal131(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained).features
    readout_net = nn.Conv2d(2688, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def dpnsal131_multipath(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained).features
    readout_net = nn.Conv2d(5376, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_coarse(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained).features
    readout_net = nn.Conv2d(2688, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_coarse_path=True)
    return model

def dpnsal131_dilation(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = nn.Conv2d(2688, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def dpnsal131_dilation_coarse(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = nn.Conv2d(2688, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_coarse_path=True)
    return model

def dpnsal131_dilation_multipath(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = nn.Conv2d(5376, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_NIx1(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.NearestInterpolationModule(5376, scale_factor=2)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_NIx2(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.NearestInterpolationModule(5376, scale_factor=4)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_NIx3(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.NearestInterpolationModule(5376, scale_factor=8)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_BIx1(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.BilinearInterpolationModule(5376, scale_factor=2)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_BIx2(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.BilinearInterpolationModule(5376, scale_factor=4)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_BIx3(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.BilinearInterpolationModule(5376, scale_factor=8)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_DCx1(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.DeconvolutionModule(5376, num_upsampling_layers=1)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_DCx2(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.DeconvolutionModule(5376, num_upsampling_layers=2)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_DCx3(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.DeconvolutionModule(5376, num_upsampling_layers=3)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_SPCx1(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.SubPixelConvModule(5376, num_upsampling_layers=1)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_SPCx2(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.SubPixelConvModule(5376, num_upsampling_layers=2)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_SPCx3(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.SubPixelConvModule(5376, num_upsampling_layers=3)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal131_dilation_multipath_Convx3(pretrained=True):
    main_net = create_model('dpn131', pretrained=pretrained, dilation=True).features
    _dilation(main_net)
    readout_net = upsampling_modules.ConvModule(5376, num_layers=3)
    model = BaseModel(main_net, readout_net)
    return model

def dpnsal107(pretrained=True):
    main_net = create_model('dpn107', pretrained=pretrained).features
    readout_net = nn.Conv2d(2688, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

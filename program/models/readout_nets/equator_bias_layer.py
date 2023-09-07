import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
import numpy as np


class EquatorBiasLayer(nn.Module):
    def __init__(self, num_channels, size=(20,20), use_cuda=True, constant_bias_path=None):
        super(EquatorBiasLayer, self).__init__()
        self.num_channels = num_channels
        if constant_bias_path==None:
            # to learn equator bias or load checkpoint
            weights = [nn.Parameter(torch.ones(1, 1, size[0], size[1])) for _ in range(self.num_channels)] # (1, 1, height, width)
        else:
            # to use constant equator bias by training
            extracted_constant_bias = np.load(constant_bias_path)*1000
            print("--debug--")
            print("extracted_cosntant_bias.max()", extracted_constant_bias.max())
            weights = [nn.Parameter(torch.from_numpy(resize(extracted_constant_bias[i], (size[0], size[1]), order=0, mode="constant")[np.newaxis, np.newaxis, :, :, 0].astype(np.float32))) for i in range(self.num_channels)]
        self.weights = nn.ParameterList(weights)

    def forward(self, x, eqbl_idx):
        """
        Parameters
        ----------
        x : tensor (n_samples, n_channels, height, width)
        eqbl_idx : int

        Returns
        ----------
        tensor (n_samples, n_channels, height, width)
        """
        weight = F.interpolate(self.weights[eqbl_idx], size=(x.size(2), x.size(3)), mode='bilinear') # adjust height and width
        out = x * weight[0]
        return F.relu(out, inplace=True)

    def load_center_bias(self, checkpoint):
        """
        load center bias from pretrained model.

        Parameters
        ----------
        checkpoint
        """
        weights = [nn.Parameter(checkpoint["center_bias_state_dict"]["weight"].clone()) for i in range(self.num_channels)]
        self.weights = nn.ParameterList(weights)

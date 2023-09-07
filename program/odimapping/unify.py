import numpy as np

import torch
from torch.nn import functional as F


def cropping(target_view_angle, view_angle, input_t):
    # print('in size: ',input_t.size())
    img_size = input_t.size(2)
    target_view_angle = int(target_view_angle)
    view_angle = int(view_angle)
    # print(target_view_angle,view_angle)
    L = (img_size/2) / np.tan(np.radians(target_view_angle/2))
    H = 2 * L * np.tan(np.radians(view_angle/2))
    target_size = img_size * (img_size / H)
    sart_position = int(round((img_size - target_size)/2))
    end_position = int(round((img_size + target_size)/2))
    # print(sart_position,end_position)
    cropping_t = input_t[:, :, sart_position:end_position, sart_position:end_position]
    # print('cropping size: ',cropping_t.size())
    return resize(cropping_t, img_size)
    
def resize(t, out_size):
    output = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=True)
    return output     

import cv2
import numpy as np
import torch


class Compose(object):
    """
    compose transform's object

    Parameters
    -----------
    transforms : list
        list of transform's object from this module
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class SimulCompose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor(object):
    """
    ndarray to torch.Tensor
    """
    def __call__(self, img, *target_img):
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if len(target_img):
            target_img = [torch.from_numpy(t_img.copy()).float() for t_img in target_img]
            return [img.float().div(255)] + target_img
        else:
            return img.float().div(255)


class Resize(object):
    """
    resize image

    Parameters
    -----------
    size: tuple, (height, width)
    scale_factor: float
    """
    def __init__(self, size=None, scale_factor=1.0):

        self.size = size
        self.scale_factor = scale_factor

    def __call__(self, img):
        if self.size is None:
            h, w, _ = img.shape
            img = cv2.resize(img, dsize=(int(w*self.scale_factor), int(h*self.scale_factor)), interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        return img


class ResizeToInnerRectangle(object):

    def __init__(self, rec_long_side=640, rec_short_side=480):
        self.rec_long_side = rec_long_side
        self.rec_short_side = rec_short_side

    def __call__(self, img):
        h, w, _ = img.shape
        src_long_side = np.maximum(h, w)
        src_short_side = np.minimum(h, w)
        if src_long_side > self.rec_long_side:
            scale = self.rec_long_side / float(src_long_side)
            dst_h, dst_w = int(scale*h), int(scale*w)
            if np.minimum(dst_h, dst_w) > self.rec_short_side:
                scale = self.rec_short_side / float(src_short_side)
                dst_h, dst_w = int(scale*h), int(scale*w)
            img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        return img

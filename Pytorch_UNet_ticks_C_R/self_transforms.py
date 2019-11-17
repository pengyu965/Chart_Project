import torch
import math
import sys
import random
from PIL import Image
import numpy as np
import numbers
import types
import collections
import warnings

def rotate(img, angle, resample=False, expand=False, center=None, fill = 0):
    """Rotate the image by angle."""


    if isinstance(fill, int):
        if img.mode == "RGBA":
            fill = tuple([fill] * 4)
        elif img.mode == "RGB":
            fill = tuple([fill] * 3)

    return img.rotate(angle, resample, expand, center, fill)

class RandomRotation(object):
    """Rotate the image by random angle."""


    def __init__(self, degrees, resample=False, expand=False, center=None, fill = 0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
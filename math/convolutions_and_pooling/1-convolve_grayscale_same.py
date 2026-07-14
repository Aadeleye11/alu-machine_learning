#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    images: numpy.ndarray (m, h, w) - multiple grayscale images
    kernel: numpy.ndarray (kh, kw) - convolution kernel

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                     mode='constant', constant_values=0)

    convolved = np.zeros((m, h, w))

    for i in range(kh):
        for j in range(kw):
            convolved += kernel[i, j] * padded[:, i:i + h, j:j + w]

    return convolved

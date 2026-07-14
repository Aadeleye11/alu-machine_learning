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

    pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
    padded = np.pad(images, pad_width, mode='constant', constant_values=0)

    convolved = np.zeros((m, h, w))

    for i in range(kh):
        for j in range(kw):
            window = padded[:, i:i + h, j:j + w]
            convolved += kernel[i, j] * window

    return convolved

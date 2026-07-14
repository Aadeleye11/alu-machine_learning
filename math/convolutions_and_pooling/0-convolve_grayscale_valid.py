#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images: numpy.ndarray (m, h, w) - multiple grayscale images
    kernel: numpy.ndarray (kh, kw) - convolution kernel

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(kh):
        for j in range(kw):
            convolved += kernel[i, j] * images[:, i:i + output_h, j:j + output_w]

    return convolved

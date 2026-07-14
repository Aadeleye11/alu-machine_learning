#!/usr/bin/env python3
"""Performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images: numpy.ndarray (m, h, w) - multiple grayscale images
    kernel: numpy.ndarray (kh, kw) - convolution kernel
    padding: tuple (ph, pw) - padding for height and width

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                     mode='constant', constant_values=0)

    convolved = np.zeros((m, output_h, output_w))

    for i in range(kh):
        for j in range(kw):
            convolved += kernel[i, j] * padded[:, i:i + output_h, j:j + output_w]

    return convolved

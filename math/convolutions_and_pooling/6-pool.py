#!/usr/bin/env python3
"""Performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images: numpy.ndarray (m, h, w, c) - multiple images
    kernel_shape: tuple (kh, kw) - pooling window shape
    stride: tuple (sh, sw) - stride for height and width
    mode: 'max' or 'avg'

    Returns: numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    pooled = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            window = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(window, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.average(window, axis=(1, 2))

    return pooled

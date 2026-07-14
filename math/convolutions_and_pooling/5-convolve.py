#!/usr/bin/env python3
"""Performs a convolution on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    images: numpy.ndarray (m, h, w, c) - multiple images
    kernels: numpy.ndarray (kh, kw, c, nc) - kernels for the convolution
    padding: tuple (ph, pw), 'same', or 'valid'
    stride: tuple (sh, sw) - stride for height and width

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    pad_width = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    padded = np.pad(images, pad_width, mode='constant', constant_values=0)

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                window = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                convolved[:, i, j, k] = np.sum(
                    window * kernels[:, :, :, k], axis=(1, 2, 3))

    return convolved

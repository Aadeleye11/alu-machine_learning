# Convolutions and Pooling

This project is part of the `alu-machine-learning` repository, under `math/convolutions_and_pooling`. It covers the fundamentals of convolutions and pooling operations on images — core building blocks of Convolutional Neural Networks (CNNs).

## Description

Convolutions are used throughout deep learning for tasks like edge detection, blurring, sharpening, and feature extraction in images. This project implements convolution operations on grayscale and color images from scratch using NumPy, without relying on deep learning frameworks.

## Requirements

- Python 3.x
- NumPy
- Ubuntu 20.04 LTS
- All files should be executable
- All code follows `pycodestyle` (PEP8) style guidelines
- Only `numpy` may be imported unless otherwise specified
- The length of your files will be tested using `wc`

## Tasks

### 0. Valid Convolution

**File:** `0-convolve_grayscale_valid.py`

Performs a "valid" convolution on grayscale images — meaning no padding is applied, so the output dimensions shrink relative to the input based on the kernel size.

**Function:** `def convolve_grayscale_valid(images, kernel):`

- `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
  - `m` is the number of images
  - `h` is the height in pixels of the images
  - `w` is the width in pixels of the images
- `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
  - `kh` is the height of the kernel
  - `kw` is the width of the kernel
- Only two `for` loops are allowed (looping over the kernel dimensions); no other loops of any kind
- Returns: a `numpy.ndarray` containing the convolved images

**Output shape:** `(m, h - kh + 1, w - kw + 1)`

**Example usage:**

```bash
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 0-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
```

**Expected output:**

```
(50000, 28, 28)
(50000, 26, 26)
```

## Repo

- **GitHub repository:** `alu-machine-learning`
- **Directory:** `math/convolutions_and_pooling`
- **File:** `0-convolve_grayscale_valid.py`

## Author

Ayomide — Software Engineering student at African Leadership University (ALU)

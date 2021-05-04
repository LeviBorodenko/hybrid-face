from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image


class Filter(ABC):
    """
    Generic FFT filter class. Once you provide a name and a kernel function, this filter can be called with
    any PIL.Image and will convolve it with the kernel_function in the frequency domain. Note that
    the kernel will be sampled on [0,1]^2.
    """

    @property
    @abstractmethod
    def __name__(self) -> str:
        return "General FFT Filter"

    epsilon: float = 0.00001

    def __init__(self, sigma: float = 0.0015):
        self.sigma = sigma

    @abstractmethod
    def kernel_function(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        pass

    @lru_cache
    def get_kernel(self, image_shape: Tuple[int, int], show_kernel: bool = False) -> np.ndarray:

        n, m = image_shape

        # double resolution due to padding
        x_range, y_range = np.linspace(-1, 1, 2 * n), np.linspace(-1, 1, 2 * m)
        xx, yy = np.meshgrid(x_range, y_range, indexing="ij", sparse=True)

        kernel = self.kernel_function(xx, yy)

        # zero if kernel is below precision
        if kernel.sum() < self.epsilon:
            return np.zeros_like(kernel)

        # display in inverse grey scale if wanted
        if show_kernel:
            Image.fromarray(256 - (kernel / kernel.max() * 256)).show()

        return kernel

    @lru_cache
    def get_shift_matrix(self, image_shape: Tuple[int, int]) -> np.ndarray:
        P, Q = 2 * image_shape[0], 2 * image_shape[1]
        i = np.arange(P).reshape((P, 1))
        j = np.arange(Q).reshape((1, Q))

        return (-1) ** (i + j)  # (P, Q) via broadcasting

    def filter(self, image_data: np.ndarray) -> np.ndarray:

        # assert image_data has valid shape
        if len(image_data.shape) != 2:
            raise ValueError(f"image data must have shape (n, m). Got {image_data.shape}")

        n, m = image_data.shape

        # pad image
        padded_data = np.pad(image_data, ((0, n), (0, m)))  # (2N, 2M)

        # center image freq
        shift_matrix = self.get_shift_matrix(image_data.shape)
        centered_padded_data = padded_data * shift_matrix

        # move to freq domain
        fft_data = np.fft.fft2(centered_padded_data)

        # get freq filter
        freq_filter = self.get_kernel(image_data.shape)

        # apply filter
        filtered_data = fft_data * freq_filter

        # move back to spacial domain
        centered_padded_result = np.fft.ifft2(filtered_data)

        # undo centering
        padded_result = centered_padded_result * shift_matrix

        # undo padding
        filtered_image = padded_result[0:n, 0:m]

        # remove complex leakage
        filtered_image = np.real(filtered_image)

        return filtered_image

    def __call__(self, image: Image) -> Image:

        # get image greyness data
        grey_scale_image = image.convert("L")
        image_data = np.asarray(grey_scale_image)

        # apply filter and return rgba image
        filtered_image_data = self.filter(image_data)
        return Image.fromarray(filtered_image_data).convert("RGBA")

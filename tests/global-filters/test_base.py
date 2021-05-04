from typing import Tuple

import numpy as np

from hybrid_face.filters import Filter, LowPassFilter


def test_initiates_properly(global_filter: Filter, sigma: float):
    assert global_filter.sigma == sigma


def test_filtering_has_correct_api(global_filter: Filter, random_image_data: np.ndarray):
    filtered_image_data = global_filter.filter(random_image_data)
    assert isinstance(filtered_image_data, np.ndarray)


def test_filters_conserve_shape(low_pass_filter: Filter, random_image_data: np.ndarray):
    initial_shape = random_image_data.shape
    filtered_image = low_pass_filter.filter(random_image_data)

    assert initial_shape == filtered_image.shape


def test_kernel_padding(global_filter: Filter, image_shape: Tuple[int, int]):
    kernel = global_filter.get_kernel(image_shape)

    assert kernel.shape == (2 * image_shape[0], 2 * image_shape[1])  # (2N, 2M)


def test_kernel_basics(global_filter: Filter, image_shape: Tuple[int, int]):

    kernel: np.ndarray = global_filter.get_kernel(image_shape)
    assert isinstance(kernel, np.ndarray)
    assert 0 <= kernel.min() <= kernel.max() <= 1

    # either the global_filter is normalised or it is zero
    assert kernel.sum() > global_filter.epsilon or kernel.sum() == 0


def test_kernel_max(low_pass_filter: LowPassFilter, image_shape: Tuple[int, int]):
    kernel: np.ndarray = low_pass_filter.get_kernel(image_shape)
    idx_of_arg_max = np.unravel_index(np.argmax(kernel), kernel.shape)

    n, m = image_shape
    n_max, m_max = idx_of_arg_max

    # ignore zero kernels
    if kernel.sum() != 0:

        # should be centered around the center of the padded image
        assert abs(n_max - n) < 2 and abs(m_max - m) < 2


def test_shift_matrix(
    global_filter: Filter,
    random_image_data: np.ndarray,
):
    shift_matrix = global_filter.get_shift_matrix(random_image_data.shape)
    assert shift_matrix.shape == (2 * random_image_data.shape[0], 2 * random_image_data.shape[1])
    assert set(np.unique(shift_matrix)) == set([-1, 1])
    assert shift_matrix[0, 0] == 1
    assert shift_matrix.sum() == 0

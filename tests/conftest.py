from typing import Tuple

import numpy as np
from pytest import fixture


@fixture(params=[0.001, 0.25, 0.75, 1, 2, 1000])
def sigma(request) -> float:
    return request.param


@fixture(params=[1, 10, 50, 100, 500])
def image_width(request) -> int:
    return request.param


@fixture(params=[1, 10, 50, 100, 500])
def image_length(request) -> int:
    return request.param


@fixture
def image_shape(image_length, image_width) -> Tuple[int, int]:
    return (image_length, image_width)


@fixture
def random_image_data(image_shape: Tuple[int, int]) -> np.ndarray:
    return np.random.randint(0, 256, image_shape)

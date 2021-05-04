import numpy as np
from PIL import Image
from pytest import fixture

from hybrid_face.filters import Filter, HighPassFilter, LowPassFilter


@fixture(params=[LowPassFilter, HighPassFilter])
def global_filter(request, sigma: int) -> Filter:
    return request.param(sigma)


@fixture
def low_pass_filter(sigma: int) -> LowPassFilter:
    return LowPassFilter(sigma)


@fixture
def random_image(random_image_data: np.ndarray) -> Image:
    return Image.fromarray(random_image_data, "L")

from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from pytest import fixture

from hybrid_face.filters import FaceFilter, HighPassFaceFilter, LowPassFaceFilter

sample_face_files = list(Path("tests/images/faces/").iterdir())
face_file_pairs = list(combinations(sample_face_files, 2))
face_filters = [HighPassFaceFilter, LowPassFaceFilter]


@fixture(params=sample_face_files)
def face_image(request) -> Image:
    return Image.open(request.param)


@fixture(params=face_file_pairs)
def two_face_images(request):
    return (Image.open(request.param[0]), Image.open(request.param[1]))


@fixture
def face_image_data(face_image: Image) -> np.ndarray:
    return np.asarray(face_image)


@fixture(params=face_filters)
def face_filter(request, sigma: float) -> FaceFilter:
    return request.param(sigma)

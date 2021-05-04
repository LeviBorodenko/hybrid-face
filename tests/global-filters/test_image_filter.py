from PIL.Image import Image

from hybrid_face.filters import Filter


def test_filters_can_be_called_with_images(global_filter: Filter, random_image: Image):
    filtered_image = global_filter(random_image)
    assert isinstance(filtered_image, Image)
    assert filtered_image.size == random_image.size
    assert filtered_image.mode == "RGBA"

from PIL.Image import Image


def test_face_filters_work(face_filter, face_image):
    filtered_face = face_filter(face_image)
    face_m, face_n = filtered_face.size
    m, n = face_image.size

    assert isinstance(filtered_face, Image)
    assert filtered_face.mode == "RGBA"

    # finds face
    assert 10 < face_m <= m
    assert 10 < face_n <= n

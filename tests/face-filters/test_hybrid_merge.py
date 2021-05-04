from PIL.Image import Image

from hybrid_face.hybrid_merge import hybrid_merge


def test_hybrid_face_merge(two_face_images, sigma):
    face1, face2 = two_face_images
    hybrid_blend = hybrid_merge(face1, face2, sigma=sigma)
    assert isinstance(hybrid_blend, Image)
    assert hybrid_blend.size[0] > 10
    assert hybrid_blend.size[1] > 10

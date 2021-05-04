from PIL import Image, ImageOps

from hybrid_face import console
from hybrid_face.filters import (
    HighPassFaceFilter,
    HighPassFilter,
    LowPassFaceFilter,
    LowPassFilter,
)


def hybrid_merge(
    image1: Image,
    image2: Image,
    sigma=0.002,
    alpha=0.5,
    ignore_faces: bool = False,
    crop_margin: int = 15,
) -> Image:
    """Creates the hybrid image of the two provided images.

    Args:
        image1 (Image): PIL.Image instance that will become to blurred image
        image2 (Image): PIL.Image instance that will become the sharp image
        sigma (float, optional): The soft cut-off frequency of the high/low pass filters. Defaults to 0.002.
        alpha (float, optional): The alpha blending parameter used to blend the two filtered images. Defaults to 0.5.
        ignore_faces (bool, optional): Whether to not crop for faces. Defaults to False.
        crop_margin (int, optional): How many pixles to cut-off from the margin before blending. Defaults to 15.

    Returns:
        Image: PIL.Image instance of the blended result image
    """

    if ignore_faces:
        # initiate filters
        low_pass_filter = LowPassFilter(sigma)
        high_pass_filter = HighPassFilter(sigma)

        # apply filters
        low_image = low_pass_filter(image1)
        high_image = high_pass_filter(image2)

        # remove convolution artifacts on border
        console.log(f"Removing {crop_margin} px from each side to avoid convolution artifacts")
        high_image = ImageOps.crop(high_image, crop_margin)
        low_image = ImageOps.crop(low_image, crop_margin)

        # resize/pad high face to fit onto to low face
        console.log(
            f"resizing {high_image.filename} from {high_image.size} to {low_image.size} \
             to so it can be merged with {low_image.filename}"
        )
        high_image = ImageOps.pad(high_image, low_image.size).convert("L").convert("RGBA")

        # merge images
        return Image.blend(low_image, high_image, alpha)

    # initate filters
    console.rule("[bold red]Step 1 - Initiate Filters")
    low_pass_face_filter = LowPassFaceFilter(sigma)
    console.log(f"Initiated [bold]{low_pass_face_filter.__name__} (σ = {low_pass_face_filter.sigma}).")
    high_pass_face_filter = HighPassFaceFilter(sigma)
    console.log(f"Initiated [bold]{high_pass_face_filter.__name__} (σ = {high_pass_face_filter.sigma}).")

    # get faces and apply filter
    console.rule("[bold red]Step 2 - Apply Filters")
    low_face = low_pass_face_filter(image1)
    low_face_aspect_ratio = low_face.size[1] / low_face.size[0]
    high_face = high_pass_face_filter(image2, min_aspect_ratio=low_face_aspect_ratio)

    # remove convolution artifacts on border
    console.rule("[bold red]Step 3 - Cosmetic Adjustments")
    console.log(f"Removing {crop_margin} px from each side to avoid convolution artifacts.")
    high_face = ImageOps.crop(high_face, crop_margin)
    low_face = ImageOps.crop(low_face, crop_margin)

    # resize/pad high face to fit onto to low face
    console.log(
        f"Resize face in {image2.filename} from {high_face.size} to {low_face.size}",
        f"so it matches the dimensions of the face in {image1.filename}.",
    )
    high_face = ImageOps.pad(high_face, low_face.size).convert("L").convert("RGBA")

    # merge images
    console.rule("[bold red]Step 4 - Blend Faces")
    console.log(f"Blending the filtered faces from {image1.filename} and {image2.filename} together.")
    return Image.blend(low_face, high_face, alpha)

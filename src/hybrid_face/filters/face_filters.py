import numpy as np
from face_recognition import face_locations
from PIL import Image

from hybrid_face import console
from hybrid_face.filters.base import Filter


class FaceFilter(Filter):
    """
    Same as the generic Filter ABC but this one will first crop to only contain an image of the face
    before doing any processing.
    """

    @property
    def __name__(self) -> str:
        return "face-aware FFT filter"

    def __call__(self, image: Image, min_aspect_ratio: float = None) -> Image:
        with console.status(f"Applying {self.__name__} to [bold]{image.filename}"):
            # get image greyness array
            grey_scale_image = image.convert("L")
            image_data = np.asarray(grey_scale_image)

            # detect faces
            face_loc = face_locations(image_data)

            # make sure there is only one face
            if len(face_loc) == 0:
                raise ValueError(f"Cannot find face in {image.filename}")
            elif len(face_loc) > 1:
                raise NotImplementedError(f"Found more than one face in {image.filename}")

            # get face location and image size (distances of the sides from their corresponding image border)
            top, right, bottom, left = face_loc[0]
            width, height = grey_scale_image.size

            # detected face size
            delta_y = abs(bottom - top)
            delta_x = abs(right - left)
            console.log(
                f"Detected a {delta_x} x {delta_y} face with an aspect ratio of ",
                f"{round(delta_y / delta_x, 2)} located at {face_loc[0]}",
            )

            # the rectangle returned by face_recognition is a bit too tight. So we extend the rectangle
            crop_top = max(0, top - delta_y * 0.8)
            crop_bottom = min(height, bottom + delta_y * 0.2)
            crop_right = min(width, right + delta_x * 0.3)
            crop_left = max(0, left - delta_x * 0.3)

            # if we want the face cutout to have a certain aspect ration
            if min_aspect_ratio is not None:

                crop_delta_y = abs(crop_bottom - crop_top)
                crop_delta_x = abs(crop_right - crop_left)

                min_y = int(crop_delta_x * min_aspect_ratio)

                if crop_delta_y < min_y:

                    # pixels needed to match aspect ratio
                    padding = int(min_y - crop_delta_x)
                    crop_top = max(0, top - padding // 2)
                    crop_bottom = min(height, bottom + padding // 2)
                    console.log(
                        f"The face has an aspect ratio of {delta_y / delta_x} but it is requested to be at least",
                        f"{min_aspect_ratio}. After padding we achieve an aspect ratio of ",
                        f"{round((crop_bottom - crop_top) / crop_delta_x, 2)}",
                    )

            # convert to (x1, y1, x2, y2) coords
            crop_loc = (crop_left, crop_top, crop_right, crop_bottom)

            # crop and apply filter
            face_image = grey_scale_image.crop(crop_loc)
            face_data = np.asarray(face_image)
            console.log(f"Applying {self.__name__} to cropped facial region")
            filtered_face_data = self.filter(face_data)

            console.print(f"[green] Finished processing [bold]{image.filename}")
            # return data as rgba to allow blending
            return Image.fromarray(filtered_face_data).convert("RGBA")


class LowPassFaceFilter(FaceFilter):
    @property
    def __name__(self) -> str:
        return "face-aware low-pass filter"

    def kernel_function(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (xx ** 2 + yy ** 2) / (self.sigma + self.epsilon))


class HighPassFaceFilter(FaceFilter):
    @property
    def __name__(self) -> str:
        return "face-aware high-pass filter"

    def kernel_function(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return 1 - np.exp(-0.5 * (xx ** 2 + yy ** 2) / (self.sigma + self.epsilon))

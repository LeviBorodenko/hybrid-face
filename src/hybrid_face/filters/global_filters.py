import numpy as np

from hybrid_face.filters.base import Filter


class LowPassFilter(Filter):
    @property
    def __name__(self) -> str:
        return "low-pass filter"

    def kernel_function(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (xx ** 2 + yy ** 2) / (self.sigma + self.epsilon))


class HighPassFilter(Filter):
    @property
    def __name__(self) -> str:
        return "high-pass filter"

    def kernel_function(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return 1 - np.exp(-0.5 * (xx ** 2 + yy ** 2) / (self.sigma + self.epsilon))

import util
import numpy as np


class BandSelect(object):
    """Downscale the band in a sample with the selected bands.

    Args:
        indices (np.ndarray): Indices of the bands to be selected
        keep_size (Boolean): Default -> True, True to retain original size by padding zeros behind the vector.
    """

    def __init__(self, indices: np.ndarray, keep_size: bool = True):

        self.indices = indices
        self.keep_size = keep_size

    def __call__(self, band):

        ans = band[self.indices]

        if self.keep_size:
            # Upscale by padding zeros behind it
            padding = np.zeros([2402 - len(self.indices)])
            ans = np.concatenate((ans, padding))

        return ans.astype('float32')


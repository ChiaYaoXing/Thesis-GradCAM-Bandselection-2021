import util
import numpy as np


class GaussianBasedSmoothing(object):
    """Downscale the band in a sample to a given size with Gaussian Bell based smoothing.

    Args:
        output_size (int): Desired output size.
        keep_size (Boolean): Default -> True, True to retain original size by padding zeros behind the vector.
    """

    def __init__(self, output_size: int = 800, vec_size: int = 6, sigma: float = 1, keep_size: bool = True):

        self.output_size = output_size
        self.band_matrix = None
        self.keep_size = keep_size
        self.vec_size = vec_size
        self.sigma = sigma

    def __call__(self, band):
        assert self.output_size <= len(band), "Output size should be smaller than the size of the band"
        assert self.vec_size <= len(band), "Size of Gaussian vector should be smaller than the size of the band"

        band_size = len(band)
        if self.band_matrix is None:
            gap = (band_size - self.vec_size - 1) / (self.output_size - 1)

            indices = [round(x * gap) for x in range(self.output_size)]

            if indices[-1] + self.vec_size != band_size:
                indices[-1] = band_size - self.vec_size - 1

            self.band_matrix = util.mat_generation(indices, util.gaussian_kernel(self.vec_size, self.sigma))

        ans = np.matmul(self.band_matrix, np.expand_dims(band, 1))

        if self.keep_size:
            # Upscale by padding zeros behind it
            padding = np.zeros([band_size - self.output_size, 1])
            ans = np.concatenate((ans, padding))

        ans = ans.squeeze(1)
        return ans.astype('float32')

    def inv(self, GradHM):
        """
        Upscale the input (Heatmap) into the same dimension as before feature reduced
        :param GradHM: Heatmap to upscale
        :return: Upscaled heatmap
        """
        to_be_upscaled = GradHM[0:self.output_size]
        tranposed_matrix = np.transpose(self.band_matrix)
        return np.matmul(tranposed_matrix, to_be_upscaled)




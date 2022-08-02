import util
import numpy as np
import torch as torch


class BandwiseAveraging(object):
    """Downscale the band in a sample to a given size with block wise averaging.

    Args:
        output_size (int): Desired output size.
        keep_size (Boolean): Default-> True, True to retain original size by padding zeros behind the vector.
    """

    def __init__(self, output_size, keep_size=True):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.block_matrix = None
        self.keep_size = keep_size

    def __call__(self, band):
        assert self.output_size <= len(band)

        if self.block_matrix is None:
            self.block_matrix = util.block_mat(self.output_size)

        self.band_size = len(band)

        ans = np.matmul(self.block_matrix, np.expand_dims(band, 1))

        if self.keep_size:
            # Upscale by padding zeros behind it
            padding = np.zeros([self.band_size - self.output_size, 1])
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
        tranposed_matrix = np.transpose(self.block_matrix)
        return np.matmul(tranposed_matrix, to_be_upscaled)


# if __name__ == "__main__":
#     GBS = GaussianBasedSmoothing(800)
#     b = torch.rand(2402)
#     GBS(b)

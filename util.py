import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


def block_mat(w: int) -> np.ndarray:
    """

    :param w: window size
    :return: Block Matrix with output size of (w, 2402)
    """
    from scipy.linalg import block_diag
    NUMBER_OF_BANDS = 2402
    num_of_bands_in_blocks = NUMBER_OF_BANDS // w
    remaining_bands = NUMBER_OF_BANDS % w

    pattern1 = np.ones(num_of_bands_in_blocks + 1) / num_of_bands_in_blocks
    pattern2 = np.ones(num_of_bands_in_blocks) / num_of_bands_in_blocks

    # First few remaining_bands will have a window size of num_of_bands_in_blocks + 1
    n_pattern = [pattern1 for x in range(remaining_bands)]

    # The remaining will all have the window size of num_of_bands_in_blocks
    for p in range(w - remaining_bands):
        n_pattern.append(pattern2)

    return block_diag(*n_pattern)


def gaussian_kernel(n: int, sigma: float) -> np.ndarray:
    """

    :param n: Width of the Gaussian Bell
    :param sigma: Standard deviation
    :return: gaussian kernel of width n and standard deviation of sigma.

    """

    assert n > 0, "n should be a positive non zero value"

    g = np.zeros(n)
    if n % 2 == 0:
        g[n // 2 - 1] = 1

    g[n // 2] = 1
    return gaussian_filter1d(g, sigma)


def mat_generation(indices: list, vec: np.ndarray) -> np.ndarray:
    """

    :param indices: indices to put the kernels
    :param vec: kernel for the matrix rows
    :return: matrix with kernel placed on the selected indices.
    """


    assert isinstance(indices, list)
    assert isinstance(vec, np.ndarray)

    vector_width = len(vec)
    size = (len(indices), indices[-1] + vector_width + 1)
    mat = np.zeros(size)

    for i in range(len(indices)):
        mat[i][indices[i]: indices[i] + vector_width] = vec

    return mat


# class ActivationsAndGradients:
#     """ Class for extracting activations and
#     registering gradients from targetted intermediate layers """
#
#     def __init__(self, model, target_layers, reshape_transform):
#         self.model = model
#         self.gradients = []
#         self.activations = []
#         self.reshape_transform = reshape_transform
#         self.handles = []
#         for target_layer in target_layers:
#             self.handles.append(
#                 target_layer.register_forward_hook(self.save_activation))
#             self.handles.append(
#                 target_layer.register_forward_hook(self.save_gradient))
#
#     def save_activation(self, module, input, output):
#         activation = output
#
#         if self.reshape_transform is not None:
#             activation = self.reshape_transform(activation)
#         self.activations.append(activation.cpu().detach())
#
#     def save_gradient(self, module, input, output):
#         if not hasattr(output, "requires_grad") or not output.requires_grad:
#             # You can only register hooks on tensor requires grad.
#             print("no requires grad attribute. ")
#             return
#
#         # Gradients are computed in reverse order
#         def _store_grad(grad):
#             if self.reshape_transform is not None:
#                 grad = self.reshape_transform(grad)
#             self.gradients = [grad.cpu().detach()] + self.gradients
#
#         output.register_hook(_store_grad)
#
#     def __call__(self, x):
#         self.gradients = []
#         self.activations = []
#         return self.model(x)
#
#     def release(self):
#         for handle in self.handles:
#             handle.remove()






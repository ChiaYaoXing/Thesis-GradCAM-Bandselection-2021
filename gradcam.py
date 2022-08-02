import torch
import torch.nn.functional as f
import numpy as np
import cv2


class GradCam:
    def __init__(self, model):
        self.model = model
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        """

        :param x: band to apply GradCAM
        :return: GradCAM Heatmap
        """
        band = x[0].data.cpu().numpy()
        band = band - np.min(band)
        if np.max(band) != 0:
            band = band / np.max(band)

        feature = x.unsqueeze(0)

        # Pass the input band through all of the layer.
        for name, module in self.model.named_children():
            if name == "fc":
                feature = feature.view(-1)

            feature = module(feature)

            if name == "conv5":
                # Save the gradient of the last convolutional layer.
                feature.register_hook(self.save_gradient)
                self.feature = feature.detach()

            if name != "pool" and name != "fc":
                feature = f.relu(feature)

        classes = f.softmax(feature, dim=0)
        # Get the most probable label class
        a, b = classes.max(dim=-1)
        one_hot = torch.zeros_like(feature)
        one_hot.scatter_(0, b, 1.0)
        self.model.zero_grad()
        feature.backward(gradient=one_hot)

        weight = self.gradient.mean(dim=-1, keepdim=True)

        heatmap = f.relu((weight * self.feature).sum(dim=1)).squeeze(0)

        # Normalise the heatmap between zero and one.
        heatmap = heatmap - torch.min(heatmap)
        if torch.max(heatmap) != 0:
            heatmap = heatmap / torch.max(heatmap)

        heatmap = heatmap.unsqueeze(0)
        size = (band.size, 1)

        # Upscale the heatmap to the same dimension as the input.
        heatmap = cv2.resize(heatmap.data.cpu().numpy(), size, interpolation=cv2.INTER_LINEAR)[0]

        return heatmap

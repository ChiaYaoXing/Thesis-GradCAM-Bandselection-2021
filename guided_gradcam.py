import torch

import gradcam
import Guided_BackProp
import numpy as np
import torch.nn.functional as f

class GuidedGradCam:

    def __init__(self, model):
        self.model = model
        self.gcam = gradcam.GradCam(self.model)
        self.gbp = Guided_BackProp.GuidedBackProp(self.model)

    def __call__(self, band, transform=None):
        """

        :param band: band to apply Guided GradCAM
        :param transform: transform function for the heatmaps
        :return: Guided GradHMs obtained by Hadamard product of both Guided Backpropagation and GradCAM
        """

        hm = self.gcam(band)

        probs, ids = self.gbp.forward(band)
        self.gbp.backward(ids[:, [0]])
        gbp = self.gbp.generate().cpu().numpy()
        gbp = np.clip(gbp, a_min=0, a_max=None)

        if transform is not None:
            hm = transform(hm)
            gbp = transform(gbp)

        gghm = np.multiply(hm, gbp)

        if np.max(gghm) != 0:
            gghm = gghm / np.max(gghm)


        return gghm

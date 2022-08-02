import torch
import torch.nn.functional as f
import torch.nn as nn


class GuidedBackProp:

    def __init__(self, model):

        self.model = model
        self.handlers = []

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.), )

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.handlers.append(module.register_backward_hook(backward_hook))

    def forward(self, band):
        """
        Forward pass of a band to the model.
        :param band: band to apply guided backpropagation
        :return: probability of each label along with its id
        """

        self.band = band.unsqueeze(0)
        self.band = self.band.requires_grad_()
        self.outputs = self.model(self.band)
        self.probs = f.softmax(self.outputs, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def generate(self):
        """
        Generate a gradient at the first layer
        :return: first layer gradient
        """
        gradient = self.band.grad.clone()
        return gradient[0][0]

    def backward(self, ids):
        """
        Class-specific backpropagation
        :param ids: one hot id of the class
        """

        one_hot = torch.zeros_like(self.outputs)
        one_hot.scatter_(1, ids, 1.0)
        self.model.zero_grad()
        self.outputs.backward(gradient=one_hot, retain_graph=True)

    def __call__(self, band):
        """
        :param band: band to apply guided backpropagation
        :return: results of guided backpropagation
        """
        probs, ids = self.forward(band)
        self.backward(ids[:, [0]])
        gbp = self.generate()
        return gbp


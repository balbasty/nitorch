# file to implement monte-carlo dropout and maybe laplace approximation

import torch
from nitorch.core import utils
from nitorch.nn.base import Module
from nitorch.nn.activations.base import activation_from_name, make_activation_from_name


class MCDropout(Module):
    """
    Class to implement Monte-Carlo dropout for a given model.
    Model must already have dropout layers used during training.
    Based on suggested answer in https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
    """
    def __init__(self):
        super().__init__()

    def enable(self, model):
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() # set dropout layers to train
        return model

    def forward(self, model, x, nb_iter=10, activation=None):
        """
        Returns:
            preds, torch.tensor(B, C, *Spatial, nb_iter)
        """
        preds = []
        if isinstance(activation, 'str'):
            activation = activation_from_name(activation)
        elif activation is None:
            activation = None
        for i in range(nb_iter):
            model = self.enable(model)
            with torch.no_grad():
                pred = model(x)
                if activation:
                    pred = activation(pred)
            preds.append(pred)
        preds = torch.stack(preds, -1).to(x.device)
        return preds

    def get_mu_std(self, model, x, nb_iter=10, activation=None):
        preds = self.forward(model, x, nb_iter, activation)
        return preds.mean(-1), preds.std(-1)

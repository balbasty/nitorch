import torch
from nitorch.core import utils
from ..base import Module


class AffineQuantiles(Module):
    """
    Apply an affine transform such that specific quantiles match specific values.
    """

    def __init__(self, qmin=0.05, qmax=0.95, vmin=0, vmax=1, bins=None):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins

    def forward(self, image, **overload):
        qmin = overload.get('qmin', self.qmin)
        qmax = overload.get('qmax', self.qmax)
        vmin = overload.get('vmin', self.vmin)
        vmax = overload.get('vmax', self.vmax)
        dim = image.dim() - 2

        mn, mx = utils.quantile(image, (qmin, qmax),
                                dim=range(-dim, 0),
                                keepdim=True,
                                bins=self.bins).unbind(-1)
        image = (image - mn) * ((vmax - vmin) / (mx - mn)) + vmin
        return image


import torch
from nitorch.core import utils
from ..base import Module


class LabelToOneHot(Module):

    def __init__(self, implicit=False, dtype=None, nb_classes=None):
        super().__init__()
        self.implicit = implicit
        self.dtype = dtype
        self.nb_classes = nb_classes

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (B, 1, *spatial)

        Returns
        -------
        x : (B, C, *spatial)

        """
        nb_classes = overload.get('nb_classes', self.nb_classes)
        dtype = self.dtype or torch.get_default_dtype()
        if nb_classes is None:
            max_label = None
        else:
            max_label = self.nb_classes - 1
        x = utils.one_hot(x, dim=1, implicit=self.implicit, dtype=dtype,
                          max_label=max_label)
        x = x.squeeze(2)  # previous channel dimension
        return x


class OneHotToLabel(Module):

    def __init__(self, implicit=False, dtype=None):
        super().__init__()
        self.implicit = implicit
        self.dtype = dtype

    def forward(self, x):
        """

        Parameters
        ----------
        x : (B, C, *spatial)

        Returns
        -------
        x : (B, 1, *spatial)

        """
        if self.implicit:
            implicit = x.sum(dim=1, keepdim=True).neg_().add_(1)
            pmax, x = x.max(dim=1, keepdim=True)
            x += 1
            x[pmax < implicit] = 0
        else:
            x = x.argmax(dim=1, keepdim=True)
        x = x.to(self.dtype)
        return x

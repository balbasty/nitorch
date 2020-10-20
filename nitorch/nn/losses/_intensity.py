"""
Losses for intensity (continuous) data.
"""

import torch
import torch.nn as tnn
import math
from ._base import Loss
from ...core.pyutils import make_list
from ...core.utils import unsqueeze
from ...core.constants import eps


class MutualInfoLoss(Loss):
    """Mutual information loss.

    This mutual information can be normalized or not, and local or global.

    Notes
    -----
    .. In the local case (`patch_size is not None`), it is advised
       to use a smaller number of bins, as it is much more memory-hungry
       and less values need to be modelled.
    """

    def __init__(self, min_val=None, max_val=None, nb_bins=32, fwhm=1,
                 normalize='arithmetic', patch_size=None, patch_stride=None,
                 *args, **kwargs):
        """

        Parameters
        ----------
        min_val : float or [float, float], default=min of input
            Minimum value in the joint histogram (per image if list).
        max_val : float or [float, float], default=max of input
            Maximum value in the joint histogram (per image if list).
        nb_bins : int or [int, int], default=32
            Number of bins in the histogram (per image if list).
        fwhm : float or [float, float], default=1
            Full-width half-max of the Gaussian window, in bins.
        normalize : bool or str or callable, default='arithmetic'
            Compute the normalized mutual information.
            * if None or 'none', do not normalize.
            * if in ('min', 'max', 'arithmetic', 'geometric'), use this
              method to average H[X] and H[Y] in the normalization term.
            * if callable, should be an averaging function of two values,
              used to average H[X] and H[Y] in the normalization term.
        patch_size : int or list[int], optional
            Patch size for local mutual information.
            If None, compute the global mutual information.
        patch_stride : int or list[int], optional
            Stride between patches for local mutual information.
            If None, same as patch_size
        """
        super().__init__(*args, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.nb_bins = nb_bins
        self.fwhm = fwhm
        self.normalize = normalize
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x, y, **overload):
        """

        Parameters
        ----------
        x : tensor (batch, 1, *spatial)
        y : tensor (batch, 1, *spatial)
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        # check inputs
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)
        dtype = x.dtype
        device = x.device
        nb_dim = x.dim() - 2
        if x.shape[1] != 1 or y.shape[1] != 1:
            raise ValueError('Mutual info is only implemented for '
                             'single channel tensors.')

        # get parameters
        x_min, y_min = make_list(overload.get('min_val', self.min_val), 2)
        x_max, y_max = make_list(overload.get('max_val', self.max_val), 2)
        x_nbins, y_nbins = make_list(overload.get('nb_bins', self.nb_bins), 2)
        x_fwhm, y_fwhm = make_list(overload.get('fwhm', self.fwhm), 2)
        normalize = overload.get('normalize', self.normalize)
        patch_size = overload.get('patch_size', self.patch_size)
        patch_stride = overload.get('patch_stride', self.patch_stride)

        # reshape
        if patch_size is not None:
            # extract patches about each voxel
            patch_size = make_list(patch_size, nb_dim)
            patch_stride = make_list(patch_stride, nb_dim)
            patch_stride = [sz if st is None else st
                            for sz, st in zip(patch_size, patch_stride)]
            x = x[:, 0, ...]
            y = y[:, 0, ...]
            for d, (sz, st) in enumerate(zip(patch_size, patch_stride)):
                x = x.unfold(dimension=d + 1, size=sz, step=st)
                y = y.unfold(dimension=d + 1, size=sz, step=st)
            x = x.reshape((x.shape[0], -1, *patch_size))
            y = y.reshape((y.shape[0], -1, *patch_size))
            # now, the spatial dimension of x and y is `patch_size` and
            # their channel dimension is the number of patches
        # collapse spatial dimensions -> we don't need them anymore
        x = x.reshape((*x.shape[:2], -1))
        y = y.reshape((*y.shape[:2], -1))

        def get_bins(x, min, max, nbins):
            """Compute the histogram bins."""
            # TODO: It's suboptimal to have bin centers fall at the
            #   min and max. Better to shift them slightly inside.
            min = x.min(dim=-1).values if min is None else min
            min = torch.as_tensor(min, dtype=dtype, device=device)
            min = unsqueeze(min, dim=2, ndim=4 - min.dim())
            # -> shape = [B, C, 1, 1]
            max = x.max(dim=-1).values if max is None else max
            max = torch.as_tensor(max, dtype=dtype, device=device)
            max = unsqueeze(max, dim=2, ndim=4 - max.dim())
            # -> shape = [B, C, 1, 1]
            bins = torch.linspace(0, 1, nbins, dtype=dtype, device=device)
            bins = unsqueeze(bins, dim=0, ndim=3)   # -> [1, 1, 1, nb_bins]
            bins = min + bins * (max - min)         # -> [B, C, 1, nb_bins]
            binwidth = (max-min)/(nbins-1)          # -> [B, C, 1, 1]
            return bins, binwidth

        # prepare bins
        x_bins, x_binwidth = get_bins(x, x_min, x_max, x_nbins)
        y_bins, y_binwidth = get_bins(y, y_min, y_max, y_nbins)

        # compute distances and collapse
        x = x[..., None]                            # -> [B, C, N, 1]
        y = y[..., None]                            # -> [B, C, N, 1]
        x_var = ((x_fwhm * x_binwidth) ** 2) / (8 * math.log(2))
        x_var = x_var.clamp(min=eps(x.dtype))
        x = -(x - x_bins).square() / (2 * x_var)    # -> [B, C, N, nb_bins]
        x = x.exp().sum(dim=-2)                     # -> [B, C, nb_bins]
        y_var = ((y_fwhm * y_binwidth) ** 2) / (8 * math.log(2))
        y_var = y_var.clamp(min=eps(y.dtype))
        y = -(y - y_bins).square() / (2 * y_var)
        y = y.exp().sum(dim=-2)
        # -> shape [B, C, nb_bins]

        def pnorm(x, dims=-1):
            """Normalize a tensor so that it's sum across `dims` is one."""
            dims = make_list(dims)
            x = x.clamp(min=eps(x.dtype))
            x = x / x.sum(dim=dims, keepdim=True)
            return x

        # compute probabilities
        p_x = pnorm(x)                            # -> [B, C, nb_bins]
        p_y = pnorm(y)                            # -> [B, C, nb_bins]
        x = x[..., None]                          # -> [B, C, nb_bins, 1]
        y = y[..., None, :]                       # -> [B, C, 1, nb_bins]
        p_xy = pnorm(x*y, [-1, -2])               # -> [B, C, nb_bins, nb_bins]

        # compute entropies
        h_x = -(p_x * p_x.log()).sum(dim=-1)            # -> [B, C]
        h_y = -(p_y * p_y.log()).sum(dim=-1)            # -> [B, C]
        h_xy = -(p_xy * p_xy.log()).sum(dim=[-1, -2])   # -> [B, C]

        # mutual information
        mi = h_x + h_y - h_xy

        # normalize
        if normalize not in (None, 'none'):
            normalize = (lambda a, b: (a+b)/2) if normalize == 'arithmetic' else \
                        (lambda a, b: (a*b).sqrt()) if normalize == 'geometric' else \
                        torch.min if normalize == 'min' else \
                        torch.max if normalize == 'max' else \
                        normalize
            mi = mi / normalize(h_x, h_y)

        # reduce
        return super().forward(-mi)  # negate (loss)

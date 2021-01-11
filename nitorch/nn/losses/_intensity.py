"""
Losses for intensity (continuous) data.
"""

import torch
import torch.nn as tnn
import math
from ._base import Loss
from nitorch.core.pyutils import make_list
from nitorch.core.utils import unsqueeze, channel2last, last2channel
from nitorch.core.constants import eps, nan, inf, pi
from nitorch.core.math import nanmin, nanmax, nansum
from nitorch.core.linalg import matvec
from nitorch.vb.mixtures import GMM


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
                 mask=None, *args, **kwargs):
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
        mask : float or [float, float] or callable or [callable, callable], optional
            * If float: exclude all values at or below this threshold
            * if callable: take tensor as input and output an exclusion mask
        """
        super().__init__(*args, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.nb_bins = nb_bins
        self.fwhm = fwhm
        self.normalize = normalize
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.mask = mask

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
        shape = x.shape[2:]

        # get parameters
        x_min, y_min = make_list(overload.get('min_val', self.min_val), 2)
        x_max, y_max = make_list(overload.get('max_val', self.max_val), 2)
        x_nbins, y_nbins = make_list(overload.get('nb_bins', self.nb_bins), 2)
        x_fwhm, y_fwhm = make_list(overload.get('fwhm', self.fwhm), 2)
        normalize = overload.get('normalize', self.normalize)
        patch_size = overload.get('patch_size', self.patch_size)
        patch_stride = overload.get('patch_stride', self.patch_stride)
        mask = overload.get('mask', self.mask)

        # reshape
        if patch_size is not None:
            # extract patches about each voxel
            patch_size = make_list(patch_size, nb_dim)
            patch_size = [pch or dim for pch, dim in zip(patch_size, shape)]
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

        # exclude masked values
        mask_x, mask_y = make_list(mask, 2)
        mask = None
        if callable(mask_x):
            mask = mask_x(x)
        elif mask_x is not None:
            mask = x <= mask_x
        if callable(mask_y):
            mask = (mask & mask_y(y)) if mask is not None else mask_y(y)
        elif mask_y is not None:
            mask = (mask & (y <= mask_y)) if mask is not None else (y <= mask_y)

        def get_bins(x, min, max, nbins):
            """Compute the histogram bins."""
            # TODO: It's suboptimal to have bin centers fall at the
            #   min and max. Better to shift them slightly inside.
            if mask is not None:
                # we set masked values to nan so that we can exclude them when
                # computing min/max
                val_nan = torch.as_tensor(nan, dtype=x.dtype, device=x.device)
                x = torch.where(mask, val_nan, x)
                min_fn = nanmin
                max_fn = nanmax
            else:
                min_fn = torch.min
                max_fn = torch.max
            min = min_fn(x, dim=-1).values if min is None else min
            min = torch.as_tensor(min, dtype=dtype, device=device)
            min = unsqueeze(min, dim=2, ndim=4 - min.dim())
            # -> shape = [B, C, 1, 1]
            max = max_fn(x, dim=-1).values if max is None else max
            max = torch.as_tensor(max, dtype=dtype, device=device)
            max = unsqueeze(max, dim=2, ndim=4 - max.dim())
            # -> shape = [B, C, 1, 1]
            bins = torch.linspace(0, 1, nbins, dtype=dtype, device=device)
            bins = unsqueeze(bins, dim=0, ndim=3)   # -> [1, 1, 1, nb_bins]
            bins = min + bins * (max - min)         # -> [B, C, 1, nb_bins]
            binwidth = (max-min)/(nbins-1)          # -> [B, C, 1, 1]
            return bins, binwidth

        # prepare bins
        x_bins, x_binwidth = get_bins(x.detach(), x_min, x_max, x_nbins)
        y_bins, y_binwidth = get_bins(y.detach(), y_min, y_max, y_nbins)

        # we transform our nans into inf so that they get zero-weight
        # in the histogram
        if mask is not None:
            val_inf = torch.as_tensor(inf, dtype=x.dtype, device=x.device)
            x = torch.where(mask, val_inf, x)
            y = torch.where(mask, val_inf, y)

        # compute distances and collapse
        x = x[..., None]                            # -> [B, C, N, 1]
        y = y[..., None]                            # -> [B, C, N, 1]
        x_var = ((x_fwhm * x_binwidth) ** 2) / (8 * math.log(2))
        x_var = x_var.clamp(min=eps(x.dtype))
        x = -(x - x_bins).square() / (2 * x_var)
        x = x.exp()
        y_var = ((y_fwhm * y_binwidth) ** 2) / (8 * math.log(2))
        y_var = y_var.clamp(min=eps(y.dtype))
        y = -(y - y_bins).square() / (2 * y_var)
        y = y.exp()
        # -> [B, C, N, nb_bins]

        def pnorm(x, dims=-1):
            """Normalize a tensor so that it's sum across `dims` is one."""
            dims = make_list(dims)
            x = x.clamp(min=eps(x.dtype))
            x = x / nansum(x, dim=dims, keepdim=True)
            return x

        # compute probabilities
        p_x = pnorm(x.sum(dim=2))                 # -> [B, C, nb_bins]
        p_y = pnorm(y.sum(dim=2))                 # -> [B, C, nb_bins]
        x = x.transpose(-1, -2)                   # -> [B, C, nb_bins, N]
        p_xy = torch.matmul(x, y)                 # -> [B, C, nb_bins, nb_bins]
        p_xy = pnorm(p_xy, [-1, -2])

        # compute entropies
        h_x = -(p_x * p_x.log()).sum(dim=-1)            # -> [B, C]
        h_y = -(p_y * p_y.log()).sum(dim=-1)            # -> [B, C]
        h_xy = -(p_xy * p_xy.log()).sum(dim=[-1, -2])   # -> [B, C]

        # negative mutual information
        mi = h_xy - (h_x + h_y)

        # normalize
        if normalize not in (None, 'none'):
            normalize = (lambda a, b: (a+b)/2) if normalize == 'arithmetic' else \
                        (lambda a, b: (a*b).sqrt()) if normalize == 'geometric' else \
                        torch.min if normalize == 'min' else \
                        torch.max if normalize == 'max' else \
                        normalize
            mi = mi / normalize(h_x, h_y)

        # reduce
        return super().forward(mi)


class GMMLoss(Loss):

    def __init__(self, nb_classes=9, max_iter=100, tolerance=1e-3):
        super().__init__()
        self.nb_classes = nb_classes
        self.max_iter = max_iter
        self.tolerance = tolerance

    def forward(self, *image, **opt):

        opt['nb_classes'] = opt.get('nb_classes', self.nb_classes)
        opt['max_iter'] = opt.get('max_iter', self.nb_classes)
        opt['tolerance'] = opt.get('tolerance', self.nb_classes)
        if len(image) > 1:
            image = torch.cat(image, dim=1)
        else:
            image = image[0]

        # first: estimate parameters without computing gradients
        with torch.no_grad():
            means, precisions, proportions = self.fit(image, **opt)

        # second: compute log-likelihood
        resp, log_resp = self.responsibilities(image, means, precisions, proportions)
        loss = self.nll(image, resp, means, precisions)
        loss = loss + self.kl(resp, log_resp, proportions)

        return super().forward(loss)

    @staticmethod
    def nll(image, resp, means, precisions):
        # aliases
        x = image
        z = resp
        m = means
        A = precisions
        nb_dim = image.dim() - 2
        del image, resp, means, precisions

        x = channel2last(x).unsqueeze(-2)       # [B, ...,  1, C]
        z = channel2last(z)                     # [B, ..., K]
        m = unsqueeze(m, dim=1, ndim=nb_dim)    # [B, ones, K, C]
        A = unsqueeze(A, dim=1, ndim=nb_dim)    # [B, ones, K, C, C]
        x = x - m
        loss = matvec(A, x)
        loss = (loss * x).sum(dim=-1)           # [B, ..., K]
        loss = (loss * z).sum(dim=-1)           # [B, ...]
        loss = loss * 0.5
        return loss

    @staticmethod
    def kl(resp, log_resp, proportions):
        # aliases
        z = resp
        logz = log_resp
        p = proportions
        nb_dim = resp.dim() - 2
        del resp, log_resp, proportions

        p = unsqueeze(p, dim=-1, ndim=nb_dim)       # [B, K, ones]
        loss = z * (logz - p.log())                 # [B, K, ...]
        loss = loss.sum(dim=1)                      # [B, ...]
        return loss

    @staticmethod
    def fit(images, nb_classes, max_iter, tolerance):
        means = []
        precisions = []
        proportions = []
        for n, image in enumerate(images):
            gmm = GMM(nb_classes)
            gmm.fit(image.reshape([image.shape[0], -1]).T,
                    max_iter=max_iter, tol=tolerance, verbose=False)
            m, C = gmm.get_means_variances()
            m = m.T
            A = C.permute([2, 0, 1]).inverse()
            p = gmm.mp
            means.append(m)
            precisions.append(A)
            proportions.append(p)
        means = torch.stack(means)
        precisions = torch.stack(precisions)
        proportions = torch.stack(proportions)
        return means, precisions, proportions

    @staticmethod
    def responsibilities(image, means, precisions, proportions):
        # aliases
        x = image
        m = means
        A = precisions
        p = proportions
        nb_dim = image.dim() - 2
        del image, means, precisions, proportions

        # voxel-wise term
        x = channel2last(x).unsqueeze(-2)       # [B, ...,  1, C]
        p = unsqueeze(p, dim=1, ndim=nb_dim)    # [B, ones, K]
        m = unsqueeze(m, dim=1, ndim=nb_dim)    # [B, ones, K, C]
        A = unsqueeze(A, dim=1, ndim=nb_dim)    # [B, ones, K, C, C]
        x = x - m
        z = matvec(A, x)
        z = (z * x).sum(dim=-1)                 # [B, ..., K]
        z = -0.5 * z

        # constant term
        twopi = torch.as_tensor(2*pi, dtype=A.dtype, device=A.device)
        nrm = torch.logdet(A) - A.shape[-1] * twopi.log()
        nrm = 0.5 * nrm + p.log()
        z = z + nrm

        # softmax
        z = last2channel(z)
        logz = torch.nn.functional.log_softmax(z, dim=1)
        z = torch.nn.functional.softmax(z, dim=1)

        return z, logz


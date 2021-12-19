import torch
import os
from ._fit import UniSeg
from ._cleanup import cleanup as cleanup_fn
from nitorch.core import utils
from nitorch.core.datasets import download, cache_dir
from nitorch import io


def uniseg(x, w=None, affine=None, device=None,
           nb_classes=None, prior=None, affine_prior=None,
           do_bias=True, do_warp=True, do_mixing=True, do_mrf=True,
           cleanup=None, spacing=3, lam_bias=0.1, lam_warp=0.1,
           max_iter=30, tol=1e-3, verbose=1, plot=0, return_parameters=False):
    """Unified Segmentation using a deformable spatial prior.

    Inputs
    ------
    x : ([C], *spatial) tensor | [list of] str
        Input multi-channel tensor
    w : (*spatial) tensor | str, optional
        Tensor of weights for each voxel
    affine : (D+1, D+1) tensor, optional
        Orientation matrix of the input tensor(s)

    Atlas
    -----
    prior : (K|K-1, *spatial_prior) tensor | str, optional
        Deformable template. If it contains only K-1 channels, the
        first channel is implicitly defined such that probabilities sum to one.
        The SPM template is used by default
    affine_prior : (D+1, D+1) tensor, optional
        Orientation matrix of the prior.

    Components
    ----------
    nb_classes : int or sequence[int], default=(2, 1, 1, 2, 3, 4)
        If an int: number of classes
        If a sequence: Number of clusters per class
    do_bias : bool, default=True
        Optimize a smooth intensity bias field
    do_warp : bool, default=True
        Optimize a nonlinear warp of the spatial prior
    do_mixing : bool, default=True
        Optimize global missing proportions
    do_mrf : {False, 'once', 'always', 'learn' or True}, default='learn'
        Include a Markov Random Field in the model.
        - 'once' : only at the end
        - 'always' : at each iteration
        - 'learn' : at each iteration and optimize its weights
    cleanup : bool, optional
        Perform an ad-hoc clean-up procedure at the end.
        By default, it is activated if the SPM template is used else
        it is not used.
    spacing : float, default=3
        Space (in mm) between sampled points.
        Smaller is more accurate but slower.

    Optimization
    ------------
    lam_bias : float, default=0.1
        Regularization of the bias field: larger == stiffer
    lam_warp : float, default=0.1
        Regularization of the warp field: larger == stiffer
    max_iter : int, default=30
        Maximum  number of EM iterations
    tol : float, default=1-e3
        Tolerance for early stopping

    Verbosity
    ---------
    verbose : int, default=1
        0: None
        1: Print summary when finished
        2: Print convergence (excluding gmm)
        3: Print convergence (including gmm)
    plot : bool or int, default=False
        0: None
        1: Show lower bound and images at the end
        2: Show live lower bound
        3: Show live images (excluding gmm)
        4: Show live images (including gmm)
    return_parameters : bool, default=False
        Return all fitted parameters (GMM, bias, warp...)

    Returns
    -------
    Z : (K, *spatial) tensor
        Posterior probabilities
    lb : () tensor
        Final value of the lower bound / log likelihood
    parameters : dict, if `return_parameters`
        Fitted parameters
        - 'mean'   : (K, C) tensor
        - 'cov'    : (K, C, C) tensor
        - 'warp'   : (*spatial, D) tensor
        - 'bias'   : (C, *spatial) tensor
        - 'mixing' : (K) tensor
        - 'mrf'    : (K, K) tensor
    """
    if cleanup is None:
        cleanup = (prior is None)  # only cleanup if (default) SPM template

    backend = get_backend(x, prior, device)
    prior, affine_prior = get_prior(prior, affine_prior, **backend)
    dim = prior.dim() - 1
    x, w, affine = get_data(x, w, affine, dim, **backend)
    affine_prior = affine_prior.to(x.dtype)

    if not nb_classes:
        if len(prior) == 5:
            nb_classes = (2, 1, 1, 2, 3, 4)
        else:
            nb_classes = len(prior) + 1

    model = UniSeg(
        nb_classes, prior=prior, affine_prior=affine_prior,
        do_bias=do_bias, do_warp=do_warp, do_mixing=do_mixing, do_mrf=do_mrf,
        lam_bias=lam_bias, lam_warp=lam_warp, spacing=spacing,
        max_iter=max_iter, tol=tol, verbose=verbose, plot=plot,
    )
    z, lb = model.fit(x, w, aff=affine)
    if cleanup:
        z = cleanup_fn(z)

    if not return_parameters:
        return z, lb

    parameters = {}
    parameters['mean'] = model.mu
    parameters['cov'] = model.sigma
    if do_warp:
        parameters['warp'] = model.warp
    if do_bias:
        parameters['bias'] = model.bias
    if do_mixing:
        parameters['mixing'] = model.mixing
    if do_mrf in ('learn', True):
        parameters['mrf'] = model.mrf

    return z, lb, parameters


def uniseg_batch(x, w=None, affine=None, device=None,
                 nb_classes=None, prior=None, affine_prior=None,
                 do_bias=True, do_warp=True, do_mixing=True, do_mrf=True,
                 cleanup=None, spacing=3, lam_bias=0.1, lam_warp=0.1,
                 max_iter=30, tol=1e-3, verbose=1, plot=0,
                 return_parameters=False):
    """Batched Unified Segmentation using a deformable spatial prior.

    Inputs
    ------
    x : (B, C, *spatial) tensor | [list of] list of str
        Input multi-channel tensor
    w : (B, *spatial) tensor | [list of] str, optional
        Tensor of weights for each voxel
    affine : (B, D+1, D+1) tensor, optional
        Orientation matrix of the input tensor(s)

    Other Parameters
    ----------------
    See `uniseg`

    Returns
    -------
    Z : (B, K, *spatial) tensor
        Posterior probabilities
    lb : (B,) tensor
        Final value of the lower bound / log likelihood
    parameters : dict, if `return_parameters`
        Fitted parameters
        - 'mean'   : (B, K, C) tensor
        - 'cov'    : (B, K, C , C) tensor
        - 'warp'   : (B, *spatial, D) tensor
        - 'bias'   : (B, C, *spatial) tensor
        - 'mixing' : (B, K) tensor
        - 'mrf'    : (B, K, K) tensor
    """
    if cleanup is None:
        cleanup = (prior is None)  # only cleanup if (default) SPM template

    backend = get_backend(x, prior, device)
    prior, affine_prior = get_prior(prior, affine_prior, **backend)

    if w is None:
        w = [None] * len(x)
    if affine is None:
        affine = [None] * len(x)

    z = []
    lb = []
    parameters = {}
    parameters['mean'] = []
    parameters['cov'] = []
    if do_warp:
        parameters['warp'] = []
    if do_bias:
        parameters['bias'] = []
    if do_mixing:
        parameters['mixing'] = []
    if do_mrf in (True, 'learn'):
        parameters['mrf'] = []

    for x1, w1, aff1 in zip(x, w, affine):
        out1 = uniseg(
            x1, w1, aff1, device=backend['device'],
            nb_classes=nb_classes, prior=prior, affine_prior=affine_prior,
            do_bias=do_bias, do_warp=do_warp, do_mixing=do_mixing, do_mrf=do_mrf,
            spacing=spacing, cleanup=cleanup, lam_bias=lam_bias, lam_warp=lam_warp,
            max_iter=max_iter, tol=tol, verbose=verbose, plot=plot,
            return_parameters=return_parameters)
        z.append(out1[0])
        lb.append(out1[1])
        if return_parameters:
            for k, v in out1[2].items():
                parameters[k].append(v)

    z = torch.stack(x)
    lb = torch.stack(lb)
    if not return_parameters:
        return z, lb
    for k, v in parameters.items():
        parameters[k] = torch.stack(v)
    return z, lb, parameters


# ======================================================================
#                               HELPERS
# ======================================================================


def get_spm_prior(**backend):
    url = 'https://github.com/spm/spm12/raw/master/tpm/TPM.nii'
    fname = os.path.join(cache_dir, 'SPM12_TPM.nii')
    if not os.path.exists(fname):
        fname = download(url, fname)
    f = io.map(fname).movedim(-1, 0)[:-1]  # drop background
    aff = f.affine
    dat = f.fdata(**backend)
    aff = aff.to(**utils.backend(dat))
    return dat, aff


def get_backend(x, prior, device=None):
    if torch.is_tensor(x):
        backend = utils.backend(x)
    elif torch.is_tensor(prior):
        backend = utils.backend(prior)
    else:
        backend = dict(dtype=torch.get_default_dtype(), device='cpu')
    if device:
        backend['device'] = device
    return backend


def get_prior(prior, affine_prior, **backend):
    if prior is None:
        prior, _affine_prior = get_spm_prior(**backend)
        if affine_prior is None:
            affine_prior = _affine_prior
    elif isinstance(prior, str):
        prior = io.map(prior).movedim(-1, 0)
        if affine_prior is None:
            affine_prior = prior.affine
        prior = prior.fdata(**backend)
    else:
        prior = prior.to(**backend)
    return prior, affine_prior


def get_data(x, w, affine, dim, **backend):
    if not torch.is_tensor(x):
        if isinstance(x, str):
            f = io.map(x)
            if affine is None:
                affine = f.affine
            if f.dim > dim:
                if f.shape[dim] == 1:
                    f = f.squeeze(dim)
                if f.dim > dim + 1:
                    raise ValueError('Too many dimensions')
            if f.dim > dim:
                f = f.movedim(-1, 0)
            else:
                f = f[None]
            x = f.fdata(**backend, missing=0)
        else:
            f = io.stack([io.map(x1) for x1 in x])
            if affine is None:
                f = f.affine[0]
            x = f.fdata(**backend, missing=0)

    if x.dim() > dim + 1:
        x = x.unsqeeze(-1)
    if x.dim() > dim + 1:
        raise ValueError('Too many dimensions')
    if x.dim() == dim:
        x = x[None]

    if not torch.is_tensor(w) and w is not None:
        w = io.loadf(w, **backend)
        if x.dim() > dim:
            w = w.squeeze(-1)
        if x.dim() > dim:
            raise ValueError('Too many dimensions')

    return x, w, affine

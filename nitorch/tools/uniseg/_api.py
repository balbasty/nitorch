import torch
import os
from ._fit import UniSeg
from ._cleanup import cleanup as cleanup_fn
from nitorch.core import utils
from nitorch.core.datasets import download, cache_dir
from nitorch import io
from nitorch.tools.registration.affine_tpm import align_tpm
from nitorch.core.utils import min_intensity_step


def uniseg(x, w=None, affine=None, device=None,
           nb_classes=None, prior=None, affine_prior=None,
           do_bias=True, do_warp=True, do_affine=True, do_mixing=True,
           do_mrf='once', wishart=None, cleanup=None, spacing=3, flexi=False,
           lam_prior=1, lam_bias=0.1, lam_warp=0.1, lam_mixing=100, lam_mrf=10, lam_wishart=1,
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
    device : torch.device, default=x.device
        Device to use during fitting.

    Atlas
    -----
    prior : (K|K-1, *spatial_prior) tensor | str, optional
        Deformable template. If it contains only K-1 channels, the
        first channel is implicitly defined such that probabilities sum to one.
        If None (default), the SPM template is used.
        If False, no prior is used.
    affine_prior : (D+1, D+1) tensor, optional
        Orientation matrix of the prior.

    Components
    ----------
    nb_classes : int or sequence[int], default=(4, 2, 2, 2, 2, 3)
        If an int: number of classes
        If a sequence: Number of clusters per class
    do_bias : bool, default=True
        Optimize a smooth intensity bias field
    do_warp : bool, default=True
        Optimize a nonlinear warp of the spatial prior
    do_affine : {False, 'once', 'always' or True}, default='always'
        Optimize an affine warp of the spatial prior
    do_mixing : bool, default=True
        Optimize global missing proportions
    do_mrf : {False, 'once', 'always', 'learn' or True}, default='once'
        Include a Markov Random Field in the model.
        - 'once' : only at the end
        - 'always' : at each iteration
        - 'learn' : at each iteration and optimize its weights
    wishart : {False, True, 'preproc8'}, default='preproc8' or True
        Regularises the estimated covariances using statistics derived
        from the image. If 'preproc8', a coarser mask derived from the TPM
        is used to compute these statistics; only works with the SPM prior.
        If the SPM prior is used, 'preproc8' is activated by default.
    cleanup : bool, optional
        Perform an ad-hoc clean-up procedure at the end.
        By default, it is activated if the SPM template is used else
        it is not used.
    spacing : float, default=3
        Space (in mm) between sampled points. None (or 0) uses all the voxels.
        Smaller is more accurate but slower.
    flexi : bool, default=False
        Try to find the correct orientation of the input data.

    Optimization
    ------------
    lam_prior : float, default=1
        Strength of the spatial prior
    lam_bias : float, default=0.1
        Regularization of the bias field: larger == stiffer
    lam_warp : float, default=0.1
        Regularization of the warp field: larger == stiffer
    lam_mixing : float, default=100
        Regularization of the mixing proportions: larger == closer to atlas
    lam_mrf : float, default=10
        Regularization of the MRF weights: larger == less smooth
    lam_wishart : float, default=1
        Modulation of the Wishart degrees of freedom: larger == fixed variances
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
    if wishart is None:
        wishart = 'preproc8' if (prior is None) else True

    backend = get_backend(x, prior, device)
    if prior is not False:
        prior, affine_prior = get_prior(prior, affine_prior, **backend)
        dim = prior.dim() - 1
    else:
        prior = affine_prior = None
        if torch.is_tensor(x):
            dim = x.dim() - 1
        else:
            dim = 3
        do_warp = False
        do_affine = False
    x, w, affine = get_data(x, w, affine, dim, **backend)
    if affine_prior is not None:
        affine_prior = affine_prior.to(x.dtype)

    if not nb_classes:
        if prior is None:
            raise ValueError('If no prior is provided, the number of '
                             'classes must be provided.')
        if len(prior) == 5:
            nb_classes = (4, 2, 2, 2, 2, 3)
        else:
            nb_classes = len(prior)

    # --- align --------------------------------------------------------
    aff = None
    if do_affine:
        K = nb_classes if isinstance(nb_classes, int) else len(nb_classes)
        if len(prior) == K - 1:
            prior_for_align = prior.new_empty([K, *prior.shape[1:]])
            prior_for_align[:-1] = prior
            torch.sum(prior, 0, out=prior_for_align[-1])
            prior_for_align[-1].neg_().add_(1)
        else:
            prior_for_align = prior
        aff = align_tpm((x, affine), (prior_for_align, affine_prior), w,
                        verbose=verbose-1, flexi=flexi)
        affine = aff.to(affine).matmul(affine)
        del prior_for_align

    # --- fit ----------------------------------------------------------
    do_affine = do_affine in (True, 'always')
    model = UniSeg(
        nb_classes, prior=prior, affine_prior=affine_prior,
        do_bias=do_bias, do_warp=do_warp, do_affine=do_affine,
        do_mixing=do_mixing, do_mrf=do_mrf, lam_prior=lam_prior, lam_bias=lam_bias,
        lam_warp=lam_warp, lam_mixing=lam_mixing, lam_mrf=lam_mrf,
        spacing=spacing, max_iter=max_iter, tol=tol, verbose=verbose,
        plot=plot, wishart=wishart, lam_wishart=lam_wishart,
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
    if do_affine:
        parameters['affine'] = model.affine @ aff
    elif aff is not None:
        parameters['affine'] = aff
    if do_mrf in ('learn', True):
        parameters['mrf'] = model.mrf
    if prior is not None:
        parameters['warped'] = model.warp_tpm(
            aff=affine, mode='softmax', shape=z.shape[1:])

    return z, lb, parameters


def uniseg_batch(x, w=None, affine=None, device=None,
                 nb_classes=None, prior=None, affine_prior=None,
                 do_bias=True, do_warp=True, do_mixing=True, do_mrf=True,
                 wishart=None, cleanup=None, spacing=3, flexi=False,
                 lam_prior=1, lam_bias=0.1, lam_warp=0.1, lam_mixing=100,
                 lam_mrf=10, lam_wishart=1, max_iter=30, tol=1e-3, verbose=1, plot=0,
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
            spacing=spacing, cleanup=cleanup, wishart=wishart,
            lam_prior=lam_prior,  lam_bias=lam_bias, lam_warp=lam_warp,
            lam_mrf=lam_mrf, lam_wishart=lam_wishart, lam_mixing=lam_mixing,
            max_iter=max_iter, tol=tol, verbose=verbose, plot=plot, flexi=flexi,
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


def path_spm_prior():
    url = 'https://github.com/spm/spm12/raw/master/tpm/TPM.nii'
    fname = os.path.join(cache_dir, 'SPM12_TPM.nii')
    if not os.path.exists(fname):
        os.makedirs(cache_dir, exist_ok=True)
        fname = download(url, fname)
    return fname


def get_spm_prior(**backend):
    fname = path_spm_prior()
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
    elif isinstance(prior, (list, tuple)):
        def ensure_4d(x):
            while x.dim < 4:
                x = x.unsqueeze(-1)
            while x.dim > 4:
                x = x.squeeze(-1)
            return x
        prior = io.cat(list(map(lambda x: ensure_4d(io.volumes.map(x)), prior)), -1)
        prior = prior.movedim(-1, 0)
    elif isinstance(prior, str):
        prior = io.volumes.map(prior).movedim(-1, 0)
    if isinstance(prior, io.MappedArray):
        if affine_prior is None:
            affine_prior = prior.affine
            if isinstance(affine_prior, (list, tuple)):
                affine_prior = affine_prior[0]
        prior = prior.fdata(**backend)
    else:
        prior = prior.to(**backend)
    return prior, affine_prior


def get_data(x, w, affine, dim, **backend):
    def ensure_dim(f):
        def ndim(f):
            return f.dim() if callable(f.dim) else f.dim
        if ndim(f) > dim:
            if f.shape[dim] == 1:
                f = f.squeeze(dim)
            if ndim(f) > dim + 1:
                raise ValueError('Too many dimensions')
        if f.dim > dim:
            f = f.movedim(-1, 0)
        else:
            f = f[None]
        return f

    if not torch.is_tensor(x):
        if isinstance(x, str):
            f = io.map(x)
            if affine is None:
                affine = f.affine
            f = ensure_dim(f)
        else:
            f = io.cat([ensure_dim(io.map(x1)) for x1 in x])
            if affine is None:
                affine = f.affine[0]
        x = f.fdata(**backend, rand=True, missing=0)
    else:
        x = ensure_dim(x)

    if not torch.is_tensor(w) and w is not None:
        w = io.loadf(w, **backend)
        if w.dim() > dim:
            w = w.squeeze(-1)
        if w.dim() > dim:
            raise ValueError('Too many dimensions')

    step = min_intensity_step(x)
    if step > 0.5:
        x = torch.rand_like(x).mul_(step).mul_(x > 0).add_(x)

    x = x.contiguous()
    if w is not None:
        w = w.contiguous()
    return x, w, affine

r"""
This loss has the same behaviour as `spm_maff8`

General idea
------------
. Let 𝒙 ∈ ⟦1,J⟧ᴺ be an *observed* label map.
  Typically, an image is discretized into J bins, and 𝑥ₙ indexes
  the bin in which voxel n falls. But it could also be any segmentation.
. Let 𝒚 ∈ ⟦1,K⟧ᴺ be a *hidden* label map (with a potentially different
  labelling scheme: J != K), such that p(𝑥ₙ = 𝑗 | 𝑦ₙ = 𝑘) = p(𝑥 = 𝑗) 𝐻ⱼₖ
  (see below for the normalization of 𝑯).
. Let 𝜇 ∈ ℝᴺᴷ be a (deformable) prior probability to see
  label k in voxel n, and 𝜙 ∈ ℝᴺᴰ be the corresponding
  deformation field, such that: p(𝑦ₙ = 𝑘) = (𝜇 ∘ 𝜙)ₙₖ
  Here, 𝜇 is a tissue probability map (TPM), 𝜙 is affine, and
  𝜇 ∘ 𝜙 is the softmax of the interpolated log-TPM.
. We want to find the parameters {𝐻, 𝜙} that maximize the marginal
  likelihood:
      p(𝒙; 𝑯, 𝜇, 𝜙) = ∏ₙ ∑ₖ p(𝑥ₙ | 𝑦ₙ = 𝑘; 𝑯) p(𝑦ₙ = 𝑘; 𝜇, 𝜙)
  where we use ";" to separate random variables from parameters.
. Note that 𝜇 is assumed constant in this problem: only 𝑯 and 𝜙 are
  optimized. There is no prior on the parameters of 𝜙: this is a
  maximum-likelihood fit.
. In order to maximize 𝑯, we use an EM algorithm (warm-started from the
  previous 𝑯). We introduce an approximate parameter 𝑯 and compute the
  approximate posterior distribution:
      q(𝑦ₙ = 𝑘 | 𝑥ₙ = 𝑗) ∝ 𝐻ⱼₖ (𝜇 ∘ 𝜙)ₙₖ                   (E-step)
  We then compute the expected value of the joint log-likelihood with respect
  to q, and maximize it with respect to 𝑯:
      p(𝑥 = 𝑗, 𝑦 = 𝑘) ∝ ∑ₙ 𝛿(𝑥ₙ = 𝑗) q(𝑦ₙ = 𝑘)              (M-step)
      𝐻ⱼₖ = p(𝑥 = 𝑗, 𝑦 = 𝑘) / (p(𝑥 = 𝑗) p(𝑦 = 𝑘))
. 𝑯 is therefore the "normalized" joint histogram p(x,y) / (p(x) p(y))
  = p(x|y) / p(x), not the conditional p(x|y). The factor 1/p(𝑥 = 𝑗)
  does not depend on k, so it cancels in the E-step. In 𝓛 (below) it
  only adds the constant -∑ₙ ln p(𝑥 = 𝑥ₙ), which depends on the image
  histogram only. With this normalization, when 𝜇 ∘ 𝜙 is one-hot (and
  the histogram is not smoothed), 𝓛/N is the mutual information
  between 𝒙 and 𝒚.
. The joint histogram is smoothed along the image-bin axis (j) after
  each M-step (`fwhm`). The M-step is then not an exact maximizer of 𝓛,
  and EM is not guaranteed to be monotonic.
. With 𝑯 fixed, in order to maximize 𝜙, we go back to the
  marginal log-likelihood:
      𝓛 = ln p(𝒙) = ∑ₙ ln ∑ₖ 𝐻{[𝑥ₙ],𝑘} (𝜇 ∘ 𝜙)ₙₖ
  which we differentiate with respect to (the parameters of) 𝜙, taking
  Gauss-Newton steps on -𝓛. If 𝑯 were the exact maximizer of 𝓛 for
  the current 𝜙, the gradient with 𝑯 fixed would be the gradient of the
  profile log-likelihood max_𝑯 𝓛 (envelope theorem). Because the
  histogram is smoothed and EM is stopped early, this is only
  approximate.
. The value used by the line search and the stopping criteria is 𝓛/N,
  evaluated at the 𝑯 re-estimated by EM at each trial 𝜙. (The mutual
  information of the expected joint histogram, ∑ⱼₖ p(𝑥 = 𝑗, 𝑦 = 𝑘) ln 𝐻ⱼₖ,
  is, up to the change of 𝑯 in the last M-step, 𝓛/N plus the mean
  KL(q ‖ 𝜇 ∘ 𝜙): it is not the function whose gradient is used, and it
  is nearly flat around the optimum.)
. If we do not have an observed hard segmentation, but a (fixed) separable
  distribution q over 𝒙, we can maximize the expected log-likelihood instead:
      𝓛 = 𝔼_q[ln p(𝒙)] = ∑ₙⱼ q(𝑥ₙ = 𝑗) ln ∑ₖ 𝐻ⱼₖ (𝜇 ∘ 𝜙)ₙₖ
  EM then treats each pair (n, j) as an observation with weight
  q(𝑥ₙ = 𝑗): the E-step computes q(𝑦ₙ = 𝑘 | 𝑥ₙ = 𝑗) as above, for all j,
  and the M-step becomes
      p(𝑥 = 𝑗, 𝑦 = 𝑘) ∝ ∑ₙ q(𝑥ₙ = 𝑗) q(𝑦ₙ = 𝑘 | 𝑥ₙ = 𝑗)       (M-step)

Mapping to the code
-------------------
. `prior[..., j, k]` is 𝐻ⱼₖ, of shape (B, J, K); `fixed` holds 𝒙
  (or q); `moving[:, k]` holds (𝜇 ∘ 𝜙)ₙₖ; `prm` holds the Lie
  parameters of the affine 𝜙.
"""
import torch
from nitorch.core import linalg, utils, py, math
from nitorch import spatial, io
from .utils import jg, jhj, affine_grid_backward
import nitorch.plot as niplt
from nitorch.core.datasets import download, cache_dir
import os
import math as pymath


# Batched linalg functions
mm = torch.matmul
mv = linalg.matvec
lmdiv = linalg.lmdiv
rmdiv = linalg.rmdiv
t = lambda x: x.transpose(-1, -2)
tiny = 1e-16


# ======================================================================
#
#                               API
#
# ======================================================================

def align_tpm(dat, tpm=None, weights=None, spacing=(8, 4), device=None,
              basis='affine', progressive=False, bins=256,
              fwhm=None, max_iter_gn=100, max_iter_em=32, max_line_search=6,
              tolerance=1e-6, flexi=False, verbose=1):
    r"""Align a Tissue Probability Map to an image

    Input Parameters
    ----------------
    dat : file(s) or tensor or (tensor, affine)
        Input image(s)
    tpm : file(s) or tensor or (tensor, affine), optional
        Input tissue probability map. Uses SPM's TPM by default.
        If a tensor, must have shape `(K, *spatial)`.
    weights : file(s) or tensor
        Input mask or weight map
    device : torch.device, optional
        Specify device
    verbose : int, default=1
        0 = Write nothing
        1 = Outer loop
        2 = Line search
        3 = Inner loop

    Option Parameters
    -----------------
    spacing : float(s), default=(8, 4)
        Sampling  distance in mm. If multiple value, coarse to fine fit.
        Larger is faster but less accurate.
    basis : {'trans', 'rot', 'rigid', 'sim', 'aff'}, default='affine'
        Transformation model
    progressive : bool, default=False
        Fit prameters progressively (translation then rigid then affine)
    flexi : bool, default=False
        Assume that the header can be wrong and try all possible orientations

    Optimization parameters
    -----------------------
    bins : int, default=256
        Number of bins to use to discretize the input image
    fwhm : float, default=bins/64
        Full-width at half-maximum used to smooth the joint histogram
        (along the image-bin axis)
    max_iter_gn : int, default=100
        Maximumm number of Gauss-Newton iterations
    max_iter_em : int, default=32
        Maximum number of EM iterations
    max_line_search : int, default=6
        Maximum number of line search steps (step halvings).
    tolerance : float, default=1e-6
        Stop Gauss-Newton when the gain in 𝓛/N (average
        log-likelihood per voxel) is less than `tolerance`.

    Returns
    -------
    aff : (4, 4) tensor
        Affine matrix.
        Can be applied to the TPM by `aff \ tpm.affine`
        or to the image by `aff @ dat.affine`.

    """
    # ------------------------------------------------------------------
    #       LOAD DATA
    # ------------------------------------------------------------------
    affine_dat = affine_tpm = None
    if isinstance(dat, (list, tuple)) and torch.is_tensor(dat[0]):
        affine_dat = dat[1] if len(dat) > 1 else None
        dat = dat[0]
    if isinstance(tpm, (list, tuple)) and torch.is_tensor(tpm[0]):
        affine_tpm = tpm[1] if len(tpm) > 1 else None
        tpm = tpm[0]
    backend = get_backend(dat, tpm, device)
    tpm, affine_tpm = get_prior(tpm, affine_tpm, **backend)
    dim = tpm.dim() - 1
    dat, weights, affine_dat = get_data(dat, weights, affine_dat, dim, **backend)

    if weights is None:
        weights = 1
    weights = weights * torch.isfinite(dat)

    dat = dat.unsqueeze(1)
    weights = weights.unsqueeze(1)
    # dat:      [B, 1, *spatial]
    # weights:  [B, 1, *spatial]
    # tpm:      [K, *spatial]

    # ------------------------------------------------------------------
    #       DEFAULT ORIENTATION MATRICES
    # ------------------------------------------------------------------
    if affine_tpm is not None:
        affine_tpm = affine_tpm.to(dat.dtype)
    else:
        affine_tpm = spatial.affine_default(tpm.shape[-dim:], **backend)
    if affine_dat is None:
        affine_dat = spatial.affine_default(dat.shape[-dim:], **backend)
    affine_dat = affine_dat.to(tpm)
    affine_tpm = affine_tpm.to(tpm)

    # ------------------------------------------------------------------
    #       DISCRETIZE
    # ------------------------------------------------------------------
    dat = discretize(dat, nbins=bins, mask=weights)

    # ------------------------------------------------------------------
    #       PREFILTER TPM
    # ------------------------------------------------------------------
    logtpm = tpm.clone()
    # ensure normalized
    logtpm = logtpm.clamp(tiny, 1-tiny).div_(logtpm.sum(0, keepdim=True))
    # transform to logits
    logtpm = logtpm.add_(1e-4).log_()
    # spline prefilter
    splineopt = dict(interpolation=2, bound='replicate')
    logtpm = spatial.spline_coeff_nd(logtpm, dim=dim, inplace=True, **splineopt)

    # ------------------------------------------------------------------
    #       OPTIONS
    # ------------------------------------------------------------------
    opt = dict(
        basis=basis,
        progressive=progressive,
        fwhm=fwhm,
        max_iter_gn=max_iter_gn,
        max_iter_em=max_iter_em,
        max_line_search=max_line_search,
        tolerance=tolerance,
        verbose=verbose,
    )

    spacing = py.make_list(spacing) or [0]
    dat0, affine_dat0, weights0 = dat, affine_dat, weights
    vx = spatial.voxel_size(affine_dat0)

    def do_spacing(sp):
        if not sp:
            return dat0, affine_dat0, weights0
        sp = [max(1, int(pymath.floor(sp / vx1))) for vx1 in vx]
        sp = tuple([slice(None, None, sp1) for sp1 in sp])
        affine_dat, _ = spatial.affine_sub(affine_dat0, dat0.shape[-dim:], sp)
        dat = dat0[(Ellipsis, *sp)]
        if weights0 is not None:
            weights = weights0[(Ellipsis, *sp)]
        else:
            weights = None
        return dat, affine_dat, weights

    # ------------------------------------------------------------------
    #       FLEXIBLE INITIALIZATION
    # ------------------------------------------------------------------
    prm0 = reorient0 = None
    if flexi:
        opt1 = dict(opt)
        opt1['verbose'] = 0

        # first: trust header
        sp = spacing[0] * 2
        dat, affine_dat1, weights = do_spacing(sp)
        dat_com1 = torch.as_tensor(dat.shape[2:]).to(affine_dat1).sub_(1).div_(2)
        dat_com1 = mv(affine_dat1[:-1, :-1], dat_com1).add_(affine_dat1[:-1, -1])
        best_name = spatial.volume_layout_to_name(spatial.affine_to_layout(affine_dat0))
        # NOTE: fit_affine_tpm expects the prefiltered log-TPM, and the
        #       orientation matrix of the subsampled data.
        mi0, _, prm0 = fit_affine_tpm(dat, logtpm, affine_dat1, affine_tpm,
                                      weights, **opt1)
        mi0 = mi0.sum().item()
        if verbose:
            print(f'best: {best_name} ({mi0:4.2f})', end='\r')

        # RAS coordinate of the center of the field of view
        tpm_com = torch.as_tensor(tpm.shape[1:]).to(affine_tpm).sub_(1).div_(2)
        tpm_com = mv(affine_tpm[:-1, :-1], tpm_com).add_(affine_tpm[:-1, -1])

        # next: try all possible orientations
        for layout in spatial.iter_layouts(dim, device=dat.device):
            reorient = spatial.layout_matrix(layout).to(affine_dat1)
            reorient[:-1, -1] += tpm_com - mv(reorient[:-1, :-1], dat_com1)
            affine_dat = mm(reorient, affine_dat1)
            name1 = spatial.volume_layout_to_name(spatial.affine_to_layout(affine_dat))
            if verbose:
                print(f'best: {best_name} ({mi0:4.2f}) | {name1}', end='')
            mi, _, prm = fit_affine_tpm(dat, logtpm, affine_dat, affine_tpm,
                                        weights, **opt1)
            mi = mi.sum().item()
            if verbose:
                print(f' ({mi:4.2f})', end='\r')
            # update if better
            if mi > mi0:
                prm0, reorient0, mi0, best_name = prm, reorient, mi, name1
        if verbose:
            print(f'best: {best_name} ({mi0:4.2f})')
        opt['progressive'] = False

    prm, reorient = prm0, reorient0

    # ------------------------------------------------------------------
    #       SPACING
    # ------------------------------------------------------------------
    for sp in spacing:
        dat, affine_dat, weights = do_spacing(sp)
        if reorient is not None:
            affine_dat = reorient.matmul(affine_dat)

        mi, aff, prm = fit_affine_tpm(dat, logtpm, affine_dat, affine_tpm,
                                      weights, **opt, prm=prm)

    if reorient is not None:
        aff = aff.matmul(reorient.to(aff))
    return aff.squeeze()


# ======================================================================
#
#                           MAIN ROUTINE
#
# ======================================================================


def fit_affine_tpm(dat, tpm, affine=None, affine_tpm=None, weights=None,
                   basis='affine', fwhm=None, prm=None,
                   max_iter_gn=100, max_iter_em=32, max_line_search=6,
                   tolerance=1e-6, progressive=False, verbose=1):
    """Fit an affine transform between a log-TPM and an image

    Parameters
    ----------
    dat : ([B], 1|J, *spatial) tensor
        Image: either discretized into J bins (integer, one channel)
        or a soft (float) distribution over J bins.
    tpm : (K, *spatial) tensor
        Spline-prefiltered (order 2) log-probabilities of the TPM 𝜇.
    affine : (4, 4) tensor
        Orientation matrix of `dat`.
    affine_tpm : (4, 4) tensor
        Orientation matrix of `tpm`.
    weights : ([B], 1, *spatial) tensor
    basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
    fwhm : float, default=J/64
        Smoothing of the joint histogram along the image-bin axis.
    prm : (F) tensor, optional
        Initial Lie parameters.
    max_iter_gn : int, default=100
    max_iter_em : int, default=32
    max_line_search : int, default=6
        Maximum number of step halvings. Each Gauss-Newton iteration
        starts with a full step.
    tolerance : float, default=1e-6
        Stop when the gain in 𝓛/N (average log-likelihood per voxel)
        between two Gauss-Newton iterations is less than `tolerance`,
        i.e., when the gain in 𝓛 is less than `tolerance * N`.
    progressive : bool, default=False

    Returns
    -------
    ll : ([B]) tensor
        Average marginal log-likelihood 𝓛/N at the solution
        (higher is better), with 𝑯 = p(x,y)/(p(x)p(y)).
    aff : (4, 4) tensor
        Affine matrix `expm(prm)`.
    prm : (F) tensor
        Lie parameters.

    """
    # !!! NOTE: `tpm` must contain spline-prefiltered log-probabilities

    dim = tpm.dim() - 1

    # ------------------------------------------------------------------
    #       RECURSIVE PROGRESSIVE FIT
    # ------------------------------------------------------------------
    if progressive:
        nb_se = dim * (dim + 1) // 2
        nb_aff = dim * (dim + 1)
        basis_recursion = {'Aff+': 'CSO', 'CSO': 'SE', 'SE': 'T'}
        basis_nb_feat = {'Aff+': nb_aff, 'CSO': nb_se + 1, 'SE': nb_se}
        basis = convert_basis(basis)
        next_basis = basis_recursion.get(basis, None)
        if next_basis:
            *_, prm = fit_affine_tpm(
                dat, tpm, affine, affine_tpm, weights,
                basis=next_basis, fwhm=fwhm, prm=prm,
                max_iter_gn=max_iter_gn, max_iter_em=max_iter_em,
                max_line_search=max_line_search, tolerance=tolerance,
                progressive=True, verbose=verbose)
            F = basis_nb_feat[basis]
            prm0 = prm
            prm = prm0.new_zeros([F])
            if basis == 'SE':
                prm[:dim] = prm0[:dim]
            else:
                nb_se = dim*(dim+1)//2
                prm[:nb_se] = prm0[:nb_se]
                if basis == 'Aff+':
                    prm[nb_se:nb_se+dim] = prm0[nb_se] * (dim**(-0.5))

    basis_name = basis

    # ------------------------------------------------------------------
    #       PREPARE
    # ------------------------------------------------------------------

    if dat.ndim == dim:
        dat = dat[None]
    if dat.ndim == dim+1:
        dat = dat[None]
    tpm = tpm[None]

    if affine is None:
        affine = spatial.affine_default(dat.shape[-dim:])
    if affine_tpm is None:
        affine_tpm = spatial.affine_default(tpm.shape[-dim:])
    affine = affine.to(**utils.backend(tpm))
    affine_tpm = affine_tpm.to(**utils.backend(tpm))
    shape = dat.shape[-dim:]

    tpm = tpm.to(dat.device)
    basis = make_basis(basis, dim, **utils.backend(tpm))
    F = len(basis)

    if prm is None:
        prm = tpm.new_zeros([F])
    aff, gaff = linalg._expm(prm, basis, grad_X=True)

    em_opt = dict(fwhm=fwhm, max_iter=max_iter_em, weights=weights,
                  verbose=verbose-2)
    drv_opt = dict(weights=weights)
    pull_opt = dict(bound='replicate', extrapolate=True, interpolation=2)

    # ------------------------------------------------------------------
    #       OPTIMIZE
    # ------------------------------------------------------------------
    prior = None
    mi = torch.as_tensor(-float('inf'))
    delta = torch.zeros_like(prm)
    for n_iter in range(max_iter_gn):

        # --------------------------------------------------------------
        #       LINE SEARCH
        # --------------------------------------------------------------
        prior0, prm0, mi0 = prior, prm, mi
        # Each GN iteration starts with a full step, which is halved
        # until the objective improves (successful steps are not
        # enlarged: the next iteration starts again from a full step).
        armijo = 1
        success = False
        for n_ls in range(max_line_search):

            # --- take a step ------------------------------------------
            prm = prm0 - armijo * delta

            # --- build transformation field ---------------------------
            aff, gaff = linalg._expm(prm, basis, grad_X=True)
            phi = lmdiv(affine_tpm, mm(aff, affine))
            phi = spatial.affine_grid(phi, shape)

            # --- warp TPM ---------------------------------------------
            mov = spatial.grid_pull(tpm, phi, **pull_opt)
            mov = math.softmax(mov, dim=1)

            # --- log-likelihood (with H re-estimated by EM) -----------
            mi, Nm, prior = em_prior(mov, dat, prior0, **em_opt)
            mi = mi / Nm

            success = mi.sum() > mi0.sum()
            if verbose >= 2:
                end = '\n' if verbose >= 3 else '\r'
                happy = ':D' if success else ':('
                print(f'(search) | {n_ls:02d} | {mi.mean():12.6g} | {happy}', end=end)

            if success:
                break
            armijo *= 0.5
        # if verbose == 2:
        #     print('')

        # --------------------------------------------------------------
        #       DID IT WORK?
        # --------------------------------------------------------------

        # DEBUG
        # plot_registration(dat, mov, f'{basis_name} | {n_iter}')

        if not success:
            # revert to the last accepted parameters (and matrix)
            prior, prm, mi = prior0, prm0, mi0
            aff = linalg._expm(prm, basis)
            break

        space = ' ' * max(0, 6 - len(basis_name))
        if verbose >= 1:
            end = '\n' if verbose >= 2 else '\r'
            print(f'({basis_name[:6]}){space} | {n_iter:02d} | {mi.mean():12.6g}', end=end)

        # `mi` is 𝓛/N, so this is a test on the gain in 𝓛 relative to N
        if mi.mean() - mi0.mean() < tolerance:
            break

        # --------------------------------------------------------------
        #       GAUSS-NEWTON
        # --------------------------------------------------------------

        # --- derivatives ----------------------------------------------
        # gradient and (Gauss-Newton) Hessian of -𝓛 with H fixed
        g, h = derivatives_intensity(mov, dat, prior, **drv_opt)
        g = g.sum(0)
        h = h.sum(0)

        # --- spatial derivatives --------------------------------------
        # d(𝜇 ∘ 𝜙)/d𝜙 through the softmax: 𝜇 * (d logit - <𝜇, d logit>)
        mov = mov.unsqueeze(-1)
        gmov = spatial.grid_grad(tpm, phi, **pull_opt)
        gmov = mov * (gmov - (mov * gmov).sum(1, keepdim=True))
        mov = mov.squeeze(-1)

        # --- chain rule -----------------------------------------------
        gaff = lmdiv(affine_tpm, mm(gaff, affine))
        g, h = chain_rule(g, h, gmov, gaff, maj=False)
        del gmov

        # --- Gauss-Newton ---------------------------------------------
        h.diagonal(0, -1, -2).add_(h.diagonal(0, -1, -2).abs().max() * 1e-5)
        delta = lmdiv(h, g.unsqueeze(-1)).squeeze(-1)

    # plot_registration(dat, mov, f'{basis_name} | {n_iter}')

    if verbose == 1:
        print('')
    return mi, aff, prm


# ======================================================================
#
#                           COMPONENTS
#
# ======================================================================


def chain_rule(g, h, g_mu, g_aff, maj=True):
    """
    parameters
    g, h : (K|K2, *spatial) - gradient/Hessian of loss wrt moving image
    g_mu : (K, *spatial, D) - spatial gradients of moving image
    g_aff : (F, D+1, D+1) - gradient of affine matrix wrt Lie parameters
    maj : bool - use a diagonal majorizer of the Hessian
    returns
    g : (F,) tensor - gradient of loss wrt Lie parameters
    h : (F, F) tensor - (Gauss-Newton) Hessian of loss wrt Lie parameters
    """
    # Note that `h` can be `None`, but the functions I
    # use deal with this case correctly.
    ndim = g.dim() - 1
    K, *shape, D = g_mu.shape
    F, _, _ = g_aff.shape
    D2 = D * (D + 1)

    g = jg(g_mu, g, dim=ndim)
    h = jhj(g_mu, h, dim=ndim)
    g, h = affine_grid_backward(g, h)
    g = g.reshape([D2])
    g_aff = g_aff[:, :-1, :].reshape([F, D2])
    g = mv(g_aff, g)
    if h is not None:
        h = h.reshape([D2, D2])
        h = mm(mm(g_aff, h), t(g_aff))
        if maj:
            h = h.abs().sum(-1).diag_embed()
        # h = h.diagonal(0, -1, -2).abs().diag_embed()
    return g, h


def em_prior(moving, fixed, prior=None, weights=None, fwhm=None,
             max_iter=32, tolerance=1e-5, verbose=0):
    """Estimate the normalized joint histogram H by Expectation Maximization

    𝐻ⱼₖ = p(𝑥 = 𝑗, 𝑦 = 𝑘) / (p(𝑥 = 𝑗) p(𝑦 = 𝑘)), where 𝒙 is the
    discretized image and 𝒚 the TPM classes (see the module docstring).

    The objective function is the marginal log-likelihood
        𝓛 = ∑ₙ ln ∑ₖ 𝐻{[𝑥ₙ],𝑘} (𝜇 ∘ 𝜙)ₙₖ                    (hard labels)
    or, if `fixed` is a soft (float) distribution q over bins,
        𝓛 = ∑ₙⱼ q(𝑥ₙ = 𝑗) ln ∑ₖ 𝐻ⱼₖ (𝜇 ∘ 𝜙)ₙₖ               (soft labels)

    Parameters
    ----------
    moving : (1|B, K, *spatial) tensor
        Deformed TPM 𝜇 ∘ 𝜙 (probabilities).
    fixed : (B, 1|J, *spatial) tensor
        Discretized image 𝒙 (integer, one channel), or
        soft distribution q over J bins (float, J channels).
    prior : (B, J, K) tensor, optional
        Initial 𝑯 (warm start). Default: uniform.
    weights : (B, 1, *spatial) tensor, optional
        Voxel weights.
    fwhm : float, default=J/64
        Width of the smoothing of the joint histogram along the
        image-bin axis (j), applied after each M-step.
    max_iter : int, default=32
        Maximum number of EM iterations.
        If 0, only evaluate 𝓛 at `prior`.
    tolerance : float, default=1e-5
        Stop when the absolute change of 𝓛 between two EM iterations
        is less than `tolerance * N`. (Because of the smoothing, EM is not
        guaranteed to be monotonic, so a decrease does not stop it.)
    verbose : int, default=0

    Returns
    -------
    ll : (B,) tensor[double]
        Weighted marginal log-likelihood 𝓛 (summed over voxels),
        evaluated at the returned 𝑯.
    N : (B,) tensor
        Total observation weight
    prior : (B, J, K) tensor
        Regularized normalized joint histogram 𝑯 = p(x,y) / (p(x) p(y))

    """
    # ------------------------------------------------------------------
    #       PREPARATION
    # ------------------------------------------------------------------
    dim = fixed.dim() - 2
    moving, fixed, weights = spatial_prepare(moving, fixed, weights)
    B, K, J, shape = spatial_shapes(moving, fixed, prior)
    N = shape.numel()
    Nm = spatial_sum(weights, dim).squeeze(-1)
    hard = not fixed.dtype.is_floating_point

    # Flatten
    moving = moving.reshape([*moving.shape[:2], -1])
    fixed = fixed.reshape([*fixed.shape[:2], -1])
    weights = weights.reshape([*weights.shape[:2], -1])

    if fwhm is None:
        fwhm = J / 64

    # ------------------------------------------------------------------
    # Initialize normalized histogram
    #   `prior[j, k]` contains the "normalized" joint histogram
    #       𝐻ⱼₖ = p(𝑥 = 𝑗, 𝑦 = 𝑘) / (p(𝑥 = 𝑗) p(𝑦 = 𝑘))
    #           = p(𝑥 = 𝑗 | 𝑦 = 𝑘) / p(𝑥 = 𝑗)
    #   In the E-step, it is used in place of the conditional p(x|y):
    #   the factor 1/p(𝑥 = 𝑗) does not depend on k and therefore
    #   disappears when the responsibilities are normalized. In the
    #   log-likelihood it adds the constant -∑ₙ ln p(𝑥 = 𝑥ₙ), which only
    #   depends on the (smoothed) histogram of the fixed image.
    # ------------------------------------------------------------------
    if prior is None:
        prior = moving.new_ones([B, J, K])
        prior /= prior.sum(dim=[-1, -2], keepdim=True) + tiny
        prior /= prior.sum(dim=-2, keepdim=True) * prior.sum(dim=-1, keepdim=True)
    else:
        prior = prior.clone()

    if hard:
        z = moving.new_zeros([B, K, N])
        hist = torch.empty_like(prior)
    else:
        # each pair (n, j) is an observation with weight q(𝑥ₙ = 𝑗)
        # (times the voxel weight)
        fixed = fixed * weights

    def loglikelihood(prior):
        # Hard: returns (𝓛, `z` = q(𝑦ₙ = 𝑘 | 𝑥ₙ))
        # Soft: returns (𝓛, q(𝑥ₙ = 𝑗) / ∑ₖ 𝐻ⱼₖ (𝜇 ∘ 𝜙)ₙₖ)
        if hard:
            sample_prior(prior, fixed, z)
            z.mul_(moving)
            norm = z.sum(-2, keepdim=True).add_(tiny)
            z.div_(norm)
            ll = norm.log_().mul_(weights).sum([-1, -2], dtype=torch.double)
            return ll, z
        else:
            norm = torch.matmul(prior, moving).add_(tiny)   # [B, J, N]
            ll = (fixed * norm.log()).sum([-1, -2], dtype=torch.double)
            return ll, fixed / norm

    # ------------------------------------------------------------------
    #       INFER PRIOR (EM)
    # ------------------------------------------------------------------
    ll_prev = None
    for n_iter in range(max_iter):
        # --------------------------------------------------------------
        # E-step
        # ------
        # Hard: estimate responsibilities of each TPM class for each
        # image voxel using Bayes' rule:
        #   q(𝑦ₙ = 𝑘 | 𝑥ₙ) ∝ p(𝑥ₙ | 𝑦ₙ = 𝑘) p(𝑦ₙ = 𝑘)
        #                  ∝ 𝐻{[𝑥ₙ],𝑘} (𝜇 ∘ 𝜙)ₙₖ
        # Soft: same thing, for each pair (n, j):
        #   q(𝑦ₙ = 𝑘 | 𝑥ₙ = 𝑗) ∝ 𝐻ⱼₖ (𝜇 ∘ 𝜙)ₙₖ
        #
        # The normalization term gives the marginal log-likelihood
        #   𝓛 = ∑ₙ ln ∑ₖ 𝐻{[𝑥ₙ],𝑘} (𝜇 ∘ 𝜙)ₙₖ
        # evaluated at the current 𝑯.
        # --------------------------------------------------------------
        ll, z = loglikelihood(prior)

        # --------------------------------------------------------------
        # M-step
        # ------
        # estimate the joint histogram by maximizing
        # 𝔼_q[ln p(𝒙, 𝒚; 𝑯)]:
        #   Hard: p(𝑥 = 𝑗, 𝑦 = 𝑘) ∝ ∑ₙ 𝛿(𝑥ₙ = 𝑗) q(𝑦ₙ = 𝑘 | 𝑥ₙ)
        #   Soft: p(𝑥 = 𝑗, 𝑦 = 𝑘) ∝ ∑ₙ q(𝑥ₙ = 𝑗) q(𝑦ₙ = 𝑘 | 𝑥ₙ = 𝑗)
        # --------------------------------------------------------------
        if hard:
            z *= weights
            scatter_prior(hist, fixed, z)  # hist[fixed] <- z
        else:
            hist = prior * torch.matmul(z, moving.transpose(-1, -2))
        prior.copy_(hist).add_(tiny)
        # make it a joint distribution
        prior /= prior.sum(dim=[-1, -2], keepdim=True) + tiny

        if fwhm:
            # smooth the joint histogram along the image-bin axis (j)
            prior = prior.transpose(-1, -2)
            prior = spatial.smooth(prior, dim=1, basis=0, fwhm=fwhm, bound='replicate')
            prior = prior.transpose(-1, -2)

        # normalize: 𝐻ⱼₖ = p(𝑥 = 𝑗, 𝑦 = 𝑘) / (p(𝑥 = 𝑗) p(𝑦 = 𝑘))
        prior /= prior.sum(dim=-2, keepdim=True) * prior.sum(dim=-1, keepdim=True) + tiny

        # --------------------------------------------------------------
        # Stopping criterion
        # ------------------
        # Because of the smoothing, EM is not monotonic, so we only stop
        # when the (absolute) change is small.
        # --------------------------------------------------------------
        if verbose > 0:
            success = ll_prev is None or ll.sum() > ll_prev.sum()
            happy = ':D' if success else ':('
            end = '\n' if verbose > 1 else '\r'
            print(f'(em)     | {n_iter:02d} | {(ll/Nm).mean():12.6g} | {happy}', end=end)
        if not torch.isfinite(ll).all():
            break
        if ll_prev is not None:
            if ((ll - ll_prev).abs() <= tolerance * Nm).all():
                break
        ll_prev = ll

    # ------------------------------------------------------------------
    # Objective: marginal log-likelihood 𝓛 evaluated at the final 𝑯.
    # `derivatives_intensity` returns the gradient of -𝓛 with respect
    # to 𝜇 ∘ 𝜙 (with 𝑯 fixed).
    #
    # NOTE: An earlier version returned N ∑ⱼₖ p(𝑥 = 𝑗, 𝑦 = 𝑘) ln 𝐻ⱼₖ
    # (the mutual information of the expected joint histogram), which
    # is about 𝓛 + ∑ₙ KL(q(𝑦ₙ | 𝑥ₙ) ‖ (𝜇 ∘ 𝜙)ₙ) and is therefore not the
    # function whose gradient is used.
    # ------------------------------------------------------------------
    ll, _ = loglikelihood(prior)

    return ll, Nm, prior


def derivatives_intensity(moving, fixed, prior, weights=None):
    """Derivatives of the negative log-likelihood with respect to 𝜇 ∘ 𝜙

    Computes the gradient and Hessian of -𝓛 with respect to the
    deformed TPM (𝜇 ∘ 𝜙)ₙₖ, with 𝑯 fixed, where
        𝓛 = ∑ₙ ln ∑ₖ 𝐻{[𝑥ₙ],𝑘} (𝜇 ∘ 𝜙)ₙₖ                    (hard labels)
        𝓛 = ∑ₙⱼ q(𝑥ₙ = 𝑗) ln ∑ₖ 𝐻ⱼₖ (𝜇 ∘ 𝜙)ₙₖ               (soft labels)
    The Hessian with respect to 𝜇 ∘ 𝜙 is exact (and positive
    semi-definite).

    Parameters
    ----------
    moving : (1|B, K, *spatial) tensor
        Deformed TPM 𝜇 ∘ 𝜙 (probabilities).
    fixed : (B, 1|J, *spatial) tensor
        Discretized image 𝒙 (integer) or soft image q (float).
    prior : (B, J, K) tensor
        Normalized joint histogram 𝑯.
    weights : (B, 1, *spatial) tensor, optional

    Returns
    -------
    grad : (B, K, *spatial) tensor
        Gradient of -𝓛
    hess : (B, K*(K+1)//2, *spatial) tensor
        Hessian of -𝓛, in packed symmetric form: the K diagonal
        elements (k, k) first, then the off-diagonal elements (k, kk),
        k < kk, in row-major order.

    """
    # ------------------------------------------------------------------
    #       PREPARATION
    # ------------------------------------------------------------------
    moving, fixed, weights = spatial_prepare(moving, fixed, weights)
    B, K, J, spatial = spatial_shapes(moving, fixed, prior)
    N = spatial.numel()

    # Flatten
    moving = moving.reshape([*moving.shape[:2], -1])
    fixed = fixed.reshape([*fixed.shape[:2], -1])
    weights = weights.reshape([*weights.shape[:2], -1])

    # ------------------------------------------------------------------
    # We compute the derivatives of
    #   -𝓛 = -∑ₙ ln ∑ₖ 𝐻{[𝑥ₙ],𝑘} (𝜇 ∘ 𝜙)ₙₖ
    # Writing (𝑯(𝜇 ∘ 𝜙))ₙⱼ = ∑ₖ 𝐻ⱼₖ (𝜇 ∘ 𝜙)ₙₖ, the gradient `g` and
    # Hessian `h` with respect to (𝜇 ∘ 𝜙)ₙₖ, (𝜇 ∘ 𝜙)ₙ,ₖₖ are:
    #   g[n, k]     = -𝐻{[𝑥ₙ],𝑘} / (𝑯(𝜇 ∘ 𝜙))ₙ,[𝑥ₙ]
    #   h[n, k, kk] =  𝐻{[𝑥ₙ],𝑘} 𝐻{[𝑥ₙ],𝑘𝑘} / (𝑯(𝜇 ∘ 𝜙))ₙ,[𝑥ₙ]²
    # In the soft case:
    #   g[n, k]     = -∑ⱼ q(𝑥ₙ = 𝑗) 𝐻ⱼₖ / (𝑯(𝜇 ∘ 𝜙))ₙⱼ
    #   h[n, k, kk] =  ∑ⱼ q(𝑥ₙ = 𝑗) 𝐻ⱼₖ 𝐻ⱼ,ₖₖ / (𝑯(𝜇 ∘ 𝜙))ₙⱼ²
    # ------------------------------------------------------------------
    K2 = K * (K + 1) // 2
    g = moving.new_zeros([B, K, N])
    h = moving.new_zeros([B, K2, N])

    # ------------------------------------------------------------------
    #       VERSION 1: DISCRETE LABELS
    # ------------------------------------------------------------------
    if not fixed.dtype.is_floating_point:
        sample_prior(prior, fixed, g)
        norm = linalg.dot(t(g), t(moving)).unsqueeze(1)
        norm = norm.add_(tiny).reciprocal_()
        g *= norm

        torch.mul(g[:, :K], g[:, :K], out=h[:, :K])
        c = K
        for k in range(K):
            for kk in range(k + 1, K):
                torch.mul(g[:, k], g[:, kk], out=h[:, c])
                c += 1

    # ------------------------------------------------------------------
    #       VERSION 2: SOFT LABELS
    # ------------------------------------------------------------------
    else:
        norm = torch.matmul(prior, moving).add_(tiny)       # [B, J, N]
        resp = fixed / norm                                 # q / m
        g = torch.matmul(t(prior), resp)                    # [B, K, N]
        resp = resp.div_(norm)                              # q / m²
        h[:, :K] = torch.matmul(t(prior.square()), resp)
        c = K
        for k in range(K):
            for kk in range(k + 1, K):
                pp = prior[:, :, k] * prior[:, :, kk]       # [B, J]
                h[:, c] = torch.matmul(pp.unsqueeze(-2), resp).squeeze(-2)
                c += 1

    g *= weights
    g.neg_()  # gradient of -𝓛
    g = g.reshape([B, K, *spatial])
    h *= weights
    h = h.reshape([B, K2, *spatial])

    return g, h


# ======================================================================
#
#                           UTILITIES
#
# ======================================================================


def plot_registration(dat, mov, title=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dat = dat[0, 0]
    mov = mov[0, 0]
    ndim = dat.dim()
    nrow = 3 if ndim == 3 else 1
    if ndim == 2:
        dat = dat.unsqueeze(-1)
        mov = mov.unsqueeze(-1)

    f = plt.figure()
    f.clf()
    for d in range(ndim):
        slicer = [slice(None)] * ndim
        slicer[d] = dat.shape[d]//2
        slicer = tuple(slicer)
        dat1 = dat[slicer].float()
        mov1 = mov[slicer].float()

        _mri = niplt.intensity_to_rgb(dat1)
        _tpm = niplt.intensity_to_rgb(mov1, colormap='inferno')

        plt.subplot(nrow, 3, d*3 + 1)
        plt.imshow(_mri)
        plt.axis('off')
        plt.subplot(nrow, 3, d*3 + 2)
        plt.imshow(_tpm)
        plt.axis('off')
        plt.subplot(nrow, 3, d*3 + 3)
        _tpm = niplt.set_alpha(_tpm, mov1)
        _tpm = niplt.stack_layers([_mri, _tpm])
        plt.imshow(_tpm)
        plt.axis('off')
    if title:
        plt.suptitle(title)
    f.canvas.flush_events()
    plt.show(block=True)


def sample_prior(prior, fixed, out=None):
    """
    prior : (*B, J, K)
    fixed : (*B, 1|J, N)
    out : (*B, K, N)
    """
    fn = (sample_prior_soft if fixed.dtype.is_floating_point else
          sample_prior_hard)
    return fn(prior, fixed, out)


def sample_prior_hard(prior, fixed, out=None):
    """
    prior : (*B, J, K)
    fixed : (*B, 1, N)
    out : (*B, K, N)
    """
    *B, J, K = prior.shape
    *_, _, N = fixed.shape
    prior = prior.unsqueeze(-1).expand([*B, J, K, N])
    fixed = fixed.unsqueeze(-2).expand([*B, 1, K, N])
    if out is not None:
        out = out.unsqueeze(-3)
    out = torch.gather(prior, -3, fixed, out=out).squeeze(-3)
    return out


def sample_prior_soft(prior, fixed, out=None):
    """
    prior : (*B, J, K)
    fixed : (*B, J, N)
    out : (*B, K, N)
    """
    prior = prior.transpose(-1, -2).unsqueeze(-2)  # [*B, K, 1, J]
    fixed = fixed.transpose(-1, -2).unsqueeze(-3)  # [*B, 1, N, J]
    out = linalg.dot(prior, fixed, out=out)
    return out


def scatter_prior(prior, fixed, z):
    """
    prior : (*B, J, K)
    fixed : (*B, 1|J, N)
    z : (*B, K, N)
    """
    fn = (scatter_prior_soft if fixed.dtype.is_floating_point else
          scatter_prior_hard)
    return fn(prior, fixed, z)


def scatter_prior_hard(prior, fixed, z):
    """
    prior : (*B, J, K)
    fixed : (*B, 1, N)
    z : (*B, K, N)
    """
    *B, J, K = prior.shape
    *_, _, N = fixed.shape
    fixed = fixed.expand([*B, K, N])
    prior.zero_()
    prior.transpose(-1, -2).scatter_add_(-1, fixed, z)
    return prior


def scatter_prior_soft(prior, fixed, z):
    """
    prior : (*B, J, K)
    fixed : (*B, J, N)
    z : (*B, K, N)
    """
    z = z.unsqueeze(-3)             # [*B, 1, K, N]
    fixed = fixed.unsqueeze(-2)     # [*B, J, 1, N]
    prior = linalg.dot(z, fixed, out=prior)
    return prior


def convert_basis(name):
    name = name.upper()
    if name[0] == 'T':
        name = 'T'
    elif name.startswith('ROT') or name == 'O':
        name = 'SO'
    elif name[0] == 'R':
        name = 'SE'
    elif name.startswith('SIM'):
        name = 'CSO'
    elif name[0] == 'A':
        name = 'Aff+'
    return name


def make_basis(name, dim, **backend):
    name = convert_basis(name)
    return spatial.affine_basis(name, dim, **backend)


def spatial_prepare(moving, fixed, weights):
    if not fixed.dtype.is_floating_point:
        # torch requires indexing tensors to be long :(
        fixed = fixed.long()

    # mask/weight missing values
    if weights is None:
        weights = 1
    weights = weights * (moving != 0).any(1, keepdim=True)
    if fixed.dtype.is_floating_point:
        weights *= (fixed != 0).any(1, keepdim=True)
        weights *= torch.isfinite(fixed).all(1, keepdim=True)
        fixed = make_finite_(fixed)

    return moving, fixed, weights


def spatial_shapes(moving, fixed, prior):
    dim = moving.dim() - 2
    if prior is not None:
        J = prior.shape[-2]
        K = prior.shape[-1]
    else:
        K = moving.shape[-dim-1]
        if fixed.dtype.is_floating_point:
            J = fixed.shape[-dim-1]
        else:
            J = fixed.max() + 1
    B = len(fixed)
    N = moving.shape[-dim:]
    return B, K, J, N


def spatial_sum(x, dim, keepdim=False):
    dims = list(range(-dim, 0))
    return x.sum(dims, keepdim=keepdim)


def make_finite_(x):
    x = x.masked_fill_(torch.isfinite(x).bitwise_not_(), 0)
    return x


def discretize(dat, nbins=256, mask=None):
    """Discretize an image into a number of bins"""
    dim = dat.dim() - 2
    dims = range(-dim, 0)
    mn, mx = utils.quantile(dat, (0.0005, 0.9995), dim=dims,
                            keepdim=True, mask=mask).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn).clamp_(0, 1).mul_(nbins-1)
    dat = make_finite_(dat)
    dat = dat.long()
    return dat


# ======================================================================
#                               DATA
# ======================================================================


def get_spm_prior(**backend):
    """Download the SPM prior"""
    # url = 'https://github.com/spm/spm12/raw/master/tpm/TPM.nii'
    url = 'https://github.com/spm/spm12/raw/refs/heads/main/tpm/TPM.nii'
    fname = os.path.join(cache_dir, 'SPM12_TPM.nii')
    if not os.path.exists(fname):
        os.makedirs(cache_dir, exist_ok=True)
        fname = download(url, fname)
    f = io.map(fname).movedim(-1, 0) #[:-1]  # drop background
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
    """Load TPM data

    Parameters
    ----------
    prior : [list of] str or tensor
    affine_prior : (4, 4) tensor, optional

    Returns
    -------
    prior : (K, *spatial) tensor
    affine_prior : (4, 4) tensor
    """
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
    """Load image data

    Parameters
    ----------
    x : [list of] str or tensor
    w : [list of] str or tensor
    affine : (4, 4) tensor, optional
    dim : int

    Returns
    -------
    x : (B, *spatial) tensor
    w : (B, *spatial) tensor
    affine : (4, 4) tensor
    """
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
            x = f.fdata(**backend, rand=True, missing=0)
        else:
            f = io.stack([io.map(x1) for x1 in x])
            if affine is None:
                affine = f.affine[0]
            x = f.fdata(**backend, rand=True, missing=0)

    if x.dim() > dim + 1:
        # drop a trailing singleton dimension
        x = x.squeeze(-1)
    if x.dim() > dim + 1:
        raise ValueError('Too many dimensions')
    if x.dim() == dim:
        x = x[None]

    if not torch.is_tensor(w) and w is not None:
        w = io.loadf(w, **backend)
        if w.dim() > dim + 1:
            w = w.squeeze(-1)
        if w.dim() > dim + 1:
            raise ValueError('Too many dimensions')

    x = x.contiguous()
    if w is not None:
        w = w.contiguous()
    return x, w, affine

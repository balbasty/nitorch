r"""
This loss has the same behaviour as `spm_maff8`

General idea
------------
. Let 𝒙 ∈ ⟦1,K⟧ᴺ be an *observed* label map.
  Typically, an image is discretized into K bins, and 𝑥ₙ indexes
  the bin in which voxel n falls. But it could also be any segmentation.
. Let 𝒚 ∈ ⟦1,J⟧ᴺ be a *hidden* label map (with a potentially different
  labelling scheme: J != K), such that p(𝑥ₙ = 𝑖 | 𝑦ₙ = 𝑗) = 𝐻ᵢⱼ
. Let 𝜇 ∈ ℝᴺᴶ be a (deformable) prior probability to see
  label j in voxel n, and 𝜙 ∈ ℝᴺᴰ be the corresponding
  deformation field, such that: p(𝑦ₙ = 𝑗) = (𝜇 ∘ 𝜙)ₙⱼ
. We want to find the parameters {𝐻, 𝜙} that maximize the marginal
  likelihood:
      p(𝒙; 𝑯, 𝜇, 𝜙) = ∏ₙ ∑ⱼ p(𝑥ₙ | 𝑦ₙ = 𝑗; 𝑯) p(𝑦ₙ = 𝑗; 𝜇, 𝜙)
  where we use ";" to separate random variables from parameters.
. Note that 𝜇 is assumed constant in this problem: only 𝑯 and 𝜙 are
  optimized.
. In order to maximize 𝑯, we use an EM algorithm:
  We introduce an approximate parameter 𝑯 and compute the approximate
  posterior distribution:
      q(𝑦ₙ = 𝑗 | 𝑥ₙ = 𝑖) ∝ 𝐻ᵢⱼ (𝜇 ∘ 𝜙)ₙⱼ                   (E-step)
  We then compute the expected value of the joint log-likelihood with respect
  to q, and maximize it with respect to 𝑯:
      𝐻ᵢⱼ ∝ ∑ₙ 𝛿(𝑥ₙ = 𝑖) q(𝑦ₙ = 𝑗)                          (M-step)
. With 𝑯 fixed, in order to maximize 𝜙, we go back to the
  marginal log-likelihood:
      𝓛 = ln p(𝒙) = ∑ₙ ln ∑ⱼ 𝐻{[𝑥ₙ],𝑗} (𝜇 ∘ 𝜙)ₙⱼ
  which we differentiate with respect to 𝜙.
. If we do not have an observed hard segmentation, but a (fixed) separable
  distribution q over 𝒙, we can maximize the expected log-likelihood instead:
      𝓛 = 𝔼_q[ln p(𝒙)] = ∑ₙᵢ q(𝑥ₙ = 𝑖) ln ∑ⱼ 𝐻ᵢⱼ (𝜇 ∘ 𝜙)ₙⱼ
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
              flexi=False, verbose=1):
    """Align a Tissue Probability Map to an image

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
    spacing : float(s), default=(8, 3)
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
    max_iter_gn : int, default=100
        Maximumm number of Gauss-Newton iterations
    max_iter_em : int, default=32
        Maximum number of EM iterations
    max_line_search : int, default=6
        Maximum number of line search steps

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
    logtpm = spatial.spline_coeff_nd(logtpm, dim=3, inplace=True, **splineopt)

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
        mi0, _, prm0 = fit_affine_tpm(dat, tpm, affine_dat0, affine_tpm, weights, **opt1)
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
            mi, _, prm = fit_affine_tpm(dat, tpm, affine_dat, affine_tpm, weights, **opt1)
            mi = mi.sum().item()
            print(f' ({mi:4.2f})', end='\r')
            # update if better
            if mi > mi0:
                prm0, reorient0, mi0, best_name = prm, reorient, mi, name1
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
                   progressive=False, verbose=1):
    """

    Parameters
    ----------
    dat : ([B], 1|J, *spatial) tensor
    tpm : (K, *spatial) tensor
    affine : (4, 4) tensor
    affine_tpm : (4, 4) tensor
    weights : (*spatial) tensor
    basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
    fwhm : float, default=J/64
    max_iter_gn : int, default=100
    max_iter_em : int, default=32
    max_line_search : int, default=12
    progressive : bool, default=False

    Returns
    -------
    mi : ([B]) tensor
    aff : (4, 4) tensor
    prm : (F) tensor

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
                max_line_search=max_line_search, progressive=True,
                verbose=verbose)
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

            # --- mutual info ------------------------------------------
            mi, Nm, prior = em_prior(mov, dat, prior0, **em_opt)
            mi = mi / Nm

            success = mi.sum() > mi0.sum()
            if verbose >= 2:
                end = '\n' if verbose >= 3 else '\r'
                happy = ':D' if success else ':('
                print(f'(search) | {n_ls:02d} | {mi.mean():12.6g} | {happy}', end=end)

            if success:
                armijo *= 2
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
            prior, prm, mi = prior0, prm0, mi0
            break

        space = ' ' * max(0, 6 - len(basis_name))
        if verbose >= 1:
            end = '\n' if verbose >= 2 else '\r'
            print(f'({basis_name[:6]}){space} | {n_iter:02d} | {mi.mean():12.6g}', end=end)

        if mi.mean() - mi0.mean() < 1e-4:
            # print('converged', mi.mean() - mi0.mean())
            break

        # --------------------------------------------------------------
        #       GAUSS-NEWTON
        # --------------------------------------------------------------

        # --- derivatives ----------------------------------------------
        g, h = derivatives_intensity(mov, dat, prior, **drv_opt)
        g = g.sum(0)
        h = h.sum(0)

        # --- spatial derivatives --------------------------------------
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
    g_mu : (K|K2, *spatial, 3) - spatial gradients of moving image
    g_aff : (F, 4, 4) - gradient of affine matrix wrt Lie parameters
    returns
    g : (D*(D+1),) tensor - gradient of loss wrt Lie parameters
    h : (D*(D+1),D*(D+1)) tensor - Hessian of loss wrt Lie parameters
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
    """Estimate H_jk = P[x == j, z == k] by Expectation Maximization

    The objective function is
        Π_n p(x ; H, mu) == Π_n Σ_k p(x | z == k; H) p(z == k; mu)

    Parameters
    ----------
    moving : (K, *spatial) tensor
    fixed : (B, 1|J, *spatial) tensor
    prior : (B, J, K) tensor, optional
    weights : (B, 1, *spatial) tensor, optional
    fwhm : float, optional
    max_iter : int, default=32

    Returns
    -------
    mi : (B,)
        Mutual information
    N : (B,)
        Total observation weight
    prior : (B, J, K)  tensor
        Regularized Joint histogram P[X,Z] / (P[X]*P[Z])

    """
    # ------------------------------------------------------------------
    #       PREPARATION
    # ------------------------------------------------------------------
    dim = fixed.dim() - 2
    moving, fixed, weights = spatial_prepare(moving, fixed, weights)
    B, K, J, shape = spatial_shapes(moving, fixed, prior)
    N = shape.numel()
    Nm = spatial_sum(weights, dim).squeeze(-1)

    # Flatten
    moving = moving.reshape([*moving.shape[:2], -1])
    fixed = fixed.reshape([*fixed.shape[:2], -1])
    weights = weights.reshape([*weights.shape[:2], -1])

    if fwhm is None:
        fwhm = J / 64

    # ------------------------------------------------------------------
    # initialize normalized histogram
    #   `prior` contains the "normalized" joint histogram p[x,y] / (p[x] p[y])
    #    However, it is used as the conditional histogram p[x|y] during
    #    the E-step. This is because the normalized histogram is equal to
    #    the conditional histogram up to a constant term that does not
    #    depend on x, and therefore disappears after normalization of the
    #    responsibilities.
    # ------------------------------------------------------------------
    if prior is None:
        prior = moving.new_ones([B, J, K])
        prior /= prior.sum(dim=[-1, -2], keepdim=True) + tiny
        prior /= prior.sum(dim=-2, keepdim=True) * prior.sum(dim=-1, keepdim=True)
    else:
        prior = prior.clone()
    # prior /= prior.sum(dim=-2, keepdim=True)  # conditional X | Z
    # prior /= prior.sum(dim=-2, keepdim=True) * prior.sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    #       INFER PRIOR (EM)
    # ------------------------------------------------------------------
    ll_prev = -float('inf')
    z = moving.new_zeros([B, K, N])
    prior0 = torch.empty_like(prior)
    for n_iter in range(max_iter):
        # --------------------------------------------------------------
        # E-step
        # ------
        # estimate responsibilities of each moving cluster for each
        # fixed voxel using Bayes' rule:
        #   p(z[n] == k | x[n] == j[n]) ∝ p(x[n] == j[n] | z[n] == k) p(z[n] == k)
        #
        # . j[n] is the discretized fixed image
        # . p(z[n] == k) is the moving template
        # . p(x[n] == j[n] | z[n] == k) is the conditional prior evaluated at (j[n], k)
        # --------------------------------------------------------------
        sample_prior(prior, fixed, z)
        z *= moving

        # --------------------------------------------------------------
        # compute log-likelihood (log_sum of the posterior)
        # ll = Σ_n log p(x[n] == j[n])
        #    = Σ_n log Σ_k p(x[n] == j[n] | z[n] == k)  p(z[n] == k)
        #    = Σ_n log Σ_k p(z[n] == k | x[n] == j) + constant{\z}
        # --------------------------------------------------------------
        ll = z.sum(-2, keepdim=True) + tiny
        z /= ll
        ll = ll.log_().mul_(weights).sum([-1, -2], dtype=torch.double)
        z *= weights

        # --------------------------------------------------------------
        # M-step
        # ------
        # estimate joint prior by maximizing Q = E_{Z;H,mu}[ln p(X, Z; H)]
        # => H_jk = p(x == j, z == k) ∝ Σ_n p(z[n] == k | x[n] == j) delta(x[n] == j)
        # --------------------------------------------------------------
        scatter_prior(prior0, fixed, z)  # prior[fixed] <- z
        prior.copy_(prior0).add_(tiny)
        # make it a joint distribution
        prior /= prior.sum(dim=[-1, -2], keepdim=True) + tiny

        if fwhm:
            # smooth "prior" for the prior
            prior = prior.transpose(-1, -2)
            prior = spatial.smooth(prior, dim=1, basis=0, fwhm=fwhm, bound='replicate')
            prior = prior.transpose(-1, -2)

        # prior /= prior.sum(dim=-2, keepdim=True)
        prior /= prior.sum(dim=-2, keepdim=True) * prior.sum(dim=-1, keepdim=True) + tiny

        if verbose > 0:
            success = ll.sum() > ll_prev
            happy = ':D' if success else ':('
            end = '\n' if verbose > 1 else '\r'
            print(f'(em)     | {n_iter:02d} | {(ll/Nm).mean():12.6g} | {happy}', end=end)
        if ll.sum() - ll_prev < tolerance * Nm.sum():
            break
        ll_prev = ll.sum()
    # if verbose == 1:
    #     print('')

    # NOTE:
    # We could return: `joint_prior / (prior_x * prior_z)` instead of
    # `conditional_prior = joint_prior / prior_z` as we currently do, which
    # would correspond to optimizing the mutual information instead of
    # the conditional likelihood. But the additional term only depends on
    # the fixed image, so does not have an impact for registration.
    #
    # Both have the same computational cost, and MI might have a slightly
    # nicer range so we could do that eventually.

    mi = (prior0 * prior.log()).sum()

    return mi, Nm, prior


def derivatives_intensity(moving, fixed, prior, weights=None):
    """

    Parameters
    ----------
    moving : (B, K, N) tensor
    fixed : (B, J|1, N) tensor
    prior : (B, J, K) tensor
    weights : (B, 1, N) tensor, optional

    Returns
    -------
    grad : (B, K, *spatial) tensor, if `grad`
    hess : (B, K, *spatial) tensor, if `hess`

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

    # compute gradients
    # Keeping only terms that depend on y, the mutual information is H[y]-H[x,y]
    # The objective function is \sum_n E[y_n]
    # > ll = \sum_n log p(x[n] == j[n] ; H, mu)
    #      = \sum_n log \sum_k p(x[n] == j[n] | z[n] == k; H) p(z[n] == k; mu)
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
        for j in range(J):
            norm = 0
            tmp = torch.zeros_like(g)
            for k in range(K):
                prior1 = prior[:, j, k, None]
                norm += prior1 * moving[:, k, :]
                tmp[:, k, :] = prior1
            tmp /= norm.add_(tiny)
            g += tmp * fixed[:, j, None, :]

            h[:, :K, :] += tmp.square() * fixed[:, j, None, :]
            h[:, :K, :] -= tmp * fixed[:, j, None, :]
            c = K
            for k in range(K):
                for kk in range(k + 1, K):
                    h[:, c, :] += tmp[:, k, :] * tmp[:, kk, :] * fixed[:, j, :]
                    c += 1

    g *= weights
    g.neg_()
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

    x = x.contiguous()
    if w is not None:
        w = w.contiguous()
    return x, w, affine

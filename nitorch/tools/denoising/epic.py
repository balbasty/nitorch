from nitorch.core import py, utils
from nitorch.spatial import (solve_field, regulariser, membrane_weights,
                             add_identity_grid, diff1d, grid_pull, grid_push,
                             grid_grad)
from nitorch.tools.img_statistics import estimate_noise
from nitorch import io
import torch


def epic(echoes, reverse_echoes=None, fieldmap=None, extrapolate=True,
         bandwidth=1, polarity='+', readout=-1, slicewise=True, lam=1,
         max_iter=(10, 32), tol=1e-5, verbose=False, device=None):
    """Edge-Preserving B0 inhomogeneity correction (EPIC)

    References
    ----------
    .. "A new distortion correction approach for multi-contrast MRI",
        Divya Varadarajan, et al. ISMRM (2020)

    Parameters
    ----------
    echoes : list[file_like] or (N, *spatial) tensor
        Echoes acquired with bipolar readout, Readout direction should be last.
    reverse_echoes : list[file_like] or (N, *spatial) tensor
        Echoes acquired with reverse bipolar readout. Else: synthesized.
    fieldmap : file_like or (*spatial) tensor
        Fieldmap (if bandwidth != 1) or voxel shift map
        Voxel shift map used to deform towards even (0, 2, ...) echoes.
        Its inverse is used to deform towards odd (1, 3, ...) echoes.
    extrapolate : bool
        Extrapolate first/last echo when reverse_echoes is True.
        Otherwise, only use interpolated echoes.
    bandwidth : float, Bandwidth of the fieldmap, in Hz/pixel
    polarity : '+' or '-', Readout polarity of the first echo
    readout : int, Index of the readout dimension
    slicewise : bool, Run the algorithm slicewise
    lam : [list of] float, Regularization factor (per echo)
    max_iter : [pair of] int, Maximum number of RLS and CG iterations
    tol : float, Tolerance for early stopping
    verbose : int, Verbosity level
    device : {'cpu', 'cuda'}

    Returns
    -------
    echoes : (N, *spatial) tensor
        Undistorted + denoised echoes

    """
    device = torch.device('cuda' if device == 'gpu' else device)
    loadopt = dict(rand=True, missing=0, device=device)

    echoes = map_files(echoes)
    reverse_echoes = map_files(reverse_echoes)
    fieldmap = map_files(fieldmap)[0]
    ndim = len(echoes.shape) - 1

    # estimate noise variance + scale regularization
    noise, tissue = 1, []
    for echo in echoes:
        noise1, tissue1 = estimate_noise(echo.fdata(**loadopt))
        noise *= noise1['sd']
        tissue.append(tissue1['mean'])
    noise = noise ** (1 / len(echoes))
    lam = py.make_list(lam, len(echoes))
    lam = [l / mu for l, mu in zip(lam, tissue)]

    # ensure readout dimension is last
    if readout > 0:
        readout = readout - ndim if readout > 0 else readout
        echoes = echoes.movedim(readout, -1)
        fieldmap = fieldmap.movedim(readout, -1)
        if reverse_echoes is not None:
            reverse_echoes = reverse_echoes.movedim(readout, -1)

    if slicewise:
        dz = -2 if echoes.shape[-2] < echoes.shape[-3] else -3
        echoes = echoes.movedim(dz, -2)
        fieldmap = fieldmap.movedim(dz, -2)
        if reverse_echoes is not None:
            reverse_echoes = reverse_echoes.movedim(dz, -2)
        nz = echoes.shape[-2]

        fit = torch.zeros(echoes.shape, dtype=torch.float32)
        for z in nz:
            if verbose > 0:
                print(f'processing slice: {z:03d}/{nz:03d}')

            # load data
            echoes1 = echoes[..., z, :].fdata(**loadopt)
            fieldmap1 = fieldmap[..., z, :].fdata(**loadopt)
            if reverse_echoes is not None:
                reverse_echoes1 = reverse_echoes[..., z, :].fdata(**loadopt)
            else:
                reverse_echoes1 = None

            # rescale fieldmap
            fieldmap1 = fieldmap1 / bandwidth
            if polarity == '-':
                fieldmap1 = -fieldmap1

            fit1 = run_epic(echoes1, reverse_echoes1, fieldmap1,
                            extrapolate=extrapolate, lam=lam, sigma=noise,
                            max_iter=max_iter, tol=tol, verbose=verbose)
            fit[..., z, :] = fit1.cpu()

    else:
        # load data
        echoes = echoes.fdata(**loadopt)
        fieldmap = fieldmap.fdata(**loadopt)
        if reverse_echoes is not None:
            reverse_echoes = reverse_echoes.fdata(**loadopt)

        # rescale fieldmap
        fieldmap = fieldmap / bandwidth
        if polarity == '-':
            fieldmap = -fieldmap

        fit = run_epic(echoes, reverse_echoes, fieldmap,
                       extrapolate=extrapolate, lam=lam, sigma=noise,
                       max_iter=max_iter, tol=tol, verbose=verbose)

    return fit


def map_files(files):
    # TODO
    return files


def run_epic(echoes, reverse_echoes=None, voxshift=None, extrapolate=True,
             lam=1, sigma=1, max_iter=(10, 32), tol=1e-5, verbose=False):
    """Run EPIC on pre-loaded tensors.

    Parameters
    ----------
    echoes : (N, *spatial) tensor
        Echoes acquired with bipolar readout, Readout direction should be last.
    reverse_echoes : (N, *spatial)
        Echoes acquired with reverse bipolar readout. Else: synthesized.
    voxshift : (*spatial) tensor
        Voxel shift map used to deform towards even (0, 2, ...) echoes.
        Its inverse is used to deform towards odd (1, 3, ...) echoes.
    extrapolate : bool
        Extrapolate first/last echo when reverse_echoes is True.
        Otherwise, only use interpolated echoes.
    lam : [list of] float
        Regularization factor (per echo)
    sigma : float
        Noise standard deviation
    max_iter : [pair of] int
        Maximum number of RLS and CG iterations
    tol : float
        Tolerance for early stopping
    verbose : int,
        Verbosity level

    Returns
    -------
    echoes : (N, *spatial) tensor
        Undistorted + denoised echoes

    """
    ne = len(echoes)                    # number of echoes
    nv = echoes.shape[1:].numel()       # number of voxels
    nd = echoes.dim() - 1               # number of dimensions

    # synthesize echoes
    if reverse_echoes is None:
        neg = synthesize_neg(echoes[0::2])
        pos = synthesize_pos(echoes[1::2])
        reverse_echoes = torch.stack([x for y in zip(pos, neg) for x in y])
        del pos, neg
    else:
        extrapolate = True

    # prepare parameters
    max_iter, sub_iter = py.make_list(max_iter, 2)
    tol, sub_tol = py.make_list(tol, 2)
    lam = [l / ne for l in py.make_list(lam, ne)]
    isigma2 = 1 / (sigma * sigma)

    # initialize denoised echoes
    fit = (echoes + reverse_echoes).div_(2)
    if not extrapolate:
        fit[0] = echoes[0]
        fit[-1] = echoes[-1]
    fwd_fit = torch.empty_like(fit)
    bwd_fit = torch.empty_like(fit)

    # prepare voxel shift maps
    ivoxshift = add_identity_grid(-voxshift)
    voxshift = add_identity_grid(voxshift)

    # compute hessian once and for all
    if extrapolate:
        h = torch.ones_like(voxshift)[None]
        h = push1d(pull1d(h, voxshift), voxshift)
        h += push1d(pull1d(h, ivoxshift), ivoxshift)
        h *= 2
    else:
        h = torch.zeros_like(fit)
        h[:-1] += push1d(pull1d(h, voxshift), voxshift)
        h[1:] += push1d(pull1d(h, ivoxshift), ivoxshift)
    h *= isigma2

    for n_iter in range(max_iter):

        # update weights
        w, jtv = membrane_weights(fit, factor=lam, return_sum=True)

        # gradient of likelihood
        fwd_fit[0::2] = pull1d(fit[0::2], voxshift)
        fwd_fit[1::2] = pull1d(fit[1::2], ivoxshift)
        fwd_fit = (fwd_fit - echoes)
        ll = fwd_fit.flatten().dot(fwd_fit.flatten())
        fwd_fit[0::2] = push1d(fwd_fit[0::2], voxshift)
        fwd_fit[1::2] = push1d(fwd_fit[1::2], ivoxshift)

        if extrapolate:
            bwd_fit[0::2] = pull1d(fit[0::2], ivoxshift)
            bwd_fit[1::2] = pull1d(fit[1::2], voxshift)
            bwd_fit = (bwd_fit - echoes)
            ll += bwd_fit.flatten().dot(bwd_fit.flatten())
            bwd_fit[0::2] = push1d(bwd_fit[0::2], ivoxshift)
            bwd_fit[1::2] = push1d(bwd_fit[1::2], voxshift)
        else:
            bwd_fit[2::2] = pull1d(fit[2::2], ivoxshift)
            bwd_fit[1:-1:2] = pull1d(fit[1:-1:2], voxshift)
            bwd_fit = (bwd_fit - echoes[1:-1])
            ll += bwd_fit.flatten().dot(bwd_fit.flatten())
            bwd_fit[2::2] = push1d(bwd_fit[2::2], ivoxshift)
            bwd_fit[1:-1:2] = push1d(bwd_fit[1:-1:2], voxshift)

        g = fwd_fit.add_(bwd_fit)
        ll *= 0.5 * isigma2

        # gradient of prior
        g += regulariser(fit, membrane=1, factor=lam, weights=w)

        # solve
        fit -= solve_field(h, g, w, membrane=1, factor=lam,
                           max_iter=sub_iter, tolerance=sub_tol)

        # track objective
        ll, jtv = ll.item() / (ne*nv), jtv.item() / (ne*nv)
        loss = ll + jtv
        if n_iter:
            loss_prev = loss
            gain = (loss_prev - loss) / max((loss_max - loss), 1e-8)
        else:
            gain = float('inf')
            loss_max = max(loss_max, loss)
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:02d} | {ll:12.6g} + {jtv:12.6g} = {loss:12.6g} '
                  f'| gain = {gain:12.6g}', end=end)

    if verbose == 1:
        print('\n')


def interpolate_echo(x1, x2):
    """Interpolate the middle echo"""
    # the output is distributed as
    return (x1 * x2).sqrt_()


def extrapolate_next_echo(x1, x2):
    """Extrapolate the next echo"""
    m = (x1 <= 0)
    return x2.square().div_(x1).masked_fill_(m, 0)


def extrapolate_previous_echo(x1, x2):
    """Extrapolate the previous echo"""
    return extrapolate_next_echo(x2, x1)


def synthesize_neg(xpos):
    """Synthesize negative polarity echoes from positive polarity echoes"""
    xneg = []
    for i in range(len(xpos)-1):
        xneg.append(interpolate_echo(xpos[i], xpos[i+1]))
    xneg.append(extrapolate_next_echo(xpos[-2], xpos[-1]))
    return xneg


def synthesize_pos(xneg):
    """Synthesize positive polarity echoes from negative polarity echoes"""
    xpos = [extrapolate_previous_echo(xneg[0], xneg[1])]
    for i in range(len(xneg)-1):
        xpos.append(interpolate_echo(xneg[i], xneg[i+1]))
    return xpos


def pull1d(img, grid, grad=False, **kwargs):
    """Pull an image by a transform along the last dimension

    Parameters
    ----------
    img : (K, *spatial) tensor, Image
    grid : (*spatial) tensor, Sampling grid
    grad : bool, Sample gradients

    Returns
    -------
    warped_img : (K, *spatial) tensor
    warped_grad : (K, *spatial) tensor, if `grad`

    """
    if grid is None:
        if grad:
            bound = kwargs.get('bound', 'dft')
            return img, diff1d(img, dim=-1, bound=bound, side='c')
        else:
            return img, None
    kwargs.setdefault('extrapolate', True)
    kwargs.setdefault('bound', 'dft')
    img, grid = img.unsqueeze(-2), grid.unsqueeze(-1)
    warped = grid_pull(img, grid, **kwargs).squeeze(-2)
    if not grad:
        return warped, None
    grad = grid_grad(img, grid, **kwargs)
    grad = grad.squeeze(-1).squeeze(-2)
    return warped, grad


def push1d(img, grid, **kwargs):
    """Push an image by a transform along the last dimension

    This is the adjoint of `pull1d`.

    Parameters
    ----------
    img : (K, *spatial) tensor, Image
    grid : (*spatial) tensor, Sampling grid

    Returns
    -------
    pushed_img : (K, *spatial) tensor

    """
    if grid is None:
        return img
    kwargs.setdefault('extrapolate', True)
    kwargs.setdefault('bound', 'dft')
    img, grid = img.unsqueeze(-2), grid.unsqueeze(-1)
    pushed = grid_push(img, grid, **kwargs).squeeze(-2)
    return pushed



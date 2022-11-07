from nitorch.core import py, utils
from nitorch.spatial import (solve_field, regulariser, membrane_weights,
                             add_identity_grid, diff1d, grid_pull, grid_push,
                             grid_grad)
from nitorch.tools.img_statistics import estimate_noise
from nitorch import io
import torch
import math
import multiprocessing


def epic(echoes, reverse_echoes=True, fieldmap=None, extrapolate=False,
         bandwidth=1, polarity=1, readout=-1, slicewise=False, lam=1e2,
         max_iter=(10, 32), tol=1e-5, verbose=False, device=None):
    """Edge-Preserving B0 inhomogeneity correction (EPIC)

    References
    ----------
    .. "A new distortion correction approach for multi-contrast MRI",
        Divya Varadarajan, et al. ISMRM (2020)

    Parameters
    ----------
    echoes : list[file_like] or (N, *spatial) tensor,
        Echoes acquired with a bipolar readout.
    reverse_echoes : bool or list[file_like] or (N, *spatial) tensor
        Echoes acquired with reverse bipolar readout. If True: synthesized.
    fieldmap : file_like or (*spatial) tensor, Fieldmap or voxel shift map
    extrapolate : bool, Extrapolate first/last echo when reverse_echoes is None
    bandwidth : float, Bandwidth of the input echoes, in Hz/pixel
    polarity : +1 or -1, Readout polarity of the first echo
    readout : int, Index of the readout dimension
    slicewise : bool or int, Run the algorithm slicewise. If int, chunk size.
    lam : [list of] float, Regularization factor (per echo)
    max_iter : [pair of] int, Maximum number of RLS and CG iterations
    tol : float, Tolerance for early stopping
    verbose : int, Verbosity level
    device : {'cpu', 'cuda'} or torch.device

    Returns
    -------
    echoes : (N, *spatial) tensor
        Undistorted + denoised echoes

    """
    device = torch.device('cuda' if device == 'gpu' else device)
    backend = dict(dtype=torch.float32, device=device)

    echoes = map_files(echoes)
    reverse_echoes = map_files(reverse_echoes)
    fieldmap = map_files(fieldmap, nobatch=True)
    ndim = len(echoes.shape) - 1

    # estimate noise variance + scale regularization
    noise, tissue = 1, []
    for echo in echoes:
        noise1, tissue1 = estimate_noise(load(echo, **backend))
        noise *= noise1['sd']
        tissue.append(tissue1['mean'])
    noise = noise ** (1 / len(echoes))
    lam = py.make_list(lam, len(echoes))
    lam = [l / mu for l, mu in zip(lam, tissue)]

    # ensure readout dimension is last
    readout = readout - ndim if readout > 0 else readout
    echoes = movedim(echoes, readout, -1)
    fieldmap = movedim(fieldmap, readout, -1)
    reverse_echoes = movedim(reverse_echoes, readout, -1)

    if slicewise:
        # --- loop over slices -----------------------------------------

        # ensure slice direction is second to last
        dz = -2 if echoes.shape[-2] < echoes.shape[-3] else -3
        echoes = movedim(echoes, dz, -2)
        fieldmap = movedim(fieldmap, dz, -2)
        reverse_echoes = movedim(reverse_echoes, dz, -2)
        nz = echoes.shape[-2]

        # allocate output
        if torch.is_tensor(echoes):
            out_backend = dict(dtype=echoes.dtype, device=echoes.device)
        else:
            out_backend = dict(dtype=torch.float32, device='cpu')
        fit = torch.zeros(echoes.shape, **out_backend)

        # prepare runner
        slicewise = int(slicewise)
        run_slicewise = RunSlicewise(
            slicewise, echoes, reverse_echoes, fieldmap,
            bandwidth, polarity, lam, noise, extrapolate, max_iter, tol,
            backend, out_backend, verbose)

        # parallel process
        with multiprocessing.Pool(torch.get_num_threads()) as pool:
            slices = pool.imap_unordered(run_slicewise, range(0, nz, slicewise))
            for chunk in slices:
                chunk, fitchunk = chunk
                fit[..., chunk, :] = fitchunk

        # unpermute slice
        fit = movedim(fit, -2, dz)

    else:
        # --- process full volume --------------------------------------

        # load data
        echoes = load(echoes, **backend)
        fieldmap = load(fieldmap, **backend)
        reverse_echoes = load(reverse_echoes, **backend)

        # rescale fieldmap
        if fieldmap is not None:
            fieldmap = fieldmap / bandwidth
            if polarity < 0:
                fieldmap = -fieldmap

        # run EPIC
        fit = run_epic(echoes, reverse_echoes, fieldmap,
                       extrapolate=extrapolate, lam=lam, sigma=noise,
                       max_iter=max_iter, tol=tol, verbose=verbose)

    # unpermute readout
    fit = movedim(fit, -1, readout)

    return fit


class RunSlicewise:

    def __init__(self, chunksize, echoes, reverse_echoes, fieldmap,
                 bandwidth, polarity, lam, sigma, extrapolate, max_iter, tol,
                 backend, out_backend, verbose):
        self.chunksize = chunksize
        self.echoes = echoes
        self.reverse_echoes = reverse_echoes
        self.fieldmap = fieldmap
        self.bandwidth = bandwidth
        self.polarity = polarity
        self.lam = lam
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.extrapolate = extrapolate
        self.backend = backend
        self.out_backend = out_backend
        self.verbose = verbose

    def __call__(self, z):
        torch.set_num_threads(1)

        nz = self.echoes.shape[-2]
        chunk = z if self.chunksize == 1 else slice(z, z + self.chunksize)
        if self.verbose > 0:
            print(f'processing slice: {z:03d}/{nz:03d}')

        echoes1 = load(self.echoes[..., chunk, :], **self.backend)

        if self.fieldmap is not None:
            fieldmap1 = load(self.fieldmap[..., chunk, :], **self.backend)
            # rescale fieldmap
            fieldmap1 = fieldmap1 / self.bandwidth
            if self.polarity < 0:
                fieldmap1 = -fieldmap1
        else:
            fieldmap1 = self.fieldmap

        if hasattr(self.reverse_echoes, '__getitem__'):
            # Tensor or MappedArray
            reverse_echoes1 = load(self.reverse_echoes[..., chunk, :],
                                   **self.backend)
        else:
            # None or True or False
            reverse_echoes1 = self.reverse_echoes

        # run EPIC
        fit1 = run_epic(echoes1, reverse_echoes1, fieldmap1,
                        extrapolate=self.extrapolate, lam=self.lam,
                        sigma=self.sigma, max_iter=self.max_iter, tol=self.tol,
                        verbose=self.verbose)
        return chunk, fit1.to(**self.out_backend)


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
        Extrapolate first/last echo when reverse_echoes is None.
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
    if reverse_echoes is False:
        return run_epic_noreverse(echoes, voxshift,
                                  lam, sigma, max_iter, tol, verbose)

    ne = len(echoes)                    # number of echoes
    nv = echoes.shape[1:].numel()       # number of voxels
    nd = echoes.dim() - 1               # number of dimensions

    # synthesize echoes
    synth = not torch.is_tensor(reverse_echoes)
    if synth:
        neg = synthesize_neg(echoes[0::2])
        pos = synthesize_pos(echoes[1::2])
        reverse_echoes = torch.stack([x for y in zip(pos, neg) for x in y])
        del pos, neg
    else:
        extrapolate = True

    # initialize denoised echoes
    fit = (echoes + reverse_echoes).div_(2)
    if not extrapolate:
        fit[0] = echoes[0]
        fit[-1] = echoes[-1]
    fwd_fit = torch.zeros_like(fit)
    bwd_fit = torch.zeros_like(fit)

    # prepare voxel shift maps
    if voxshift is not None:
        ivoxshift = add_identity_1d(-voxshift)
        voxshift = add_identity_1d(voxshift)
    else:
        ivoxshift = None

    # prepare parameters
    max_iter, sub_iter = py.make_list(max_iter, 2)
    tol, sub_tol = py.make_list(tol, 2)
    lam = [l / ne for l in py.make_list(lam, ne)]
    isigma2 = 1 / (sigma * sigma)

    # compute hessian once and for all
    if voxshift is not None:
        one = torch.ones_like(voxshift)[None]
        if extrapolate:
            h = push1d(pull1d(one, voxshift), voxshift)
            h += push1d(pull1d(one, ivoxshift), ivoxshift)
            weight_ = lambda x: x.mul_(0.5)
            halfweight_ = lambda x: x.mul_(math.sqrt(0.5))
        else:
            h = torch.zeros_like(fit)
            h[:-1] += push1d(pull1d(one, voxshift), voxshift)
            h[1:] += push1d(pull1d(one, ivoxshift), ivoxshift)
            weight_ = lambda x: x[1:-1].mul_(0.5)
            halfweight_ = lambda x: x[1:-1].mul_(math.sqrt(0.5))
        del one
        weight_(h)
    else:
        h = fit.new_ones([ne] + [1] * nd)
        if extrapolate:
            h *= 2
            weight_ = lambda x: x.mul_(0.5)
            halfweight_ = lambda x: x.mul_(math.sqrt(0.5))
        else:
            h[1:-1] *= 2
            weight_ = lambda x: x[1:-1].mul_(0.5)
            halfweight_ = lambda x: x[1:-1].mul_(math.sqrt(0.5))
    weight_(h)
    h *= isigma2

    loss = float('inf')
    for n_iter in range(max_iter):

        # update weights
        w, jtv = membrane_weights(fit, factor=lam, return_sum=True)

        # gradient of likelihood (forward)
        pull_forward(fit, voxshift, ivoxshift, out=fwd_fit)
        fwd_fit.sub_(echoes)
        halfweight_(fwd_fit)
        ll = ssq(fwd_fit)
        halfweight_(fwd_fit)
        push_forward(fwd_fit, voxshift, ivoxshift, out=fwd_fit)

        # gradient of likelihood (reversed)
        pull_backward(fit, voxshift, ivoxshift, extrapolate, out=bwd_fit)
        if extrapolate:
            bwd_fit.sub_(reverse_echoes)
        else:
            bwd_fit[1:-1].sub_(reverse_echoes[1:-1])
        halfweight_(bwd_fit)
        ll += ssq(bwd_fit)
        halfweight_(bwd_fit)
        push_backward(bwd_fit, voxshift, ivoxshift, extrapolate, out=bwd_fit)

        g = fwd_fit.add_(bwd_fit).mul_(isigma2)
        ll *= 0.5 * isigma2

        # gradient of prior
        g += regulariser(fit, membrane=1, factor=lam, weights=w)

        # solve
        fit -= solve_field(h, g, w, membrane=1, factor=lam,
                           max_iter=sub_iter, tolerance=sub_tol)

        # track objective
        ll, jtv = ll.item() / (ne*nv), jtv.item() / (ne*nv)
        loss, loss_prev = ll + jtv, loss
        if n_iter:
            gain = (loss_prev - loss) / max((loss_max - loss), 1e-8)
        else:
            gain = float('inf')
            loss_max = loss
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:02d} | {ll:12.6g} + {jtv:12.6g} = {loss:12.6g} '
                  f'| gain = {gain:12.6g}', end=end)
        if gain < tol:
            break

    if verbose == 1:
        print('')

    return fit


def run_epic_noreverse(echoes, voxshift=None, lam=1, sigma=1,
                       max_iter=(10, 32), tol=1e-5, verbose=False):
    """Run EPIC on pre-loaded tensors (no reverse gradients or synthesis)

    Parameters
    ----------
    echoes : (N, *spatial) tensor
        Echoes acquired with bipolar readout, Readout direction should be last.
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

    # initialize denoised echoes
    fit = echoes.clone()
    fwd_fit = torch.empty_like(fit)

    # prepare voxel shift maps
    do_voxshift = voxshift is not None
    if do_voxshift:
        ivoxshift = add_identity_1d(-voxshift)
        voxshift = add_identity_1d(voxshift)
    else:
        ivoxshift = None

    # prepare parameters
    max_iter, sub_iter = py.make_list(max_iter, 2)
    tol, sub_tol = py.make_list(tol, 2)
    lam = [l / ne for l in py.make_list(lam, ne)]
    isigma2 = 1 / (sigma * sigma)

    # compute hessian once and for all
    if do_voxshift:
        one = torch.empty_like(voxshift)[None]
        h = torch.empty_like(echoes)
        h[0::2] = push1d(pull1d(one, voxshift), voxshift)
        h[1::2] = push1d(pull1d(one, ivoxshift), ivoxshift)
        del one
    else:
        h = fit.new_ones([1] * (nd + 1))
    h *= isigma2

    loss = float('inf')
    for n_iter in range(max_iter):

        # update weights
        w, jtv = membrane_weights(fit, factor=lam, return_sum=True)

        # gradient of likelihood
        pull_forward(fit, voxshift, ivoxshift, out=fwd_fit)
        fwd_fit.sub_(echoes)
        ll = ssq(fwd_fit)
        push_forward(fwd_fit, voxshift, ivoxshift, out=fwd_fit)

        g = fwd_fit.mul_(isigma2)
        ll *= 0.5 * isigma2

        # gradient of prior
        g += regulariser(fit, membrane=1, factor=lam, weights=w)

        # solve
        fit -= solve_field(h, g, w, membrane=1, factor=lam,
                           max_iter=sub_iter, tolerance=sub_tol)

        # track objective
        ll, jtv = ll.item() / (ne*nv), jtv.item() / (ne*nv)
        loss, loss_prev = ll + jtv, loss
        if n_iter:
            gain = (loss_prev - loss) / max((loss_max - loss), 1e-8)
        else:
            gain = float('inf')
            loss_max = loss
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:02d} | {ll:12.6g} + {jtv:12.6g} = {loss:12.6g} '
                  f'| gain = {gain:12.6g}', end=end)
        if gain < tol:
            break

    if verbose == 1:
        print('')

    return fit


def map_files(files, nobatch=False, keep_open=False):
    """Concat and map input files or tensors"""
    if isinstance(files, str):
        files = io.volumes.map(files, keep_open=keep_open)
        if files.dim == 5 and files.shape[3] == 1 and files.dim == 5:
            # channel along the 5th dimension (C), drop 4th (T) dimension
            files = files[:, :, :, 0, :]
        elif files.dim == 5 and files.shape[4] == 1:
            # channel along the 4th dimension (T), drop 5th (C) dimension
            files = files[:, :, :, :, 0]
        elif files.dim == 4:
            # channel along the 4th dimension (T)
            pass
        elif files.dim == 3:
            files = files[..., None]
        else:
            raise ValueError(f'Unsupported input shape {list(files.shape)}')
        files = files.movedim(-1, 0)
    elif isinstance(files, (list, tuple)):
        files = list(map(map_files, files))
        if isinstance(files[0], io.MappedArray):
            files = io.cat(files)
        else:
            files = torch.cat(files)
    if nobatch and (torch.is_tensor(files) or isinstance(files, io.MappedArray)):
        files = files[0]
    return files


def load(dat, rand=True, missing=0, **backend):
    """Load a tensor from a file (if input is a mapped file)"""
    if torch.is_tensor(dat):
        return dat.to(**backend)
    elif isinstance(dat, io.MappedArray):
        dat = dat.fdata(rand=rand, missing=missing, **backend)
        dat.masked_fill_(torch.isfinite(dat).logical_not_(), 0)
        return dat
    else:
        return dat


def movedim(x, *a, **k):
    if torch.is_tensor(x):
        return torch.movedim(x, *a, **k)
    elif isinstance(x, io.MappedArray):
        return x.movedim(*a, **k)
    else:
        return x


def ssq(x):
    """Sum of squares"""
    return x.flatten().dot(x.flatten())


def pull_forward(echoes, voxshift, ivoxshift, out=None):
    """Pull echoes "forward" (nonreversed gradient)"""
    if out is None:
        out = torch.empty_like(echoes)
    if voxshift is None:
        if not utils.same_storage(out, echoes):
            out.copy_(echoes)
        return out
    out[0::2] = pull1d(echoes[0::2], voxshift)
    out[1::2] = pull1d(echoes[1::2], ivoxshift)
    return out


def push_forward(echoes, voxshift, ivoxshift, out=None):
    """Push echoes "forward" (nonreversed gradient)"""
    if out is None:
        out = torch.empty_like(echoes)
    if voxshift is None:
        if not utils.same_storage(out, echoes):
            out.copy_(echoes)
        return out
    out[0::2] = push1d(echoes[0::2], voxshift)
    out[1::2] = push1d(echoes[1::2], ivoxshift)
    return out


def pull_backward(echoes, voxshift, ivoxshift, extrapolate, out=None):
    """Pull echoes "backward" (reversed gradient)"""
    if out is None:
        out = torch.zeros_like(echoes)
    if extrapolate:
        if voxshift is None:
            if not utils.same_storage(out, echoes):
                out.copy_(echoes)
            return out
        out[0::2] = pull1d(echoes[0::2], ivoxshift)
        out[1::2] = pull1d(echoes[1::2], voxshift)
    else:
        if voxshift is None:
            if not utils.same_storage(out, echoes):
                out[1:-1].copy_(echoes[1:-1])
            return out
        out[2::2] = pull1d(echoes[2::2], ivoxshift)
        out[1:-1:2] = pull1d(echoes[1:-1:2], voxshift)
    return out


def push_backward(echoes, voxshift, ivoxshift, extrapolate, out=None):
    """Push echoes "backward" (reversed gradient)"""
    if out is None:
        out = torch.zeros_like(echoes)
    if extrapolate:
        if voxshift is None:
            if not utils.same_storage(out, echoes):
                out.copy_(echoes)
            return out
        out[0::2] = push1d(echoes[0::2], ivoxshift)
        out[1::2] = push1d(echoes[1::2], voxshift)
    else:
        if voxshift is None:
            if not utils.same_storage(out, echoes):
                out[1:-1].copy_(echoes[1:-1])
            return out
        out[2::2] = push1d(echoes[2::2], ivoxshift)
        out[1:-1:2] = push1d(echoes[1:-1:2], voxshift)
    return out


def interpolate_echo(x1, x2):
    """Interpolate the middle echo"""
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
        return warped
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


def add_identity_1d(grid):
    """Add 1D identity"""
    return add_identity_grid(grid[..., None])[..., 0]

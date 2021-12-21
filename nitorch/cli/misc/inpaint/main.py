from nitorch import spatial, io
from nitorch.core import py, utils, optim
from nitorch.core.constants import nan
import os
import torch


def ensure_4d(f):
    while len(f.shape) > 4:
        for d in reversed(range(len(f.shape))):
            if f.shape[d] == 1:
                f.squeeze(d)
                continue
        raise ValueError('Too many channel dimensions')
    while len(f.shape) < 4:
        f = f.unsqueeze(-1)
    return f


def inpaint(*inputs, missing='nan', output=None, device=None, verbose=1,
            max_iter_rls=10, max_iter_cg=32, tol_rls=1e-5, tol_cg=1e-5):
    """Inpaint missing values by minimizing Joint Total Variation.

    Parameters
    ----------
    *inputs : str or tensor or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    missing : 'nan' or scalar or callable, default='nan'
        Mask of the missing data. If a scalar, all voxels with that value
        are considered missing. If a function, it should return the mask
        of missing values when applied to the multi-channel data. Else,
        non-finite values are assumed missing.
    output : [sequence of] str, optional
        Output filename(s).
        If the input is not a path, the unstacked data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.pool{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file,
        `i` is the coordinate (starting at 1) of the slice.
    verbose : int, default=1
    device : torch.device, optional
    max_iter_rls : int, default=10
    max_iter_cg : int, default=32
    tol_rls : float, default=1e-5
    tol_cg : float, default=1e-5

    Returns
    -------
    *output : str or (tensor, tensor)
        If the input is a path, the output path is returned.
        Else, the pooled data and orientation matrix are returned.

    """
    # Preprocess
    dirs = []
    bases = []
    exts = []
    fnames = []
    nchannels = []
    dat = []
    aff = None
    for i, inp in enumerate(inputs):
        is_file = isinstance(inp, str)
        if is_file:
            fname = inp
            dir, base, ext = py.fileparts(fname)
            fnames.append(inp)
            dirs.append(dir)
            bases.append(base)
            exts.append(ext)

            f = io.volumes.map(fname)
            if aff is None:
                aff = f.affine
            f = ensure_4d(f)
            dat.append(f.fdata(device=device))

        else:
            fnames.append(None)
            dirs.append('')
            bases.append(f'{i+1}')
            exts.append('.nii.gz')
            if isinstance(inp, (list, tuple)):
                if aff is None:
                    dat1, aff = inp
                else:
                    dat1, _ = inp
            else:
                dat1 = inp
            dat.append(torch.as_tensor(dat1, device=device))
            del dat1
        nchannels.append(dat[-1].shape[-1])
    dat = utils.to(*dat, dtype=torch.float, device=utils.max_device(dat))
    if not torch.is_tensor(dat):
        dat = torch.cat(dat, dim=-1)
    dat = utils.movedim(dat, -1, 0)  # (channels, *spatial)

    # Set missing data
    if missing != 'nan':
        if not callable(missing):
            missingval = utils.make_vector(missing, dtype=dat.dtype, device=dat.device)
            if missingval.numel() > 1:
                missing = lambda x: utils.isin(x, missingval)
            else:
                missing = lambda x: x == missingval[0]
        dat[missing(dat)] = nan
    dat[torch.isfinite(dat).bitwise_not_()] = nan

    # Do it
    if aff is not None:
        vx = spatial.voxel_size(aff)
    else:
        vx = 1
    dat = do_inpaint(dat, voxel_size=vx, verbose=verbose,
                     max_iter_rls=max_iter_rls, tol_rls=tol_rls,
                     max_iter_cg=max_iter_cg, tol_cg=tol_cg)

    # Postprocess
    dat = utils.movedim(dat, 0, -1)
    dat = dat.split(nchannels, dim=-1)
    output = py.make_list(output, len(dat))
    for i in range(len(dat)):
        if fnames[i] and not output[i]:
            output[i] = '{dir}{sep}{base}.inpaint{ext}'
        if output[i]:
            if fnames[i]:
                output[i] = output[i].format(dir=dirs[i] or '.', base=bases[i],
                                             ext=exts[i], sep=os.path.sep)
                io.volumes.save(dat[i], output[i], like=fnames[i], affine=aff)
            else:
                output[i] = output[i].format(sep=os.path.sep)
                io.volumes.save(dat[i], output[i], affine=aff)

    dat = [output[i] if fnames[i] else
           (dat[i], aff) if aff is not None
           else dat[i] for i in range(len(dat))]
    if len(dat) == 1:
        dat = dat[0]
    return dat


def do_inpaint(dat, voxel_size=1, max_iter_rls=50, max_iter_cg=32,
               tol_rls=1e-5, tol_cg=1e-5, verbose=1):
    """Perform inpainting

    Parameters
    ----------
    dat : (channels, *spatial) tensor
        Tensor with missing data marked as NaN

    Returns
    -------
    dat : (channels, *spatial) tensor

    """
    dat = torch.as_tensor(dat)
    backend = utils.backend(dat)
    voxel_size = utils.make_vector(voxel_size, dat.dim()-1, **backend)
    voxel_size = voxel_size / voxel_size.mean()

    # initialize
    mask = torch.isnan(dat)
    for channel in dat:
        channel[torch.isnan(channel)] = channel[~torch.isnan(channel)].median()
    weights = dat.new_ones([1, *dat.shape[1:]])

    ll_w = weights.sum(dtype=torch.double)
    ll_x = spatial.membrane(dat, dim=3, voxel_size=voxel_size, weights=weights)
    ll_x = (ll_x * dat).sum(dtype=torch.double)
    if verbose > 0:
        print(f'{0:3d} | {ll_x} + {ll_w} = {ll_x + ll_w}')

    # Reweighted least squares loop
    zeros = torch.zeros_like(dat)
    nmask = mask.sum()
    grad_buf = dat.new_zeros(nmask)
    precond_buf = dat.new_zeros(nmask)
    forward_buf = dat.new_zeros(nmask)
    for n_iter_rls in range(1, max_iter_rls+1):

        ll_x0 = ll_x
        ll_w0 = ll_w

        grad = spatial.membrane(dat, dim=3, voxel_size=voxel_size, weights=weights)
        grad = torch.masked_select(grad, mask, out=grad_buf)

        def precond(x):
            p = spatial.membrane_diag(dim=3, voxel_size=voxel_size, weights=weights)
            p = torch.masked_select(p.expand_as(dat), mask, out=precond_buf)
            p = torch.div(x, p, out=p)
            return p

        def forward(x):
            m = zeros
            m.masked_scatter_(mask, x)
            m = spatial.membrane(m, dim=3, voxel_size=voxel_size, weights=weights)
            m = torch.masked_select(m, mask, out=forward_buf)
            return m

        delta = optim.cg(forward, grad, precond=precond, max_iter=max_iter_cg,
                         tolerance=tol_cg, verbose=verbose > 1)
        subdat = torch.masked_select(dat, mask, out=forward_buf)
        subdat -= delta
        dat.masked_scatter_(mask, subdat)

        weights, ll_w = spatial.membrane_weights(dat, dim=3, voxel_size=voxel_size, return_sum=True)
        ll_x = spatial.membrane(dat, dim=3, voxel_size=voxel_size, weights=weights)
        ll_x = (ll_x * dat).sum(dtype=torch.double)
        if verbose > 0:
            print(f'{n_iter_rls:3d} | {ll_x} + {ll_w} = {ll_x + ll_w}')

        if abs(((ll_x0-ll_w0) - (ll_x+ll_w))/(ll_x0-ll_w0)) < tol_rls:
            if verbose > 0:
                print('Converged')
            break

    return dat

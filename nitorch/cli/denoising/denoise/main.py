from nitorch import io
from nitorch.core import py, utils
from nitorch.tools.denoising import tv
from nitorch.tools.img_statistics import estimate_noise
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


def denoise(*inputs, lam, sigma=None, jtv=True,
            output=None, device=None, verbose=1,
            max_iter=(10, 32), tol=1e-4, optim='cg'):

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

    # estimate noise
    sigma = py.make_list(sigma or [None], len(dat))
    for c, (sigma1, dat1) in enumerate(zip(sigma, dat)):
        if sigma1:
            continue
        sigma[c] = estimate_noise(dat1)[0]['sd']

    # Prepare options
    lam = py.make_list(lam, len(dat))
    # lam = [l * (s*s) for l, s in zip(lam, sigma)]
    lam = [l.item() if hasattr(l, 'item') else l for l in lam]
    max_iter, sub_iter = py.make_list(max_iter, 2)
    tol, sub_tol = py.make_list(tol, 2)

    # vx = 1 if aff is None else spatial.voxel_size(aff)
    vx = 1

    dat = tv.denoise(dat, lam=lam, sigma=sigma, jtv=jtv, dim=3, 
                     max_iter=max_iter, sub_iter=sub_iter, tol=tol, sub_tol=sub_tol,
                     optim=optim, plot=verbose > 2, voxel_size=vx)

    # Postprocess
    dat = utils.movedim(dat, 0, -1)
    dat = dat.split(nchannels, dim=-1)
    output = py.make_list(output, len(dat))
    for i in range(len(dat)):
        if fnames[i] and not output[i]:
            output[i] = '{dir}{sep}{base}.denoised{ext}'
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

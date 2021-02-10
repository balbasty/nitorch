import sys
import torch
from nitorch import io, spatial
from nitorch.core import utils, linalg
from nitorch.spatial import (
    mean_space, affine_conv, affine_resize, affine_matmul, affine_lmdiv,
    affine_grid, affine_matvec, grid_inv, affine_inv)
from .parser import parse
from .helpers import ffd_exp, samespace, smalldef, pull_grid, pull
from . import struct
from ._utils import BacktrackingLineSearch


def autoreg(argv=None):

    # parse arguments
    argv = argv or list(sys.argv)
    options = parse(list(argv))
    if not options:
        return

    if not options.optimizers:
        options.optimizers.append(struct.Adam())
    options.propagate_defaults()
    options.read_info()
    options.propagate_filenames()
    print(repr(options))

    load_data(options)
    load_transforms(options)

    while not all_optimized(options):
        add_freedom(options)
        init_optimizers(options)
        optimize(options)

    free_data(options)
    write_transforms(options)
    write_data(options)


def load_data(s):
    """Loads data and prepare loss functions"""

    def load(files, is_label):
        """Load one multi-channel multi-file volume.
        Returns a (channels, *spatial) tensor
        """
        dats = []
        for file in files:
            if is_label:
                dat = io.volumes.load(file.fname, dtype=torch.int32)
            else:
                dat = io.volumes.loadf(file.fname, dtype=torch.float32)
            dat = dat.reshape([*file.shape, file.channels])
            dat = utils.movedim(dat, -1, 0)
            dats.append(dat)
        return torch.cat(dats, dim=0)

    def pyramid(dat, affine, levels):
        """Compute an image pyramid using mean pooling.
        Returns a list of tensors, ordered form fine to coarse.
        """
        dats = [(dat, affine)] if 1 in levels else []
        for l in range(2, max(levels)+1):
            shape = dat.shape[1:]
            ker = [min(2, d) for d in shape]
            dat = torch.nn.functional.avg_pool3d(dat[None], ker)[0]
            affine, _ = affine_conv(affine, shape, ker, ker)
            if l in levels:
                dats.append((dat, affine))
        return dats

    levels = list(sorted(set(s.pyramid)))

    for loss in s.losses:
        if isinstance(loss, struct.NoLoss):
            continue
        loss.fixed.dat = load(loss.fixed.files, loss.fixed.type == 'labels')
        loss.moving.dat = load(loss.moving.files, loss.moving.type == 'labels')
        lvl = levels if loss.fixed.type != 'labels' else 1
        loss.fixed.dat = pyramid(loss.fixed.dat, loss.fixed.affine, lvl)
        loss.moving.dat = pyramid(loss.moving.dat, loss.moving.affine, lvl)


def load_transforms(s):
    """Initialize transforms"""

    def reshape3d(dat, channels=None):
        """Reshape as (*spatial) or (C, *spatial) or (*spatial, C).
        `channels` should be in ('first', 'last', None).
        """
        while len(dat.shape) > 3:
            if dat.shape[-1] == 1:
                dat = dat[..., 0]
                continue
            elif dat.shape[3] == 1:
                dat = dat[:, :, :, 0, ...]
                continue
            else:
                break
        if len(dat.shape) > 3 + bool(channels):
            raise ValueError('Too many channel dimensions')
        if channels and len(dat.shape) == 3:
            dat = dat[..., None]
        if channels == 'first':
            dat = utils.movedim(dat, -1, 0)
        return dat

    for trf in s.transformations:
        if isinstance(trf, struct.Linear):
            # Affine
            if isinstance(trf.init, str):
                trf.dat = io.transforms.loadf(trf.init, dtype=torch.float32)
            else:
                trf.dat = torch.zeros(trf.nb_prm(3), dtype=torch.float32)
        else:
            # compute mean space
            all_affines = []
            all_shapes = []
            for loss in s.losses:
                if isinstance(loss, struct.NoLoss):
                    continue
                all_shapes.append(loss.fixed.shape)
                all_shapes.append(loss.fixed.affine)
                all_shapes.append(loss.moving.shape)
                all_shapes.append(loss.moving.affine)
            affine, shape = mean_space(all_affines, all_shapes)

            # FFD/Diffeo
            if isinstance(trf.init, str):
                f = io.volumes.map(trf.init)
                trf.dat = reshape3d(f.loadf(dtype=torch.float32), 'last')
                if len(trf.dat) != 3:
                    raise ValueError('Field should have 3 channels')
                factor = [int(s//g) for g, s in zip(trf.shape[:-1], shape)]
                trf.affine, trf.shape = affine_resize(trf.affine, trf.shape[:-1], factor)
            else:
                trf.dat = torch.zeros([*shape, 3], dtype=torch.float32)
                trf.affine = affine
                trf.shape = shape


def init_optimizers(options):
    """Initialize optimizers and their step function."""
    params = []
    for trf in options.transformations:
        if hasattr(trf, 'optdat'):
            param = trf.optdat
            params.append({'params': param, 'lr': trf.lr})

    for optim in options.optimizers:

        params1 = []
        for param in params:
            params1.append({'params': param['params'],
                            'lr': param['lr'] * optim.lr})
        optim_obj = optim.call(params1, lr=optim.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_obj)
        if optim.ls != 0:
            if optim.ls is True:
                optim.ls = 6
            optim_obj = BacktrackingLineSearch(optim_obj, max_iter=optim.ls)

        def optim_step(fwd):
            optim_obj.zero_grad()
            loss = fwd()
            loss.backward()
            optim_obj.step()
            scheduler.step(loss)
            return loss

        def current_lr():
            lr = []
            for p0, p1 in zip(params, optim_obj.param_groups):
                lr.append(p1['lr'] / (p0['lr'] * optim.lr))
            return lr

        optim.step = optim_step
        optim.current_lr = current_lr


def all_optimized(options):
    """True if all parameters have been optimized"""

    for trf in options.transformations:
        if trf.freeable():
            return False
    return True


def add_freedom(options):
    """Progressively free parameters."""

    sortfn = lambda x: 0 if isinstance(x, struct.Linear) else 1
    options.transformations = list(sorted(options.transformations, key=sortfn))

    if options.progressive:
        for trf in options.transformations:
            if trf.freeable():
                trf.free()
                return
    else:
        for trf in options.transformations:
            trf.optdat = torch.nn.Parameter(trf.dat, requires_grad=True)
            trf.dat = trf.optdat


def optimize(options):
    """Optimization loop."""

    device = torch.device(options.device)
    backend = dict(dtype=torch.float, device=device)

    def forward():
        """Forward pass up to the loss"""

        loss = 0

        # affine matrix
        A = None
        for trf in options.transformations:
            if isinstance(trf, struct.Linear):
                q = trf.dat.to(**backend)
                B = trf.basis.to(**backend)
                A = linalg.expm(q, B)
                for loss1 in trf.losses:
                    loss += loss1.call(q)
                break

        # non-linear displacement field
        d = None
        d_aff = None
        for trf in options.transformations:
            if isinstance(trf, struct.FFD):
                d = trf.dat.to(**backend)
                d = ffd_exp(d, trf.shape, returns='disp')
                for loss1 in trf.losses:
                    loss += loss1.call(d)
                d_aff = trf.affine.to(**backend)
                break
            elif isinstance(trf, struct.Diffeo):
                d = trf.dat.to(**backend)
                for loss1 in trf.losses:
                    loss += loss1.call(d)
                d = spatial.exp(d, displacement=True)
                d_aff = trf.affine.to(**backend)
                break

        # loop over image pairs
        for match in options.losses:
            if not match.fixed:
                continue
            nb_levels = len(match.fixed.dat)
            prm = dict(interpolation=match.moving.interpolation,
                       bound=match.moving.bound,
                       extrapolate=match.moving.extrapolate)
            # loop over pyramid levels
            for moving, fixed in zip(match.moving.dat, match.fixed.dat):
                moving_dat, moving_aff = moving
                fixed_dat, fixed_aff = fixed

                moving_dat = moving_dat.to(**backend)
                moving_aff = moving_aff.to(**backend)
                fixed_dat = fixed_dat.to(**backend)
                fixed_aff = fixed_aff.to(**backend)

                # affine-corrected moving space
                if A is not None:
                    Ms = affine_matmul(A, moving_aff)
                else:
                    Ms = moving_aff

                if d is not None:
                    # fixed to param
                    Mt = affine_lmdiv(d_aff, fixed_aff)
                    if samespace(Mt, d.shape[:-1], fixed_dat.shape[1:]):
                        g = smalldef(d)
                    else:
                        g = affine_grid(Mt, fixed_dat.shape[1:])
                        g += pull_grid(d, g)
                    # param to moving
                    Ms = affine_lmdiv(Ms, d_aff)
                    g = affine_matvec(Ms, g)
                else:
                    # fixed to moving
                    Mt = fixed_aff
                    Ms = affine_lmdiv(Ms, Mt)
                    g = affine_grid(Ms, fixed_dat.shape[1:])

                # pull moving image
                warped_dat = pull(moving_dat, g, **prm)
                loss += match.call(warped_dat, fixed_dat) / float(nb_levels)

                # import matplotlib.pyplot as plt
                # plt.subplot(1, 2, 1)
                # plt.imshow(fixed_dat[0, :, :, fixed_dat.shape[-1]//2].detach())
                # plt.axis('off')
                # plt.subplot(1, 2, 2)
                # plt.imshow(warped_dat[0, :, :, warped_dat.shape[-1]//2].detach())
                # plt.axis('off')
                # plt.show()

        return loss

    # optimization loop
    for optimizer in options.optimizers:
        for n_iter in range(1, optimizer.max_iter):
            loss = optimizer.step(forward)
            current_lr = optimizer.current_lr()
            print(f'{n_iter:4d} | {loss.item():12.6f} | '
                  f'lr/lr0 = {sum(current_lr)/len(current_lr):7.3g}', end='\r')
            if all(lr < optimizer.stop for lr in current_lr):
                break
    print('')


def free_data(options):
    """Delete all handles to pyramid data"""
    for loss in options.losses:
        if loss.fixed:
            if hasattr(loss.fixed, 'dat'):
                delattr(loss.fixed, 'dat')
        if loss.moving:
            if hasattr(loss.moving, 'dat'):
                delattr(loss.moving, 'dat')


def write_transforms(options):
    """Write transformations (affine and nonlin) on disk"""
    nonlin = None
    affine = None
    for trf in options.transformations:
        if isinstance(trf, struct.NonLinear):
            nonlin = trf
        else:
            affine = trf

    if affine:
        q = affine.dat
        B = affine.basis
        lin = linalg.expm(q, B)
        io.transforms.savef(lin, affine.output, type=2)

    if nonlin:
        affine = nonlin.affine
        shape = nonlin.shape
        if isinstance(nonlin, struct.FFD):
            factor = [s/g for s, g in zip(shape, nonlin.dat.shape[:-1])]
            affine, _ = spatial.affine_resize(affine, shape, factor)
        io.volumes.savef(nonlin.dat, nonlin.output, affine=affine)


def update(moving, fname, inv=False, lin=None, nonlin=None,
           interpolation=1, bound='dct2', extrapolate=False):
    nonlin = nonlin or dict(disp=None, affine=None)
    prm = dict(interpolation=interpolation, bound=bound, extrapolate=extrapolate)

    if inv:
        # affine-corrected fixed space
        if lin is not None:
            new_affine = affine_lmdiv(lin, moving.affine)
        else:
            new_affine = moving.affine

        if nonlin.fwd is not None:
            # moving voxels to param voxels (warps param to moving)
            mov2nlin = affine_lmdiv(nonlin['affine'], moving.affine)
            if samespace(mov2nlin, nonlin['disp'].shape[:-1], moving.shape):
                g = smalldef(nonlin['disp'])
            else:
                g = affine_grid(mov2nlin, moving.shape)
                g += pull_grid(nonlin['disp'], g)
            # param to moving
            nonlin2mov = affine_inv(mov2nlin)
            g = affine_matvec(nonlin2mov, g)
        else:
            g = None

    else:
        # affine-corrected moving space
        if lin is not None:
            new_affine = affine_matmul(lin, moving.affine)
        else:
            new_affine = moving.affine

        if nonlin['disp'] is not None:
            # moving voxels to param voxels (warps param to moving)
            mov2nlin = affine_lmdiv(nonlin['affine'], new_affine)
            if samespace(mov2nlin, nonlin['disp'].shape[:-1], moving.shape):
                g = smalldef(nonlin['disp'])
            else:
                g = affine_grid(mov2nlin, moving.shape)
                g += pull_grid(nonlin['disp'], g)
            # param to moving
            nonlin2mov = affine_inv(mov2nlin)
            g = affine_matvec(nonlin2mov, g)
        else:
            g = None

    for file, ofname in zip(moving.files, fname):
        dat = io.volumes.loadf(file.fname)
        dat = dat.reshape([*file.shape, file.channels])
        if g is not None:
            dat = utils.movedim(dat, -1, 0)
            dat = pull(dat, g, **prm)
            dat = utils.movedim(dat, 0, -1)
        io.savef(dat, ofname, like=file.fname, affine=new_affine)


def reslice(moving, fname, like, inv=False, lin=None, nonlin=None,
           interpolation=1, bound='dct2', extrapolate=False):
    lin = lin or None
    nonlin = nonlin or dict(disp=None, affine=None)
    prm = dict(interpolation=interpolation, bound=bound, extrapolate=extrapolate)

    if inv:
        # affine-corrected fixed space
        if lin is not None:
            fix2lin = affine_lmdiv(lin, like.affine)
        else:
            fix2lin = moving.affine

        if nonlin['disp'] is not None:
            # fixed voxels to param voxels (warps param to fixed)
            fix2nlin = affine_lmdiv(nonlin['affine'], fix2lin)
            if samespace(fix2nlin, nonlin['disp'].shape[:-1], like.shape):
                g = smalldef(nonlin['disp'])
            else:
                g = affine_grid(fix2nlin, like.shape)
                g += pull_grid(nonlin['disp'], g)
            # param to moving
            nlin2mov = affine_lmdiv(moving.affine, nonlin['affine'])
            g = affine_matvec(nlin2mov, g)
        else:
            g = None

    else:
        # affine-corrected moving space
        if lin is not None:
            mov2lin = affine_matmul(lin, moving.affine)
        else:
            mov2lin = moving.affine

        if nonlin.fwd is not None:
            # fixed voxels to param voxels (warps param to fixed)
            fix2nlin = affine_lmdiv(nonlin['affine'], like.affine)
            if samespace(fix2nlin, nonlin['disp'].shape[:-1], like.shape):
                g = smalldef(nonlin['disp'])
            else:
                g = affine_grid(fix2nlin, like.shape)
                g += pull_grid(nonlin['disp'], g)
            # param voxels to moving voxels (warps moving to fixed)
            nonlin2mov = affine_inv(mov2lin)
            g = affine_matvec(nonlin2mov, g)
        else:
            g = None

    for file, ofname in zip(moving.files, fname):
        dat = io.volumes.loadf(file.fname)
        dat = dat.reshape([*file.shape, file.channels])
        if g is not None:
            dat = utils.movedim(dat, -1, 0)
            dat = pull(dat, g, **prm)
            dat = utils.movedim(dat, 0, -1)
        io.savef(dat, ofname, like=file.fname, affine=like.affine)


def write_data(options):

    device = torch.device(options.device)
    backend = dict(dtype=torch.float, device=device)

    need_inv = False
    for loss in options.losses:
        if loss.fixed and (loss.fixed.resliced or loss.fixed.updated):
            need_inv = True
            break

    # affine matrix
    lin = None
    for trf in options.transformations:
        if isinstance(trf, struct.Linear):
            q = trf.dat.to(**backend)
            B = trf.basis.to(**backend)
            lin = linalg.expm(q, B)
            break

    # non-linear displacement field
    d = None
    id = None
    d_aff = None
    for trf in options.transformations:
        if isinstance(trf, struct.FFD):
            d = trf.dat.to(**backend)
            d = ffd_exp(d, trf.shape, returns='disp')
            if need_inv:
                id = grid_inv(d)
            d_aff = trf.affine.to(**backend)
            break
        elif isinstance(trf, struct.Diffeo):
            d = trf.dat.to(**backend)
            if need_inv:
                id = spatial.exp(d, displacement=True, inverse=True)
            d = spatial.exp(d, displacement=True)
            d_aff = trf.affine.to(**backend)
            break

    # loop over image pairs
    for match in options.losses:

        moving = match.moving
        fixed = match.fixed
        prm = dict(interpolation=moving.interpolation,
                   bound=moving.bound,
                   extrapolate=moving.extrapolate)
        nonlin = dict(disp=d, affine=d_aff)
        if moving.updated:
            update(moving, moving.updated, lin=lin, nonlin=nonlin, **prm)
        if moving.resliced:
            reslice(moving, moving.resliced, like=fixed, lin=lin, nonlin=nonlin, **prm)
        if not fixed:
            continue
        prm = dict(interpolation=fixed.interpolation,
                   bound=fixed.bound,
                   extrapolate=fixed.extrapolate)
        nonlin = dict(disp=id, affine=d_aff)
        if fixed.updated:
            update(fixed, fixed.updated, inv=True, lin=lin, nonlin=nonlin, **prm)
        if fixed.resliced:
            reslice(fixed, fixed.resliced, inv=True, like=moving, lin=lin, nonlin=nonlin, **prm)

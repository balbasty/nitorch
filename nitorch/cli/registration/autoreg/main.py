"""This file implements the entry points as well as the fitting functions."""
import sys
import torch
from warnings import warn
from nitorch import io, spatial
from nitorch.core import utils, linalg
from nitorch.cli.cli import commands
from nitorch.spatial import (
    mean_space, affine_conv, affine_resize, affine_matmul, affine_lmdiv,
    affine_grid, affine_matvec, grid_inv, affine_inv, greens_apply)
from .parser import parse, ParseError, help
from nitorch.cli.registration.helpers import (
    ffd_exp, samespace, smalldef, pull_grid, pull, BacktrackingLineSearch)
from . import struct


def autoreg(argv=None):

    argv = argv or sys.argv[1:]

    try:
        _autoreg(argv)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['autoreg'] = autoreg


def _autoreg(argv=None):
    """Autograd Registration

    This is a command-line utility.
    """

    # parse arguments
    argv = argv or list(sys.argv)
    options = parse(list(argv))
    if not options:
        return

    # add a couple of defaults
    for trf in options.transformations:
        if isinstance(trf, struct.NonLinear) and not trf.losses:
            trf.losses.append(struct.AbsoluteLoss(factor=0.0001))
            trf.losses.append(struct.MembraneLoss(factor=0.001))
            trf.losses.append(struct.BendingLoss(factor=0.2))
            trf.losses.append(struct.LinearElasticLoss(factor=(0.05, 0.2)))
        trf.losses = [collapse_losses(trf.losses)]
    if not options.optimizers:
        options.optimizers.append(struct.Adam())

    options.propagate_defaults()
    options.read_info()
    options.propagate_filenames()

    if options.verbose >= 2:
        print(repr(options))

    load_data(options)
    load_transforms(options)

    print('Losses:')
    for loss in options.losses:
        print(f' - {loss.name}')
        for f, m in zip(loss.fixed.dat, loss.moving.dat):
            print(f'   -| {list(m[0].shape)}, {spatial.voxel_size(m[1]).tolist()}')
            print(f'   -> {list(f[0].shape)}, {spatial.voxel_size(f[1]).tolist()}')
    print('Transforms')
    for trf in options.transformations:
        print(f' - {trf.name}')
        if isinstance(trf, struct.NonLinear):
            pyramid0 = trf.pyramid[-1]
            for pyramid in reversed(trf.pyramid):
                factor = 2**(pyramid0 - pyramid)
                shape = [s*factor for s in trf.dat.shape]
                vx = spatial.voxel_size(trf.affine) / factor
                print(f'   - {list(shape)}, {vx.tolist()}')

    while not all_optimized(options):
        add_freedom(options)
        init_optimizers(options)
        optimize(options)

    free_data(options)
    detach_transforms(options)
    write_transforms(options)
    write_data(options)


def collapse_losses(losses):
    """Collapse losses on dense fields into a GridLoss"""
    gridloss = struct.GridLoss()
    for loss in losses:
        if isinstance(loss, struct.AbsoluteLoss):
            gridloss.absolute = loss.factor
        if isinstance(loss, struct.MembraneLoss):
            gridloss.membrane = loss.factor
        if isinstance(loss, struct.BendingLoss):
            gridloss.bending = loss.factor
        if isinstance(loss, struct.LinearElasticLoss):
            gridloss.lame = loss.factor
    return gridloss


def load_data(s):
    """Loads data and prepare loss functions"""

    device = torch.device(s.device)

    def load(files, is_label=False):
        """Load one multi-channel multi-file volume.
        Returns a (channels, *spatial) tensor
        """
        dats = []
        for file in files:
            if is_label:
                dat = io.volumes.load(file.fname,
                                      dtype=torch.int32, device=device)
            else:
                dat = io.volumes.loadf(file.fname, rand=True,
                                       dtype=torch.float32, device=device)
            dat = dat.reshape([*file.shape, file.channels])
            dat = dat[..., file.subchannels]
            dat = utils.movedim(dat, -1, 0)
            dim = dat.dim() - 1
            qt = utils.quantile(dat, (0.01, 0.95), dim=range(-dim, 0), keepdim=True)
            mn, mx = qt.unbind(-1)
            dat = dat.sub_(mn).div_(mx-mn)
            dats.append(dat)
            del dat
        dats = torch.cat(dats, dim=0)
        if is_label and len(dats) > 1:
            warn('Multi-channel label images are not accepted. '
                 'Using only the first channel')
            dats = dats[:1]
        return dats

    def split(dat, labels):
        # transforming labels into probabilities
        if not labels:
            labels = torch.unique(dat, sorted=True)
            labels = labels[labels != 0].tolist()
        out = dat.new_empty(dtype=torch.float32)
        for i, l in enumerate(labels):
            out[labels == l] = 1
        return out, labels

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

    def compute_grad(dat):
        med = dat.reshape([dat.shape[0], -1]).median(dim=-1).values
        med = utils.unsqueeze(med, -1, 3)
        dat /= 0.5*med
        dat = spatial.diff(dat, dim=[1, 2, 3]).square().sum(-1)
        return dat

    for loss in s.losses:
        if isinstance(loss, struct.NoLoss):
            continue
        loss.fixed.dat = load(loss.fixed.files, loss.fixed.type == 'labels')
        loss.moving.dat = load(loss.moving.files, loss.moving.type == 'labels')
        if loss.moving.type == 'labels':
            loss.moving.dat, loss.labels = split(loss.moving.dat, loss.labels)
        lvl = (list(sorted(set(loss.fixed.pyramid)))
               if loss.fixed.type != 'labels' else [1])
        loss.fixed.dat = pyramid(loss.fixed.dat,
                                 loss.fixed.affine.to(device), lvl)
        lvl = (list(sorted(set(loss.moving.pyramid)))
               if loss.moving.type != 'labels' else [1])
        loss.moving.dat = pyramid(loss.moving.dat,
                                  loss.moving.affine.to(device), lvl)
        if isinstance(loss, struct.JTVLoss):
            loss.moving.dat = [(compute_grad(dat), aff)
                               for dat, aff in loss.moving.dat]


def load_transforms(s):
    """Initialize transforms"""
    device = torch.device(s.device)

    def reshape3d(dat, channels=None, dim=3):
        """Reshape as (*spatial) or (C, *spatial) or (*spatial, C).
        `channels` should be in ('first', 'last', None).
        """
        while len(dat.shape) > dim:
            if dat.shape[-1] == 1:
                dat = dat[..., 0]
                continue
            elif dat.shape[dim] == 1:
                dat = dat[:, :, :, 0, ...]
                continue
            else:
                break
        if len(dat.shape) > dim + bool(channels):
            raise ValueError('Too many channel dimensions')
        if channels and len(dat.shape) == dim:
            dat = dat[..., None]
        if channels == 'first':
            dat = utils.movedim(dat, -1, 0)
        return dat

    # compute mean space
    #   it is used to define the space of the nonlinear transform, but
    #   also to shift the center of rotation of the linear transform.
    all_affines = []
    all_shapes = []
    all_affines_fixed = []
    all_shapes_fixed = []
    for loss in s.losses:
        if isinstance(loss, struct.NoLoss):
            continue
        if getattr(loss, 'exclude', False):
            continue
        all_shapes_fixed.append(loss.fixed.shape)
        all_affines_fixed.append(loss.fixed.affine)
        all_shapes.append(loss.fixed.shape)
        all_affines.append(loss.fixed.affine)
        all_shapes.append(loss.moving.shape)
        all_affines.append(loss.moving.affine)
    affine0, shape0 = mean_space(all_affines, all_shapes,
                                 pad=s.pad, pad_unit=s.pad_unit)
    affinef, shapef = mean_space(all_affines_fixed, all_shapes_fixed,
                                 pad=s.pad, pad_unit=s.pad_unit)
    backend = dict(dtype=affine0.dtype, device=affine0.device)

    for trf in s.transformations:
        for reg in trf.losses:
            if isinstance(reg.factor, (list, tuple)):
                reg.factor = [f * trf.factor for f in reg.factor]
            else:
                reg.factor = reg.factor * trf.factor
        if isinstance(trf, struct.Linear):
            # Affine
            if isinstance(trf.init, str):
                trf.dat = io.transforms.loadf(trf.init, dtype=torch.float32,
                                              device=device)
            else:
                trf.dat = torch.zeros(trf.nb_prm(3), dtype=torch.float32,
                                      device=device)
            if trf.shift:
                shift = torch.as_tensor(shapef, **backend) * 0.5
                trf.shift = -spatial.affine_matvec(affinef, shift)
            else:
                trf.shift = 0.
        else:
            affine, shape = (affine0, shape0)
            trf.pyramid = list(sorted(trf.pyramid))
            max_level = max(trf.pyramid)
            factor = 2**(max_level-1)
            affine, shape = affine_resize(affine, shape, 1/factor)

            # FFD/Diffeo
            if isinstance(trf.init, str):
                f = io.volumes.map(trf.init)
                trf.dat = reshape3d(f.loadf(dtype=torch.float32, device=device),
                                    'last')
                if len(trf.dat) != trf.dim:
                    raise ValueError('Field should have 3 channels')
                factor = [int(s//g) for g, s in zip(trf.shape[:-1], shape)]
                trf.affine, trf.shape = affine_resize(trf.affine, trf.shape[:-1], factor)
            else:
                trf.dat = torch.zeros([*shape, trf.dim], dtype=torch.float32,
                                      device=device)
                trf.affine = affine
                trf.shape = shape


def init_optimizers(options):
    """Initialize optimizers and their step function."""
    def get_state(optim):
        state = None
        if not hasattr(optim, 'param_groups'):
            return state
        for group in optim.param_groups:
            for p in group['params']:
                if p.dim() > 1:
                    return optim.state[p]
        return group['params'], state

    def get_shape(params):
        for p in params:
            if p['params'].dim() > 1:
                return p['params'].shape[:-1]
        return None

    params = []
    for trf in options.transformations:
        if hasattr(trf, 'optdat'):
            param = trf.optdat
            params.append({'params': param,
                           'lr': trf.lr,
                           'weight_decay': trf.weight_decay})

    for optim in options.optimizers:
        params1 = []
        for param in params:
            params1.append({'params': param['params'],
                            'lr': param['lr'] * optim.lr,
                           'weight_decay': param['weight_decay'] * optim.weight_decay})
        # if hasattr(optim, 'obj') and isinstance(optim.obj, torch.optim.Adam):
        #     state = get_state(optim)
        # else:
        #     state = None
        optim.obj = optim.call(params1, lr=optim.lr,
                               weight_decay=optim.weight_decay)
        # if state is not None:
        #     p, new_shape = get_shape(params1)
        #     state['step'] = 0
        #     state['exp_avg'] = spatial.resize_grid(
        #         state['exp_avg'][None], shape=new_shape, type='displacement')[0]
        #     state['exp_avg_sq'] = spatial.resize_grid(
        #         state['exp_avg_sq'][None], shape=new_shape, type='displacement')[0]
        #     optim.state[p] = state

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim.obj)
        if optim.ls != 0:
            if optim.ls is True:
                optim.ls = 6
            optim.obj = BacktrackingLineSearch(optim.obj, max_iter=optim.ls)
        optim.first_iter = True

        def optim_step(fwd, greens=None):
            def closure():
                optim.obj.zero_grad()
                loss = fwd()
                loss.backward()
                if greens is not None:
                    for group in optim.obj.param_groups:
                        for param in group['params']:
                            if param.dim() >= 3:
                                param.grad = greens_apply(param.grad, greens)
                return loss
            loss = optim.obj.step(closure)

            if optim.first_iter:
                optim.first_iter = False
            else:
                scheduler.step(loss)
            return loss

        def current_lr():
            lr = []
            for p0, p1 in zip(params, optim.obj.param_groups):
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

    greens = None
    for trf in options.transformations:
        if isinstance(trf, struct.Diffeo):
            if trf.losses:
                greens = trf.losses[0].greens(trf.dat.shape[:-1], **backend)

    def forward():
        """Forward pass up to the loss"""

        loss = 0

        # affine matrix
        A = None
        for trf in options.transformations:
            trf.update()
            if isinstance(trf, struct.Linear):
                q = trf.optdat.to(**backend)
                # print(q.tolist())
                B = trf.basis.to(**backend)
                A = linalg.expm(q, B)
                if torch.is_tensor(trf.shift):
                    # include shift
                    shift = trf.shift.to(**backend)
                    eye = torch.eye(options.dim, **backend)
                    A = A.clone()  # needed because expm is a custom autograd.Function
                    A[:-1, -1] += torch.matmul(A[:-1, :-1] - eye, shift)
                for loss1 in trf.losses:
                    loss += loss1.call(q)
                break

        # non-linear displacement field
        d = None
        d_aff = None
        for trf in options.transformations:
            if not trf.isfree():
                continue
            if isinstance(trf, struct.FFD):
                d = trf.dat.to(**backend)
                d = ffd_exp(d, trf.shape, returns='disp')
                for loss1 in trf.losses:
                    loss += loss1.call(d)
                d_aff = trf.affine.to(**backend)
                break
            elif isinstance(trf, struct.Diffeo):
                d = trf.dat.to(**backend)
                if not trf.smalldef:
                    # penalty on velocity fields
                    for loss1 in trf.losses:
                        loss += loss1.call(d)
                d = spatial.exp(d[None], displacement=True)[0]
                if trf.smalldef:
                    # penalty on exponentiated transform
                    for loss1 in trf.losses:
                        loss += loss1.call(d)
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
                        g = g + pull_grid(d, g)
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
        for n_iter in range(1, optimizer.max_iter+1):
            loss = optimizer.step(forward, greens=greens)
            current_lr = optimizer.current_lr()
            if options.verbose:
                print(f'{n_iter:4d} | {loss.item():12.6f} | '
                      f'lr/lr0 = {sum(current_lr)/len(current_lr):7.3g}', end='\r')
                if n_iter == 1:
                    print('')
            if all(lr < optimizer.stop for lr in current_lr):
                break
    if options.verbose:
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


def detach_transforms(options):
    for trf in options.transformations:
        trf.update()
        trf.dat = trf.dat.detach()
        delattr(trf, 'optdat')


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
        if torch.is_tensor(affine.shift):
            # include shift
            shift = affine.shift.to(dtype=lin.dtype, device=lin.device)
            eye = torch.eye(3, dtype=lin.dtype, device=lin.device)
            lin[:-1, -1] += torch.matmul(lin[:-1, :-1] - eye, shift)
        io.transforms.savef(lin.cpu(), affine.output, type=2)

    if nonlin:
        affine = nonlin.affine
        shape = nonlin.shape
        if isinstance(nonlin, struct.FFD):
            factor = [s/g for s, g in zip(shape, nonlin.dat.shape[:-1])]
            affine, _ = spatial.affine_resize(affine, shape, factor)
        io.volumes.savef(nonlin.dat.cpu(), nonlin.output, affine=affine.cpu())


def update(moving, fname, inv=False, lin=None, nonlin=None,
           interpolation=1, bound='dct2', extrapolate=False, device='cpu',
           verbose=True):
    """Apply the linear transform to the header of a file +
    apply the non-linear component in the original space.

    Notes
    -----
    .. The shape and general orientation of the moving image is kept untouched.
    .. The linear transform is composed with the original orientation matrix.
    .. The non-linear component is "wrapped" in the input space, where it
       is applied.
    .. This function writes a new file (it does not modify the input files
       in place).

    Parameters
    ----------
    moving : ImageFile
        An object describing an image to wrap.
    fname : list of str
        Output filename for each input file of the moving image
        (since images can be encoded over multiple volumes)
    inv : bool, default=False
        True if we are warping the fixed image to the moving space.
        In the case, `moving` should be a `FixedImageFile`.
        Else it should be a `MovingImageFile`.
    lin : (4, 4) tensor, optional
        Linear (or rather affine) transformation
    nonlin : dict, optional
        Non-linear displacement field, with keys:
        disp : (..., 3) tensor
            Displacement field (in voxels)
        affine : (4, 4) tensor
            Orientation matrix of the displacement field
    interpolation : int, default=1
    bound : str, default='dct2'
    extrapolate : bool, default = False
    device : default='cpu'

    """
    nonlin = nonlin or dict(disp=None, affine=None)
    prm = dict(interpolation=interpolation, bound=bound, extrapolate=extrapolate)

    moving_affine = moving.affine.to(device)

    if inv:
        # affine-corrected fixed space
        if lin is not None:
            new_affine = affine_lmdiv(lin, moving_affine)
        else:
            new_affine = moving.affine

        if nonlin['disp'] is not None:
            # moving voxels to param voxels (warps param to moving)
            mov2nlin = affine_lmdiv(nonlin['affine'].to(device), moving_affine)
            if samespace(mov2nlin, nonlin['disp'].shape[:-1], moving.shape):
                g = smalldef(nonlin['disp'].to(device))
            else:
                g = affine_grid(mov2nlin, moving.shape)
                g += pull_grid(nonlin['disp'].to(device), g)
            # param to moving
            nonlin2mov = affine_inv(mov2nlin)
            g = affine_matvec(nonlin2mov, g)
        else:
            g = None

    else:
        # affine-corrected moving space
        if lin is not None:
            new_affine = affine_matmul(lin, moving_affine)
        else:
            new_affine = moving_affine

        if nonlin['disp'] is not None:
            # moving voxels to param voxels (warps param to moving)
            mov2nlin = affine_lmdiv(nonlin['affine'].to(device), new_affine)
            if samespace(mov2nlin, nonlin['disp'].shape[:-1], moving.shape):
                g = smalldef(nonlin['disp'].to(device))
            else:
                g = affine_grid(mov2nlin, moving.shape)
                g += pull_grid(nonlin['disp'].to(device), g)
            # param to moving
            nonlin2mov = affine_inv(mov2nlin)
            g = affine_matvec(nonlin2mov, g)
        else:
            g = None

    for file, ofname in zip(moving.files, fname):
        if verbose:
            print(f'Registered: {file.fname}\n'
                  f'         -> {ofname}')
        dat = io.volumes.loadf(file.fname, rand=True, device=device)
        dat = dat.reshape([*file.shape, file.channels])
        if g is not None:
            dat = utils.movedim(dat, -1, 0)
            dat = pull(dat, g, **prm)
            dat = utils.movedim(dat, 0, -1)
        io.savef(dat.cpu(), ofname, like=file.fname, affine=new_affine.cpu())


def reslice(moving, fname, like, inv=False, lin=None, nonlin=None,
           interpolation=1, bound='dct2', extrapolate=False, device=None,
           verbose=True):
    """Apply the linear and non-linear components of the transform and
    reslice the image to the target space.

    Notes
    -----
    .. The shape and general orientation of the moving image is kept untouched.
    .. The linear transform is composed with the original orientation matrix.
    .. The non-linear component is "wrapped" in the input space, where it
       is applied.
    .. This function writes a new file (it does not modify the input files
       in place).

    Parameters
    ----------
    moving : ImageFile
        An object describing a moving image.
    fname : list of str
        Output filename for each input file of the moving image
        (since images can be encoded over multiple volumes)
    like : ImageFile
        An object describing the target space
    inv : bool, default=False
        True if we are warping the fixed image to the moving space.
        In the case, `moving` should be a `FixedImageFile` and `like` a
        `MovingImageFile`. Else it should be a `MovingImageFile` and `'like`
        a `FixedImageFile`.
    lin : (4, 4) tensor, optional
        Linear (or rather affine) transformation
    nonlin : dict, optional
        Non-linear displacement field, with keys:
        disp : (..., 3) tensor
            Displacement field (in voxels)
        affine : (4, 4) tensor
            Orientation matrix of the displacement field
    interpolation : int, default=1
    bound : str, default='dct2'
    extrapolate : bool, default = False
    device : default='cpu'

    """
    nonlin = nonlin or dict(disp=None, affine=None)
    prm = dict(interpolation=interpolation, bound=bound, extrapolate=extrapolate)

    moving_affine = moving.affine.to(device)
    fixed_affine = like.affine.to(device)

    if inv:
        # affine-corrected fixed space
        if lin is not None:
            fix2lin = affine_matmul(lin, fixed_affine)
        else:
            fix2lin = fixed_affine

        if nonlin['disp'] is not None:
            # fixed voxels to param voxels (warps param to fixed)
            fix2nlin = affine_lmdiv(nonlin['affine'].to(device), fix2lin)
            if samespace(fix2nlin, nonlin['disp'].shape[:-1], like.shape):
                g = smalldef(nonlin['disp'].to(device))
            else:
                g = affine_grid(fix2nlin, like.shape)
                g += pull_grid(nonlin['disp'].to(device), g)
            # param to moving
            nlin2mov = affine_lmdiv(moving_affine, nonlin['affine'].to(device))
            g = affine_matvec(nlin2mov, g)
        else:
            g = affine_lmdiv(moving_affine, fix2lin)
            g = affine_grid(g, like.shape)

    else:
        # affine-corrected moving space
        if lin is not None:
            mov2nlin = affine_matmul(lin, moving_affine)
        else:
            mov2nlin = moving_affine

        if nonlin['disp'] is not None:
            # fixed voxels to param voxels (warps param to fixed)
            fix2nlin = affine_lmdiv(nonlin['affine'].to(device), fixed_affine)
            if samespace(fix2nlin, nonlin['disp'].shape[:-1], like.shape):
                g = smalldef(nonlin['disp'].to(device))
            else:
                g = affine_grid(fix2nlin, like.shape)
                g += pull_grid(nonlin['disp'].to(device), g)
            # param voxels to moving voxels (warps moving to fixed)
            nlin2mov = affine_lmdiv(mov2nlin, nonlin['affine'].to(device))
            g = affine_matvec(nlin2mov, g)
        else:
            g = affine_lmdiv(mov2nlin, fixed_affine)
            g = affine_grid(g, like.shape)

    if moving.type == 'labels':
        prm['interpolation'] = 0
    for file, ofname in zip(moving.files, fname):
        if verbose:
            print(f'Resliced:   {file.fname}\n'
                  f'         -> {ofname}')
        dat = io.volumes.loadf(file.fname, rand=True, device=device)
        dat = dat.reshape([*file.shape, file.channels])
        if g is not None:
            dat = utils.movedim(dat, -1, 0)
            dat = pull(dat, g, **prm)
            dat = utils.movedim(dat, 0, -1)
        io.savef(dat.cpu(), ofname, like=file.fname, affine=like.affine.cpu())


def write_data(options):

    device = torch.device(options.device)
    backend = dict(dtype=torch.float, device='cpu')

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
            if torch.is_tensor(trf.shift):
                # include shift
                shift = trf.shift.to(**backend)
                eye = torch.eye(3, **backend)
                lin[:-1, -1] += torch.matmul(lin[:-1, :-1] - eye, shift)
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
                id = spatial.exp(d[None], displacement=True, inverse=True)[0]
            d = spatial.exp(d[None], displacement=True)[0]
            d_aff = trf.affine.to(**backend)
            break

    # loop over image pairs
    for match in options.losses:

        moving = match.moving
        fixed = match.fixed
        prm = dict(interpolation=moving.interpolation,
                   bound=moving.bound,
                   extrapolate=moving.extrapolate,
                   device='cpu',
                   verbose=options.verbose)
        nonlin = dict(disp=d, affine=d_aff)
        if moving.updated:
            update(moving, moving.updated, lin=lin, nonlin=nonlin, **prm)
        if moving.resliced:
            reslice(moving, moving.resliced, like=fixed, lin=lin, nonlin=nonlin, **prm)
        if not fixed:
            continue
        prm = dict(interpolation=fixed.interpolation,
                   bound=fixed.bound,
                   extrapolate=fixed.extrapolate,
                   device='cpu',
                   verbose=options.verbose)
        nonlin = dict(disp=id, affine=d_aff)
        if fixed.updated:
            update(fixed, fixed.updated, inv=True, lin=lin, nonlin=nonlin, **prm)
        if fixed.resliced:
            reslice(fixed, fixed.resliced, inv=True, like=moving, lin=lin, nonlin=nonlin, **prm)

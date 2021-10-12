from nitorch import spatial, io
from nitorch.core import utils
from nitorch.core.dtypes import dtype as nitype
import torch
import os
from .parser import FileWithInfo, Linear, Velocity, Displacement
from .. import helpers


def compose(options):
    read_info(options)
    collapse_transforms(options)
    write_data(options)


def squeeze_to_nd(dat, dim=3, channels=1):
    if isinstance(dat, (tuple, list)):
        shape = dat
        dat = None
    else:
        shape = dat.shape
    shape = list(shape)
    while len(shape) > dim + channels:
        has_deleted = False
        for d in reversed(range(dim, len(shape))):
            if shape[d] == 1:
                del shape[d]
                has_deleted = True
        if has_deleted:
            continue
        for d in reversed(range(len(shape)-channels)):
            if shape[d] == 1:
                del shape[d]
                has_deleted = True
                break
        if has_deleted:
            continue

        raise ValueError(f'Cannot squeeze shape so that it has '
                         f'{dim} spatial dimensions and {channels} '
                         f'channels.')
    shape = shape + [1] * max(0, dim-len(shape))
    if len(shape) < dim + channels:
        ones = [1] * max(0, dim+channels-len(shape))
        shape = shape[:dim] + ones + shape[dim:]
    shape = tuple(shape)
    if dat is not None:
        dat = dat.reshape(shape)
        return dat
    return shape


def read_info(options):
    """Load affine transforms and space info of other volumes"""

    def read_file(fname):
        o = FileWithInfo()
        o.fname = fname
        o.dir = os.path.dirname(fname) or '.'
        o.base = os.path.basename(fname)
        o.base, o.ext = os.path.splitext(o.base)
        if o.ext in ('.gz', '.bz2'):
            zext = o.ext
            o.base, o.ext = os.path.splitext(o.base)
            o.ext += zext
        f = io.volumes.map(fname)
        o.float = nitype(f.dtype).is_floating_point
        o.shape = squeeze_to_nd(f.shape, dim=3, channels=1)
        o.channels = o.shape[-1]
        o.shape = o.shape[:3]
        o.affine = f.affine.float()
        return o

    def read_affine(fname):
        mat = io.transforms.loadf(fname).float()
        return squeeze_to_nd(mat, 0, 2)

    def read_field(fname):
        f = io.volumes.map(fname)
        return f.affine.float(), f.shape[:3]

    for trf in options.transformations:
        if isinstance(trf, Linear):
            trf.affine = read_affine(trf.file)
        else:
            trf.affine, trf.shape = read_field(trf.file)
    if options.target:
        options.target = read_file(options.target)
    else:
        for trf in reversed(options.transformations):
            if isinstance(trf, (Velocity, Displacement)):
                options.target = read_file(trf.file)
                break


def collapse_transforms(options):
    """Pre-invert affines and combine sequential affines"""
    trfs = []
    last_trf = None
    for trf in options.transformations:
        if isinstance(trf, Linear):
            if trf.inv:
                trf.affine = spatial.affine_inv(trf.affine)
                trf.inv = False
            if isinstance(last_trf, Linear):
                last_trf.affine = spatial.affine_matmul(last_trf.affine, trf.affine)
            else:
                last_trf = trf
        else:
            if isinstance(last_trf, Linear):
                trfs.append(last_trf)
                last_trf = None
            trfs.append(trf)
    if isinstance(last_trf, Linear):
        trfs.append(last_trf)
    options.transformations = trfs


def write_data(options):

    backend = dict(dtype=torch.float32, device=options.device)

    # Pre-exponentiate velocities
    for trf in options.transformations:
        if isinstance(trf, Velocity):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            trf.dat = spatial.exp(trf.dat[None], displacement=True,
                                  inverse=trf.inv)[0]
            trf.inv = False
            trf.order = 1
        elif isinstance(trf, Displacement):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.unit == 'mm':
                # convert mm displacement to vox displacement
                trf.dat = spatial.affine_lmdiv(trf.affine, trf.dat[..., None])
                trf.dat = trf.dat[..., 0]
                trf.unit = 'vox'

    def build_from_target(target):
        """Compose all transformations, starting from the final orientation"""
        grid = spatial.affine_grid(target.affine.to(**backend),
                                   target.shape)
        for trf in reversed(options.transformations):
            if isinstance(trf, Linear):
                grid = spatial.affine_matvec(trf.affine.to(**backend), grid)
            else:
                mat = trf.affine.to(**backend)
                if trf.inv:
                    vx0 = spatial.voxel_size(mat)
                    vx1 = spatial.voxel_size(target.affine.to(**backend))
                    factor = vx0 / vx1
                    disp, mat = spatial.resize_grid(trf.dat[None], factor,
                                                    affine=mat,
                                                    interpolation=trf.spline)
                    disp = spatial.grid_inv(disp[0], type='disp')
                    order = 1
                else:
                    disp = trf.dat
                    order = trf.spline
                imat = spatial.affine_inv(mat)
                grid = spatial.affine_matvec(imat, grid)
                grid += helpers.pull_grid(disp, grid, interpolation=order)
                grid = spatial.affine_matvec(mat, grid)
        return grid

    if options.target:
        # If target is provided, we build a dense transformation field
        grid = build_from_target(options.target)
        oaffine = options.target.affine
        if options.output_unit[0] == 'v':
            grid = spatial.affine_matvec(spatial.affine_inv(oaffine), grid)
            grid = grid - spatial.identity_grid(grid.shape[:-1],
                                                **utils.backend(grid))
        else:
            grid = grid - spatial.affine_grid(oaffine.to(**utils.backend(grid)),
                                              grid.shape[:-1])
        io.volumes.savef(grid, options.output.format(ext='.nii.gz'), affine=oaffine)
    else:
        if len(options.transformations) > 1:
            raise RuntimeError('Something weird happened: '
                               'multiple transforms and no target')
        io.transforms.savef(options.transformations[0].affine,
                            options.output.format(ext='.lta'))





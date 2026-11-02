import sys
import os
import json
import copy
import torch
import itertools
import math as pymath
from nitorch import io, spatial
from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.core import utils, py
from nitorch.core.dtypes import dtype as nitype
from nitorch.core.linalg import sqrtm
from nitorch.core.math import logit, softmax
from . import struct
from .parser import parse, help
from .. import helpers


def reslice(argv=None):
    """Generic reslicing

    This is a command-line utility: errors are printed to stderr and
    reported by returning 1 (see `_reslice` for a version that raises).
    """

    try:
        return _reslice(argv)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1


def _reslice(argv=None):
    """Generic reslicing, without exception handling"""
    argv = argv or sys.argv[1:]
    options = parse(list(argv))
    if not options:
        return

    read_info(options)

    options_volumes = options
    options_streamlines = copy.deepcopy(options)
    options_streamlines.transformations \
        = invert_transforms(options_streamlines.transformations)

    if options_volumes.volumes:
        collapse(options_volumes)
        write_volumes(options_volumes)
        del options_volumes

    if options_streamlines.streamlines:
        collapse(options_streamlines)
        write_streamlines(options_streamlines)
        del options_streamlines


commands['reslice'] = reslice


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
        f = io.map(fname)
        if isinstance(f, io.MappedArray):
            return read_volume(fname, f)
        elif isinstance(f, io.MappedStreamlines):
            return read_streamlines(fname, f)
        else:
            raise TypeError(type(f))

    def read_streamlines(fname, f=None):
        o = struct.FileWithInfo()
        o.type = "streamlines"
        o.fname = fname
        o.dir = os.path.dirname(fname) or '.'
        o.base = os.path.basename(fname)
        o.base, o.ext = os.path.splitext(o.base)
        if o.ext in ('.gz', '.bz2'):
            zext = o.ext
            o.base, o.ext = os.path.splitext(o.base)
            o.ext += zext
        f = f or io.streamlines.map(f)
        o.affine = f.affine.float()
        return o

    def read_volume(fname, f=None):
        o = struct.FileWithInfo()
        o.type = "volume"
        o.fname = fname
        o.dir = os.path.dirname(fname) or '.'
        o.base = os.path.basename(fname)
        o.base, o.ext = os.path.splitext(o.base)
        if o.ext in ('.gz', '.bz2'):
            zext = o.ext
            o.base, o.ext = os.path.splitext(o.base)
            o.ext += zext
        f = f or io.volumes.map(fname)
        o.float = nitype(f.dtype).is_floating_point
        o.shape = squeeze_to_nd(f.shape, dim=3, channels=1)
        o.channels = o.channels_in = o.shape[-1]
        o.shape = o.shape[:3]
        o.affine = f.affine.float()

        if options.channels is not None:
            o.channel_index = channel_index(options.channels, o.channels_in)
            o.channels = len(o.channel_index)

        return o

    def read_affine(fname):
        mat = io.transforms.loadf(fname).float()
        return squeeze_to_nd(mat, 0, 2)

    def read_field(fname):
        f = io.volumes.map(fname)
        return f.affine.float(), f.shape[:3]

    options.files = [read_file(file) for file in options.files]
    options.streamlines = [
        file for file in options.files if file.type == "streamlines"
    ]
    options.volumes = [
        file for file in options.files if file.type == "volume"
    ]
    for trf in options.transformations:
        if isinstance(trf, struct.Linear):
            trf.affine = read_affine(trf.file)
        else:
            trf.affine, trf.shape = read_field(trf.file)
    if options.target:
        options.target = read_file(options.target)
        if options.voxel_size:
            options.voxel_size = utils.make_vector(options.voxel_size, 3,
                                                   dtype=options.target.affine.dtype)
            factor = spatial.voxel_size(options.target.affine) / options.voxel_size
            options.target.affine, options.target.shape = \
                spatial.affine_resize(options.target.affine, options.target.shape,
                                      factor=factor, anchor='f')


def channel_index(channels, nb_channels):
    """Convert a list of channel specifications into a list of indices

    Parameters
    ----------
    channels : [list of] int or range or slice
        Channel indices and/or Python-style ranges.
    nb_channels : int
        Number of channels in the volume.

    Returns
    -------
    index : list[int]
        Non-negative channel indices.
    """
    all_channels = list(range(nb_channels))
    index = []
    for c in py.make_list(channels):
        if isinstance(c, range):
            # Python slicing semantics (negative/omitted bounds)
            c = slice(c.start, c.stop, c.step)
        try:
            if isinstance(c, slice):
                index.extend(all_channels[c])
            else:
                index.append(all_channels[int(c)])
        except (IndexError, TypeError, ValueError):
            raise ValueError(f'Invalid channel {c} for a volume with '
                             f'{nb_channels} channels')
    if not index:
        raise ValueError(f'No channel selected by {channels}')
    return index


def collapse(options):
    options.transformations = collapse_transforms(options.transformations)


def collapse_transforms(transformations):
    """Pre-invert affines and combine sequential affines"""
    trfs = []
    last_trf = None
    for trf in transformations:
        if isinstance(trf, struct.Linear):
            if trf.square:
                trf.affine = sqrtm(trf.affine)
            if trf.inv:
                trf.affine = spatial.affine_inv(trf.affine)
                trf.inv = False
            if isinstance(last_trf, struct.Linear):
                last_trf.affine = spatial.affine_matmul(last_trf.affine, trf.affine)
            else:
                last_trf = trf
        else:
            if isinstance(last_trf, struct.Linear):
                trfs.append(last_trf)
                last_trf = None
            trfs.append(trf)
    if isinstance(last_trf, struct.Linear):
        trfs.append(last_trf)
    return trfs


# keys of the JSON sidecar (written by `nitorch register`) that are
# forwarded to `spatial.shoot`
_shoot_keys = ('absolute', 'membrane', 'bending', 'lame', 'factor',
               'steps', 'fast')


def load_shoot_parameters(fname):
    """Read shooting parameters from a JSON sidecar"""
    with open(fname) as f:
        prm = json.load(f)
    return {key: val for key, val in prm.items() if key in _shoot_keys}


def exponentiate_transforms(transformations, **backend):
    for trf in transformations:
        if isinstance(trf, struct.Velocity):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.json:
                if trf.square:
                    trf.dat.mul_(0.5)
                prm = load_shoot_parameters(trf.json)
                prm['voxel_size'] = spatial.voxel_size(trf.affine)
                trf.dat = spatial.shoot(trf.dat[None], displacement=True,
                                        return_inverse=trf.inv, **prm)
                if trf.inv:
                    trf.dat = trf.dat[-1]
            else:
                if trf.square:
                    trf.dat.mul_(0.5)
                trf.dat = spatial.exp(trf.dat[None], displacement=True,
                                      inverse=trf.inv)
            trf.dat = trf.dat[0]  # drop batch dimension
            trf.inv = False
            trf.square = False
            trf.order = 1
        elif isinstance(trf, struct.Displacement):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.unit == 'mm':
                # convert mm displacement to vox displacement
                lin = trf.affine[:3, :3].to(trf.dat)
                trf.dat = trf.dat.matmul(lin.inverse().transpose(-1, -2))
                trf.unit = 'vox'
    return transformations


def invert_transforms(transformations):
    transformations = [copy.deepcopy(trf) for trf in reversed(transformations)]
    for trf in transformations:
        trf.inv = not trf.inv
    return transformations


def _logit_implicit(options):
    return (isinstance(options.logit, str) and
            options.logit[:1].lower() == 'i')


def pre_transform_intensities(dat, options):
    """Map intensities to log/logit space. `dat` is channel-first."""
    if not (options.log or options.logit):
        return dat
    pos = dat[dat.isfinite() & (dat > 0)]
    mn = pos.min() if pos.numel() else 1e-5
    dat = dat.clamp_min_(mn*1e-3)
    if options.logit and _logit_implicit(options):
        dat = logit(dat, dim=0, implicit=True, inplace=True)
    else:
        # --log, or explicit --logit: log-probabilities are mapped
        # back with a softmax across channels
        dat = dat.log_()
    return dat


def post_transform_intensities(dat, options):
    """Map intensities back from log/logit space. `dat` is channel-first."""
    if options.logit:
        dat = softmax(dat, dim=0, implicit=_logit_implicit(options),
                      inplace=True)
    elif options.log:
        dat = dat.exp_()
    return dat


def write_volumes(options):

    backend = dict(dtype=torch.float32, device=options.device)

    if options.chunk:
        options.chunk = py.make_list(options.chunk, 3)

    # 1) Pre-exponentiate velocities
    exponentiate_transforms(options.transformations, **backend)

    # 2) If the first transform is linear, compose it with the input
    #    orientation matrix
    if (options.transformations and
            isinstance(options.transformations[0], struct.Linear)):
        trf = options.transformations[0]
        for file in options.volumes:
            mat = file.affine.to(**backend)
            aff = trf.affine.to(**backend)
            file.affine = spatial.affine_lmdiv(aff, mat)
        options.transformations = options.transformations[1:]

    def build_from_target(affine, shape, smart=False):
        """Compose all transformations, starting from the final orientation"""
        if smart and all(isinstance(trf, struct.Linear) for trf in options.transformations):
            return None
        grid = spatial.affine_grid(affine.to(**backend), shape)
        for trf in reversed(options.transformations):
            if isinstance(trf, struct.Linear):
                grid = spatial.affine_matvec(trf.affine.to(**backend), grid)
            else:
                mat = trf.affine.to(**backend)
                if trf.inv:
                    vx0 = spatial.voxel_size(mat)
                    vx1 = spatial.voxel_size(affine.to(**backend))
                    factor = vx0 / vx1
                    disp, mat = spatial.resize_grid(trf.dat[None], factor,
                                                    affine=mat,
                                                    interpolation=trf.order)
                    disp = spatial.grid_inv(disp[0], type='disp')
                    order = 1
                else:
                    disp = trf.dat
                    order = trf.order
                imat = spatial.affine_inv(mat)
                grid = spatial.affine_matvec(imat, grid)
                grid += helpers.pull_grid(disp, grid, interpolation=order)
                grid = spatial.affine_matvec(mat, grid)
        return grid

    # 3) If target is provided, we can build most of the transform once
    #    and just multiply it with a input-wise affine matrix.
    #    (built lazily, as it is not needed if processing chunkwise)
    target_grid = None

    # 4) Loop across input files
    opt_pull0 = dict(interpolation=options.interpolation,
                     bound=options.bound,
                     extrapolate=options.extrapolate)
    opt_coeff = dict(interpolation=options.interpolation,
                     bound=options.bound,
                     dim=3,
                     inplace=True)
    output = py.make_list(options.output, len(options.volumes))
    for file, ofname in zip(options.volumes, output):
        is_label = isinstance(options.interpolation, str) and options.interpolation == 'l'
        ofname = ofname.format(dir=file.dir, base=file.base, ext=file.ext, sep=os.path.sep)
        if options.verbose:
            print(f'Reslicing:   {file.fname}\n'
                  f'          -> {ofname}')
        vol = io.volumes.map(file.fname)
        if is_label:
            backend_int = dict(dtype=torch.long, device=backend['device'])
            # dat = io.volumes.load(file.fname, **backend_int)
            opt_pull = dict(opt_pull0)
            opt_pull['interpolation'] = 1
        else:
            # dat = io.volumes.loadf(file.fname, rand=False, **backend)
            opt_pull = opt_pull0
        if file.channel_index is not None:
            channels = file.channel_index
        else:
            channels = slice(None)

        resample = True
        if not options.target:
            oaffine = file.affine
            oshape = file.shape
            if options.voxel_size:
                ovx = utils.make_vector(options.voxel_size, 3,
                                        dtype=oaffine.dtype)
                factor = spatial.voxel_size(oaffine) / ovx
                oaffine, oshape = spatial.affine_resize(oaffine, oshape, factor=factor, anchor='f')
            else:
                resample = not all(
                    isinstance(t, struct.Linear)
                    for t in options.transformations
                )
        else:
            oaffine = options.target.affine
            oshape = options.target.shape

        # --------------------------------------------------------------
        # chunkwise processing
        # --------------------------------------------------------------
        need_chunk = bool(options.chunk)
        if need_chunk:
            chunk = py.make_list(options.chunk, 3)
            need_chunk = need_chunk and any(c < x for x, c in zip(oshape, chunk))

        if resample and need_chunk:

            odat = utils.movedim(torch.empty(
                [*oshape, file.channels],
                **(backend_int if is_label else backend)
            ), -1, 0)

            ncells = [
                int(pymath.ceil(x/c))
                for x, c in zip(oshape, chunk)
            ]
            pcells = [list(range(x)) for x in ncells]
            for cell in itertools.product(*pcells):
                oslicer = tuple(slice(i*c, (i+1)*c) for i, c in zip(cell, chunk))

                # output chunk
                odatc = odat[(Ellipsis, *oslicer)]
                oshapec = odatc.shape[-3:]
                oaffinec, _ = spatial.affine_sub(oaffine, oshape, oslicer)

                # grid
                grid = build_from_target(oaffinec, oshapec)
                mat = file.affine.to(**backend)
                imat = spatial.affine_inv(mat)
                grid = spatial.affine_matvec(imat, grid)

                mn = grid.reshape([-1, 3]).min(dim=0).values
                mx = grid.reshape([-1, 3]).max(dim=0).values
                for i in range(3):
                    mn[i].sub_(5).floor_().clamp_(0, file.shape[i]-1)
                    mx[i].add_(5).ceil_().clamp_(0, file.shape[i]-1)
                grid -= mn

                # input chunk
                mn, mx = mn.long(), mx.long()
                islicer = tuple(
                    slice(mn1.item(), mx1.item()+1)
                    for mn1, mx1 in zip(mn, mx)
                )
                idatc = vol[(*islicer, Ellipsis)]
                if is_label:
                    idatc = idatc.data(**backend_int)
                else:
                    idatc = idatc.fdata(rand=False, **backend)
                ishapec = idatc.shape[:3]
                idatc = idatc.reshape([*ishapec, file.channels_in])
                idatc = idatc[..., channels]
                idatc = utils.movedim(idatc, -1, 0)

                if options.clip:
                    imin = idatc[idatc.isfinite()].min()
                    imax = idatc[idatc.isfinite()].max()

                # pre-transform intensities
                idatc = pre_transform_intensities(idatc, options)

                # reslice
                if options.prefilter and not is_label:
                    idatc = spatial.spline_coeff_nd(idatc, **opt_coeff)
                idatc = helpers.pull(idatc, grid, **opt_pull)

                # post-transform intensities
                idatc = post_transform_intensities(idatc, options)

                if options.clip:
                    idatc.clamp_(imin, imax)

                # copy to output
                odatc.copy_(idatc)

            dat = utils.movedim(odat, 0, -1)

        # --------------------------------------------------------------
        # one-shot processing
        # --------------------------------------------------------------
        else:
            if is_label:
                dat = vol.data(**backend_int)
            else:
                dat = vol.fdata(rand=False, **backend)
            dat = dat.reshape([*file.shape, file.channels_in])
            dat = dat[..., channels]
            dat = utils.movedim(dat, -1, 0)

            if options.target:
                if target_grid is None:
                    target_grid = build_from_target(oaffine, oshape)
                grid = target_grid
            else:
                grid = build_from_target(oaffine, oshape, smart=not options.voxel_size)
            if grid is not None:
                mat = file.affine.to(**backend)
                imat = spatial.affine_inv(mat)

                if options.clip:
                    imin = dat[dat.isfinite()].min()
                    imax = dat[dat.isfinite()].max()

                # pre-transform intensities
                dat = pre_transform_intensities(dat, options)

                # reslice
                if options.prefilter and not is_label:
                    dat = spatial.spline_coeff_nd(dat, **opt_coeff)
                dat = helpers.pull(dat, spatial.affine_matvec(imat, grid), **opt_pull)

                # post-transform intensities
                dat = post_transform_intensities(dat, options)

                if options.clip:
                    dat.clamp_(imin, imax)

            dat = utils.movedim(dat, 0, -1)

        if is_label:
            io.volumes.save(dat, ofname, like=file.fname, affine=oaffine, dtype=options.dtype)
        else:
            io.volumes.savef(dat, ofname, like=file.fname, affine=oaffine, dtype=options.dtype)


def write_streamlines(options):

    backend = dict(dtype=torch.float32, device=options.device)

    # 1) Pre-exponentiate velocities
    exponentiate_transforms(options.transformations, **backend)

    def deform_streamlines(streamlines):
        """Compose all transformations, starting from the final orientation"""
        for trf in reversed(options.transformations):
            if isinstance(trf, struct.Linear):
                streamlines = spatial.affine_matvec(trf.affine.to(streamlines), streamlines)
            else:
                mat = trf.affine.to(**backend)
                if trf.inv:
                    disp = spatial.grid_inv(trf.dat, type='disp')
                    order = 1
                else:
                    disp = trf.dat
                    order = trf.order
                imat = spatial.affine_inv(mat)
                streamlines = spatial.affine_matvec(imat.to(streamlines), streamlines)
                streamlines = streamlines + helpers.pull_grid(disp, streamlines[None, None], interpolation=order)[0, 0]
                streamlines = spatial.affine_matvec(mat.to(streamlines), streamlines)
        return streamlines

    # 4) Loop across input files
    output = py.make_list(options.output, len(options.streamlines))
    for file, ofname in zip(options.streamlines, output):
        ofname = ofname.format(dir=file.dir, base=file.base, ext=file.ext, sep=os.path.sep)
        if options.verbose:
            print(f'Reslicing:   {file.fname}\n'
                  f'          -> {ofname}')
        dat = list(io.streamlines.loadf(file.fname, **backend))
        offsets = py.cumsum(map(len, dat), exclusive=True)
        dat = torch.cat(list(dat))

        dat = deform_streamlines(dat)
        dat = [
            dat[offsets[i]:(offsets[i+1] if i+1 < len(offsets) else None)]
            for i in range(len(offsets))
        ]

        io.streamlines.savef(dat, ofname, like=file.fname)

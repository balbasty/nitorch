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
from nitorch.core.linalg import logm, expm
from nitorch.core.math import logit, softmax
from . import struct
from .parser import parse, help
from .. import helpers


def reslice(argv=None):
    """Generic reslicing

    This is a command-line utility.
    """

    try:
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

    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


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
        o.channels = o.shape[-1]
        o.shape = o.shape[:3]
        o.affine = f.affine.float()

        if options.channels is not None:
            channels = py.make_list(options.channels)
            channels = [
                list(c) if isinstance(c, range) else
                list(range(o.channels))[c] if isinstance(c, slice) else
                c for c in channels
            ]
            if not all([isinstance(c, int) for c in channels]):
                raise ValueError('Channel list should be a list of integers')
            o.channels = len(channels)

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


def collapse(options):
    options.transformations = collapse_transforms(options.transformations)


def collapse_transforms(transformations):
    """Pre-invert affines and combine sequential affines"""
    trfs = []
    last_trf = None
    for trf in transformations:
        if isinstance(trf, struct.Linear):
            if trf.square:
                trf.affine = expm(logm(trf.affine).mul_(0.5))
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
                with open(trf.json) as f:
                    prm = json.load(f)
                prm['voxel_size'] = spatial.voxel_size(trf.affine)
                trf.dat = spatial.shoot(trf.dat[None], displacement=True,
                                        return_inverse=trf.inv)
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
                trf.dat = spatial.affine_lmdiv(trf.affine, trf.dat[..., None])
                trf.dat = trf.dat[..., 0]
                trf.unit = 'vox'
    return transformations


def invert_transforms(transformations):
    transformations = [copy.deepcopy(trf) for trf in reversed(transformations)]
    for trf in transformations:
        trf.inv = not trf.inv
    return transformations


def write_volumes(options):

    backend = dict(dtype=torch.float32, device=options.device)

    if options.chunk:
        options.chunk = py.make_list(options.chunk, 3)

    # 1) Pre-exponentiate velocities
    for trf in options.transformations:
        if isinstance(trf, struct.Velocity):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.json:
                if trf.square:
                    trf.dat.mul_(0.5)
                with open(trf.json) as f:
                    prm = json.load(f)
                prm['voxel_size'] = spatial.voxel_size(trf.affine)
                trf.dat = spatial.shoot(trf.dat[None], displacement=True,
                                        return_inverse=trf.inv)
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
                trf.dat = spatial.affine_lmdiv(trf.affine, trf.dat[..., None])
                trf.dat = trf.dat[..., 0]
                trf.unit = 'vox'

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
    if options.target:
        if not options.chunk:
            grid = build_from_target(options.target.affine, options.target.shape)
        oaffine = options.target.affine

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
        ofname = ofname.format(dir=file.dir, base=file.base, ext=file.ext)
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
        if options.channels is not None:
            channels = py.make_list(options.channels)
            channels = [
                list(c) if isinstance(c, range) else
                list(range(vol.shape[-1]))[c] if isinstance(c, slice) else
                c for c in channels
            ]
            if not all([isinstance(c, int) for c in channels]):
                raise ValueError('Channel list should be a list of integers')
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
            need_chunk = need_chunk and any(c < x for x, c in zip(file.shape, chunk))

        if resample and need_chunk:

            odat = utils.movedim(torch.empty(
                [*oshape, file.channels],
                **(backend_int if is_label else backend)
            ), -1, 0)

            ncells = [
                int(pymath.ceil(x/c))
                for x, c in zip(file.shape, chunk)
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
                idatc = idatc[..., channels]
                ishapec = idatc.shape[:3]
                idatc = idatc.reshape([*ishapec, file.channels])
                idatc = utils.movedim(idatc, -1, 0)

                if options.clip:
                    imin = idatc[idatc.isfinite()].min()
                    imax = idatc[idatc.isfinite()].max()

                # pre-transform intensities
                if options.log or options.logit:
                    mx = idatc.abs().max()
                    idatc = idatc.clamp_min_(mx*1e-8)
                    idatc = idatc.log_()

                    if (
                        options.logit and
                        isinstance(options.logit, str) and
                        options.logit[:1].lower() == "i"
                    ):
                        idatc = logit(idatc, implicit=True, inplace=True)
                    else:
                        idatc = idatc.log_()

                # reslice
                if options.prefilter and not is_label:
                    idatc = spatial.spline_coeff_nd(idatc, **opt_coeff)
                idatc = helpers.pull(idatc, grid, **opt_pull)

                # post-transform intensities
                if options.log or options.logit:
                    if options.logit:
                        implicit = (
                            isinstance(options.logit, str) and
                            options.logit[:1].lower() == "i"
                        )
                        idatc = softmax(idatc, implicit=implicit, inplace=True)
                    else:
                        idatc = idatc.exp_()

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
            dat = dat[..., channels]
            dat = dat.reshape([*file.shape, file.channels])
            dat = utils.movedim(dat, -1, 0)

            if not options.target:
                grid = build_from_target(oaffine, oshape, smart=not options.voxel_size)
            if grid is not None:
                mat = file.affine.to(**backend)
                imat = spatial.affine_inv(mat)

                if options.clip:
                    imin = dat[dat.isfinite()].min()
                    imax = dat[dat.isfinite()].max()

                # pre-transform intensities
                if options.log or options.logit:
                    mn = dat[dat.isfinite() & (dat > 0)].min()
                    dat = dat.clamp_min_(mn*1e-3)

                    if (
                        options.logit and
                        isinstance(options.logit, str) and
                        options.logit[:1].lower() == "i"
                    ):
                        dat = logit(dat, implicit=True, inplace=True)
                    else:
                        dat = dat.log_()

                # reslice
                if options.prefilter and not is_label:
                    dat = spatial.spline_coeff_nd(dat, **opt_coeff)
                dat = helpers.pull(dat, spatial.affine_matvec(imat, grid), **opt_pull)

                # post-transform intensities
                if options.log or options.logit:
                    if options.logit:
                        implicit = (
                            isinstance(options.logit, str) and
                            options.logit[:1].lower() == "i"
                        )
                        dat = softmax(dat, implicit=implicit, inplace=True)
                    else:
                        dat = dat.exp_()

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
    for trf in options.transformations:
        if isinstance(trf, struct.Velocity):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.json:
                if trf.square:
                    trf.dat.mul_(0.5)
                with open(trf.json) as f:
                    prm = json.load(f)
                prm['voxel_size'] = spatial.voxel_size(trf.affine)
                trf.dat = spatial.shoot(trf.dat[None], displacement=True,
                                        return_inverse=trf.inv)
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
                trf.dat = spatial.affine_lmdiv(trf.affine, trf.dat[..., None])
                trf.dat = trf.dat[..., 0]
                trf.unit = 'vox'

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

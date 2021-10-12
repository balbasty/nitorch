import sys
import os
import torch
from nitorch import io, spatial
from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure
from nitorch.core import utils, py
from nitorch.core.dtypes import dtype as nitype
from .parser import parser, help


def resize(argv=None):
    """Generic reslicing

    This is a command-line utility.
    """

    try:
        argv = argv or sys.argv[1:]
        options = parser.parse(list(argv))
        if not options:
            return
        if options.help:
            print(help)
            return

        files = read_info(options.files)
        write_data(files, options)

    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['resize'] = resize


class FileWithInfo(Structure):
    fname: str = None               # Full path
    shape: tuple = None             # Spatial shape
    affine = None                   # Orientation matrix
    dir: str = None                 # Directory
    base: str = None                # Base name (without extension)
    ext: str = None                 # Extension
    channels: int = None            # Number of channels
    float: bool = True              # Is raw dtype floating point


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


def read_info(files):
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

    return [read_file(file) for file in files]


def write_data(files, options):

    device = options.gpu if isinstance(options.gpu, str) else f'cuda:{options.gpu}'
    backend = dict(dtype=torch.float32, device=device)
    ofiles = py.make_list(options.output, len(files))

    for file, ofile in zip(files, ofiles):

        ofile = ofile.format(dir=file.dir, base=file.base, ext=file.ext, sep=os.sep)
        print(f'Resizing:   {file.fname}\n'
              f'         -> {ofile}')

        dat = io.loadf(file.fname, **backend)
        dat = dat.reshape([*file.shape, file.channels])

        # compute resizing factor
        input_vx = spatial.voxel_size(file.affine)
        if options.voxel_size:
            if options.factor:
                raise ValueError('Cannot use both factor and voxel size')
            factor = input_vx / utils.make_vector(options.voxel_size, 3)
        elif options.factor:
            factor = utils.make_vector(options.factor, 3)
        elif options.shape:
            input_shape = utils.make_vector(dat.shape[:-1], 3, dtype=torch.float32)
            output_shape = utils.make_vector(options.shape, 3, dtype=torch.float32)
            factor = output_shape / input_shape
        else:
            raise ValueError('Need at least one of factor/voxel_size/shape')
        factor = factor.tolist()

        # check if output shape is provided
        if options.shape:
            output_shape = py.ensure_list(options.shape, 3)
        else:
            output_shape = None

        # Perform resize
        opt = dict(
            anchor=options.anchor,
            bound=options.bound,
            interpolation=options.interpolation,
            prefilter=options.prefilter,
        )
        if options.grid:
            dat, affine = spatial.resize_grid(dat[None], factor, output_shape,
                                              type=options.grid,
                                              affine=file.affine, **opt)[0]
        else:
            dat = utils.movedim(dat, -1, 0)
            dat, affine = spatial.resize(dat[None], factor, output_shape,
                                         affine=file.affine, **opt)
            dat = utils.movedim(dat[0], 0, -1)

        # Write output file
        io.volumes.savef(dat, ofile, like=file.fname, affine=affine)


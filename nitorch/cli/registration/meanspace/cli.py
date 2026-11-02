import sys
import shutil
import warnings
import torch
import os.path as op
from tempfile import mkdtemp

from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch import io
from nitorch.cli.registration.reslice.main import reslice
from nitorch.spatial import mean_space
from nitorch.core.py import make_list, fileparts
from .parser import parser, help


def cli(args=None):
    f"""Command-line interface for `meanspace`

    {help}

    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except ParseError as e:
        print(help)
        print('[ERROR]', e)
        return
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['meanspace'] = cli


def setup_device(device='cpu', ndevice=None):
    if device == 'gpu' and not torch.cuda.is_available():
        warnings.warn('CUDA not available. Switching to CPU.')
        device, ndevice = 'cpu', None
    if device == 'cpu':
        if ndevice:
            torch.set_num_threads(ndevice)
        return torch.device('cpu')
    if ndevice is not None:
        return torch.device(f'cuda:{ndevice}')
    return torch.device('cuda')


def _device_args(device):
    """Device option to forward to `reslice`"""
    if device.type == 'cuda':
        return ['--gpu'] + ([str(device.index)] if device.index is not None else [])
    return ['--cpu']


def _cli(args):
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help)
        return

    device = setup_device(*options.device)

    # get all shape and matrices
    shapes = []
    affines = []
    for path in options.input:
        f = io.map(path)
        shapes += [f.shape[:3]]
        affines += [torch.as_tensor(f.affine, device=device)]

    ndim = max(map(len, shapes))

    # parse voxel size
    voxel_size = options.voxel_size
    voxel_size = make_list(voxel_size or [])
    if voxel_size and isinstance(voxel_size[-1], str):
        *voxel_size, vx_unit = voxel_size
    else:
        vx_unit = 'mm'
    if voxel_size:
        voxel_size = make_list(voxel_size, ndim)
    else:
        voxel_size = None

    # parse padding
    pad = options.pad
    pad = make_list(pad or [])
    if pad and isinstance(pad[-1], str):
        *pad, pad_unit = pad
    else:
        pad_unit = '%'
    if pad:
        pad = make_list(pad, ndim)
    else:
        pad = None

    # compute mean space
    affine, shape = mean_space(
        affines,
        shapes,
        voxel_size=voxel_size,
        vx_unit=vx_unit,
        layout=options.layout,
        pad=pad,
        pad_unit=pad_unit
    )
    print(shape)
    print(affine.cpu().numpy())

    dir, base, ext = fileparts(op.abspath(options.input[0]))

    # write mean space image
    write_meanspace = options.output is not False
    if not write_meanspace and not options.resliced:
        return
    tmpdir = None
    if not write_meanspace:
        # only needed as a reslicing target: write it to a temporary
        # directory that is removed once all inputs are resliced
        tmpdir = mkdtemp()
        options.output = op.join(tmpdir, 'meanspace' + ext)

    options.output = options.output.format(dir=dir, base=base, ext=ext)

    try:
        io.savef(
            torch.zeros(shape),
            options.output,
            affine=affine.cpu()
        )

        # write resliced images
        if options.resliced:
            for input in options.input:
                input = op.abspath(input)
                dir, base, ext = fileparts(input)
                output = options.resliced.format(dir=dir, base=base, ext=ext)
                print(output)
                reslice([input, '-o', output, '-t', options.output,
                         *_device_args(device)])
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

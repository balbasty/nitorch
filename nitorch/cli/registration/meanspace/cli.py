import sys
import torch
import os.path as op
from tempfile import NamedTemporaryFile

from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch import io
from nitorch.cli.registration.reslice.main import reslice
from nitorch.spatial import mean_space
from nitorch.core.py import make_list, fileparts
from .parser import parser


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

help = r"""
Compute an average "oriented voxel space" from a series of oriented images.

usage:
    nitorch meanspace --input <path> [--input ...]

arguments:
    -i, --input         Affine transform for one pair of images
                          <fix>   Index (or label) of fixed image
                          <mov>   Index (or label) of moving image
                          <path>  Path to an LTA file that warps <mov> to <fix>
    -o, --output        Path to output transforms (default: {label}_optimal.lta)
    -l, --log           Minimize L2 in Lie algebra (default: L2 in matrix space)
    -a, --affine        Assume transforms are all affine (default)
    -s, --similitude    Assume transforms are all similitude
    -r, --rigid         Assume transforms are all rigid

example:
    optimal_affine \
      -i mtw pdw mtw_to_pdw.lta \
      -i mtw t1w mtw_to_t1w.lta \
      -i pdw mtw pdw_to_mtw.lta \
      -i pdw t1w pdw_to_t1w.lta \
      -i t1w mtw t1w_to_mtw.lta \
      -i t1w pdw t1w_to_pdw.lta \
      -o out/{label}_to_mean.lta

references:
    "Consistent multi-time-point brain atrophy estimation from the
    boundary shift integral"
    Leung, Ridgway, Ourselin, Fox
    NeuroImage (2011)

    "Symmetric Diffeomorphic Modeling of Longitudinal Structural MRI"
    Ashburner, Ridgway
    Front Neurosci. (2012)
"""


def _cli(args):
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help)
        return

    # get all shape and matrices
    shapes = []
    affines = []
    for path in options.input:
        f = io.map(path)
        shapes += [f.shape[:3]]
        affines += [f.affine]

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
        pad=pad,
        pad_unit=pad_unit
    )
    print(shape)
    print(affine.numpy())

    dir, base, ext = fileparts(op.abspath(options.input[0]))

    # write mean space image
    write_meanspace = options.output is not False
    if not write_meanspace:
        tmp = NamedTemporaryFile("wb", delete=False, suffix=ext)
        options.output = tmp.name

    options.output = options.output.format(dir=dir, base=base, ext=ext)

    io.savef(
        torch.zeros(shape),
        options.output,
        affine=affine
    )

    # write resliced images
    if options.resliced:
        for input in options.input:
            input = op.abspath(input)
            dir, base, ext = fileparts(input)
            output = options.resliced.format(dir=dir, base=base, ext=ext)
            print(output)
            reslice([input, '-o', output, '-t', options.output])

        if not write_meanspace:
            tmp.close()

import os.path

import torch.cuda

from nitorch.core.py import make_list, fileparts
from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.spatial import smooth, voxel_size
from nitorch.core.utils import movedim_front2back, movedim_back2front
from nitorch import io
from .parser import parser, help
import sys


def cli(args=None):
    f"""Command-line interface for `smooth`
    
    {help}
    
    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


def _cli(args):
    """Command-line interface for `smooth` without exception handling"""
    args = args or sys.argv[1:]

    options = parser(args)
    if options.help:
        print(help)
        return

    fwhm = options.fwhm
    unit = 'mm'
    if isinstance(fwhm[-1], str):
        *fwhm, unit = fwhm
    fwhm = make_list(fwhm, 3)
    device = 'cpu'
    if options.gpu[0] == 'gpu' and torch.cuda.is_available():
        device = f'cuda: {options.gpu[1]}'

    options.output = make_list(options.output, len(options.files))
    for fname, ofname in zip(options.files, options.output):
        f = io.map(fname)
        vx = voxel_size(f.affine).tolist()
        dim = len(vx)
        if unit == 'mm':
            fwhm1 = [f/v for f, v in zip(fwhm, vx)]
        else:
            fwhm1 = fwhm[:len(vx)]

        dat = f.fdata(device=device)
        dat = movedim_front2back(dat, dim)
        dat = smooth(dat, type=options.method, fwhm=fwhm1, basis=options.basis,
                     bound=options.padding, dim=dim)
        dat = movedim_back2front(dat, dim)

        folder, base, ext = fileparts(fname)
        ofname = ofname.format(dir=folder or '.', base=base, ext=ext, sep=os.path.sep)
        io.savef(dat, ofname, like=f)


commands['smooth'] = cli

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
        return 1
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1


# CLI names -> names/orders understood by `nitorch.core.kernels.smooth`
method_aliases = {'tri': 'triangle'}
basis_aliases = {'rect': 0, 'tri': 1}


def get_device(option):
    """Convert a `('cpu'|'gpu', N)` option into a torch device string"""
    device, index = option
    if device == 'gpu' and torch.cuda.is_available():
        return f'cuda:{index}'
    if device == 'cpu' and index:
        torch.set_num_threads(index)
    return 'cpu'


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
    device = get_device(options.gpu)
    method = method_aliases.get(options.method, options.method)
    basis = basis_aliases[options.basis]

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
        dat = smooth(dat, type=method, fwhm=fwhm1, basis=basis,
                     bound=options.padding, dim=dim)
        dat = movedim_back2front(dat, dim)

        folder, base, ext = fileparts(fname)
        ofname = ofname.format(dir=folder or '.', base=base, ext=ext, sep=os.path.sep)
        io.savef(dat, ofname, like=f)


commands['smooth'] = cli

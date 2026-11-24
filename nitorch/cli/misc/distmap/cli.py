import os.path
import sys
import warnings

import torch

from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.core.py import make_list, fileparts
from nitorch.core.utils import movedim_front2back, movedim_back2front
from nitorch.spatial import (
    voxel_size,
    euclidean_distance_transform, euclidean_signed_transform,
    l1_distance_transform, l1_signed_transform)
from nitorch import io
from .parser import parser, help


def cli(args=None):
    f"""Command-line interface for `distmap`

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


transforms = {
    ('l2', False): euclidean_distance_transform,
    ('l2', True): euclidean_signed_transform,
    ('l1', False): l1_distance_transform,
    ('l1', True): l1_signed_transform,
}


def get_device(option):
    """Convert a `('cpu'|'gpu', N)` option into a torch device string"""
    device, index = option
    if device == 'gpu' and torch.cuda.is_available():
        return f'cuda:{index}'
    if device == 'cpu' and index:
        torch.set_num_threads(index)
    return 'cpu'


def get_mask(dat, labels=None, threshold=0):
    """Binarize a volume: voxels in `labels`, or above `threshold`."""
    if labels:
        mask = torch.zeros_like(dat, dtype=torch.bool)
        for label in labels:
            mask |= dat == label
    else:
        mask = dat > threshold
    return mask


def distmap(mask, dim, vx=1, metric='l2', signed=False, invert=False):
    """Distance map of a (..., *spatial) mask"""
    if invert:
        mask = ~mask
    dist = transforms[(metric, signed)](mask, dim=dim, vx=vx)
    if invert and signed:
        dist = dist.neg_()
    return dist


def _cli(args):
    """Command-line interface for `distmap` without exception handling"""
    args = args or sys.argv[1:]

    options = parser(args)
    if options.help:
        print(help)
        return

    device = get_device(options.gpu)
    options.output = make_list(options.output, len(options.files))
    for fname, ofname in zip(options.files, options.output):
        f = io.map(fname)
        vx = voxel_size(f.affine).tolist()
        dim = len(vx)
        if options.unit != 'mm':
            vx = [1.] * dim

        dat = f.fdata(device=device)
        dat = movedim_front2back(dat, dim)
        mask = get_mask(dat, options.labels, options.threshold)
        del dat
        dist = distmap(mask, dim, vx, options.metric,
                       options.signed, options.invert)
        if not torch.isfinite(dist).all():
            warnings.warn(f'{fname}: the object or the background is empty, '
                          f'some distances are infinite.')
        dist = movedim_back2front(dist, dim)

        folder, base, ext = fileparts(fname)
        ofname = ofname.format(dir=folder or '.', base=base, ext=ext,
                               sep=os.path.sep)
        io.savef(dist, ofname, like=f, dtype='float32')


commands['distmap'] = cli

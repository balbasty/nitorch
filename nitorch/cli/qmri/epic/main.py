from nitorch.cli.cli import commands
from .parser import parser, help, ParseError
from nitorch.tools.denoising.epic import epic
from nitorch import io, spatial
from nitorch.core import py, utils
import torch
import sys
import os
import warnings


def cli(args=None):
    f"""Command-line interface for `epic`

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


commands['epic'] = cli


def _cli(args):
    """Command-line interface for `epic` without exception handling"""
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help)
        return
    if options.verbose > 3:
        print(options)
        print('')

    return main(options)


def get_device(device):
    device, ndevice = device
    if device == 'gpu' and not torch.cuda.is_available():
        warnings.warn('CUDA not available. Switching to CPU.')
        device, ndevice = 'cpu', None
    if device == 'cpu':
        device = torch.device('cpu')
        if ndevice:
            torch.set_num_threads(ndevice)
    else:
        assert device == 'gpu'
        if ndevice is not None:
            device = torch.device(f'cuda:{ndevice}')
        else:
            device = torch.device('cuda')
    return device


def main(options):

    # find readout direction
    f = io.map(options.echoes[0])
    affine, shape = f.affine, f.shape
    readout = get_readout(options.direction, affine, shape, options.verbose)

    if not options.reversed:
        reversed_echoes = options.synth
    else:
        reversed_echoes = options.reversed

    # do EPIC
    fit = epic(options.echoes,
               reverse_echoes=reversed_echoes,
               fieldmap=options.fieldmap,
               extrapolate=options.extrapolate,
               bandwidth=options.bandwidth,
               polarity=-1 if options.polarity == '-' else +1,
               readout=readout,
               slicewise=options.slicewise,
               lam=options.penalty,
               max_iter=options.maxiter,
               tol=options.tolerance,
               verbose=options.verbose,
               device=get_device(options.gpu))

    # save volumes
    input, output = options.echoes, options.output
    if len(output) != len(input):
        if len(output) == 1:
            if '{base}' in output[0]:
                output = [output[0]] * len(input)
        elif len(output) != len(fit):
            raise ValueError(f'There should be either one output file, '
                             f'or as many output files as input files, '
                             f'or as many output files as echoes. Got '
                             f'{len(output)} output files, {len(input)} '
                             f'input files, and {len(fit)} echoes.')
    if len(output) == 1:
        dir, base, ext = _fileparts(input[0])
        output = output[0]
        if '{n}' in output:
            for n, echo in enumerate(fit):
                out = output.format(dir=dir, sep=os.sep, base=base, ext=ext, n=n)
                io.savef(echo, out, like=input[0])
        else:
            output = output.format(dir=dir, sep=os.sep, base=base, ext=ext)
            io.savef(utils.movedim(fit, 0, -1), output, like=input[0])
    elif len(output) == len(input):
        for i, (inp, out) in enumerate(zip(input, output)):
            dir, base, ext = _fileparts(inp)
            out = out.format(dir=dir, sep=os.sep, base=base, ext=ext, n=i)
            ne = [*io.map(inp).shape, 1][3]
            io.savef(utils.movedim(fit[:ne], 0, -1), out, like=inp)
            fit = fit[ne:]
    else:
        assert len(output) == len(fit)
        dir, base, ext = _fileparts(input[0])
        for n, (echo, out) in enumerate(zip(fit, output)):
            out = out.format(dir=dir, sep=os.sep, base=base, ext=ext, n=n)
            io.savef(echo, out, like=input[0])


def _fileparts(fname):
    """`py.fileparts`, but with `dir='.'` when there is no directory part
    (so that `{dir}{sep}{base}` does not end up at the filesystem root)."""
    dir, base, ext = py.fileparts(fname)
    return dir or os.curdir, base, ext


_opposite = dict(r='l', l='r', a='p', p='a', s='i', i='s')


def get_readout(readout, affine, shape, verbose):
    """
    Convert the provided readout dir from semantic (R/A/S) to index (0/1/2)
        or
    Guess the readout direction as the one with largest number of voxels

    Returns a non-negative index into the (3D) spatial dimensions.
    """
    shape = list(shape)[:3]   # drop echo/channel dimensions
    layout = spatial.affine_to_layout(affine)
    layout = spatial.volume_layout_to_name(layout).lower()
    if readout is None:
        readout = py.argmax(shape)
    else:
        direction = readout.lower()
        readout = None
        for i, l in enumerate(layout):
            if l in direction or _opposite[l] in direction:
                readout = i
                break
        if readout is None:
            raise ParseError(f'Could not match readout direction '
                             f'"{direction}" with layout {layout.upper()}')
    if verbose:
        print(f'Layout: {layout.upper()} | readout direction: {layout[readout].upper()}')
    return readout

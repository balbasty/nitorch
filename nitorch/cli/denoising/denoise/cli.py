from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.core.py import make_list
from .parser import parser, help
from .main import denoise
import sys
import torch


def cli(args=None):
    f"""Command-line interface for `denoise`

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


commands['denoise'] = cli


def setup_device(device, ndevice):
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


def _cli(args):
    """Command-line interface for `nipaint` without exception handling"""
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

    device = setup_device(*options.device)

    options.output = make_list(options.output, len(options.files))
    denoise(*options.files, output=options.output,
            device=device, verbose=options.verbose,
            max_iter=options.max_iter, tol=options.tolerance,
            lam=options.lam, sigma=options.sigma, jtv=options.joint)


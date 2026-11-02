from nitorch.cli.cli import commands
from .main import denoise_mri
from .parser import parse, help, ParseError
import sys
import warnings
import torch


def cli(args=None):
    f"""Command-line interface for `denoise_mri`
    
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


commands['denoise_mri'] = cli


def setup_device(device):
    """Return a torch device, falling back to CPU if CUDA is unavailable."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        warnings.warn('CUDA not available. Switching to CPU.')
        device = 'cpu'
    return torch.device(device)


def _cli(args):
    """Command-line interface for `denoise_mri` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    device = setup_device(options.device)

    denoise_mri(*options.files, lam_scl=options.lam_scl, lr=options.learning_rate,
                max_iter=options.max_iter, tolerance=options.tolerance, verbose=options.verbose,
                device=device, do_write=options.do_write, dir_out=options.dir_out)

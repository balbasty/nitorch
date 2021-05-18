from nitorch.cli.cli import commands
from .main import denoise_mri
from .parser import parse, help, ParseError
import sys


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
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['denoise_mri'] = cli


def _cli(args):
    """Command-line interface for `denoise_mri` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    denoise_mri(*options.files, lam_scl=options.lam_scl, lr=options.learning_rate,
                max_iter=options.max_iter, tolerance=options.tolerance, verbose=options.verbose,
                device=options.device, do_write=options.do_write, dir_out=options.dir_out)

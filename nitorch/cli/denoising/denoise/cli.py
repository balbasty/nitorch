from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.core.py import make_list
from .parser import parser, help
from .main import denoise
import sys


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

    options.output = make_list(options.output, len(options.files))
    denoise(*options.files, output=options.output,
            device=options.device, verbose=options.verbose,
            max_iter=options.max_iter, tol=options.tolerance,
            lam=options.lam, sigma=options.sigma, jtv=options.jtv)


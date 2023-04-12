from nitorch.core.py import make_list
from nitorch.cli.cli import commands
from .main import pool
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `nipool`
    
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
    """Command-line interface for `nipool` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    options.output = make_list(options.output, len(options.files))
    for fname, ofname in zip(options.files, options.output):
        pool(fname, window=options.window, stride=options.stride,
             padding=options.padding or 0, bound=options.bound or 'dct2',
             method=options.method,
             dim=options.dim, output=ofname, device=options.device)


commands['pool'] = cli

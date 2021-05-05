from nitorch.core.py import make_list
from nitorch.cli.cli import commands
from .main import unstack
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `niunstack`
    
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


def _cli(args):
    """Command-line interface for `niunstack` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    options.output = make_list(options.output, len(options.files))
    options.transform = make_list(options.transform, len(options.files))
    for fname, ofname, tfname in zip(options.files, options.output, options.transform):
        unstack(fname, options.dim, ofname, transform=tfname)


commands['unstack'] = cli

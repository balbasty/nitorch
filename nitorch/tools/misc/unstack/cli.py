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

    unstack(options.file, options.dim, options.output,
            transform=options.transform)

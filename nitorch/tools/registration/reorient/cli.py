from nitorch.core.py import make_list
from .main import reorient
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `nireorient`
    
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
    """Command-line interface for `nireorient` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    for fname in options.files:
        reorient(fname, options.layout, options.output,
                 transform=options.transform)

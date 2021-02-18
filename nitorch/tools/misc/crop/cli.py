from nitorch.core.py import make_list
from .main import crop
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `nicrop`
    
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
    """Command-line interface for `nicrop` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    options.transform = make_list(options.transform, len(options.files))
    crop(options.file, size=options.size, center=options.center,
         space=(options.size_space, options.center_space), like=options.like,
         output=options.output, transform=options.transform)

from nitorch.cli.cli import commands
from .main import extract_patches
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `nichunk`
    
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
    """Command-line interface for `nichunk` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    extract_patches(options.file, options.size, options.stride,
                    output=options.output, transform=options.transform)


commands['extract_patches'] = cli

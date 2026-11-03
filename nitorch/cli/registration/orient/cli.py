from nitorch.core.py import make_list
from nitorch.cli.cli import commands
from .main import orient
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `nitorch orient`
    
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


def _cli(args):
    """Command-line interface for `nitorch orient` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    options.output = make_list(options.output, len(options.files))
    options.transform = make_list(options.transform or [None],
                                  len(options.files))
    for fname, ofname, otname in zip(options.files,
                                     options.output,
                                     options.transform):
        orient(fname,
               affine=options.affine,
               layout=options.layout,
               voxel_size=options.voxel_size,
               center=options.center,
               like=options.like,
               output=ofname,
               output_transform=otname)


commands['orient'] = cli

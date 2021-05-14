from nitorch.cli.cli import commands
from nitorch.core.py import make_list
from .main import convert
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `convert`
    
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
    """Command-line interface for `niinfo` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    options.output = make_list(options.output, len(options.files))
    for file, ofile in zip(options.files, options.output):
        convert(file, meta=options.meta, dtype=options.dtype,
                casting=options.cast, format=options.format, output=ofile)


commands['convert'] = cli

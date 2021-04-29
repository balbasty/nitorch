from nitorch.cli.cli import commands
from .main import compose
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `vexp`
    
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

    compose(options)


commands['compose'] = cli

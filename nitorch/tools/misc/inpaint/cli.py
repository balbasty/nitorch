from nitorch.core.py import make_list
from nitorch.tools.cli import commands
from .main import inpaint
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `nipaint`
    
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


commands['inpaint'] = cli


def _cli(args):
    """Command-line interface for `nipaint` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    options.output = make_list(options.output, len(options.files))
    inpaint(*options.files, missing=options.missing, output=options.output,
            device=options.device)



def parse_missing(missing):

    import ast

    if 'x' in missing:
        missing = 'lambda x: (' + missing + ')'
        missing = ast.parse(missing, mode='eval')




from nitorch.core.py import make_list
from nitorch.cli.cli import commands
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
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['inpaint'] = cli


def _cli(args):
    """Command-line interface for `nipaint` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    options.output = make_list(options.output, len(options.files))
    inpaint(*options.files, missing=options.missing, output=options.output,
            device=options.device, verbose=options.verbose,
            max_iter_rls=options.max_rls, max_iter_cg=options.max_cg,
            tol_rls=options.tol_rls, tol_cg=options.tol_cg)




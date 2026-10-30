import sys

from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from .parser import parser, help
from .main import optinorm


def cli(args=None):
    f"""Command-line interface for `optinorm`

    {help}

    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except ParseError as e:
        print(help)
        print('[ERROR]', e)
        return


commands['optinorm'] = cli


def _cli(args):
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help)
        return

    optinorm(
        options.input,
        output=options.output,
        steps=options.steps,
        both=options.both,
        cleanup=options.cleanup,
        loss=options.loss,
        affine=options.affine,
        affine_levels=options.affine_levels,
        nonlin_levels=options.nonlin_levels,
        absolute=options.absolute,
        membrane=options.membrane,
        bending=options.bending,
        lame=options.lame,
        order=options.order,
        clip=options.clip,
        verbose=options.verbose,
        device=options.device,
    )

from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from .parser import parser, help
from ..register.cli import cli as cli_register
import sys


def cli(args=None):
    f"""Command-line interface for `simplereg`

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


commands['simplereg'] = cli


def _cli(args):
    """Command-line interface for `simplereg` without exception handling"""
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help)
        return

    if options.verbose > 3:
        print(options)
        print('')

    return cli_register(build_register_args(options))


def build_register_args(options):
    """Translate parsed `simplereg` options into `nitorch register` arguments.

    Parameters
    ----------
    options : simplereg options (output of `parser.parse`)

    Returns
    -------
    args : list[str]
        Command line for `nitorch register`.
    """
    if not options.moving or not options.fixed:
        raise ParseError('Both a moving (-m) and fixed (-f) image are required.')
    do_affine = options.do_affine and options.affine != 'none'
    do_nonlin = options.do_nonlin and options.nonlin != 'none'
    if not do_affine and not do_nonlin:
        raise ParseError('Cannot disable both the affine (--affine none / '
                         '--no-affine) and nonlinear (--nonlin none / '
                         '--no-nonlin) transforms.')

    args = ['--verbose', str(options.verbose)]
    args += [f'--{options.device[0]}']
    if options.device[1] is not None:
        args += [str(options.device[1])]

    # --- loss + images ------------------------------------------------
    args += ['@loss', options.loss]
    if options.loss in ('lcc', 'lgmm'):
        args += ['-p'] + [str(p) for p in options.kernel]
    if options.symmetric:
        args += ['--symmetric']

    preproc = []
    if options.fwhm is not None:
        preproc += ['-f'] + [str(x) for x in options.fwhm]
    if options.pad is not None:
        preproc += ['-z'] + [str(x) for x in options.pad]
    if options.bound is not None:
        preproc += ['-b', options.bound]
    if options.order is not None:
        preproc += ['-n', str(options.order)]
    if options.rescale is not None:
        preproc += ['-s'] + [str(x) for x in options.rescale]
    if options.discretize:
        preproc += ['-d', str(options.discretize)]

    args += ['@@fix', options.fixed] + preproc
    args += ['@@mov', options.moving] + preproc
    if options.init:
        args += ['-a', options.init]
    args += ['-o', str(options.moved)]
    args += ['-r', str(options.resliced)]

    # --- affine -------------------------------------------------------
    if do_affine:
        args += ['@affine', options.affine]
        args += ['-o', str(options.output_affine)]
        optimizer = options.optimizer
        if not optimizer:
            # Powell for nmi (as nitorch affreg), else register's default
            optimizer = 'powell' if options.loss == 'nmi' else 'unset'
        args += ['@@optim', optimizer,
                 '-n', str(options.max_iter),
                 '-t', str(options.tolerance),
                 '-s', str(options.search)]

    # --- nonlinear ----------------------------------------------------
    if do_nonlin:
        args += ['@nonlin', options.nonlin]
        if options.penalty is not None:
            args += [str(options.penalty)]
        args += ['-o', str(options.output_warp)]
        for key, flag in (('absolute', '-a'), ('membrane', '-m'),
                          ('bending', '-b')):
            if getattr(options, key) is not None:
                args += [flag, str(getattr(options, key))]
        if options.lame is not None:
            args += ['-l'] + [str(x) for x in options.lame]
        args += ['@@optim', options.nonlin_optimizer or 'unset']
        if options.nonlin_iter is not None:
            args += ['-n', str(options.nonlin_iter)]

    # --- joint optimization -------------------------------------------
    args += ['@optim', 'interleaved']
    if options.outer_iter is not None:
        args += ['-n', str(options.outer_iter)]

    args += ['@pyramid', '--levels']
    levels = options.pyramid or [0]
    for level in levels:
        if isinstance(level, range):
            args += [str(l) for l in level]
        else:
            args += [str(level)]

    return args

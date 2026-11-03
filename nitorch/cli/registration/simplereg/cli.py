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
        # `register` reports its own failures by returning 1
        return _cli(args)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1


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


_bound_aliases = {'nearest': 'replicate', 'border': 'replicate'}


def _path_token(value):
    """Output option -> register token: True/False -> 'true'/'false'
    (register's default path / no output), else the path itself."""
    if value is True or value is False:
        return str(value).lower()
    return str(value)


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
        # nitorch.core.bounds' canonical name for nearest/border
        preproc += ['-b', _bound_aliases.get(options.bound, options.bound)]
    if options.order is not None:
        preproc += ['-n', str(options.order)]
    if options.rescale is not None:
        preproc += ['-s'] + [str(x) for x in options.rescale]
    if options.discretize:
        preproc += ['-d', str(options.discretize)]

    # simplereg only exposes the moving image's outputs (-x/-r): disable
    # the fixed image's `.moved` output, which `register` writes by default
    args += ['@@fix', options.fixed] + preproc + ['-o', 'false']
    args += ['@@mov', options.moving] + preproc
    if options.init:
        args += ['-a', options.init]
    args += ['-o', str(options.moved)]
    args += ['-r', str(options.resliced)]

    # --- affine -------------------------------------------------------
    if do_affine:
        args += ['@affine', options.affine]
        args += ['-o', _path_token(options.output_affine)]
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
        args += ['-o', _path_token(options.output_warp)]
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

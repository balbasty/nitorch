from nitorch.cli.cli import commands
from .parser import parser, help, ParseError
from nitorch.tools.qmri import estatics, ESTATICSOptions, io as qio
from nitorch import io
from nitorch.core import py
import torch
import sys
import os


def cli(args=None):
    f"""Command-line interface for `estatics`

    {help[1]}

    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except ParseError as e:
        print(help[1])
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['estatics'] = cli


def _cli(args):
    """Command-line interface for `estatics` without exception handling"""
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help[options.help])
        return

    if options.verbose > 3:
        print(options)
        print('')

    _main(options)


def _main(options):
    if isinstance(options.gpu, str):
        device = torch.device(options.gpu)
    else:
        assert isinstance(options.gpu, int)
        device = torch.device(f'cuda:{options.gpu}')
    if not torch.cuda.is_available():
        device = 'cpu'

    # prepare options
    estatics_opt = ESTATICSOptions()
    estatics_opt.likelihood = options.likelihood
    estatics_opt.verbose = options.verbose >= 1
    estatics_opt.plot = options.verbose >= 2
    estatics_opt.recon.space = options.space
    estatics_opt.backend.device = device
    estatics_opt.optim.nb_levels = options.levels
    estatics_opt.optim.max_iter_rls = options.iter
    estatics_opt.optim.tolerance = options.tol
    estatics_opt.regularization.norm = options.regularization
    estatics_opt.regularization.factor = [*options.lam_intercept, options.lam_decay]
    estatics_opt.distortion.enable = options.meetup
    estatics_opt.distortion.bending = options.lam_meetup
    estatics_opt.preproc.register = options.register

    # prepare files
    contrasts = []
    for i, c in enumerate(options.contrast):

        # read meta-parameters
        meta = {}
        if c.te:
            te, unit = c.te, ''
            if isinstance(te[-1], str):
                *te, unit = te
            if unit:
                if unit == 'ms':
                    te = [t * 1e-3 for t in te]
                elif unit not in ('s', 'sec'):
                    raise ValueError(f'TE unit: {unit}')
            if c.echo_spacing:
                delta, *unit = c.echo_spacing
                unit = unit[0] if unit else ''
                if unit == 'ms':
                    delta = delta * 1e-3
                elif unit not in ('s', 'sec'):
                    raise ValueError(f'echo spacing unit: {unit}')
                ne = len(c.echoes)
                te = [te[0] + e*delta for e in range(ne)]
            meta['te'] = te

        # map volumes
        contrasts.append(qio.GradientEchoMulti.from_fnames(c.echoes, **meta))

    # run algorithm
    [te0, r2s, *b0] = estatics(contrasts, estatics_opt)

    # write results
    odir0 = options.odir
    for i, te1 in enumerate(te0):
        ifname = contrasts[i].echo(0).volume.fname
        odir, obase, oext = py.fileparts(ifname)
        odir = odir0 or odir
        obase = obase + '_TE0'
        ofname = os.path.join(odir, obase + oext)
        io.savef(te1.volume, ofname, affine=te1.affine, like=ifname, te=0, dtype='float32')
    ifname = contrasts[0].echo(0).volume.fname
    odir, obase, oext = py.fileparts(ifname)
    odir = odir0 or odir
    io.savef(r2s.volume, os.path.join(odir, 'R2star' + oext), affine=r2s.affine, dtype='float32')
    for i, b0 in enumerate(b0):
        ifname = contrasts[i].echo(0).volume.fname
        odir, obase, oext = py.fileparts(ifname)
        odir = odir0 or odir
        obase = obase + '_B0'
        ofname = os.path.join(odir, obase + oext)
        io.savef(te1.volume, ofname, affine=te1.affine, like=ifname, te=0, dtype='float32')




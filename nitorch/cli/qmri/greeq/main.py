from nitorch.cli.cli import commands
from .parser import parser, help, ParseError
from nitorch.tools.qmri import greeq, GREEQOptions, io as qio
from nitorch import io
from nitorch.core import py
import torch
import math as pymath
import sys
import os


def cli(args=None):
    f"""Command-line interface for `greeq`

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


commands['greeq'] = cli


def _cli(args):
    """Command-line interface for `greeq` without exception handling"""
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

    return _main(options)


def _main(options):
    if isinstance(options.gpu, str):
        device = torch.device(options.gpu)
    else:
        assert isinstance(options.gpu, int)
        device = torch.device(f'cuda:{options.gpu}')
    if not torch.cuda.is_available():
        device = 'cpu'

    # prepare options
    greeq_opt = GREEQOptions()
    greeq_opt.likelihood = options.likelihood
    greeq_opt.verbose = options.verbose
    # greeq_opt.plot = options.verbose >= 2
    greeq_opt.recon.space = options.space
    if isinstance(options.space, str) and options.space != 'mean':
        for c, contrast in enumerate(options.contrast):
            if contrast.name == options.space:
                greeq_opt.recon.space = c
                break
    greeq_opt.recon.crop = options.crop
    greeq_opt.backend.device = device
    greeq_opt.uncertainty = options.uncertainty
    greeq_opt.optim.nb_levels = options.levels
    greeq_opt.optim.max_iter_rls = options.iter
    greeq_opt.optim.tolerance = options.tol
    greeq_opt.optim.tolerance_cg = options.tol
    solver, *subiter_max = options.solver
    if subiter_max:
        subiter_max = int(subiter_max[0])
    elif solver == 'fmg':
        subiter_max = 2
    else:  # cg
        subiter_max = 32
    greeq_opt.optim.solver = solver
    greeq_opt.optim.max_iter_cg = subiter_max
    greeq_opt.penalty.norm = options.regularization
    greeq_opt.penalty.factor = {
        'pd': options.lam_pd if options.lam_pd else options.lam,
        'r1': options.lam_t1 if options.lam_t1 else options.lam,
        'r2s': options.lam_t2 if options.lam_t2 else options.lam,
        'mt': options.lam_mt if options.lam_mt else options.lam,
    }
    # greeq_opt.distortion.enable = options.meetup
    # greeq_opt.distortion.bending = options.lam_meetup
    greeq_opt.preproc.register = options.register

    # prepare files
    contrasts = [None] * len(options.contrast)
    transmits = [None] * len(options.contrast)
    receives = [None] * len(options.contrast)
    b0s = [None] * len(options.contrast)
    for i, c in enumerate(options.contrast):

        # read meta-parameters
        meta = {}
        if c.fa:
            fa, *unit = c.fa
            unit = unit[0] if unit else ''
            if unit and unit == 'rad':
                fa = fa * 180 / pymath.pi
            meta['fa'] = fa
        if c.tr:
            tr, *unit = c.tr
            unit = unit[0] if unit else ''
            if unit:
                if unit == 'ms':
                    tr = tr * 1e-3
                elif unit not in ('s', 'sec'):
                    raise ValueError(f'TR unit: {unit}')
            meta['tr'] = tr
        if c.mt is not None:
            meta['mt'] = c.mt
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
                ne = sum(io.map(f).unsqueeze(-1).shape[3] for f in c.echoes)
                te = [te[0] + e*delta for e in range(ne)]
            meta['te'] = te

        # map volumes
        cc = qio.GradientEchoMulti.from_fname(c.echoes, **meta)
        contrasts[i] = cc

        if c.transmit:
            files = c.transmit
            meta = {}
            if files[-1] in ('%', 'pct', 'p.u.', 'a.u.'):
                *files, unit = files
                meta['unit'] = unit
            files, *mag = files
            if mag:
                meta['magnitude'] = mag[0]
            transmits[i] = qio.PrecomputedFieldMap(files, **meta)

        if c.receive:
            files = c.receive
            meta = {}
            if files[-1] in ('%', 'pct', 'p.u.', 'a.u.'):
                *files, unit = files
                meta['unit'] = unit
            files, *mag = files
            if mag:
                meta['magnitude'] = mag[0]
            receives[i] = qio.PrecomputedFieldMap(files, **meta)

        if c.b0:
            files = c.b0
            meta = {}
            if files[-1].lower() in ('hz', '1/s', 'vox'):
                *files, unit = files
                meta['unit'] = unit
            files, *mag = files
            if mag:
                meta['magnitude'] = mag[0]
            b0s[i] = qio.PrecomputedFieldMap(files, **meta)

    # shared fieldmaps

    if options.transmit:
        files = options.transmit
        meta = {}
        if files[-1] in ('%', 'pct', 'p.u.', 'a.u.'):
            *files, unit = files
            meta['unit'] = unit
        files, *mag = files
        if mag:
            meta['magnitude'] = mag[0]
        for i in range(len(transmits)):
            if transmits[i] is None:
                transmits[i] = qio.PrecomputedFieldMap(files, **meta)

    if options.receive:
        files = options.receive
        meta = {}
        if files[-1] in ('%', 'pct', 'p.u.', 'a.u.'):
            *files, unit = files
            meta['unit'] = unit
        files, *mag = files
        if mag:
            meta['magnitude'] = mag[0]
        for i in len(receives):
            if receives[i] is None:
                receives[i] = qio.PrecomputedFieldMap(files, **meta)

    if options.b0:
        files = options.b0
        meta = {}
        if files[-1] in ('%', 'pct', 'p.u.', 'a.u.'):
            *files, unit = files
            meta['unit'] = unit
        files, *mag = files
        if mag:
            meta['magnitude'] = mag[0]
        for i in len(b0s):
            if b0s[i] is None:
                b0s[i] = qio.PrecomputedFieldMap(files, **meta)

    # run algorithm
    [pd, r1, r2s, *mt] = greeq(contrasts, transmits, receives, greeq_opt)

    # write results
    odir0 = options.odir
    ifname = contrasts[0].echo(0).volume.fname
    odir, obase, oext = py.fileparts(ifname)
    odir = odir0 or odir
    io.savef(pd.volume, os.path.join(odir, 'PD' + oext), affine=pd.affine, dtype='float32')
    io.savef(r1.volume, os.path.join(odir, 'R1' + oext), affine=r1.affine, dtype='float32')
    io.savef(r2s.volume, os.path.join(odir, 'R2star' + oext), affine=r2s.affine, dtype='float32')
    if options.uncertainty:
        io.savef(pd.uncertainty, os.path.join(odir, 'log_PD_uncertainty' + oext), affine=pd.affine, dtype='float32')
        io.savef(r1.uncertainty, os.path.join(odir, 'log_R1_uncertainty' + oext), affine=r1.affine, dtype='float32')
        io.savef(r2s.uncertainty, os.path.join(odir, 'log_R2star_uncertainty' + oext), affine=r2s.affine, dtype='float32')
    if mt:
        mt = mt[0]
        io.savef(mt.volume, os.path.join(odir, 'MTsat' + oext), affine=mt.affine, dtype='float32')
        if options.uncertainty:
            io.savef(mt.uncertainty, os.path.join(odir, 'logit_MTsat_uncertainty' + oext), affine=mt.affine, dtype='float32')

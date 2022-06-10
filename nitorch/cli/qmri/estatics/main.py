from nitorch.cli.cli import commands
from .parser import parser, help, ParseError
from nitorch.tools.qmri import estatics, ESTATICSOptions, io as qio
from nitorch.tools.qmri.relax.utils import smart_pull, pull1d
from nitorch.tools.qmri.param import DenseDistortion
from nitorch import io, spatial
from nitorch.core import py
import torch
import sys
import os


def cli(args=None):
    f"""Command-line interface for `estatics`

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


commands['estatics'] = cli


def _cli(args):
    """Command-line interface for `estatics` without exception handling"""
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
    if isinstance(options.space, str) and  options.space != 'mean':
        for c, contrast in enumerate(options.contrast):
            if contrast.name == options.space:
                estatics_opt.recon.space = c
                break
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
    distortion = []
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
                ne = sum(io.map(f).unsqueeze(-1).shape[3] for f in c.echoes)
                te = [te[0] + e*delta for e in range(ne)]
            meta['te'] = te

        # map volumes
        contrasts.append(qio.GradientEchoMulti.from_fname(c.echoes, **meta))

        if c.readout:
            layout = spatial.affine_to_layout(contrasts[-1].affine)
            layout = spatial.volume_layout_to_name(layout)
            readout = None
            for j, l in enumerate(layout):
                if l.lower() in c.readout.lower():
                    readout = j - 3
            contrasts[-1].readout = readout

        if c.b0:
            bw = c.bandwidth
            b0, *unit = c.b0
            unit = unit[-1] if unit else 'vx'
            fb0 = b0.map(b0)
            b0 = fb0.fdata(device=device)
            b0 = spatial.reslice(b0, fb0.affine, contrasts[-1][0].affine,
                                 contrasts[-1][0].shape)
            if unit.lower() == 'hz':
                if not bw:
                    raise ValueError('Bandwidth required to convert fieldmap'
                                     'from Hz to voxel')
                b0 /= bw
            b0 = DenseDistortion(b0)
            distortion.append(b0)
        else:
            distortion.append(None)

    # run algorithm
    [te0, r2s, *b0] = estatics(contrasts, distortion, opt=estatics_opt)

    # write results

    # --- intercepts ---
    odir0 = options.odir
    for i, te1 in enumerate(te0):
        ifname = contrasts[i].echo(0).volume.fname
        odir, obase, oext = py.fileparts(ifname)
        odir = odir0 or odir
        obase = obase + '_TE0'
        ofname = os.path.join(odir, obase + oext)
        io.savef(te1.volume, ofname, affine=te1.affine, like=ifname, te=0, dtype='float32')

    # --- decay ---
    ifname = contrasts[0].echo(0).volume.fname
    odir, obase, oext = py.fileparts(ifname)
    odir = odir0 or odir
    io.savef(r2s.volume, os.path.join(odir, 'R2star' + oext), affine=r2s.affine, dtype='float32')

    # --- fieldmap + undistorted ---
    if b0:
        b0 = b0[0]
        for i, b01 in enumerate(b0):
            ifname = contrasts[i].echo(0).volume.fname
            odir, obase, oext = py.fileparts(ifname)
            odir = odir0 or odir
            obase = obase + '_B0'
            ofname = os.path.join(odir, obase + oext)
            io.savef(b01.volume, ofname, affine=b01.affine, like=ifname, te=0, dtype='float32')
        for i, (c, b) in enumerate(zip(contrasts, b0)):
            readout = c.readout
            grid_up, grid_down, jac_up, jac_down = b.exp2(
                add_identity=True, jacobian=True)
            for j, e in enumerate(c):
                blip = e.blip or (2*(j % 2) - 1)
                grid_blip = grid_down if blip > 0 else grid_up  # inverse of
                jac_blip = jac_down if blip > 0 else jac_up     # forward model
                ifname = e.volume.fname
                odir, obase, oext = py.fileparts(ifname)
                odir = odir0 or odir
                obase = obase + '_unwrapped'
                ofname = os.path.join(odir, obase + oext)
                d = e.fdata(device=device)
                d, _ = pull1d(d, grid_blip, readout)
                d *= jac_blip
                io.savef(d, ofname, affine=e.affine, like=ifname)
                del d
            del grid_up, grid_down, jac_up, jac_down
    if options.register:
        for i, c in enumerate(contrasts):
            for j, e in enumerate(c):
                ifname = e.volume.fname
                odir, obase, oext = py.fileparts(ifname)
                odir = odir0 or odir
                obase = obase + '_registered'
                ofname = os.path.join(odir, obase + oext)
                io.save(e.volume, ofname, affine=e.affine)


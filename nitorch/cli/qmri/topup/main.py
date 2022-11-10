from nitorch.cli.cli import commands
from .parser import parser_fit, parser_apply, help, ParseError
from nitorch.tools.registration.topup import topup_fit, topup_apply
from nitorch.tools.registration.losses import MSE, NCC, LNCC, LGMMH, GMMH
from nitorch.tools.img_statistics import estimate_noise
from nitorch import io, spatial
from nitorch.core import py, utils
import torch
import sys
import os
import math as pymath


def cli(args=None):
    f"""Command-line interface for `topup`

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


commands['topup'] = cli


def _cli(args):
    """Command-line interface for `topup` without exception handling"""
    args = args or sys.argv[1:]

    if args[0] == '-h':
        print(help)
        return

    mode, *args = args
    if mode == 'fit':
        options = parser_fit.parse(args)
    else:
        options = parser_apply.parse(args)
    if not options:
        return
    if options.help:
        print(help)
        return

    if options.verbose > 3:
        print(options)
        print('')

    if mode == 'fit':
        main_fit(options)
    elif mode == 'apply':
        main_apply(options)
    else:
        raise ParseError('topup subcommand should be fit or apply but got ',
                         mode)


def downsample(x, aff_in, vx_out):
    """
    Downsample an image (by an integer factor) to approximately
    match a target voxel size
    """
    vx_in = spatial.voxel_size(aff_in)
    dim = len(vx_in)
    vx_out = utils.make_vector(vx_out, dim)
    factor = (vx_out / vx_in).clamp_min(1).floor().long()
    if (factor == 1).all():
        return x, aff_in
    factor = factor.tolist()
    x, aff_out = spatial.pool(dim, x, factor, affine=aff_in)
    return x, aff_out


def upsample_vel(v, aff_in, aff_out, shape, readout):
    """
    Upsample a 1D displacement field (by a potentially non-integer factor) using
    second order spline interpolation.
    Scales the displacement field appropriately to take into account the
    change of voxel size.
    """
    if v.shape == shape:
        return v
    vx_down = spatial.voxel_size(aff_in)
    vx_down = vx_down[readout]
    vx_up = spatial.voxel_size(aff_out)[readout]
    factor = vx_down / vx_up
    v = spatial.reslice(v, aff_in, aff_out, shape,
                        bound='dct2', interpolation=2, prefilter=True,
                        extrapolate=True)
    v *= factor
    return v


def get_device(opt):
    if isinstance(opt, str):
        device = torch.device(opt)
    else:
        assert isinstance(opt, int)
        device = torch.device(f'cuda:{opt}')
    if not torch.cuda.is_available():
        device = 'cpu'
    return device


def get_readout(readout, affine, shape):
    """
    Convert the provided readout dir from semantic (R/A/S) to index (0/1/2)
        or
    Guess the readout direction as the one with largest number of voxels
    """
    dim = len(shape)
    layout = spatial.affine_to_layout(affine)
    layout = spatial.volume_layout_to_name(layout).lower()
    if readout is None:
        readout = py.argmax(shape)
    else:
        readout = readout.lower()
        for i, l in enumerate(layout):
            if l in readout:
                readout = i
                break
    print(f'Layout: {layout.upper()} | readout direction: {layout[readout].upper()}')
    if readout > 0:
        readout = readout - dim
    return readout


def ras_to_layout(x, affine):
    layout = spatial.affine_to_layout(affine)
    ras_to_layout = layout[..., 0]
    return [x[i] for i in ras_to_layout]


def get_kernel(kernel, affine, shape, level):
    """Convert the provided kernel size (RAS mm or pct) to native voxels"""
    dim = affine.shape[-1] - 1
    kernel = py.make_list(kernel)
    unit = 'pct'
    if isinstance(kernel[-1], str):
        *kernel, unit = kernel
    kernel = py.make_list(kernel, dim)
    unit = unit.lower()
    if unit[0] == 'v':  # voxels
        kernel = [p / 2**level for p in kernel]
    elif unit in ('m', 'mm', 'cm', 'um'):  # assume RAS orientation
        factor = (1e-3 if unit == 'um' else
                  1e1 if unit == 'cm' else
                  1e3 if unit == 'm' else
                  1)
        affine_ras = spatial.affine_reorient(affine, layout='RAS')
        vx_ras = spatial.voxel_size(affine_ras).tolist()
        kernel = [factor * p / v for p, v in zip(kernel, vx_ras)]
        kernel = ras_to_layout(kernel, affine)
    elif unit[0] in 'p%':    # percentage of shape
        kernel = [0.01 * p * s for p, s in zip(kernel, shape)]
    else:
        raise ValueError('Unknown patch unit:', unit)

    # ensure patch size is an integer >= 2 (else, no gradients)
    kernel = list(map(lambda x: max(int(pymath.ceil(x)), 2), kernel))
    return kernel


def main_fit(options):
    """
    Estimate a displacement field from opposite polarity  images
    """
    device = get_device(options.gpu)

    # map input files
    f0 = io.map(options.pos_file)
    f1 = io.map(options.neg_file)
    dim = f0.affine.shape[-1] - 1

    # map mask
    fm = None
    if options.mask:
        fm = io.map(options.mask)

    # detect readout direction
    readout = get_readout(options.readout, f0.affine, f0.shape[-dim:])

    # detect penalty
    penalty_type = 'bending'
    penalties = options.penalty
    if penalties and isinstance(penalties[-1], str):
        *penalties, penalty_type = penalties
    if not penalties:
        penalties = [1]
    if penalty_type[0] == 'b':
        penalty_type = 'bending'
    elif penalty_type[0] == 'm':
        penalty_type = 'membrane'
    else:
        raise ValueError('Unknown penalty type', penalty_type)

    downs = options.downsample
    max_iter = options.max_iter
    tolerance = options.tolerance
    nb_levels = max(len(penalties), len(max_iter), len(tolerance), len(downs))
    penalties = py.make_list(penalties, nb_levels)
    tolerance = py.make_list(tolerance, nb_levels)
    max_iter = py.make_list(max_iter, nb_levels)
    downs = py.make_list(downs, nb_levels)

    # load
    d00 = f0.fdata(device='cpu')
    d11 = f1.fdata(device='cpu')
    dmask = fm.fdata(device='cpu') if fm else None

    # fit
    vel = mask = None
    aff = last_aff = f0.affine
    last_dwn = None
    for penalty, n, tol, dwn in zip(penalties, max_iter, tolerance, downs):
        if dwn != last_dwn:
            d0, aff = downsample(d00.to(device), f0.affine, dwn)
            d1, _ = downsample(d11.to(device), f1.affine, dwn)
            vx = spatial.voxel_size(aff)
            if vel is not None:
                vel = upsample_vel(vel, last_aff, aff, d0.shape[-dim:], readout)
            last_aff = aff
            if fm:
                mask, _ = downsample(dmask.to(device), f1.affine, dwn)
        last_dwn = dwn
        scl = py.prod(d00.shape) / py.prod(d0.shape)
        penalty = penalty * scl

        kernel = get_kernel(options.kernel, aff, d0.shape[-dim:], dwn)

        # prepare loss
        if options.loss == 'mse':
            prm0, _ = estimate_noise(d0)
            prm1, _ = estimate_noise(d1)
            sd = ((prm0['sd'].log() + prm1['sd'].log())/2).exp()
            print(sd.item())
            loss = MSE(lam=1/(sd*sd), dim=dim)
        elif options.loss == 'lncc':
            loss = LNCC(dim=dim, patch=kernel)
        elif options.loss == 'lgmm':
            if options.bins == 1:
                loss = LNCC(dim=dim, patch=kernel)
            else:
                loss = LGMMH(dim=dim, patch=kernel, bins=options.bins)
        elif options.loss == 'gmm':
            if options.bins == 1:
                loss = NCC(dim=dim)
            else:
                loss = GMMH(dim=dim, bins=options.bins)
        else:
            loss = NCC(dim=dim)

        # fit
        vel = topup_fit(d0, d1, loss=loss, dim=readout, vx=vx, ndim=dim,
                        model=('svf' if options.diffeo else 'smalldef'),
                        lam=penalty, penalty=penalty_type, vel=vel,
                        modulation=options.modulation, max_iter=n,
                        tolerance=tol, verbose=options.verbose, mask=mask)

    del d0, d1, d00, d11

    # upsample
    vel = upsample_vel(vel, aff, f0.affine, f0.shape[-dim:], readout)

    # convert to Hz
    if options.bandwidth != 1:
        vel *= options.bandwidth

    # save
    dir, base, ext = py.fileparts(options.pos_file)
    fname = options.output
    fname = fname.format(dir=dir or '.', sep=os.sep, base=base, ext=ext)
    io.savef(vel, fname, like=options.pos_file, dtype='float32')


def _deform1d(img, phi):
    img = img.unsqueeze(-2)
    phi = phi.unsqueeze(-1)
    wrp = spatial.grid_pull(img, phi, bound='dct2', extrapolate=True)
    wrp = wrp.squeeze(-2)
    return wrp


def main_apply(options):
    """
    Unwarp distorted images using a pre-computed 1d displacement field.
    """
    device = get_device(options.gpu)

    # detect readout direction
    if options.file_pos:
        f0 = io.map(options.file_pos[0])
    else:
        f0 = io.map(options.file_neg[0])
    dim = f0.affine.shape[-1] - 1
    readout = get_readout(options.readout, f0.affine, f0.shape[-dim:])

    def do_apply(fnames, phi, jac):
        """Correct files with a given polarity"""
        for fname in fnames:
            dir, base, ext = py.fileparts(fname)
            ofname = options.output
            ofname = ofname.format(dir=dir or '.', sep=os.sep, base=base,
                                   ext=ext)
            if options.verbose:
                print(f'unwarp {fname} \n'
                      f'    -> {ofname}')

            f = io.map(fname)
            d = f.fdata(device=device)
            d = utils.movedim(d, readout, -1)
            d = _deform1d(d, phi)
            if jac is not None:
                d *= jac
            d = utils.movedim(d, -1, readout)

            io.savef(d, ofname, like=fname)

    # load and apply
    vel = io.loadf(options.dist_file, device=device)
    vel = utils.movedim(vel, readout, -1)

    if options.file_pos:
        phi = vel / options.bandwidth
        if options.diffeo:
            phi, *jac = spatial.exp1d_forward(phi, bound='dct2',
                                              jacobian=options.modulation)
            jac = jac[0] if jac else None
        else:
            jac = None
            if options.modulation:
                jac = spatial.diff1d(phi, dim=readout, bound='dct2', side='c')
                jac += 1
        phi = spatial.add_identity_grid_(phi.unsqueeze(-1)).squeeze(-1)

        do_apply(options.file_pos, phi, jac)

    if options.file_neg:
        phi = vel / (-options.bandwidth)
        if options.diffeo:
            phi, *jac = spatial.exp1d_forward(phi, bound='dct2',
                                              jacobian=options.modulation)
            jac = jac[0] if jac else None
        else:
            jac = None
            if options.modulation:
                jac = spatial.diff1d(phi, dim=readout, bound='dct2', side='c')
                jac += 1
        phi = spatial.add_identity_grid_(phi.unsqueeze(-1)).squeeze(-1)

        do_apply(options.file_neg, phi, jac)

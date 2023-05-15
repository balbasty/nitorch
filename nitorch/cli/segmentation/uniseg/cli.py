import warnings
from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from .parser import parser, help
from nitorch.tools.uniseg import uniseg, path_spm_prior, get_data
from nitorch import io, spatial
from nitorch.core import utils, py, linalg
import torch
import sys
import os
import math as pymath


def cli(args=None):
    f"""Command-line interface for `uniseg`

    {help}

    """
    try:
        cli_parse(args)
    except ParseError as e:
        print(help[1])
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['uniseg'] = cli


def cli_parse(args):
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
    main(options)


def setup_device(device, ndevice):
    if device == 'gpu' and not torch.cuda.is_available():
        warnings.warn('CUDA not available. Switching to CPU.')
        device, ndevice = 'cpu', None
    if device == 'cpu':
        device = torch.device('cpu')
        if ndevice:
            torch.set_num_threads(ndevice)
    else:
        assert device == 'gpu'
        if ndevice is not None:
            device = torch.device(f'cuda:{ndevice}')
        else:
            device = torch.device('cuda')
    return device


def main(options):

    if options.plot > 0:
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except ImportError:
            options.plot = 0

    wishart_mode = options.wish
    if wishart_mode and not options.tpm:
        wishart_mode = 'preproc8'

    if options.clusters and len(options.clusters) == 1:
        options.clusters = options.clusters[0]

    z, _, prm = uniseg(
        options.input, options.mask,
        device=setup_device(*options.device),
        nb_classes=options.clusters,
        prior=options.tpm,
        do_bias=options.bias,
        do_warp=options.warp,
        do_mixing=options.mix,
        do_mrf=options.mrf,
        wishart=wishart_mode,
        cleanup=options.clean,
        spacing=options.spacing,
        flexi=options.flexi,
        lam_prior=options.lam_prior,
        lam_bias=options.lam_bias,
        lam_warp=options.lam_warp,
        lam_mixing=options.lam_mix,
        lam_mrf=options.lam_mrf,
        lam_wishart=options.lam_wish,
        max_iter=options.iter,
        tol=options.tolerance,
        verbose=options.verbose,
        plot=options.plot,
        return_parameters=True
    )

    write_outputs(z, prm, options)


def get_format_dict(fname, dir0):
    dir, base, ext = py.fileparts(fname)
    dir = dir0 or dir or '.'
    return dict(dir=dir, sep=os.sep, base=base, ext=ext)


def write_outputs(z, prm, options):

    # prepare filenames
    ref_native = options.input[0]
    ref_mni = options.tpm[0] if options.tpm else path_spm_prior()
    format_dict = get_format_dict(ref_native, options.output)

    # move channels to back
    backend = utils.backend(z)
    if (options.nobias_nat or options.nobias_mni or options.nobias_wrp
            or options.all_nat or options.all_mni or options.all_wrp):
        dat, _, affine = get_data(options.input, options.mask, None, 3, **backend)

    # --- native space -------------------------------------------------

    if options.prob_nat or (options.all_nat and options.prob_nat is not False):
        fname = options.prob_nat or '{dir}{sep}{base}.prob.nat{ext}'
        fname = fname.format(**format_dict)
        if options.verbose > 0:
            print('prob.nat     ->', fname)
        io.savef(torch.movedim(z, 0, -1), fname, like=ref_native, dtype='float32')

    if options.labels_nat or (options.all_nat and options.labels_nat is not False):
        fname = options.labels_nat or '{dir}{sep}{base}.labels.nat{ext}'
        fname = fname.format(**format_dict)
        if options.verbose > 0:
            print('labels.nat   ->', fname)
        io.save(z.argmax(0), fname, like=ref_native, dtype='int16')

    if (options.bias_nat or options.all_nat) and options.bias:
        bias = prm['bias']
        fname = options.bias_nat or '{dir}{sep}{base}.bias.nat{ext}'
        if len(options.input) == 1:
            fname = fname.format(**format_dict)
            if options.verbose > 0:
                print('bias.nat     ->', fname)
            io.savef(torch.movedim(bias, 0, -1), fname, like=ref_native, dtype='float32')
        else:
            for c, (bias1, ref1) in enumerate(zip(bias, options.input)):
                format_dict1 = get_format_dict(ref1, options.output)
                fname = fname.format(**format_dict1)
                if options.verbose > 0:
                    print(f'bias.nat.{c+1}   ->', fname)
                io.savef(bias1, fname, like=ref1, dtype='float32')
        del bias

    if (options.nobias_nat or options.all_nat) and options.bias:
        nobias = dat * prm['bias']
        fname = options.nobias_nat or '{dir}{sep}{base}.nobias.nat{ext}'
        if len(options.input) == 1:
            fname = fname.format(**format_dict)
            if options.verbose > 0:
                print('nobias.nat   ->', fname)
            io.savef(torch.movedim(nobias, 0, -1), fname, like=ref_native)
        else:
            for c, (nobias1, ref1) in enumerate(zip(bias, options.input)):
                format_dict1 = get_format_dict(ref1, options.output)
                fname = fname.format(**format_dict1)
                if options.verbose > 0:
                    print(f'nobias.nat.{c+1} ->', fname)
                io.savef(nobias1, fname, like=ref1)
        del nobias

    if (options.warp_nat or options.all_nat) and options.warp:
        warp = prm['warp']
        fname = options.warp_nat or '{dir}{sep}{base}.warp.nat{ext}'
        fname = fname.format(**format_dict)
        if options.verbose > 0:
            print('warp.nat     ->', fname)
        io.savef(warp, fname, like=ref_native, dtype='float32')

    if options.prior_nat or options.all_nat:
        fname = options.prior_nat or '{dir}{sep}{base}.prior.nat{ext}'
        fname = fname.format(**format_dict)
        if options.verbose > 0:
            print('prior.nat     ->', fname)
        io.savef(torch.movedim(prm['warped'], 0, -1), fname, like=ref_native, dtype='float32')

    # --- MNI space ----------------------------------------------------
    if options.tpm is False:
        # No template -> no MNI space
        return

    fref = io.map(ref_mni)
    mni_affine, mni_shape = fref.affine, fref.shape[:3]
    dat_affine = io.map(ref_native).affine
    mni_affine = mni_affine.to(**backend)
    dat_affine = dat_affine.to(**backend)
    prm_affine = prm['affine'].to(**backend)
    dat_affine = prm_affine @ dat_affine
    if options.mni_vx:
        vx = spatial.voxel_size(mni_affine)
        scl = vx / options.mni_vx
        mni_affine, mni_shape = spatial.affine_resize(
            mni_affine, mni_shape, scl, anchor='f')

    if options.prob_mni or options.labels_mni or options.all_mni:
        z_mni = spatial.reslice(z, dat_affine, mni_affine, mni_shape)
        if options.prob_mni:
            fname = options.prob_mni or '{dir}{sep}{base}.prob.mni{ext}'
            fname = fname.format(**format_dict)
            if options.verbose > 0:
                print('prob.mni     ->', fname)
            io.savef(torch.movedim(z_mni, 0, -1), fname, like=ref_native,
                     affine=mni_affine, dtype='float32')
        if options.labels_mni:
            fname = options.labels_mni or '{dir}{sep}{base}.labels.mni{ext}'
            fname = fname.format(**format_dict)
            if options.verbose > 0:
                print('labels.mni   ->', fname)
            io.save(z_mni.argmax(0), fname, like=ref_native,
                    affine=mni_affine, dtype='int16')
        del z_mni

    if options.prior_mni or options.all_mni:
        mu_mni = spatial.reslice(prm['warped'], dat_affine, mni_affine, mni_shape)
        fname = options.prob_mni or '{dir}{sep}{base}.prior.mni{ext}'
        fname = fname.format(**format_dict)
        if options.verbose > 0:
            print('prior.mni     ->', fname)
        io.savef(torch.movedim(mu_mni, 0, -1), fname, like=ref_native,
                 affine=mni_affine, dtype='float32')
        del mu_mni

    if options.bias and (options.bias_mni or options.nobias_mni or options.all_mni):
        bias = spatial.reslice(prm['bias'], dat_affine, mni_affine, mni_shape,
                               interpolation=3, prefilter=False, bound='dct2')

        if options.bias_mni or options.all_mni:
            fname = options.bias_mni or '{dir}{sep}{base}.bias.mni{ext}'
            if len(options.input) == 1:
                fname = fname.format(**format_dict)
                if options.verbose > 0:
                    print('bias.mni     ->', fname)
                io.savef(torch.movedim(bias, 0, -1), fname, like=ref_native,
                         affine=mni_affine, dtype='float32')
            else:
                for c, (bias1, ref1) in enumerate(zip(bias, options.input)):
                    format_dict1 = get_format_dict(ref1, options.output)
                    fname = fname.format(**format_dict1)
                    if options.verbose > 0:
                        print(f'bias.mni.{c+1}   ->', fname)
                    io.savef(bias1, fname, like=ref1, affine=mni_affine,
                             dtype='float32')

        if options.nobias_mni or options.all_mni:
            nobias = spatial.reslice(dat, dat_affine, mni_affine, mni_shape)
            nobias *= bias
            fname = options.bias_mni or '{dir}{sep}{base}.nobias.mni{ext}'
            if len(options.input) == 1:
                fname = fname.format(**format_dict)
                if options.verbose > 0:
                    print('nobias.mni   ->', fname)
                io.savef(torch.movedim(nobias, 0, -1), fname, like=ref_native,
                         affine=mni_affine)
            else:
                for c, (nobias1, ref1) in enumerate(zip(bias, options.input)):
                    format_dict1 = get_format_dict(ref1, options.output)
                    fname = fname.format(**format_dict1)
                    if options.verbose > 0:
                        print(f'nobias.mni.{c+1} ->', fname)
                    io.savef(nobias1, fname, like=ref1, affine=mni_affine)
            del nobias

        del bias

    need_iwarp = (options.warp_mni or options.prob_wrp or options.labels_wrp or
                  options.bias_wrp or options.nobias_wrp or options.all_mni or
                  options.all_wrp)
    need_iwarp = need_iwarp and options.warp
    if not need_iwarp:
        return

    iwarp = spatial.grid_inv(prm['warp'], type='disp')
    iwarp = iwarp.movedim(-1, 0)
    iwarp = spatial.reslice(iwarp, dat_affine, mni_affine, mni_shape,
                            interpolation=2, bound='dft', extrapolate=True)
    iwarp = iwarp.movedim(0, -1)
    iaff = mni_affine.inverse() @ dat_affine
    iwarp = linalg.matvec(iaff[:3, :3], iwarp)

    if (options.warp_mni or options.all_mni) and options.warp:
        fname = options.warp_mni or '{dir}{sep}{base}.warp.mni{ext}'
        fname = fname.format(**format_dict)
        if options.verbose > 0:
            print('warp.mni     ->', fname)
        io.savef(iwarp, fname, like=ref_native, affine=mni_affine,
                 dtype='float32')

    # --- Warped space -------------------------------------------------
    iwarp = spatial.add_identity_grid_(iwarp)
    iwarp = spatial.affine_matvec(dat_affine.inverse() @ mni_affine, iwarp)

    if options.prob_wrp or options.labels_wrp or options.all_wrp:
        z_mni = spatial.grid_pull(z, iwarp)
        if options.prob_mni or options.all_wrp:
            fname = options.prob_mni or '{dir}{sep}{base}.prob.wrp{ext}'
            fname = fname.format(**format_dict)
            if options.verbose > 0:
                print('prob.wrp     ->', fname)
            io.savef(torch.movedim(z_mni, 0, -1), fname, like=ref_native,
                     affine=mni_affine, dtype='float32')
        if options.labels_mni or options.all_wrp:
            fname = options.labels_mni or '{dir}{sep}{base}.labels.wrp{ext}'
            fname = fname.format(**format_dict)
            if options.verbose > 0:
                print('labels.wrp   ->', fname)
            io.save(z_mni.argmax(0), fname, like=ref_native,
                    affine=mni_affine, dtype='int16')
        del z_mni

    if options.prior_wrp or options.all_wrp:
        mu_mni = spatial.grid_pull(prm['warped'], iwarp)
        fname = options.prob_mni or '{dir}{sep}{base}.prior.wrp{ext}'
        fname = fname.format(**format_dict)
        if options.verbose > 0:
            print('prior.wrp     ->', fname)
        io.savef(torch.movedim(mu_mni, 0, -1), fname, like=ref_native,
                 affine=mni_affine, dtype='float32')
        del z_mni

    if options.bias and (options.bias_wrp or options.nobias_wrp or options.all_wrp):
        bias = spatial.grid_pull(prm['bias'], iwarp,
                                 interpolation=3, prefilter=False, bound='dct2')
        if options.bias_wrp or options.all_wrp:
            fname = options.bias_wrp or '{dir}{sep}{base}.bias.wrp{ext}'
            if len(options.input) == 1:
                fname = fname.format(**format_dict)
                if options.verbose > 0:
                    print('bias.wrp     ->', fname)
                io.savef(torch.movedim(bias, 0, -1), fname, like=ref_native,
                         affine=mni_affine, dtype='float32')
            else:
                for c, (bias1, ref1) in enumerate(zip(bias, options.input)):
                    format_dict1 = get_format_dict(ref1, options.output)
                    fname = fname.format(**format_dict1)
                    if options.verbose > 0:
                        print(f'bias.wrp.{c+1}   ->', fname)
                    io.savef(bias1, fname, like=ref1, affine=mni_affine,
                             dtype='float32')

        if options.nobias_wrp or options.all_wrp:
            nobias = spatial.grid_pull(dat, iwarp)
            nobias *= bias
            fname = options.nobias_wrp or '{dir}{sep}{base}.nobias.wrp{ext}'
            if len(options.input) == 1:
                fname = fname.format(**format_dict)
                if options.verbose > 0:
                    print('nobias.wrp   ->', fname)
                io.savef(torch.movedim(nobias, 0, -1), fname, like=ref_native,
                         affine=mni_affine)
            else:
                for c, (nobias1, ref1) in enumerate(zip(bias, options.input)):
                    format_dict1 = get_format_dict(ref1, options.output)
                    fname = fname.format(**format_dict1)
                    if options.verbose > 0:
                        print(f'nobias.wrp.{c+1} ->', fname)
                    io.savef(nobias1, fname, like=ref1, affine=mni_affine)
            del nobias

        del bias

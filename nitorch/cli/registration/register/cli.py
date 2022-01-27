from nitorch.cli.cli import commands
from .parser import parser, help, ParseError
from nitorch.tools.registration import (pairwise, losses, optim,
                                        utils as regutils, objects)
from nitorch import io, spatial
from nitorch.core import utils, py, dtypes
import torch
import sys
import os


def cli(args=None):
    f"""Command-line interface for `register`
    
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


commands['register'] = cli


def _cli(args):
    """Command-line interface for `register` without exception handling"""
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


def _map_image(fnames, dim=None):
    """Load a N-D image from disk"""
    affine = None
    imgs = []
    for fname in fnames:
        img = io.map(fname)
        if affine is None:
            affine = img.affine
        if dim is None:
            dim = img.affine.shape[-1] - 1
        # img = img.fdata(rand=True, device=device)
        if img.dim > dim:
            img = img.movedim(-1, 0)
        else:
            img = img[None]
        img = img.unsqueeze(-1, dim + 1 - img.dim)
        if img.dim > dim + 1:
            raise ValueError(f'Don\'t know how to deal with an image of '
                             f'shape {tuple(img.shape)}')
        imgs.append(img)
        del img
    imgs = io.cat(imgs, dim=0)
    return imgs, affine


def _load_image(fnames, dim=None, device=None, label=False):
    """Load a N-D image from disk"""
    dat, affine = _map_image(fnames, dim)
    if label:
        dtype = dat.dtype
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        dtype = dtypes.as_torch(dtype, upcast=True)
        dat0 = dat.data(device=device, dtype=dtype)[0]  # assume single channel
        if label is True:
            label = dat0.unique(sorted=True)
            label = label[label != 0].tolist()
        dat = torch.zeros([len(label), *dat0.shape], device=device)
        for i, l in enumerate(label):
            dat[i] = dat0 == l
    else:
        dat = dat.fdata(device=device, rand=True)
    affine = affine.to(dat.device, torch.float32)
    return dat, affine


def _rescale_image(dat, quantiles):
    """Rescale an image based on two quantiles"""
    dim = dat.dim() - 1
    if not isinstance(quantiles, (list, tuple)):
        quantiles = [quantiles]
    if len(quantiles) == 0:
        mn = 0
        mx = 95
    elif len(quantiles) == 1:
        mn = 0
        mx = quantiles[0]
    else:
        mn, mx = quantiles
    mx = mx / 100
    mn, mx = utils.quantile(dat, (mn, mx), dim=range(-dim, 0), keepdim=True).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn)
    return dat


def _discretize_image(dat, nbins=256):
    """Discretize an image into a number of bins"""
    dim = dat.dim() - 1
    mn, mx = utils.quantile(dat, (0.0005, 0.9995), dim=range(-dim, 0), keepdim=True).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn).clamp_(0, 1).mul_(nbins-1)
    dat = dat.long()
    return dat


def _make_image(option, dim=None, device=None):
    """Return an ImagePyramid object"""
    dat, affine = _load_image(option.files, dim=dim, device=device,
                              label=option.label)
    dim = dat.dim() - 1
    if option.mask:
        mask, _ = _load_image([option.mask], dim=dim, device=device,
                              label=option.label)
        if mask.shape[-dim:] != dat.shape[-dim:]:
            raise ValueError('Mask should have the same shape as the image. '
                             f'Got {mask.shape[-dim:]} and {dat.shape[-dim:]}')
    else:
        mask = None
    if option.world:  # overwrite orientation matrix
        affine = io.transforms.map(option.world).fdata()
    for transform in (option.affine or []):
        transform = io.transforms.map(transform).fdata()
        affine = spatial.affine_lmdiv(transform, affine)
    if not option.discretize and option.rescale:
        dat = _rescale_image(dat, option.rescale)
    if option.pad:
        pad = option.pad
        if isinstance(pad[-1], str):
            *pad, unit = pad
        else:
            unit = 'vox'
        if unit == 'mm':
            voxel_size = spatial.voxel_size(affine)
            pad = torch.as_tensor(pad, **utils.backend(voxel_size))
            pad = pad / voxel_size
            pad = pad.floor().int().tolist()
        else:
            pad = [int(p) for p in pad]
        pad = py.make_list(pad, dim)
        affine = spatial.affine_pad(affine, dat.shape[-dim:], pad, side='both')
        dat = utils.pad(dat, pad, side='both', mode=option.bound)
        if mask is not None:
            mask = utils.pad(mask, pad, side='both', mode=option.bound)
    if option.fwhm:
        fwhm = option.fwhm
        if isinstance(fwhm[-1], str):
            *fwhm, unit = fwhm
        else:
            unit = 'vox'
        if unit == 'mm':
            voxel_size = spatial.voxel_size(affine)
            fwhm = torch.as_tensor(fwhm, **utils.backend(voxel_size))
            fwhm = fwhm / voxel_size
        dat = spatial.smooth(dat, dim=dim, fwhm=fwhm, bound=option.bound)
    pyramid = []
    for level in option.pyramid:
        if isinstance(level, int):
            pyramid.append(level)
        else:
            pyramid.extend(list(level))
    image = objects.ImagePyramid(dat, levels=pyramid, affine=affine,
                                 dim=dim, bound=option.bound, mask=mask,
                                 extrapolate=option.extrapolate)
    if not option.label and option.discretize:
        for level in image:
            level.dat = _discretize_image(level.dat, option.discretize)
    return image


def _almost_identity(aff):
    return torch.allclose(aff, torch.eye(*aff.shape, **utils.backend(aff)))


def _warp_image(option, affine=None, nonlin=None, dim=None, device=None, odir=None):

    if not (option.mov.output or option.mov.resliced or
            option.fix.output or option.fix.resliced):
        return

    fix, fix_affine = _map_image(option.fix.files, dim=dim)
    mov, mov_affine = _map_image(option.mov.files, dim=dim)
    fix_affine = fix_affine.float()
    mov_affine = mov_affine.float()
    dim = dim or (fix.dim - 1)

    if option.fix.world:  # overwrite orientation matrix
        fix_affine = io.transforms.map(option.fix.world).fdata()
    for transform in (option.fix.affine or []):
        transform = io.transforms.map(transform).fdata()
        fix_affine = spatial.affine_lmdiv(transform, fix_affine)

    if option.mov.world:  # overwrite orientation matrix
        mov_affine = io.transforms.map(option.mov.world).fdata()
    for transform in (option.mov.affine or []):
        transform = io.transforms.map(transform).fdata()
        mov_affine = spatial.affine_lmdiv(transform, mov_affine)

    # moving
    if option.mov.output or option.mov.resliced:
        ifname = option.mov.files[0]
        idir, base, ext = py.fileparts(ifname)
        odir = odir or idir or '.'

        image = objects.Image(mov.fdata(rand=True, device=device), dim=dim,
                              affine=mov_affine, bound=option.mov.bound,
                              extrapolate=option.mov.extrapolate)

        if option.mov.output:
            target_affine = mov_affine
            target_shape = image.shape
            if affine and affine.position[0].lower() in 'ms':
                aff = affine.exp(recompute=False, cache_result=True)
                target_affine = spatial.affine_lmdiv(aff, target_affine)

            fname = option.mov.output.format(dir=odir, base=base, sep=os.path.sep, ext=ext)
            print(f'Minimal reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

        if option.mov.resliced:
            target_affine = fix_affine
            target_shape = fix.shape[1:]

            fname = option.mov.resliced.format(dir=odir, base=base, sep=os.path.sep, ext=ext)
            print(f'Full reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin, reslice=True)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

    # fixed
    if option.fix.output or option.fix.resliced:
        ifname = option.fix.files[0]
        idir, base, ext = py.fileparts(ifname)
        odir = odir or idir or '.'

        image = objects.Image(fix.fdata(rand=True, device=device), dim=dim,
                            affine=fix_affine, bound=option.fix.bound,
                            extrapolate=option.fix.extrapolate)

        if option.fix.output:
            target_affine = fix_affine
            target_shape = image.shape
            if affine and affine.position[0].lower() in 'fs':
                aff = affine.exp(recompute=False, cache_result=True)
                target_affine = spatial.affine_matmul(aff, target_affine)

            fname = option.fix.output.format(dir=odir, base=base, sep=os.path.sep, ext=ext)
            print(f'Minimal reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin, backward=True)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

        if option.fix.resliced:
            target_affine = mov_affine
            target_shape = mov.shape[1:]

            fname = option.mov.resliced.format(dir=odir, base=base, sep=os.path.sep, ext=ext)
            print(f'Full reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin,
                                  backward=True, reslice=True)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped


def _warp_image1(image, target, shape=None, affine=None, nonlin=None,
                 backward=False, reslice=False):
    """Returns the warped image, with channel dimension last"""
    # build transform
    aff_right = target
    aff_left = spatial.affine_inv(image.affine)
    aff = None
    if affine:
        # exp = affine.iexp if backward else affine.exp
        exp = affine.exp
        aff = exp(recompute=False, cache_result=True)
        if backward:
            aff = spatial.affine_inv(aff)
    if nonlin:
        if affine:
            if affine.position[0].lower() in ('ms' if backward else 'fs'):
                aff_right = spatial.affine_matmul(aff, aff_right)
            if affine.position[0].lower() in ('fs' if backward else 'ms'):
                aff_left = spatial.affine_matmul(aff_left, aff)
        exp = nonlin.iexp if backward else nonlin.exp
        phi = exp(recompute=False, cache_result=True)
        aff_left = spatial.affine_matmul(aff_left, nonlin.affine)
        aff_right = spatial.affine_lmdiv(nonlin.affine, aff_right)
        if _almost_identity(aff_right) and nonlin.shape == shape:
            phi = nonlin.add_identity(phi)
        else:
            tmp = spatial.affine_grid(aff_right, shape)
            phi = regutils.smart_pull_grid(phi, tmp).add_(tmp)
            del tmp
        if not _almost_identity(aff_left):
            phi = spatial.affine_matvec(aff_left, phi)
    else:
        # no nonlin: single affine even if position == 'symmetric'
        if reslice:
            aff = spatial.affine_matmul(aff, aff_right)
            aff = spatial.affine_matmul(aff_left, aff)
            phi = spatial.affine_grid(aff, shape)
        else:
            phi = None

    # warp image
    if phi is not None:
        warped = image.pull(phi)
    else:
        warped = image.dat

    # write to disk
    if len(warped) == 1:
        warped = warped[0]
    else:
        warped = utils.movedim(warped, 0, -1)
    return warped


def _main(options):
    if isinstance(options.gpu, str):
        device = torch.device(options.gpu)
    else:
        assert isinstance(options.gpu, int)
        device = torch.device(f'cuda:{options.gpu}')

    # build loss
    image_dict = {}
    loss_list = []
    for loss in options.loss:
        if not loss.fix.rescale[-1]:
            loss.fix.rescale = False
        if not loss.mov.rescale[-1]:
            loss.mov.rescale = False
        if loss.name in ('cat', 'dice'):
            loss.fix.rescale = False
            loss.mov.rescale = False
        if loss.name == 'emi':
            loss.mov.rescale = False
            loss.fix.discretize = loss.fix.discretize or 256
        fix = _make_image(loss.fix, dim=options.dim, device=device)
        mov = _make_image(loss.mov, dim=options.dim, device=device)
        image_dict[loss.fix.name or loss.fix.files[0]] = fix
        image_dict[loss.mov.name or loss.mov.files[0]] = mov
        dim = fix.dim
        if loss.name == 'mi':
            lossobj = losses.MI(bins=loss.bins, norm=loss.norm,
                                spline=loss.order, fwhm=loss.fwhm, dim=dim)
        elif loss.name == 'ent':
            lossobj = losses.Entropy(bins=loss.bins, spline=loss.order,
                                     fwhm=loss.fwhm, dim=dim)
        elif loss.name == 'mse':
            lossobj = losses.MSE(lam=loss.weight, dim=dim)
        elif loss.name == 'mad':
            lossobj = losses.MAD(lam=loss.weight, dim=dim)
        elif loss.name == 'tuk':
            lossobj = losses.Tukey(lam=loss.weight, dim=dim)
        elif loss.name == 'cc':
            lossobj = losses.CC(dim=dim)
        elif loss.name == 'lcc':
            lossobj = losses.LCC(patch=loss.patch, dim=dim, stride=loss.stride,
                                 mode=loss.kernel)
        elif loss.name == 'gmm':
            lossobj = losses.GMMH(bins=loss.bins, dim=dim,
                                  max_iter=loss.max_iter)
        elif loss.name == 'lgmm':
            lossobj = losses.LGMMH(bins=loss.bins, dim=dim,
                                   max_iter=loss.max_iter,
                                   patch=loss.patch,
                                   stride=loss.stride,
                                   mode=loss.kernel)
        elif loss.name == 'cat':
            lossobj = losses.Cat(dim=dim, log=False)
            # lossobj = losses.AutoCat()
        elif loss.name == 'dice':
            lossobj = losses.Dice(weighted=loss.weight, log=False)
        elif loss.name == 'prod':
            lossobj = losses.ProdLoss(dim=dim)
        elif loss.name == 'normprod':
            lossobj = losses.NormProdLoss(dim=dim)
        elif loss.name == 'sqz':
            lossobj = losses.SqueezedProdLoss(dim=dim, lam=loss.weight)
        elif loss.name == 'emi':
            fwhm = None
            if not loss.fix.label:
                fwhm = loss.fix.discretize // 64
            lossobj = losses.EMI(dim=dim, fwhm=fwhm)
        elif loss.name == 'extra':
            # Not a proper loss, we just want to warp these images at the end
            continue
        else:
            raise ValueError(loss.name)
        lossobj = objects.LossComponent(lossobj, mov, fix, factor=loss.factor,
                                        symmetric=loss.symmetric)
        loss_list.append(lossobj)

    # build affine
    affine = []
    affine_optim = None
    if options.affine:
        make_affine = lambda name: objects.AffineModel(
            name, options.affine.factor, position=options.affine.position)
        name = options.affine.name
        while name:
            affine = [make_affine(name), *affine]
            if name == 'affine':
                name = 'similitude'
            elif name == 'similitude':
                name = 'rigid'
            else:
                name = ''
        max_iter = options.affine.optim.max_iter
        if not max_iter:
            if options.nonlin and options.optim.name == 'interleaved':
                max_iter = 50
            else:
                max_iter = 100
        if options.affine.optim.name == 'unset':
            if all(loss.loss.order >= 2 for loss in loss_list):
                options.affine.optim.name = 'gn'
            else:
                options.affine.optim.name = 'lbfgs'
        if options.affine.optim.name == 'gd':
            affine_optim = optim.GradientDescent(lr=options.affine.optim.lr)
        elif options.affine.optim.name == 'mom':
            affine_optim = optim.Momentum(lr=options.affine.optim.lr,
                                          momentum=options.affine.optim.momentum)
        elif options.affine.optim.name == 'nes':
            affine_optim = optim.Nesterov(lr=options.affine.optim.lr,
                                          momentum=options.affine.optim.momentum,
                                          auto_restart=options.affine.optim.restart)
        elif options.affine.optim.name == 'ogm':
            affine_optim = optim.OGM(lr=options.affine.optim.lr,
                                     momentum=options.affine.optim.momentum,
                                     relax=options.affine.optim.relax,
                                     auto_restart=options.affine.optim.restart)
        elif options.affine.optim.name == 'gn':
            affine_optim = optim.GaussNewton(lr=options.affine.optim.lr,
                                             marquardt=getattr(options.affine.optim, 'marquardt', None))
        elif options.affine.optim.name == 'lbfgs':
            affine_optim = optim.LBFGS(lr=options.affine.optim.lr,
                                       history=getattr(options.affine.optim, 'history', 100),
                                       max_iter=max_iter)
            # TODO: tolerance?
        else:
            raise ValueError(options.affine.optim.name)
        if not isinstance(affine_optim, optim.LBFGS):
            affine_optim = optim.IterateOptim(affine_optim,
                                              max_iter=max_iter,
                                              tol=options.affine.optim.tolerance,
                                              ls=options.affine.optim.line_search)

    # build dense deformation
    nonlin = None
    nonlin_optim = None
    if options.nonlin:
        # build mean space
        vx = options.nonlin.voxel_size
        if isinstance(vx[-1], str):
            *vx, vx_unit = vx
        else:
            vx_unit = 'mm'
        pad = options.nonlin.pad
        if isinstance(pad[-1], str):
            *pad, pad_unit = pad
        else:
            pad_unit = '%'
        vx = py.make_list(vx, dim)
        pad = py.make_list(pad, dim)
        space = objects.MeanSpace(
            [image_dict[key] for key in (options.nonlin.fov or image_dict)],
            voxel_size=vx, vx_unit=vx_unit, pad=pad, pad_unit=pad_unit)
        prm = dict(absolute=options.nonlin.absolute,
                   membrane=options.nonlin.membrane,
                   bending=options.nonlin.bending,
                   lame=options.nonlin.lame)
        vel = objects.Displacement(space.shape, affine=space.affine, dim=dim,
                                   device=device)
        Model = objects.NonLinModel.subclass(options.nonlin.name)
        nonlin = Model(dat=vel, factor=options.nonlin.factor,
                       prm=prm, steps=getattr(options.nonlin, 'steps', None))

        max_iter = options.nonlin.optim.max_iter
        if not max_iter:
            if affine and options.optim.name == 'interleaved':
                max_iter = 10
            else:
                max_iter = 50
        if options.nonlin.optim.name == 'unset':
            if all(loss.loss.order >= 2 for loss in loss_list):
                options.nonlin.optim.name = 'gn'
            else:
                options.nonlin.optim.name = 'lbfgs'
        if options.nonlin.optim.name == 'gd':
            nonlin_optim = optim.GradientDescent(lr=options.nonlin.optim.lr)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'mom':
            nonlin_optim = optim.Momentum(lr=options.nonlin.optim.lr,
                                          momentum=options.nonlin.optim.momentum)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'nes':
            nonlin_optim = optim.Nesterov(lr=options.nonlin.optim.lr,
                                          momentum=options.nonlin.optim.momentum,
                                          auto_restart=options.nonlin.optim.restart)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'ogm':
            nonlin_optim = optim.OGM(lr=options.nonlin.optim.lr,
                                     momentum=options.nonlin.optim.momentum,
                                     relax=options.nonlin.optim.relax,
                                     auto_restart=options.nonlin.optim.restart)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'gn':
            marquardt = getattr(options.nonlin.optim, 'marquardt', None)
            sub_iter = getattr(options.nonlin.optim, 'sub_iter', None)
            if not sub_iter:
                if options.nonlin.optim.fmg:
                    sub_iter = 2
                else:
                    sub_iter = 16
            prm = {'factor': nonlin.factor / py.prod(nonlin.shape),
                   'voxel_size': nonlin.voxel_size,
                   **nonlin.prm}
            if getattr(options.nonlin.optim, 'solver', 'cg') == 'cg':
                nonlin_optim = optim.GridCG(
                    lr=options.nonlin.optim.lr,
                    marquardt=marquardt,
                    max_iter=sub_iter,
                    **prm)
            elif getattr(options.nonlin.optim, 'solver') == 'relax':
                nonlin_optim = optim.GridRelax(lr=options.nonlin.optim.lr,
                                               marquardt=marquardt,
                                               max_iter=sub_iter,
                                               **prm)
            else:
                raise ValueError(getattr(options.nonlin.optim, 'solver'))
        elif options.nonlin.optim.name == 'lbfgs':
            nonlin_optim = optim.LBFGS(
                lr=options.nonlin.optim.lr,
                history=getattr(options.nonlin.optim, 'history'),
                max_iter=max_iter)
            nonlin_optim.preconditioner = nonlin.greens_apply
            # TODO: tolerance?
        else:
            raise ValueError(options.nonlin.optim.name)
        if not isinstance(affine_optim, optim.LBFGS):
            nonlin_optim = optim.IterateOptim(
                nonlin_optim,
                max_iter=max_iter,
                tol=options.nonlin.optim.tolerance,
                ls=options.nonlin.optim.line_search)

    if not affine and not nonlin:
        raise ValueError('At least one of @affine or @nonlin must be used.')

    if options.verbose > 1:
        import matplotlib
        matplotlib.use('TkAgg')

    # LCC and related losses may benefit from selecting the best conv
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    if affine:  # start with a few rounds of progressive affine
        affines = affine
        affine_prev = None
        for i, affine in enumerate(affines):
            if affine_prev:
                n = len(affine_prev.dat.dat)
                affine.set_dat(dim=dim, device=affine_prev.dat.dat.device)
                affine.dat.dat[:n] = affine_prev.dat.dat
                if i == 2: # similitude -> affine
                    affine.dat.dat[n:n+2] = affine_prev.dat.dat[-1]
            register = pairwise.Register(loss_list, affine, None, affine_optim,
                                         verbose=options.verbose,
                                         framerate=options.framerate)
            register.fit()
            affine_prev = affine

    if nonlin:  # now do joint optimization
        # build joint optimizer
        if affine:
            if options.optim.name == 'sequential':
                joptim = optim.IterateOptimInterleaved([affine_optim, nonlin_optim],
                                                       max_iter=1, tol=0)
            else:
                joptim = optim.IterateOptimInterleaved([affine_optim, nonlin_optim],
                                                       max_iter=options.optim.max_iter,
                                                       tol=options.optim.tolerance)
        else:
            joptim = nonlin_optim

        register = pairwise.Register(loss_list, affine, nonlin, joptim,
                                     verbose=options.verbose,
                                     framerate=options.framerate)
        register.fit()

    if register.affine and options.affine.output:
        odir = options.odir or py.fileparts(options.loss[0].fix.files[0])[0] or '.'
        fname = options.affine.output.format(dir=odir, sep=os.path.sep,
                                             name=options.affine.name)
        print('Affine ->', fname)
        aff = register.affine.exp(cache_result=True, recompute=False)
        io.transforms.savef(aff.cpu(), fname, type=1)  # 1 = RAS_TO_RAS
    if register.nonlin and options.nonlin.output:
        odir = options.odir or py.fileparts(options.loss[0].fix.files[0])[0] or '.'
        fname = options.nonlin.output.format(dir=odir, sep=os.path.sep,
                                             name=options.nonlin.name)
        io.savef(register.nonlin.dat.dat, fname, affine=register.nonlin.affine)
        print('Nonlin ->', fname)
    for loss in options.loss:
        _warp_image(loss, affine=register.affine, nonlin=register.nonlin,
                    dim=dim, device=device, odir=options.odir)


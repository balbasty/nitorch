from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from .parser import parser, help
from nitorch.tools.registration import objects
from nitorch.tools.registration.pairwise_preproc import (
    map_image, load_image, preproc_image
)
from nitorch.tools.registration.pairwise_postproc import (
    warp_images
)
from nitorch.tools.registration.pairwise_pyramid import (
    pyramid_levels, concurrent_pyramid, sequential_pyramid
)
from nitorch.tools.registration.pairwise_makeobj import (
    make_image, make_loss,
    make_affine, make_affine_2d, make_affine_optim,
    make_nonlin, make_nonlin_2d, make_nonlin_optim
)
from nitorch.tools.registration.pairwise_run import run
from nitorch import io, spatial
from nitorch.core import py
import torch
import sys
import os
import json
import warnings


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
commands['register2'] = cli  # hidden key that I keep to nod break U01 scripts


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


def _main(options):
    device = setup_device(*options.device)
    dim = 3

    if options.odir:
        os.makedirs(options.odir, exist_ok=True)

    # ------------------------------------------------------------------
    #                       COMPUTE PYRAMID
    # ------------------------------------------------------------------
    vxs, shapes = get_spaces([file.files[0] for loss in options.loss
                              for file in (loss.fix, loss.mov)])
    pyramids = pyramid_levels(vxs, shapes, **options.pyramid)

    # refold as loss/image/level
    pyramids0 = list(pyramids)
    pyramid_per_loss = []
    for _ in options.loss:
        pyramid_per_image = {'fix': pyramids0[0], 'mov': pyramids0[1]}
        pyramid_per_loss.append(pyramid_per_image)
        pyramids0 = pyramids0[2:]

    # ------------------------------------------------------------------
    #                       BUILD LOSSES
    # ------------------------------------------------------------------
    losses, image_dict = build_losses(options, pyramid_per_loss, device)

    # ------------------------------------------------------------------
    #                           BUILD AFFINE
    # ------------------------------------------------------------------
    affine = None
    if options.affine:
        if options.affine.is2d is not False:
            if isinstance(losses[0], objects.SumSimilarity):
                affine0 = losses[0][0].fixed.affine
            else:
                affine0 = losses[0].fixed.affine
            affine = make_affine_2d(options.affine.is2d, affine0,
                                    options.affine.name, options.affine.position,
                                    init=options.affine.init)
        else:
            affine = make_affine(options.affine.name, options.affine.position,
                                 init=options.affine.init)

    # ------------------------------------------------------------------
    #                           BUILD DENSE
    # ------------------------------------------------------------------
    images = [image_dict[key] for key in (getattr(options.nonlin, 'fov', None) or image_dict)]
    nonlin = None
    if options.nonlin:
        options.nonlin.penalty = dict(
            absolute=options.nonlin.absolute,
            membrane=options.nonlin.membrane,
            bending=options.nonlin.bending,
            lame=options.nonlin.lame,
        )
        if options.nonlin.is2d is not False:
            if isinstance(losses[0], objects.SumSimilarity):
                affine0 = losses[0][0].fixed.affine
            else:
                affine0 = losses[0].fixed.affine
            nonlin = make_nonlin_2d(options.nonlin.is2d, affine0,
                                    images, options.nonlin.name, **options.nonlin)
        else:
            nonlin = make_nonlin(images, options.nonlin.name, **options.nonlin)

    if not affine and not nonlin:
        raise ValueError('At least one of @affine or @nonlin must be used.')

    # ------------------------------------------------------------------
    #                           BUILD OPTIM
    # ------------------------------------------------------------------
    order = min(loss.loss.order for loss in losses[0])
    interleaved = options.optim.name[0] == 'i'

    affine_optim = None
    if affine and options.affine.optim.name != 'none':
        if not options.affine.optim.max_iter:
            options.affine.optim.max_iter = 50 if interleaved else 100
        affine_optim = make_affine_optim(options.affine.optim.name, order,
                                         **options.affine.optim)
    elif affine and options.affine.optim.name == 'none':
        affine_optim = 'none'

    nonlin_optim = None
    if nonlin and options.nonlin.optim.name != 'none':
        if not options.nonlin.optim.max_iter:
            options.nonlin.optim.max_iter = 10 if interleaved else 50
        nonlin_optim = make_nonlin_optim(options.nonlin.optim.name, order,
                                         **options.nonlin.optim, nonlin=nonlin)
    elif nonlin and options.nonlin.optim.name == 'none':
        nonlin_optim = 'none'

    # ------------------------------------------------------------------
    #                           BACKEND STUFF
    # ------------------------------------------------------------------
    if options.verbose > 1:
        import matplotlib
        matplotlib.use('TkAgg')

    # local losses may benefit from selecting the best conv
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    #                      PERFORM REGISTRATION
    # ------------------------------------------------------------------
    affine, nonlin = run(
        losses, affine, nonlin, affine_optim, nonlin_optim,
        pyramid=not options.pyramid.concurrent,
        interleaved=options.optim.name != 'sequential',
        progressive=getattr(options.affine, 'progressive', False),
        max_iter=options.optim.max_iter,
        tolerance=options.optim.tolerance,
        verbose=options.verbose,
        framerate=options.framerate,
    )

    # ------------------------------------------------------------------
    #                           WRITE RESULTS
    # ------------------------------------------------------------------
    odir = options.odir or py.fileparts(options.loss[0].fix.files[0])[0] or '.'

    if affine:
        if options.affine.output:
            fname = options.affine.output.format(
                dir=odir, sep=os.path.sep, name=options.affine.name)
            print('Affine ->', fname)
            aff = affine.exp(cache_result=True, recompute=True)
            if affine.position[0] == 's':
                aff = aff.matmul(aff)
            io.transforms.savef(aff.cpu(), fname, type=1)  # 1 = RAS_TO_RAS

    if nonlin:
        if options.nonlin.output:
            fname = options.nonlin.output.format(
                dir=odir, sep=os.path.sep, name=options.nonlin.name)
            io.savef(nonlin.dat.dat, fname, affine=nonlin.affine)
            if isinstance(nonlin, objects.ShootModel):
                nldir, nlbase, _ = py.fileparts(fname)
                fname = os.path.join(nldir, nlbase + '.json')
                with open(fname, 'w') as f:
                    prm = dict(nonlin.penalty)
                    prm['factor'] = nonlin.factor / py.prod(nonlin.shape)
                    json.dump(prm, f)
            print('Nonlin ->', fname)

        if (
            options.nonlin.hessian and
            getattr(nonlin, 'hess', None) is not None
        ):
            fname = options.nonlin.hessian.format(
                dir=odir, sep=os.path.sep, name=options.nonlin.name)
            io.savef(nonlin.hess, fname, affine=nonlin.affine)
            print('Hessian ->', fname)

    for loss in options.loss:
        warp_images(loss.fix, loss.mov, affine=affine, nonlin=nonlin,
                    dim=dim, device=device, odir=options.odir)


def setup_device(device='cpu', ndevice=0):
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


def get_spaces(fnames):
    """Return the voxel size and spatial shape of all volumes"""
    vxs = []
    shapes = []
    for fname in fnames:
        f, affine = map_image(fname)
        shapes.append(f.shape[1:])
        vxs.append(spatial.voxel_size(affine).tolist())
    return vxs, shapes


def build_losses(options, pyramids, device):
    dim = 3

    losskeys = ('kernel', 'patch', 'bins', 'norm', 'fwhm',
                'spline', 'weight', 'weighted', 'max_iter', 'slicewise')
    loadkeys = ('label', 'missing', 'world', 'affine', 'rescale',
                'pad', 'bound', 'fwhm', 'mask', 'channels')
    imagekeys = ('pyramid', 'pyramid_method', 'discretize',
                 'soft', 'bound', 'extrapolate', 'mind')

    image_dict = {}
    sumloss = 0
    for loss, pyramid in zip(options.loss, pyramids):
        lossobj = make_loss(loss.name, **include(loss, losskeys))
        if not lossobj:
            # not a proper loss
            continue
        if not loss.fix.rescale[-1]:
            loss.fix.rescale = (0, 0)
        if not loss.mov.rescale[-1]:
            loss.mov.rescale = (0, 0)
        if loss.name in ('cat', 'dice'):
            loss.fix.rescale = (0, 0)
            loss.mov.rescale = (0, 0)
        if loss.name == 'emmi':
            loss.mov.rescale = (0, 0)
            loss.fix.discretize = loss.fix.discretize or 256
            loss.mov.soft_quantize = loss.mov.discretize or 16
            loss.mov.missing = []
        loss.fix.pyramid = pyramid['fix']
        loss.mov.pyramid = pyramid['mov']
        loss.fix.pyramid_method = options.pyramid.name
        loss.mov.pyramid_method = options.pyramid.name
        fix = make_image(*preproc_image(loss.fix.files, **include(loss.fix, loadkeys), dim=dim, device=device),
                         **include(loss.fix, imagekeys))
        mov = make_image(*preproc_image(loss.mov.files, **include(loss.mov, loadkeys), dim=dim, device=device),
                         **include(loss.mov, imagekeys))
        image_dict[loss.fix.name or loss.fix.files[0]] = fix
        image_dict[loss.mov.name or loss.mov.files[0]] = mov

        # Forward loss
        loss.factor = py.ensure_list(loss.factor, len(pyramid))[::-1]
        factor = loss.factor[-1] / (2 if loss.symmetric else 1)
        sumloss = sumloss + factor * objects.Similarity(lossobj, mov, fix)

        # Backward loss
        if loss.symmetric:
            lossobj = make_loss(loss.name, **include(loss, losskeys))
            if loss.name == 'emmi':
                loss.fix, loss.mov = loss.mov, loss.fix
                loss.mov.rescale = (0, 0)
                loss.fix.discretize = loss.fix.discretize or 256
                loss.mov.soft_quantize = loss.mov.discretize or 16
                fix = make_image(*load_image(loss.fix.files, **loss.fix, dim=dim, device=device), **loss.fix)
                mov = make_image(*load_image(loss.mov.files, **loss.mov, dim=dim, device=device), **loss.mov)
            sumloss = sumloss + factor * objects.Similarity(lossobj, mov, fix, backward=True)

    pyramid_fn = (concurrent_pyramid if options.pyramid.concurrent else
                  sequential_pyramid)
    loss_list = pyramid_fn(sumloss)

    if options.pyramid.concurrent:
        if not isinstance(loss_list, objects.SumSimilarity):
            loss_list = objects.SumSimilarity([loss_list])
        loss_list = list(loss_list)
        new_list = []
        nb_levels = len(loss_list) // len(options.loss)
        nb_losses = len(options.loss)
        for i in range(nb_levels):
            for j in range(nb_losses):
                rel_factor = (
                    options.loss[j].factor[i] /
                    options.loss[j].factor[-1]
                )
                new_list.append(loss_list[i*nb_losses+j] * rel_factor)
        loss_list = objects.SumSimilarity.sum(new_list)
    else:
        new_list = []
        for i, level in enumerate(loss_list):
            if not isinstance(level, objects.SumSimilarity):
                level = objects.SumSimilarity([level])
            level = list(level)
            new_level = []
            for j, loss_elem in enumerate(level):
                i = min(i, len(options.loss[j].factor) - 1)
                rel_factor = (
                    options.loss[j].factor[i] /
                    options.loss[j].factor[-1]
                )
                new_level.append(loss_elem * rel_factor)
            new_level = objects.SumSimilarity.sum(new_level)
            new_list.append(new_level)
        loss_list = new_list

    return loss_list, image_dict


def include(dict_like, keys):
    return {key: value for key, value in dict_like.items()
            if key in keys}


def exclude(dict_like, keys):
    return {key: value for key, value in dict_like.items()
            if key in keys}

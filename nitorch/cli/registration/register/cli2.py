from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from .parser import parser, help
from nitorch.tools.registration import (pairwise, losses, optim,
                                        utils as regutils, objects)
from nitorch.tools.registration.pairwise_preproc import map_image, load_image
from nitorch.tools.registration.pairwise_postproc import warp_images
from nitorch.tools.registration.pairwise_pyramid import pyramid_levels
from nitorch.tools.registration.pairwise_makeobj import (
    make_image, make_loss,
    make_affine, make_affine_2d, make_affine_optim,
    make_nonlin, make_nonlin_2d, make_nonlin_optim)
from nitorch.tools.registration.pairwise_run import run
from nitorch import io, spatial
from nitorch.core import utils, py, dtypes
import torch
import sys
import os
import json
import warnings
import copy
import math as pymath


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


commands['register2'] = cli


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

    # ------------------------------------------------------------------
    #                       COMPUTE PYRAMID
    # ------------------------------------------------------------------
    vxs, shapes = _get_spaces([file for loss in options.loss
                               for file in (loss.fix, loss.mov)])
    pyramids = pyramid_levels(vxs, shapes, **options.pyramid)

    # ------------------------------------------------------------------
    #                       BUILD LOSSES
    # ------------------------------------------------------------------
    losses, image_dict = _build_losses(options, pyramids, device)

    # ------------------------------------------------------------------
    #                           BUILD AFFINE
    # ------------------------------------------------------------------
    if options.affine.is2d is not False:
        affine = make_affine_2d(options.affine.is2d, losses[0].fixed.affine,
                                options.affine.name, options.affine.position)
    else:
        affine = make_affine(options.affine.name, options.affine.position)

    # ------------------------------------------------------------------
    #                           BUILD DENSE
    # ------------------------------------------------------------------
    images = [image_dict[key] for key in (options.nonlin.fov or image_dict)]
    if options.nonlin.is2d is not False:
        nonlin = make_nonlin_2d(options.nonlin.is2d, losses[0].fixed.affine,
                                images, options.nonlin.name, **options.nonlin)
    else:
        nonlin = make_nonlin(images, options.nonlin.name, **options.nonlin)

    if not affine and not nonlin:
        raise ValueError('At least one of @affine or @nonlin must be used.')

    # ------------------------------------------------------------------
    #                           BUILD OPTIM
    # ------------------------------------------------------------------
    order = min(loss.order for loss in losses[0])
    affine_optim = make_affine_optim(options.affine.optim.name, order,
                                     **options.affine.optim)
    nonlin_optim = make_nonlin_optim(options.nonlin.optim.name, order,
                                     **options.nonlin.optim, nonlin=nonlin)

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
    affine, nonlin = run(losses, affine, nonlin, affine_optim, nonlin_optim,
                         pyramid=not options.pyramid.concurrent,
                         interleaved=options.optim.name != 'sequential',
                         progressive=options.affine.progressive,
                         max_iter=options.optim.max_iter,
                         tolerance=options.optim.tolerance,
                         verbose=options.verbose,
                         framerate=options.framerate)

    # ------------------------------------------------------------------
    #                           WRITE RESULTS
    # ------------------------------------------------------------------

    if affine and options.affine.output:
        odir = options.odir or py.fileparts(options.loss[0].fix.files[0])[0] or '.'
        fname = options.affine.output.format(dir=odir, sep=os.path.sep,
                                             name=options.affine.name)
        print('Affine ->', fname)
        aff = affine.exp(cache_result=True, recompute=True)
        io.transforms.savef(aff.cpu(), fname, type=1)  # 1 = RAS_TO_RAS
    if nonlin and options.nonlin.output:
        odir = options.odir or py.fileparts(options.loss[0].fix.files[0])[0] or '.'
        fname = options.nonlin.output.format(dir=odir, sep=os.path.sep,
                                             name=options.nonlin.name)
        io.savef(nonlin.dat.dat, fname, affine=nonlin.affine)
        if isinstance(nonlin, objects.ShootModel):
            nldir, nlbase, _ = py.fileparts(fname)
            fname = os.path.join(nldir, nlbase + '.json')
            with open(fname, 'w') as f:
                prm = dict(nonlin.penalty)
                prm['factor'] = nonlin.factor / py.prod(nonlin.shape)
                json.dump(prm, f)
        print('Nonlin ->', fname)
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


def build_losses(options, pyramids, device):
    dim = 3

    image_dict = {}
    loss_list = []
    for loss, pyramid in zip(options.loss, pyramids):
        lossobj = make_loss(loss.name, **loss)
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
        fix = make_image(*load_image(loss.fix.files, **loss.fix, dim=dim, device=device), **loss.fix)
        mov = make_image(*load_image(loss.mov.files, **loss.mov, dim=dim, device=device), **loss.mov)
        image_dict[loss.fix.name or loss.fix.files[0]] = fix
        image_dict[loss.mov.name or loss.mov.files[0]] = mov

        # Forward loss
        factor = loss.factor / (2 if loss.symmetric else 1)
        lossobj = objects.Similarity(lossobj, mov, fix, factor=factor)
        loss_list.append(lossobj)

        # Backward loss
        if loss.symmetric:
            lossobj = make_loss(loss.name, **loss)
            if loss.name != 'emmi':
                lossobj = objects.Similarity(
                    lossobj, fix, mov, factor=factor, backward=True)
            else:
                loss.fix, loss.mov = loss.mov, loss.fix
                loss.mov.rescale = (0, 0)
                loss.fix.discretize = loss.fix.discretize or 256
                loss.mov.soft_quantize = loss.mov.discretize or 16
                lossobj = objects.Similarity(
                    lossobj, mov, fix, factor=factor, backward=True)
            loss_list.append(lossobj)

    return loss_list, image_dict

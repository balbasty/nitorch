__all__ = ['run', 'run_progressive_init', 'run_single', 'run_pyramid',
           'build_joint_optim']

from .pairwise_impl import PairwiseRegister
from .pairwise_makeobj import make_affine, make_affine_optim, make_nonlin, make_nonlin_optim
from .optim import InterleavedOptimIterator, Optim
from . import objects
from nitorch.core.py import make_list
import torch
import math


def run(
    losses,
    affine=None,
    nonlin=False,
    affine_optim=None,
    nonlin_optim=None,
    pyramid=True,
    interleaved=True,
    progressive=True,
    affine_then_nonlin=True,
    max_iter=10,
    tolerance=1e-5,
    verbose=True,
    framerate=1,
    figure=None
):
    """Run pairwise registration

    Parameters
    ----------
    losses : objects.Similarity or list[objects.Similarity]
        If `pyramid`, should be a list of list.
    affine : AffineModel, optional
        Instantiated affine transformation model
    nonlin : NonlinModel, optional
        Instantiated nonlinear transformation model
    affine_optim : Optim, optional
        Instantiated affine optimizer
    nonlin_optim : Optim, optional
        Instantiated nonlinear optimizer
    pyramid : bool, default=True
        Optimize pyramid levels in a coarse-to-fine manner
    interleaved : bool, default=False
        Fit the affine and nonlinear components in an interleaved manner
    progressive : bool
        Initialize the affine transform by progressively freeing DoF.
    affine_then_nonlin : bool
        Fit the affine from coarse to fine, then fit everything from coarse to fine.
    max_iter : int, default=10
        Maximum number of outer iterations in the interleaved case
    tolerance : float, default=1e-5
        Tolerance for early stopping the in the interleaved case
    verbose : bool or int
        Verbosity level
    framerate : float
        Framerate for live plotting

    Returns
    -------
    affine : AffineModel
    nonlin : NonlinModel

    """
    if verbose >= 3 and not figure:
        import matplotlib.pyplot as plt
        figure = plt.figure()

    line_size = 89 if nonlin else 74
    if not pyramid:
        losses = objects.SumSimilarity.sum(losses)

    first_level = losses[0] if pyramid else losses
    last_level = losses[-1] if pyramid else losses

    # --- prepare transformation models if not provided ---
    if not affine and not nonlin:
        affine = 'rigid'
    if not isinstance(affine, objects.AffineModel) and affine:
        affine = make_affine(affine)
    if not isinstance(nonlin, objects.NonLinModel) and nonlin:
        images = list(last_level.images())
        nonlin = make_nonlin(images, nonlin)

    # --- prepare optimizers if not provided ---
    if affine and not isinstance(affine_optim, Optim):
        order = min(loss.loss.order for loss in last_level)
        affine_optim = make_affine_optim(affine_optim, order)
        if not affine_optim:
            affine.frozen = True
    if affine and not hasattr(affine, "frozen"):
        affine.frozen = False
    if nonlin and not isinstance(nonlin_optim, Optim):
        order = min(loss.loss.order for loss in last_level)
        nonlin_optim = make_nonlin_optim(nonlin_optim, order, nonlin=nonlin)
        if not nonlin_optim:
            nonlin.frozen = True
    if nonlin and not hasattr(nonlin, "frozen"):
        nonlin.frozen = False

    plotopt = dict(verbose=verbose, framerate=framerate, figure=figure)

    runner = run_pyramid if pyramid else run_single

    # --- progressive affine initialization ---
    if affine and not affine.frozen and progressive:
        affine, figure = run_progressive_init(
            runner, losses, affine, affine_optim, line_size=line_size,
            **plotopt)
        plotopt["figure"] = figure

    # --- full affine ---
    if affine and not affine.frozen and (affine_then_nonlin or not nonlin):
        affine, _, _ = runner(losses, affine, None, affine_optim, **plotopt)
    if not nonlin or nonlin.frozen:
        return affine, nonlin

    # --- joint affine and nonlinear  ---
    optim = build_joint_optim(
        affine_optim, nonlin_optim,
        sequential=not interleaved, max_iter=max_iter, tolerance=tolerance)

    affine, nonlin, _ = runner(losses, affine, nonlin, optim, **plotopt)

    return affine, nonlin


def flatten_list(nested_list):
    if not isinstance(nested_list, (list, tuple)):
        return [nested_list]
    flat = []
    for elem in nested_list:
        flat += flatten_list(elem)
    return flat


def run_progressive_init(
    runner,
    losses,
    affine,
    affine_optim,
    line_size=74,
    verbose=True,
    framerate=1,
    figure=None
):
    """Initialize the affine model by running registration on a subset
    of degrees of freedom in a preogressive fashion.

    Parameters
    ----------
    losses : objects.Similarity
        List of losses
    affine : objects.AffineModel
        Target affine model
    affine_optim : optim.Optimizer
        Optimizer
    line_size : int, default=74
        Length of a logging line
    verbose : bool or int, default=True
        Verbosity level
    framerate : float, default=1
        Framerate of live plot

    Returns
    -------
    affine : objects.AffineModel
        Initialized affine model

    """
    plotopt = dict(verbose=verbose, framerate=framerate, figure=figure)

    names = []
    name = affine.basis_name
    while name:
        names = [name, *names]
        if name == 'affine':
            name = 'similitude'
        elif name == 'similitude':
            name = 'rigid'
        else:
            name = ''
    if len(names) == 1:
        return affine, None

    if verbose:
        print('-' * line_size)
        print('   PROGRESSIVE INITIALIZATION')
        print('-' * line_size)

    affine.optim = names.pop(0)
    while names:
        if verbose:
            line_pad = line_size - len(affine.optim) - 5
            print(f'--- {affine.optim} ', end='')
            print('-' * max(0, line_pad))
        affine_optim.reset_state()
        torch.cuda.empty_cache()

        affine, _, figure = runner(losses, affine, None, affine_optim, **plotopt)
        plotopt["figure"] = figure

        affine.optim = names.pop(0)

    affine.optim = None
    return affine, figure


def run_single(
    losses,
    affine,
    nonlin,
    optim,
    verbose=True,
    framerate=1,
    figure=None
):
    """Run pairwise registration at a single level"""

    losses = make_list(losses)
    optim.reset_state()
    if nonlin and not getattr(nonlin, 'frozen', False):
        if affine and not getattr(affine, 'frozen', False):
            nonlin_optim = optim[1]
        else:
            nonlin_optim = optim
        if hasattr(nonlin_optim, 'voxel_size'):
            nonlin_optim.voxel_size = nonlin.voxel_size
        if hasattr(nonlin_optim, 'penalty'):
            nonlin_optim.penalty = nonlin.penalty
    register = PairwiseRegister(losses, affine, nonlin, optim,
                                verbose=verbose, framerate=framerate,
                                figure=figure)
    torch.cuda.empty_cache()
    register.fit()
    return affine, nonlin, figure


def run_pyramid(
    losses,
    affine,
    nonlin,
    optim,
    verbose=True,
    framerate=1,
    figure=None
):
    """Run sequential pyramid registration"""
    line_size = 89 if nonlin else 74

    nonlin_optim = None
    if nonlin and not getattr(nonlin, 'frozen', False):
        if affine and not getattr(affine, 'frozen', False):
            nonlin_optim = optim[1]
        else:
            nonlin_optim = optim

    losses = make_list(losses)
    nb_levels = len(losses)

    n_level = nb_levels - 1
    if nonlin and n_level > 0:
        shapes = [nonlin.shape]
        for n in range(1, nb_levels):
            shapes.append([int(math.ceil(s/2)) for s in shapes[-1]])
        if getattr(nonlin, 'frozen', False):
            nonlin0 = nonlin
            nonlin = nonlin.downsample(shape=shapes[n_level])
        else:
            nonlin = nonlin.downsample_(shape=shapes[n_level])

    if nonlin and hasattr(nonlin_optim, 'penalty'):
        nonlin_optim.penalty = nonlin.penalty

    while losses:
        loss_level = losses.pop(0)
        print('-' * line_size)
        print(f'   PYRAMID LEVEL {n_level}')
        print('-' * line_size)
        if nonlin and hasattr(nonlin_optim, 'voxel_size'):
            nonlin_optim.voxel_size = nonlin.voxel_size
        optim.reset_state()
        register = PairwiseRegister(loss_level, affine, nonlin, optim,
                                    verbose=verbose, framerate=framerate,
                                    figure=figure)
        torch.cuda.empty_cache()
        register.fit()
        figure = register.figure

        if n_level:
            n_level -= 1
            if nonlin and nb_levels > 1:
                if getattr(nonlin, 'frozen', False):
                    if n_level == 0:
                        nonlin = nonlin0
                    else:
                        nonlin = nonlin0.downsample(shape=shapes[n_level])
                else:
                    if n_level == 0:
                        nonlin.upsample_(shape=shapes[n_level])
                    elif n_level > 0:
                        nonlin.upsample_(shape=shapes[n_level])

    return affine, nonlin, figure


def build_joint_optim(
    affine_optim,
    nonlin_optim,
    max_iter=10,
    tolerance=1e-5,
    crit='diff',
    sequential=True
):
    if nonlin_optim:
        if affine_optim:
            if sequential:
                max_iter, tolerance = 1, 0
            joptim = InterleavedOptimIterator([affine_optim, nonlin_optim],
                                              max_iter=max_iter, tol=tolerance,
                                              stop=crit)
        else:
            joptim = nonlin_optim
    else:
        joptim = affine_optim
    return joptim

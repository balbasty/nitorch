from .pairwise import PairwiseRegister
from .optim import InterleavedOptimIterator, Optim
from . import objects
import math as pymath


def run(losses, affine=None, nonlin=False, affine_optim=None, nonlin_optim=None,
        pyramid=True, interleaved=False, progressive=True,
        max_iter=10, tolerance=1e-5, verbose=True, framerate=1):
    """Run pairwise registration

    Parameters
    ----------
    losses : list[objects.Loss] or list[list[objects.Loss]]
        List of losses. If `pyramid`, should be a list of list.
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
    line_size = 89 if nonlin else 74
    if not pyramid:
        losses = [flatten_list(losses)]

    # --- prepare transformation models if not provided ---
    if not affine and not nonlin:
        affine = True
    if not isinstance(affine, objects.AffineModel):
        affine = make_affine(affine)
    if not isinstance(nonlin, objects.NonLinModel):
        nonlin = make_nonlin(nonlin)

    # --- prepare optimizers if not provided ---
    if affine and not isinstance(affine_optim, Optim):
        affine_optim = make_affine_optim(affine_optim, affine)
    if nonlin and not isinstance(nonlin_optim, Optim):
        nonlin_optim = make_nonlin_optim(nonlin_optim, nonlin)

    # --- progressive affine initialization ---
    if progressive:
        affine = run_progressive_init(
            losses, affine, affine_optim,
            line_size=line_size, verbose=verbose, framerate=framerate)

    # --- joint affine and nonlinear  ---
    optim = build_joint_optim(
        affine_optim, nonlin_optim,
        sequential=not interleaved, max_iter=max_iter, tolerance=tolerance)

    if pyramid:
        run_pyramid(losses, affine, nonlin, optim, verbose=True, framerate=1)
    else:
        run_single(losses, affine, nonlin, optim, verbose=True, framerate=1)

    return affine, nonlin


def flatten_list(nested_list):
    if not isinstance(nested_list, (list, tuple)):
        return [nested_list]
    flat = []
    for elem in nested_list:
        flat += flatten_list(elem)
    return flat


def run_progressive_init(losses, affine, affine_optim,
                         line_size=74, verbose=True, framerate=1):
    """Initialize the affine model by running registration on a subset
    of degrees of freedom in a preogressive fashion.

    Parameters
    ----------
    losses : list[objects.Loss]
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
        return affine

    if verbose:
        print('-' * line_size)
        print(f'   PROGRESSIVE INITIALIZATION')
        print('-' * line_size)

    affine = affine.switch_basis(names.pop(0))
    while names:
        if verbose:
            line_pad = line_size - len(affine.basis_name) - 5
            print(f'--- {affine.basis_name} ', end='')
            print('-' * max(0, line_pad))
        affine_optim.reset_state()
        register = PairwiseRegister(losses, affine, None, affine_optim,
                                    verbose=verbose, framerate=framerate)
        register.fit()
        affine.switch_basis(names.pop(0))

    return affine


def run_single(losses, affine, nonlin, optim, verbose=True, framerate=1):
    """Run pairwise registration at a single level"""

    optim.reset_state()
    if nonlin:
        if affine and affine.dat and affine.dat.dat is not None:
            affine.dat.dat /= 2  # take the matrix square root
        nonlin_optim = optim[1] if affine else optim
        if hasattr(nonlin_optim, 'factor'):
            nonlin_optim.factor /= pymath.prod(nonlin.shape)
    register = PairwiseRegister(losses, affine, nonlin, optim,
                                verbose=verbose, framerate=framerate)
    register.fit()
    return affine, nonlin


def run_pyramid(losses, affine, nonlin, optim, verbose=True, framerate=1):
    """Run sequential pyramid registration"""
    line_size = 89 if nonlin else 74

    nonlin_optim = None if not nonlin else optim[1] if affine else optim

    if nonlin and affine and affine.dat and affine.dat.dat is not None:
        affine.dat.dat /= 2  # take the matrix square root

    n_level = nb_levels = len(losses)
    factor = getattr(nonlin_optim, 'factor', None)
    if nonlin and n_level > 1:
        vel_shape = nonlin.shape
        nonlin = nonlin.downsample_(2**n_level)

    while losses:
        loss_level = losses.pop(0)
        n_level -= 1
        print('-' * line_size)
        print(f'   PYRAMID LEVEL {n_level}')
        print('-' * line_size)
        if nonlin and nb_levels > 1:
            if n_level == 0:
                nonlin.upsample_(shape=vel_shape, interpolation=3)
            else:
                nonlin.upsample_()
        if nonlin and factor is not None:
            nonlin_optim.factor = factor / pymath.prod(nonlin.shape)
        optim.reset_state()
        register = PairwiseRegister(loss_level, affine, nonlin, optim,
                                    verbose=verbose, framerate=framerate)
        register.fit()

    return affine, nonlin


def build_joint_optim(affine_optim, nonlin_optim, max_iter=10, tolerance=1e-5,
                      sequential=True):
    if nonlin_optim:
        if affine_optim:
            if sequential:
                max_iter, tolerance = 1, 0
            joptim = InterleavedOptimIterator([affine_optim, nonlin_optim],
                                              max_iter=max_iter, tol=tolerance)
        else:
            joptim = nonlin_optim
    else:
        joptim = affine_optim
    return joptim
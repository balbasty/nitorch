__all__ = ['make_image', 'make_loss',
           'make_affine', 'make_affine_2d', 'make_affine_optim',
           'make_nonlin', 'make_nonlin_2d', 'make_nonlin_optim']

from nitorch import spatial, io
from nitorch.core.py import make_list
from nitorch.core import utils
from . import losses, objects, optim as opt
from .pairwise_preproc import soft_quantize_image, discretize_image, preproc_image


def make_image(dat, mask=None, affine=None,
               pyramid=0, pyramid_method='gaussian',
               discretize=False, soft=False, mind=None,
               bound='zero', extrapolate=True, **kwargs):
    """Create an image pyramid (eventually with a single level)

    Parameters
    ----------
    dat : (C, *spatial) tensor
        Input image
    mask : (C|1, *spatial) tensor, optional
        Mask of voxels to include
    affine : (D+1, D+1) tensor, optional
        Orientation matrix
    pyramid : [sequence of] int, default=[0]
        Pyramid level to compute
    pyramid_method : {'gauss', 'average', 'median', 'stride'}, default='gauss'
        Method used to compute the pyramid
    discretize : int, optional
        Discretize the image at each level into this many bins
    soft : bool, default=False
        Use soft discretization instead of hard discretization
    mind : pair(float, int), optional
        Compute MIND features at each pyramid level.
        First parameter is the FWHM (default = 1)
        Second parameter is the radius (default = 0 = first-ring neighbors)
    bound : [sequence of] str
        Boundary conditions
    extrapolate : bool, default=True
        Extrapolate input data when sampling out-of-bound

    Returns
    -------
    image: ImagePyramid

    """
    pyramid = make_list(pyramid)

    if isinstance(dat, (str, io.MappedArray)):
        dat, mask0, aff0 = preproc_image(dat, **kwargs, bound=bound)
        if mask is None:
            mask = mask0
        if affine is None:
            affine = aff0
        del mask0, aff0

    dim = dat.dim() - 1
    is_label = not dat.dtype.is_floating_point
    if affine is None:
        affine = spatial.affine_default(dat.shape[1:])

    image = objects.ImagePyramid(
        dat,
        levels=pyramid,
        affine=affine,
        dim=dim,
        bound=bound,
        mask=mask,
        extrapolate=extrapolate,
        method=pyramid_method
    )

    mind = [] if mind is False else [1, 2] if mind is True else mind
    mind = make_list(mind or [])
    if mind:
        fwhm, radius = make_list(mind, 2, default=2)
        for level in image:
            level.preview = level.dat
            level.dat = spatial.rmind(level.dat, dim=dim,
                                     radius=int(radius), fwhm=fwhm,
                                     bound=level.bound)
            level.dat = utils.movedim(level.dat, -1, 0)
            level.dat = level.dat.reshape([-1, *level.shape])

    if discretize:
        if soft:
            if len(dat) > 1:
                raise ValueError('Cannot soft-quantize a multi-channel image')
            for level in image:
                level.preview = level.dat
                level.dat = soft_quantize_image(level.dat, discretize)
        elif not is_label:
            for level in image:
                level.preview = level.dat
                level.dat = discretize_image(level.dat, discretize)
    return image


def make_affine(basis='rigid', position='symmetric', penalty=None, init=None):
    """Build an AffineModel object

    Parameters
    ----------
    basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}, default='rigid'
        Affine basis name
    position : {'moving', 'fixed', 'symmetric'}, default='symmetric'
        Which image should be rotated by this transformation.
        If 'symmetric', both images are rotated by the transformation and
        its inverse, towards a mean space; thereby making the model fully
        symmetric.

    Returns
    -------
    affine : objects.AffineModel

    """
    if not basis:
        return None
    if basis is True:
        basis = 'rigid'
    if init is not None:
        if isinstance(init, str):
            init = io.map(init)
        if isinstance(init, io.transforms.MappedAffine):
            init = init.fdata().squeeze()
        if init.dim() > 1:
            init, _ = spatial.affine_parameters(init, basis)
    return objects.AffineModel(basis, position=position, penalty=penalty,
                               dat=init)


def make_affine_2d(plane, affine_ref, basis='rigid', position='symmetric',
                   penalty=None, init=None):
    """Build an Affine2dModel object

    An Affine2dModel represents an in-plane affine within a 3D space.
    The plane in which it lies is defined by an orientation matrix
    (that maps from a space in which the plane is a canonical axis to
    the "world" space) and an axis name.

    Parameters
    ----------
    plane : int or {'LR', 'IS', 'AP'}
        Plane in which the 2D transform lives.
    affine_ref : (D+1, D+1) tensor
        Orientation matrix of the reference space
    basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}, default='rigid'
        Affine basis name
    position : {'moving', 'fixed', 'symmetric'}, default='symmetric'
        Which image should be rotated by this transformation.
        If 'symmetric', both images are rotated by the transformation and
        its inverse, towards a mean space; thereby making the model fully
        symmetric.

    Returns
    -------
    affine : objects.AffineModel

    """
    if not basis:
        return None
    if basis is True:
        basis = 'rigid'
    obj = objects.Affine2dModel(basis, plane=plane, ref_affine=affine_ref,
                                position=position, penalty=penalty)
    if init is not None:
        if isinstance(init, str):
            init = io.map(init)
        if isinstance(init, io.transforms.MappedAffine):
            init = init.fdata().squeeze()
        if init.dim() > 1:
            obj.set_dat()
            rot = obj.rotation.to(init)
            init = rot.T @ init @ rot
            init, _ = spatial.affine_parameters(init, obj.basis.to(init))
        obj.set_dat(init)
    return obj


def make_nonlin(shape_or_init=None, model='svf', affine=None, voxel_size=None,
                pad=None, factor=1, penalty=None, device=None, **kwargs):
    """Build an NonlinModel object

    Parameters
    ----------
    shape or init : list[int] or list[objects.Image] or str or MappedArray
        Shape or a list of images from which to compute a mean space.
        If a str or MappedArray, read from file
    model : {'smalldef', 'svf', 'shoot'}, default='svf'
        Nonlinear model name
    affine : (D+1, D+1) tensor, optional
        Orientation matrix of the nonlinear space
    voxel_size : [list of] float, optional
        Voxel size of the nonlinear space
    pad : [list of] int or float, optional
    penalty : dict, optional
    device : torch.device, optional

    Returns
    -------
    nonlin : objects.NonlinModel

    """
    if not model:
        return None

    if kwargs.get('shape', None) is not None:
        shape_or_init = kwargs.pop('shape')
    elif kwargs.get('init', None) is not None:
        shape_or_init = kwargs.pop('init')

    if shape_or_init is None:
        raise ValueError('`shape` or `init` required')

    # check if file provided
    if isinstance(shape_or_init, str):
        shape_or_init = io.map(shape_or_init)

    if isinstance(shape_or_init, io.MappedArray):
        vel = shape_or_init.fdata(device=device)
        if vel.shape[3:4] == (1,):
            # unused time axis in some niftis
            vel = vel.squeeze(3)
        if vel.shape[-1] != (vel.ndim-1):
            # This assumes we only care about the space of the input
            vel = vel.new_zeros(vel.shape + (len(vel.shape),))
        if vel.shape[-1] not in (2, 3):
            raise ValueError("Expected a 2D or 3D displacement field")
        vel = objects.Displacement(vel, affine=shape_or_init.affine)

    else:
        shape = shape_or_init

        do_mean_space = isinstance(shape[0], objects.Image)
        dim = shape[0].dim if do_mean_space else len(shape)

        voxel_size = make_list(voxel_size or [])
        if voxel_size and isinstance(voxel_size[-1], str):
            *voxel_size, vx_unit = voxel_size
        else:
            vx_unit = 'mm'
        if voxel_size:
            voxel_size = make_list(voxel_size, dim)
        else:
            voxel_size = None

        pad = make_list(pad or [])
        if pad and isinstance(pad[-1], str):
            *pad, pad_unit = pad
        else:
            pad_unit = '%'
        if pad:
            pad = make_list(pad, dim)
        else:
            pad = None

        # build mean space
        if do_mean_space:
            space = objects.MeanSpace(shape,
                                      voxel_size=voxel_size, vx_unit=vx_unit,
                                      pad=pad, pad_unit=pad_unit)
        else:
            space = objects.Space(shape, affine, voxel_size)

        # allocate data
        vel = objects.Displacement(space.shape, affine=space.affine, device=device)

    # build model
    Model = objects.NonLinModel.subclass(model)
    if isinstance(factor, (list, tuple)):
        factor = factor[0]
    nonlin = Model(dat=vel, factor=factor, penalty=penalty, **kwargs)
    return nonlin


def make_nonlin_2d(plane, affine_ref, *args, **kwargs):
    """Build an Nonlin2dModel object

    An Nonlin2dModel represents nonlinear warp along a plane within a 3D space.
    The plane in which it lies is defined by an orientation matrix
    (that maps from a space in which the plane is a canonical axis to
    the "world" space) and an axis name.

    Parameters
    ----------
    plane : int or {'LR', 'IS', 'AP'}
        Plane in which the 2D transform lives.
    affine_ref : (D+1, D+1) tensor
        Orientation matrix of the reference space
    shape : list[int] or list[objects.Image]
        Shape or a list of images from which to compute a mean space.
    model : {'smalldef', 'svf', 'shoot'}, default='svf'
        Nonlinear model name
    affine : (D+1, D+1) tensor, optional
        Orientation matrix of the nonlinear space
    voxel_size : [list of] float, optional
        Voxel size of the nonlinear space
    padding : [list of] int or float, optional
    penalty : dict, optional
    device : torch.device, optional

    Returns
    -------
    nonlin : objects.Nonlin2dModel

    """
    nonlin = make_nonlin(*args, **kwargs)
    if not nonlin:
        return None
    nonlin = objects.Nonlin2dModel(nonlin, plane, affine_ref)
    return nonlin


def make_affine_optim(optim=None, order=None, line_search='wolfe', lr=None,
                      max_iter=50, tolerance=1e-5, crit='diff', **kwargs):
    """Build an Optimizer for the affine model

    Parameters
    ----------
    optim : {'gn', 'lbfgs', 'gd', 'cg', 'mom', 'nes', 'ogm', 'pow'}, optional
        Optimizer name. If None, use one automatically based on `order`
    order : {0, 1, 2}, optional
        Maximum differentiability order of the loss
    line_search : 'wolfe' or int, default='wolfe'
        Line search.
        If a number, use a backtracking line search with maximum this
        number of iterations.
        If 'wolfe' use a cubic line search until Wolfe conditions are
        satisfied.
    lr : float, optional
        Learning rate.
        By default: 1 for second order methods, 1e-3 for first order methods.
    max_iter : int, default=50
        Maximum number of iterations
    tolerance : float, default=1e-5
        Tolerance for early stopping
    crit : {'diff', 'gain'}, default='diff'
        Stopping criterion

    Keyword Parameters
    ------------------
    beta, if 'cg' : {'fletcher_reeves', 'polak_ribiere', 'hestenes_stiefel', 'dai_yuan'}
    momentum, if 'mom' or 'nes' or 'ogm' : float
    auto_restart, if 'nes' or 'ogm' : bool
    relaxation, if 'ogm' : float
    marquardt, if 'gn' : float
    history, if 'lbfgs' : int

    Returns
    -------
    optim : objects.Optim

    """
    optim = (optim or '').lower()
    if not optim or optim == 'unset':
        order = order or 0
        if order >= 2:
            optim = 'gn'
        elif order == 1:
            optim = 'lbfgs'
        else:
            optim = 'pow'

    if optim == 'gd':
        affine_optim = opt.GradientDescent
    elif optim == 'cg' or optim.startswith('conj'):
        affine_optim = opt.ConjugateGradientDescent
    elif optim.startswith('mom'):
        affine_optim = opt.Momentum
    elif optim.startswith('nes'):
        affine_optim = opt.Nesterov
    elif optim == 'ogm':
        affine_optim = opt.OGM
    elif optim == 'gn' or optim.startswith('gauss') or optim.startswith('newt'):
        affine_optim = opt.GaussNewton
    elif optim == 'lbfgs':
        affine_optim = opt.LBFGS
    elif optim.startswith('pow'):
        affine_optim = opt.Powell
    elif optim == 'none':
        return None
    else:
        raise ValueError(optim)

    lr = lr or (1 if issubclass(affine_optim, opt.SecondOrder) else 1e-3)
    affine_optim = affine_optim(lr, **kwargs)

    if line_search and not isinstance(affine_optim, (opt.Powell, opt.LBFGS)):
        affine_optim.search = line_search

    affine_optim.iter = opt.OptimIterator(max_iter=max_iter, tol=tolerance, stop=crit)
    return affine_optim


def make_nonlin_optim(optim=None, order=None, line_search='wolfe', lr=None,
                      max_iter=50, tolerance=1e-5, crit='diff',
                      nonlin=None, penalty=None, voxel_size=None,
                      preconditioner=None, **kwargs):
    """Build an Optimizer for the affine model

    Parameters
    ----------
    optim : {'gn', 'lbfgs', 'gd', 'cg', 'mom', 'nes', 'ogm', 'pow'}, optional
        Optimizer name. If None, use one automatically based on `order`
    order : {0, 1, 2}, optional
        Maximum differentiability order of the loss
    line_search : 'wolfe' or int, default='wolfe'
        Line search.
        If a number, use a backtracking line search with maximum this
        number of iterations.
        If 'wolfe' use a cubic line search until Wolfe conditions are
        satisfied.
    lr : float, optional
        Learning rate.
        By default: 1 for second order methods, 1e-3 for first order methods.
    max_iter : int, default=50
        Maximum number of iterations
    tolerance : float, default=1e-5
        Tolerance for early stopping
    crit : {'diff', 'gain'}, default='diff'
        Stopping criterion
    nonlin : NonlinModel, optional
        Nonlinear model, from which the penalty, voxel size and
        preconditioner can be guessed.
    penalty : dict, default=`nonlin.penalty`
    voxel_size : dict, default=`nonlin.voxel_size`
    preconditioner : callable, default=`nonlin.greens_apply`
        Preconditioner for 1st order methods.
        It is advised to use `nonlin.greens_apply`

    Keyword Parameters
    ------------------
    beta, if 'cg' : {'fletcher_reeves', 'polak_ribiere', 'hestenes_stiefel', 'dai_yuan'}
    momentum, if 'mom' or 'nes' or 'ogm' : float
    auto_restart, if 'nes' or 'ogm' : bool
    relaxation, if 'ogm' : float
    fmg, if 'gn' : bool
    solver, if 'gn' : {'cg', 'relax'}
    marquardt, if 'gn' : float
    sub_iter, if 'gn' : int
    history, if 'lbfgs' : int

    Returns
    -------
    optim : objects.Optim

    """
    optim = (optim or '').lower()
    if not optim or optim == 'unset':
        order = order or 0
        if order >= 2:
            optim = 'gn'
        elif order == 1:
            optim = 'lbfgs'
        else:
            optim = 'pow'

    if nonlin:
        preconditioner = preconditioner or nonlin.greens_apply
        penalty = penalty or dict(nonlin.penalty)
        penalty['voxel_size'] = voxel_size or nonlin.voxel_size
        if 'factor' not in penalty:
            penalty['factor'] = nonlin.factor

    if optim == 'gd':
        nonlin_optim = opt.GradientDescent
    elif optim == 'cg' or optim.startswith('conj'):
        nonlin_optim = opt.ConjugateGradientDescent
    elif optim.startswith('mom'):
        nonlin_optim = opt.Momentum
    elif optim.startswith('nes'):
        nonlin_optim = opt.Nesterov
    elif optim == 'ogm':
        nonlin_optim = opt.OGM
    elif optim == 'gn' or optim.startswith('gauss') or optim.startswith('newt'):
        fmg = kwargs.pop('fmg', True)
        kwargs['max_iter'] = kwargs.pop('sub_iter', None) or (2 if fmg else 16)
        solver = kwargs.pop('solver', 'cg')
        kwargs['penalty'] = penalty
        nonlin_optim = opt.GridCG if solver == 'cg' else opt.GridRelax
    elif optim == 'lbfgs':
        nonlin_optim = opt.LBFGS
    elif optim.startswith('pow'):
        nonlin_optim = opt.Powell
    elif optim == 'none':
        return None
    else:
        raise ValueError(optim)

    lr = lr or (1 if issubclass(nonlin_optim, opt.SecondOrder) else 1e-3)
    nonlin_optim = nonlin_optim(lr, **kwargs)

    if preconditioner and isinstance(nonlin_optim, opt.FirstOrder):
        nonlin_optim.preconditioner = preconditioner

    if line_search and not isinstance(nonlin_optim, (opt.Powell, opt.LBFGS)):
        nonlin_optim.search = line_search

    nonlin_optim.iter = opt.OptimIterator(max_iter=max_iter, tol=tolerance, stop=crit)
    return nonlin_optim


def make_loss(loss, slicewise=False, **kwargs):
    """Instantiate a loss object

    Parameters
    ----------
    loss : str
        - mi:            Mutual Information
        - nmi:           Normalized Mutual Information
        - emmi:          Mutual Information by Expectation-Maximization
        - ent[ropy]:     Joint entropy
        - mse or l2:     Mean Squared Error
        - mad or l1:     Median Absolute Deviation
        - tuk[ey]:       Tukey's biweight
        - ncc:           Normalized Cross Correlation
        - lncc:          Local Normalized Cross Correlation
        - gmm:           Joint Gaussian Mixture
        - lgmm:          Entropy of a Local Gaussian Mixture
        - cat[egorical]: Categorical cross-entropy
        - dice:          Dice
        - dot:           Dot product
        - ndot:          Normalized Dot product
        - sqz:           Squeezed Product
    slicewise : int, optional
        Make the loss slicewise

    Keyword Parameters
    ------------------
    bins, if 'mi' or 'ent' : int
    norm, if 'nmi': {'studholme', 'arithmetic'}
    fwhm, if 'mi', 'ent' or 'emmi' : float
    spline, if 'mi' or 'ent' : {0..7}
    weight, if 'mse', 'mad' or 'tuk' : float or None
    patch, if 'lcc' or 'lgmm' : [list of] float
    stride, if 'lcc' or 'lgmm' : [list of] int
    kernel, if 'lcc' or 'lgmm' : {'gaussian', 'hat'}
    max_iter, if 'gmm', 'lgmm' : int
    weighted, if 'dice' : list[float]

    Returns
    -------
    loss : Loss

    """
    loss = loss.lower()
    if loss == 'mi':
        kwargs.setdefault('norm', None)
        lossobj = losses.MI(**kwargs)
    elif loss == 'nmi':
        kwargs.setdefault('norm', 'studholme')
        lossobj = losses.MI(**kwargs)
    elif loss == 'ent':
        lossobj = losses.Entropy(**kwargs)
    elif loss in ('mse', 'l2'):
        kwargs['lam'] = kwargs.pop('weight', None)
        lossobj = losses.MSE(**kwargs)
    elif loss in ('mad', 'l1'):
        kwargs['lam'] = kwargs.pop('weight', None)
        lossobj = losses.MAD(**kwargs)
    elif loss.startswith('tuk'):
        kwargs['lam'] = kwargs.pop('weight', None)
        lossobj = losses.Tukey(**kwargs)
    elif loss in ('cc', 'ncc'):
        lossobj = losses.CC()
    elif loss in ('lcc', 'lncc'):
        lossobj = losses.LCC(**kwargs)
    elif loss == 'gmm':
        lossobj = losses.GMMH(**kwargs)
    elif loss == 'lgmm':
        lossobj = losses.LGMMH(**kwargs)
    elif loss == 'cat':
        lossobj = losses.Cat(log=False)
    elif loss == 'dice':
        lossobj = losses.Dice(**kwargs)
    elif loss == 'dot':
        lossobj = losses.ProdLoss()
    elif loss == 'ndot':
        lossobj = losses.NormProdLoss()
    elif loss == 'sqz':
        kwargs['lam'] = kwargs.pop('weight', None)
        lossobj = losses.SqueezedProdLoss(**kwargs)
    elif loss == 'emmi':
        lossobj = losses.EMMI(**kwargs)
    elif loss == 'extra':
        # Not a proper loss, we just want to warp these images at the end
        lossobj = None
    else:
        raise ValueError(loss)
    if slicewise is not False:
        lossobj = losses.SliceWiseLoss(lossobj, slicewise)
    return lossobj

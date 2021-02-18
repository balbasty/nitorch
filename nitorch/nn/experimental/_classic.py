import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nitorch import core, spatial, io
from nitorch import nn as nni
# import matplotlib.pyplot as plt


def zero_grad_(param):
    """Reset the gradients of a (list of) parameter(s)."""
    if isinstance(param, (list, tuple)):
        for p in param:
            zero_grad_(p)
        return
    if param.grad is not None:
        param.grad.detach_()
        param.grad.zero_()
    
    
def prepare(inputs, device=None):
    """Preprocess a list of tensors (and affine matrix)
    
    Parameters
    ----------
    inputs : sequence[str or tensor or (tensor, affine)]
    device : torch.device, optional
    
    Returns
    -------
    inputs : list[(tensor, affine)]
        - Each tensor is of shape (batch, channel, *spatial).
        - Each affine is of shape (batch, dim+1, dim+1).
        - All tensors and affines are on the same device.
        - All tensors and affines are of type float32.
    """

    def prepare_one(inp):
        if isinstance(inp, (list, tuple)):
            has_aff = len(inp) > 1
            if has_aff:
                aff0 = inp[1]
            inp, aff = prepare_one(inp[0])
            if has_aff:
                aff = aff0
            return [inp, aff]
        if isinstance(inp, str):
            inp = io.map(inp)[None, None]
        if isinstance(inp, io.MappedArray):
            return inp.fdata(rand=True), inp.affine[None]
        inp = torch.as_tensor(inp)
        aff = spatial.affine_default(inp.shape)[None]
        return [inp, aff]

    prepared = []
    for inp in inputs:
        prepared.append(prepare_one(inp))

    prepared[0][0] = prepared[0][0].to(device=device, dtype=torch.float32)
    device = prepared[0][0].device
    dtype = prepared[0][0].dtype
    backend = dict(dtype=dtype, device=device)
    for i in range(len(prepared)):
        prepared[i][0] = prepared[i][0].to(**backend)
        prepared[i][1] = prepared[i][1].to(**backend)
    return prepared


def get_backend(tensor):
    device = tensor.device
    dtype = tensor.dtype
    return dict(dtype=dtype, device=device)


def rescale(x):
    """Very basic min/max rescaling"""
    x = x.clone()
    min = x.min()
    max = x.max()
    x -= min
    x /= max - min
    x /= x[x > 0.01].mean()  # actually: use (robust) mean rather than max
    x.clamp_max_(2.)
    return x


def affine(source, target, group='SE', loss=None, pull=None, preproc=True,
           max_iter=1000, device=None, origin='center',
           init=None, lr=0.1, scheduler=ReduceLROnPlateau):
    """Affine registration
    
    Note
    ----
    .. Tensors must have shape (batch, channel, *spatial)
    .. Composite losses (e.g., computed on both intensity and categorical
       images) can be obtained by stacking all types of inputs across 
       the channel dimension. The loss function is then responsible 
       for unstacking the tensor and computing the appropriate 
       losses. The drawback of this approach is that all inputs
       must share the same lattice and orientation matrix, as 
       well as the same interpolation order. The advantage is that 
       it simplifies the signature of this function.

    Parameters
    ----------
    source : tensor or (tensor, affine)
    target : tensor or (tensor, affine)
    group : {'T', 'SO', 'SE', 'CSO', 'GL+', 'Aff+'}, default='SE'
    loss : Loss, default=MutualInfoLoss()
    pull : dict
        interpolation : int, default=1
        bound : bound_like, default='dct2'
        extrapolate : bool, default=False
    preproc : bool, default=True
    max_iter : int, default=1000
    device : device, optional
    origin : {'native', 'center'}, default='center'
    init : tensor_like, default=0
    lr : float, default=0.1
    scheduler : Scheduler, default=ReduceLROnPlateau

    Returns
    -------
    q : tensor
        Parameters
    aff : (D+1, D+1) tensor
        Affine transformation matrix.
        The source affine matrix can be "corrected" by left-multiplying
        it with `aff`.
    moved : tensor
        Source image moved to target space.


    """
    pull = pull or dict()
    pull['interpolation'] = pull.get('interpolation', 'linear')
    pull['bound'] = pull.get('bound', 'dct2')
    pull['extrapolate'] = pull.get('extrapolate', False)
    pull_opt = pull
    
    # prepare all data tensors
    ((source, source_aff), (target, target_aff)) = prepare([source, target], device)
    backend = get_backend(source)
    batch = source.shape[0]
    src_channels = source.shape[1]
    trg_channels = target.shape[1]
    dim = source.dim() - 2

    # Rescale to [0, 1]
    if preproc:
        source = rescale(source)
        target = rescale(target)

    # Shift origin
    if origin == 'center':
        shift = torch.as_tensor(target.shape, **backend)/2
        shift = -spatial.affine_matvec(target_aff, shift)
        target_aff[..., :-1, -1] += shift
        source_aff[..., :-1, -1] += shift

    # Prepare affine utils + Initialize parameters
    basis = spatial.affine_basis(group, dim, **backend)
    nb_prm = spatial.affine_basis_size(group, dim)
    if init is not None:
        parameters = torch.as_tensor(init, **backend).clone().detach()
        parameters = parameters.reshape([batch, nb_prm])
    else:
        parameters = torch.zeros([batch, nb_prm], **backend)
    parameters = nn.Parameter(parameters, requires_grad=True)
    identity = spatial.identity_grid(target.shape[2:], **backend)

    def pull(q):
        aff = core.linalg.expm(q, basis)
        aff = spatial.affine_matmul(aff, target_aff)
        aff = spatial.affine_lmdiv(source_aff, aff)
        expd = (slice(None),) + (None,) * dim + (slice(None), slice(None))
        grid = spatial.affine_matvec(aff[expd], identity)
        moved = spatial.grid_pull(source, grid, **pull_opt)
        return moved

    # Prepare loss and optimizer
    if loss is None:
        loss_fn = nni.MutualInfoLoss()
        loss = lambda x, y: loss_fn(x, y)

    optim = torch.optim.Adam([parameters], lr=lr)
    if scheduler is not None:
        scheduler = scheduler(optim)

    # Optim loop
    loss_val = core.constants.inf
    for n_iter in range(1, max_iter+1):

        loss_val0 = loss_val
        optim.zero_grad(set_to_none=True)
        moved = pull(parameters)
        loss_val = loss(moved, target)
        loss_val.backward()
        optim.step()
        if scheduler is not None and n_iter % 10 == 0:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss_val)
            else:
                scheduler.step()

        with torch.no_grad():
            if n_iter % 10 == 0:
                print('{:4d} {:12.6f} | lr={:g}'
                      .format(n_iter, loss_val.item(), 
                              optim.param_groups[0]['lr']), 
                      end='\r')

    print('')
    with torch.no_grad():
        moved = pull(parameters)
        aff = core.linalg.expm(parameters, basis)
        if origin == 'center':
            aff[..., :-1, -1] -= shift
            shift = core.linalg.matvec(aff[..., :-1, :-1], shift)
            aff[..., :-1, -1] += shift
        aff = aff.inverse()
        aff.requires_grad_(False)
    return parameters, aff, moved



def ffd_exp(prm, shape, order=3, bound='dft', returns='disp'):
    """Transform FFD parameters into a displacement or transformation grid.
    
    Parameters
    ----------
    prm : (..., *spatial, dim)
        FFD parameters
    shape : sequence[int]
        Exponentiated shape
    order : int, default=3
        Spline order
    bound : str, default='dft'
        Boundary condition
    returns : {'disp', 'grid', 'disp+grid'}, default='grid'
        What to return:
        - 'disp' -> displacement grid
        - 'grid' -> transformation grid
    
    Returns
    -------
    disp : (..., *shape, dim), optional
        Displacement grid
    grid : (..., *shape, dim), optional
        Transformation grid
        
    """
    backend = dict(dtype=prm.dtype, device=prm.device)
    dim = prm.shape[-1]
    batch = prm.shape[:-(dim+1)]
    prm = prm.reshape([-1, *prm.shape[-(dim+1):]])
    disp = spatial.resize_grid(prm, type='displacement',
                               shape=shape, 
                               interpolation=order, bound=bound)
    disp = disp.reshape(batch + disp.shape[1:])
    grid = disp + spatial.identity_grid(shape, **backend)
    if 'disp' in returns and 'grid' in returns:
        return disp, grid
    elif 'disp' in returns:
        return disp
    elif 'grid' in returns:
        return grid


def grid_inv(grid, type='grid', lam=0.1, bound='dft', extrapolate=True):
    """Invert a dense deformation (or displacement) grid
    
    Notes
    -----
    The deformation/displacement grid must be expressed in 
    voxels, and map from/to the same lattice.
    
    Let `f = id + d` be the transformation. The inverse 
    is obtained as `id - (k * (f.T @ d)) / (k * (f.T @ 1))`
    where `k` is a smothing kernel, `f.T @ _` is the adjoint 
    operation ("push") of `f @ _` ("pull"). and `1` is an 
    image of ones.
    
    
    Parameters
    ----------
    grid : (..., *spatial, dim)
        Transformation (or displacement) grid
    type : {'grid', 'disp'}, default='grid'
        Type of deformation.
    lam : float, default=0.1
        Regularisation
    bound : str, default='dft'
    extrapolate : bool, default=True
        
    Returns
    -------
    grid_inv : (..., *spatial, dim)
        Inverse transformation (or displacement) grid
    
    """
    # get shape components
    dim = grid.shape[-1]
    shape = grid.shape[-(dim+1):-1]
    batch = grid.shape[:-(dim+1)]
    grid = grid.reshape([-1, *shape, dim])
    backend = dict(dtype=grid.dtype, device=grid.device)
    
    # get displacement
    identity = spatial.identity_grid(shape, **backend)
    if type == 'grid':
        disp = grid - identity
    else:
        disp = grid
        grid = disp + identity
    
    # push displacement
    push_opt = dict(bound=bound, extrapolate=extrapolate)
    disp = core.utils.movedim(disp, -1, 1)
    disp = spatial.grid_push(disp, grid, **push_opt)
    count = spatial.grid_count(grid, **push_opt)
    
    # Fill missing values using regularised least squares
    disp = spatial.solve_field_sym(count, disp, membrane=0.1, 
                                   bound='dft', dim=dim)
    disp = core.utils.movedim(disp, 1, -1)
    disp = disp.reshape([*batch, *shape, dim])
    
    if type == 'grid':
        return identity - disp
    else:
        return -disp


def ffd(source, target, grid_shape=10, group='SE', 
        image_loss=None, def_loss=None, pull=None, preproc=True,
        max_iter=1000, device=None, origin='center',
        init=None, lr=1e-4, optim_affine=True, scheduler=ReduceLROnPlateau):
    """FFD (= cubic spline) registration
    
    Note
    ----
    .. Tensors must have shape (batch, channel, *spatial)
    .. Composite losses (e.g., computed on both intensity and categorical
       images) can be obtained by stacking all types of inputs across 
       the channel dimension. The loss function is then responsible 
       for unstacking the tensor and computing the appropriate 
       losses. The drawback of this approach is that all inputs
       must share the same lattice and orientation matrix, as 
       well as the same interpolation order. The advantage is that 
       it simplifies the signature of this function.

    Parameters
    ----------
    source : tensor or (tensor, affine)
    target : tensor or (tensor, affine)
    group : {'T', 'SO', 'SE', 'CSO', 'GL+', 'Aff+'}, default='SE'
    loss : Loss, default=MutualInfoLoss()
    pull : dict
        interpolation : int, default=1
        bound : bound_like, default='dct2'
        extrapolate : bool, default=False
    preproc : bool, default=True
    max_iter : int, default=1000
    device : device, optional
    origin : {'native', 'center'}, default='center'
    init : tensor_like, default=0
    lr : float, default=0.1
    scheduler : Scheduler, default=ReduceLROnPlateau

    Returns
    -------
    q : tensor
        Parameters
    aff : (D+1, D+1) tensor
        Affine transformation matrix.
        The source affine matrix can be "corrected" by left-multiplying
        it with `aff`.
    moved : tensor
        Source image moved to target space.


    """
    pull = pull or dict()
    pull['interpolation'] = pull.get('interpolation', 'linear')
    pull['bound'] = pull.get('bound', 'dft')
    pull['extrapolate'] = pull.get('extrapolate', False)
    pull_opt = pull
    
    # prepare all data tensors
    ((source, source_aff), (target, target_aff)) = prepare([source, target],
                                                           device)
    backend = get_backend(source)
    batch = source.shape[0]
    src_channels = source.shape[1]
    trg_channels = target.shape[1]
    dim = source.dim() - 2

    # Rescale to [0, 1]
    if preproc:
        source = rescale(source)
        target = rescale(target)

    # Shift origin
    if origin == 'center':
        shift = torch.as_tensor(target.shape, **backend) / 2
        shift = -spatial.affine_matvec(target_aff, shift)
        target_aff[..., :-1, -1] += shift
        source_aff[..., :-1, -1] += shift

    # Prepare affine utils + Initialize parameters
    basis = spatial.affine_basis(group, dim, **backend)
    nb_prm = spatial.affine_basis_size(group, dim)
    if init is not None:
        affine_parameters = torch.as_tensor(init, **backend).clone().detach()
        affine_parameters = affine_parameters.reshape([batch, nb_prm])
    else:
        affine_parameters = torch.zeros([batch, nb_prm], **backend)
    affine_parameters = nn.Parameter(affine_parameters, requires_grad=optim_affine)
    grid_shape = core.py.make_list(grid_shape, dim)
    grid_parameters = torch.zeros([batch, *grid_shape, dim], **backend)
    grid_parameters = nn.Parameter(grid_parameters, requires_grad=True)

    def pull(q, grid):
        aff = core.linalg.expm(q, basis)
        aff = spatial.affine_matmul(aff, target_aff)
        aff = spatial.affine_lmdiv(source_aff, aff)
        expd = (slice(None),) + (None,) * dim + (slice(None), slice(None))
        grid = spatial.affine_matvec(aff[expd], grid)
        moved = spatial.grid_pull(source, grid, **pull_opt)
        return moved

    def exp(prm):
        disp = spatial.resize_grid(prm, type='displacement',
                                   shape=target.shape[2:], 
                                   interpolation=3, bound='dft')
        grid = disp + spatial.identity_grid(target.shape[2:], **backend)
        return disp, grid
        
    
    # Prepare loss and optimizer
    if not callable(image_loss):
        image_loss_fn = nni.MutualInfoLoss()
        factor = 1. if image_loss is None else image_loss
        image_loss = lambda x, y: factor * image_loss_fn(x, y)

    if not callable(def_loss):
        def_loss_fn = nni.BendingLoss(bound='dft')
        factor = 1. if def_loss is None else def_loss
        def_loss = lambda x: factor * def_loss_fn(
            core.utils.last2channel(x))

    lr = core.utils.make_list(lr, 2)
    opt_prm = [{'params': affine_parameters, 'lr': lr[1]}, 
               {'params': grid_parameters, 'lr': lr[0]}] if optim_affine else [grid_parameters]
    optim = torch.optim.Adam(opt_prm, lr=lr[0])
    if scheduler is not None:
        scheduler = scheduler(optim, cooldown=5)

    
#     with torch.no_grad():    
#         disp, grid = exp(grid_parameters)
#         moved = pull(affine_parameters, grid) 
#         plt.imshow(torch.cat([target, moved, source], dim=1).detach().cpu())
#         plt.show()
        
    # Optim loop
    loss_val = core.constants.inf
    loss_avg = 0
    for n_iter in range(max_iter):

        loss_val0 = loss_val
        zero_grad_([affine_parameters, grid_parameters])
        disp, grid = exp(grid_parameters)
        moved = pull(affine_parameters, grid)
        loss_val = image_loss(moved, target) + def_loss(disp[0])
        loss_val.backward()
        optim.step()

        with torch.no_grad():
            loss_avg += loss_val
            
        if n_iter % 10 == 0:
#             print(affine_parameters)
#             plt.imshow(torch.cat([target, moved, source], dim=1).detach().cpu())
#             plt.show()
            
            loss_avg /= 10
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss_avg)
                else:
                    scheduler.step()

            with torch.no_grad():
                if n_iter % 10 == 0:
                    print('{:4d} {:12.6f} | lr={:g}'
                          .format(n_iter, loss_avg.item(),
                                  optim.param_groups[0]['lr']),
                          end='\r')
                    
            loss_avg = 0

    print('')
    with torch.no_grad():
        moved = pull(affine_parameters, grid)
        aff = core.linalg.expm(affine_parameters, basis)
        if origin == 'center':
            aff[..., :-1, -1] -= shift
            shift = core.linalg.matvec(aff[..., :-1, :-1], shift)
            aff[..., :-1, -1] += shift
        aff = aff.inverse()
        aff.requires_grad_(False)
    return affine_parameters, aff, grid_parameters, moved


def diffeo(source, target, group='SE', image_loss=None, vel_loss=None,
           pull=None, preproc=False, max_iter=1000, device=None, origin='center',
           init=None, lr=1e-4, optim_affine=True, scheduler=ReduceLROnPlateau):
    """

    Parameters
    ----------
    source : path or tensor or (tensor, affine)
    target : path or tensor or (tensor, affine)
    group : {'T', 'SO', 'SE', 'CSO', 'GL+', 'Aff+'}, default='SE'
    image_loss : Loss, default=MutualInfoLoss()
    pull : dict
        interpolation : int, default=1
        bound : bound_like, default='dct2'
        extrapolate : bool, default=False
    preproc : bool, default=True
    max_iter : int, default=1000
    device : device, optional
    origin : {'native', 'center'}, default='center'
    init : tensor_like, default=0
    lr: float, default=1e-4
    optim_affine : bool, default=True

    Returns
    -------
    q : tensor
        Parameters
    aff : (D+1, D+1) tensor
        Affine transformation matrix.
        The source affine matrix can be "corrected" by left-multiplying
        it with `aff`.
    vel : (D+1, D+1) tensor
        Initial velocity of the diffeomorphic transform.
        The full warp is `(aff @ aff_src).inv() @ aff_trg @ exp(vel)`
    moved : tensor
        Source image moved to target space.


    """
    pull = pull or dict()
    pull['interpolation'] = pull.get('interpolation', 'linear')
    pull['bound'] = pull.get('bound', 'dct2')
    pull['extrapolate'] = pull.get('extrapolate', False)
    pull_opt = pull
    
    # prepare all data tensors
    ((source, source_aff), (target, target_aff)) = prepare([source, target], device)
    backend = get_backend(source)
    batch = source.shape[0]
    src_channels = source.shape[1]
    trg_channels = target.shape[1]
    dim = source.dim() - 2

    # Rescale to [0, 1]
    source = rescale(source)
    targe = rescale(target)

    # Shift origin
    if origin == 'center':
        shift = torch.as_tensor(target.shape, **backend)/2
        shift = -spatial.affine_matvec(target_aff, shift)
        target_aff = target_aff.clone()
        source_aff = source_aff.clone()
        target_aff[..., :-1, -1] += shift
        source_aff[..., :-1, -1] += shift

    # Prepare affine utils + Initialize parameters
    basis = spatial.affine_basis(group, dim, **backend)
    nb_prm = spatial.affine_basis_size(group, dim)
    if init is not None:
        parameters = torch.as_tensor(init, **backend).clone().detach()
        parameters = parameters.reshape([batch, nb_prm])
    else:
        parameters = torch.zeros([batch, nb_prm], **backend)
    parameters = nn.Parameter(parameters, requires_grad=optim_affine)
    velocity = torch.zeros([batch, *target.shape[2:], dim], **backend)
    velocity = nn.Parameter(velocity, requires_grad=True)

    def pull(q, vel):
        grid = spatial.exp(vel)
        aff = core.linalg.expm(q, basis)
        aff = spatial.affine_matmul(aff, target_aff)
        aff = spatial.affine_lmdiv(source_aff, aff)
        grid = spatial.affine_matvec(aff, grid)
        moved = spatial.grid_pull(source, grid, **pull_opt)
        return moved

    # Prepare loss and optimizer
    if not callable(image_loss):
        image_loss_fn = nni.MutualInfoLoss()
        factor = 1. if image_loss is None else image_loss
        image_loss = lambda x, y: factor*image_loss_fn(x, y)
        
    if not callable(vel_loss):
        vel_loss_fn = nni.BendingLoss(bound='dft')
        factor = 1. if vel_loss is None else vel_loss
        vel_loss = lambda x: factor*vel_loss_fn(core.utils.last2channel(x))

    lr = core.utils.make_list(lr, 2)
    opt_prm = [{'params': parameters}, {'params': velocity, 'lr': lr[1]}] \
              if optim_affine else [velocity]
    optim = torch.optim.Adam(opt_prm, lr=lr[0])
    if scheduler is not None:
        scheduler = scheduler(optim, cooldown=5)

    # Optim loop
    loss_val = core.constants.inf
    loss_avg = 0
    for n_iter in range(1, max_iter+1):

        loss_val0 = loss_val
        optim.zero_grad(set_to_none=True)
        moved = pull(parameters, velocity)
        loss_val = image_loss(moved, target) + vel_loss(velocity)
        loss_val.backward()
        optim.step()
        with torch.no_grad():
            loss_avg += loss_val
            
        if n_iter % 10 == 0:
            loss_avg /= 10
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss_avg)
                else:
                    scheduler.step()

            with torch.no_grad():
                if n_iter % 10 == 0:
                    print('{:4d} {:12.6f} | lr={:g}'
                          .format(n_iter, loss_avg.item(), 
                                  optim.param_groups[0]['lr']),
                          end='\r')
                    
            loss_avg = 0

    print('')
    with torch.no_grad():
        moved = pull(parameters, velocity)
        aff = core.linalg.expm(parameters, basis)
        if origin == 'center':
            aff[..., :-1, -1] -= shift
            shift = core.linalg.matvec(aff[..., :-1, :-1], shift)
            aff[..., :-1, -1] += shift
        aff = aff.inverse()
        aff.requires_grad_(False)
    return parameters, aff, velocity, moved

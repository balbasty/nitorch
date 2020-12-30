import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from nitorch import core, spatial, io
from nitorch import nn as nni


def zero_grad_(param):
    if isinstance(param, (list, tuple)):
        for p in param:
            zero_grad_(p)
        return
    if param.grad is not None:
        param.grad.detach_()
        param.grad.zero_()
    
    
def prepare(inputs, device=None):

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
            inp = io.map(inp)
        if isinstance(inp, io.MappedArray):
            return inp.fdata(rand=True), inp.affine
        inp = torch.as_tensor(inp)
        aff = spatial.affine_default(inp.shape)
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


def affine(source, target, group='SE', loss=None,
           interpolation='linear', bound='dct2', extrapolate=False,
           max_iter=1000, tolerance=1e-5, device=None, origin='center',
           init=None, lr=0.1, scheduler=ReduceLROnPlateau):
    """

    Parameters
    ----------
    source : path or tensor or (tensor, affine)
    target : path or tensor or (tensor, affine)
    group : {'T', 'SO', 'SE', 'CSO', 'GL+', 'Aff+'}, default='SE'
    loss : Loss, default=MutualInfoLoss()
    interpolation : int, default=1
    bound : bound_like, default='dct2'
    extrapolate : bool, default=False
    max_iter : int, default=1000
    tolerance : float, default=1e-5
    device : device, optional
    origin : {'native', 'center'}, default='center'

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
    # prepare all data tensors
    ((source, source_aff), (target, target_aff)) = prepare([source, target], device)
    backend = get_backend(source)
    dim = source.dim()

    # Rescale to [0, 1]
    source_min = source.min()
    source_max = source.max()
    target_min = target.min()
    target_max = target.max()
    source -= source_min
    source /= source_max - source_min
    target -= target_min
    target /= target_max - target_min

    # Shift origin
    if origin == 'center':
        shift = torch.as_tensor(target.shape, **backend)/2
        shift = -spatial.affine_matvec(target_aff, shift)
        target_aff[:-1, -1] += shift
        source_aff[:-1, -1] += shift

    # Prepare affine utils + Initialize parameters
    basis = spatial.affine_basis(group, dim, **backend)
    nb_prm = spatial.affine_basis_size(group, dim)
    if init is not None:
        parameters = torch.as_tensor(init, **backend).clone().detach()
        parameters = parameters.reshape(nb_prm)
    else:
        parameters = torch.zeros(nb_prm, **backend)
    parameters = nn.Parameter(parameters, requires_grad=True)
    identity = spatial.identity_grid(target.shape, **backend)

    def pull(q):
        aff = core.linalg.expm(q, basis)
        aff = spatial.affine_matmul(aff, target_aff)
        aff = spatial.affine_lmdiv(source_aff, aff)
        grid = spatial.affine_matvec(aff, identity)
        moved = spatial.grid_pull(source[None, None, ...], grid[None, ...],
                                  interpolation=interpolation, bound=bound,
                                  extrapolate=extrapolate)[0, 0]
        return moved

    # Prepare loss and optimizer
    if loss is None:
        loss_fn = nni.MutualInfoLoss()
        loss = lambda x, y: loss_fn(x[None, None, ...], y[None, None, ...])

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
            crit = (loss_val0 - loss_val)
            if n_iter % 10 == 0:
                print('{:4d} {:12.6f} ({:12.6g}) | lr={:g}'
                      .format(n_iter, loss_val.item(), crit.item(), 
                              optim.param_groups[0]['lr']), 
                      end='\r')
            if crit.abs() < tolerance:
                break

    print('')
    with torch.no_grad():
        moved = pull(parameters)
        aff = core.linalg.expm(parameters, basis)
        if origin == 'center':
            aff[:-1, -1] -= shift
            shift = core.linalg.matvec(aff[:-1, :-1], shift)
            aff[:-1, -1] += shift
        aff = aff.inverse()
        aff.requires_grad_(False)
    return parameters, aff, moved


def diffeo(source, target, group='SE', image_loss=None, vel_loss=None,
           interpolation='linear', bound='dct2', extrapolate=False,
           max_iter=1000, tolerance=1e-5, device=None, origin='center',
           init=None, lr=1e-4, optim_affine=True, scheduler=ReduceLROnPlateau):
    """

    Parameters
    ----------
    source : path or tensor or (tensor, affine)
    target : path or tensor or (tensor, affine)
    group : {'T', 'SO', 'SE', 'CSO', 'GL+', 'Aff+'}, default='SE'
    image_loss : Loss, default=MutualInfoLoss()
    vel_loss : Loss, default=MembraneLoss()
    interpolation : int, default=1
    bound : bound_like, default='dct2'
    extrapolate : bool, default=False
    max_iter : int, default=1000
    tolerance : float, default=1e-5
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
    # prepare all data tensors
    ((source, source_aff), (target, target_aff)) = prepare([source, target], device)
    backend = get_backend(source)
    dim = source.dim()

    # Rescale to [0, 1]
    source_min = source.min()
    source_max = source.max()
    target_min = target.min()
    target_max = target.max()
    source -= source_min
    source /= source_max - source_min
    target -= target_min
    target /= target_max - target_min

    # Shift origin
    if origin == 'center':
        shift = torch.as_tensor(target.shape, **backend)/2
        shift = -spatial.affine_matvec(target_aff, shift)
        target_aff[:-1, -1] += shift
        source_aff[:-1, -1] += shift

    # Prepare affine utils + Initialize parameters
    basis = spatial.affine_basis(group, dim, **backend)
    nb_prm = spatial.affine_basis_size(group, dim)
    if init is not None:
        parameters = torch.as_tensor(init, **backend).clone().detach()
        parameters = parameters.reshape(nb_prm)
    else:
        parameters = torch.zeros(nb_prm, **backend)
    parameters = nn.Parameter(parameters, requires_grad=optim_affine)
    velocity = torch.zeros([*target.shape, dim], **backend)
    velocity = nn.Parameter(velocity, requires_grad=True)

    def pull(q, vel):
        grid = spatial.exp(vel[None, ...])
        aff = core.linalg.expm(q, basis)
        aff = spatial.affine_matmul(aff, target_aff)
        aff = spatial.affine_lmdiv(source_aff, aff)
        grid = spatial.affine_matvec(aff, grid)
        moved = spatial.grid_pull(source[None, None, ...], grid,
                                  interpolation=interpolation, bound=bound,
                                  extrapolate=extrapolate)[0, 0]
        return moved

    # Prepare loss and optimizer
    if not callable(image_loss):
        image_loss_fn = nni.MutualInfoLoss()
        factor = 1. if image_loss is None else image_loss
        image_loss = lambda x, y: factor*image_loss_fn(x[None, None, ...],
                                                       y[None, None, ...])
        
    if not callable(vel_loss):
        vel_loss_fn = nni.BendingLoss()
        factor = 1. if vel_loss is None else vel_loss
        vel_loss = lambda x: factor*vel_loss_fn(core.utils.last2channel(x[None, ...]))

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
                crit = (loss_val0 - loss_val)
                if n_iter % 10 == 0:
                    print('{:4d} {:12.6f} ({:12.6g}) | lr={:g}'
                          .format(n_iter, loss_avg.item(), crit.item(),
                                  optim.param_groups[0]['lr']),
                          end='\r')
                if crit.abs() < tolerance:
                    break
                    
            loss_avg = 0

    print('')
    with torch.no_grad():
        moved = pull(parameters, velocity)
        aff = core.linalg.expm(parameters, basis)
        if origin == 'center':
            aff[:-1, -1] -= shift
            shift = core.linalg.matvec(aff[:-1, :-1], shift)
            aff[:-1, -1] += shift
        aff = aff.inverse()
        aff.requires_grad_(False)
    return parameters, aff, velocity, moved

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nitorch import core, spatial
from nitorch import nn as nni
from._utils import (affine_group_converter, prepare, get_backend,
                    BacktrackingLineSearch)


def ffd(source, target, grid_shape=10, group='SE', origin='center',
        image_loss=None, def_loss=None, pull=None, optim_affine=True,
        max_iter=1000, lr=0.1, min_lr=1e-7, init=None, device=None):
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
    source :  tensor or (tensor, affine)
        The source (moving) image, with shape (batch, channel, *spatial).
    target : tensor or (tensor, affine)
        The target (fixed) image, with shape (batch, channel, *spatial).
    grid_shape : (sequence os) int, default=10
        Number of FFD ndes per dimension.
    group : {'tr', 'rot', 'rigid', 'sim', 'lin', 'aff'}, default='rigid'
        Affine sub-group to optimize.
    origin : {'native', 'center'}, default='center'
        Whether to rotate about the origin of the world-space ('native')
        or the center of the target field-of-view ('center').
        When the origin of the world-space is far off (say you are
        registering smaller blocks cropped from a larger MRI), it can
        be beneficiary to rotate about the center of the FOV.
    image_loss : callable(mov, fix) -> loss, default=MutualInfoLoss()
        A loss function that takestwo  inputs of shape
        (batch, channel, *spatial).
    def_loss : callable(mov, fix) -> loss, default=MutualInfoLoss()
        A loss function that takestwo  inputs of shape
        (batch, channel, *spatial).
    pull : dict
        interpolation : int, default=1
            Interpolation order
        bound : bound_like, default='dct2'
            Boundary condition
        extrapolate : bool, default=False
            Extrapolate out-of-bound data using the boundary conditions.
    max_iter : int, default=1000
        Maximum number of iterations
    lr : float, default=0.1
        Initial learning rate.
    min_lr : float, default=1e-7
        Minimum learning rate. The optimization is stopped once this
        learning rate is reached.
    device : {'cpu', 'cuda', 'cuda:<id>'}, optional
        Backend to use
    init : ([batch], nb_prm) tensor_like, default=0
        Initial guess for the affine parameters.

    Returns
    -------
    q : (batch, nb_prm) tensor
        Parameters
    aff : (batch, D+1, D+1) tensor
        Affine transformation matrix.
        The source affine matrix can be "corrected" by left-multiplying
        it with `aff`.
    ffd : (batch, *grid_shape, D) tensor
        FFD parameters
    moved : tensor
        Source image moved to target space.

    """
    group = affine_group_converter(group)
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
    dim = source.dim() - 2

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
    affine_parameters = nn.Parameter(affine_parameters,
                                     requires_grad=optim_affine)
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
        def_loss = lambda x: factor * def_loss_fn(core.utils.last2channel(x))

    lr = core.utils.make_list(lr, 2)
    min_lr = core.utils.make_list(min_lr, 2)
    opt_prm = [{'params': affine_parameters, 'lr': lr[1]},
               {'params': grid_parameters, 'lr': lr[0]}] \
              if optim_affine else [grid_parameters]
    optim = torch.optim.Adam(opt_prm, lr=lr[0])
    scheduler = ReduceLROnPlateau(optim)

    def forward():
        disp, grid = exp(grid_parameters)
        moved = pull(affine_parameters, grid)
        loss_val = image_loss(moved, target) + def_loss(disp[0])
        return loss_val

    # Optim loop
    loss_avg = 0
    for n_iter in range(max_iter):

        optim.zero_grad(set_to_none=True)
        loss_val = forward()
        loss_val.backward()
        loss_val = optim.step(forward)

        with torch.no_grad():
            loss_avg += loss_val

            if n_iter % 10 == 0:
                loss_avg /= 10
                scheduler.step(loss_avg)

                print('{:4d} {:12.6f} | lr={:g}'
                      .format(n_iter, loss_avg.item(),
                              optim.param_groups[0]['lr']),
                      end='\r')
                loss_avg = 0

        if (optim.param_groups[0]['lr'] < min_lr[0] and
                (len(optim.param_groups) == 1 or
                 optim.param_groups[1]['lr'] < min_lr[1])):
            print('\nConverged.')
            break

    print('')
    with torch.no_grad():
        disp, grid = exp(grid_parameters)
        moved = pull(affine_parameters, grid)
        aff = core.linalg.expm(affine_parameters, basis)
        if origin == 'center':
            aff[..., :-1, -1] -= shift
            shift = core.linalg.matvec(aff[..., :-1, :-1], shift)
            aff[..., :-1, -1] += shift
        aff = aff.inverse()
    return (affine_parameters.detach(),
            aff.detach(),
            grid_parameters.detach(),
            moved.detach())

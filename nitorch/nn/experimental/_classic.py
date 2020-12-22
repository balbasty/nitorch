import torch
from torch import nn
from torch.autograd import Variable
from nitorch import core, spatial, io
from nitorch import nn as nni


def affine(source, target, group='SE', loss=None, optim=None,
           interpolation='linear', bound='dct2', extrapolate=False,
           max_iter=1000, tolerance=1e-5, device=None, origin='center',
           init=None):
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
    aff : (D+1, D+1) tensor
        Affine transformation matrix.
        The source affine matrix can be "corrected" by left-multiplying
        it with `aff`.
    moved : tensor
        Source image moved to target space.


    """
    def prepare(inp):
        if isinstance(inp, (list, tuple)):
            has_aff = len(inp) > 1
            if has_aff:
                aff0 = inp[1]
            inp, aff = prepare(inp[0])
            if has_aff:
                aff = aff0
            return inp, aff
        if isinstance(inp, str):
            inp = io.map(inp)
        if isinstance(inp, io.MappedArray):
            return inp.fdata(rand=True), inp.affine
        inp = torch.as_tensor(inp)
        aff = spatial.affine_default(inp.shape)
        return inp, aff

    # prepare all data tensors
    source, source_aff = prepare(source)
    target, target_aff = prepare(target)
    source = source.to(device=device, dtype=torch.float32)
    device = source.device
    dtype = source.dtype
    backend = dict(dtype=dtype, device=device)
    source_aff = source_aff.to(**backend)
    target_aff = target_aff.to(**backend)
    target = target.to(**backend)
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
        parameters = torch.as_tensor(init, **backend)
        parameters = parameters.reshape(nb_prm)
    else:
        parameters = torch.zeros(nb_prm, **backend)
    parameters = Variable(parameters, requires_grad=True)
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

    if optim is None:
        optim = torch.optim.Adam
    optim = optim([parameters], lr=1e-4)

    # Optim loop
    loss_val = core.constants.inf
    for n_iter in range(max_iter):

        loss_val0 = loss_val
        moved = pull(parameters)
        loss_val = loss(moved, target)
        loss_val.backward()
        optim.step()

        with torch.no_grad():
            crit = (loss_val0 - loss_val)
            if n_iter % 10 == 0:
                print('{:4d} {:12.6f} ({:12.6g})'
                      .format(n_iter, loss_val.item(), crit.item()), 
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
        return aff, moved










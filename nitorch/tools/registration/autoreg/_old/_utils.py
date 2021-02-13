import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nitorch import core, spatial, io
from nitorch import nn as nni
# import matplotlib.pyplot as plt


class BacktrackingLineSearch(torch.optim.Optimizer):

    def __init__(self, optim, armijo=1, max_iter=6):
        self.optim = optim
        self.armijo = float(armijo)
        self.max_iter = max_iter

    def step(self, closure, loss=None):
        """

        Parameters
        ----------
        closure : callable
        loss : tensor, optional

        Returns
        -------
        tensor

        """

        def get_params():
            params = []
            for group in self.optim.param_groups:
                params.extend(group['params'])
            return params

        with torch.no_grad():
            if loss is None:
                loss = closure()
            armijo = self.armijo

            params0 = [param.detach().clone() for param in get_params()]
            self.optim.step()
            deltas = [p - p0 for p, p0 in zip(get_params(), params0)]
            self.last_ok = False
            for n_iter in range(self.max_iter):
                new_loss = closure()
                if new_loss < loss:
                    armijo = 2 * armijo
                    self.last_ok = True
                    break
                else:
                    armijo = armijo / 2.
                    for param, param0, delta in zip(get_params(), params0,
                                                    deltas):
                        param.copy_(param0 + armijo * delta)
        return new_loss

    def add_param_group(self, param_group: dict) -> None:
        return self.optim.add_param_group(param_group)

    def load_state_dict(self, state_dict: dict) -> None:
        return self.optim.load_state_dict(state_dict)

    def state_dict(self):
        return self.optim.state_dict()

    def zero_grad(self, *args, **kwargs) -> None:
        return self.optim.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups


def affine_group_converter(group):
    """Convert classic affine types to names of Lie groups."""
    if group.lower().startswith('t'):
        return 'T'
    elif group.lower().startswith('rot'):
        return 'SO'
    elif group.lower().startswith('rig'):
        return 'SE'
    elif group.lower().startswith('sim'):
        return 'CSO'
    elif group.lower().startswith('l'):
        return 'GL+'
    elif group.lower().startswith('a'):
        return 'Aff+'
    elif group in spatial.affine_basis_choices:
        return group
    else:
        raise ValueError(f'Unknown affine group {group}')


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




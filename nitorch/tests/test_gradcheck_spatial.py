import torch
from torch.autograd import gradcheck
from nitorch.spatial import grid_grad, grid_pull, grid_push, grid_count
from nitorch.spatial import identity_grid, BoundType, InterpolationType
import pytest

# global parameters
dtype = torch.double        # data type (double advised to check gradients)
shape1 = 3                  # size along each dimension

# parameters
bounds = [bound for bound in BoundType.__members__.keys() if bound != 'sliding']
orders = [order for order in InterpolationType.__members__.keys()]
devices = [('cpu', 1), ('cpu', 10), 'cuda']
dims = [1, 2, 3]


def make_data(shape, device, dtype):
    id = identity_grid(shape, dtype=dtype, device=device)
    id = id[None, ...]  # add batch dimension
    disp = torch.rand(id.shape, device=device, dtype=dtype)
    grid = id + disp
    vol = torch.rand((1, 1) + shape, device=device, dtype=dtype)
    return vol, grid


def init_device(device):
    if isinstance(device, (list, tuple)):
        device, param = device
    else:
        param = 1 if device == 'cpu' else 0
    if device == 'cuda':
        torch.cuda.set_device(param)
        torch.cuda.init()
        torch.cuda.empty_cache()
    else:
        assert device == 'cpu'
        torch.set_num_threads(param)


@pytest.mark.parametrize("device,dim,bound,interpolation",
                         devices, dims, bounds, orders)
def test_gradcheck_grid_grad(device, dim, bound, interpolation):
    init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_grad, (vol, grid, interpolation, bound, True),
                     rtol=1., raise_exception=False)


@pytest.mark.parametrize("device,dim,bound,interpolation",
                         devices, dims, bounds, orders)
def test_gradcheck_grid_pull(device, dim, bound, interpolation):
    init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_pull, (vol, grid, interpolation, bound, True),
                     rtol=1., raise_exception=False)


@pytest.mark.parametrize("device,dim,bound,interpolation",
                         devices, dims, bounds, orders)
def test_gradcheck_grid_push(device, dim, bound, interpolation):
    init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_push, (vol, grid, shape, interpolation, bound, True),
                     rtol=1., raise_exception=False)


@pytest.mark.parametrize("device,dim,bound,interpolation",
                         devices, dims, bounds, orders)
def test_gradcheck_grid_count(device, dim, bound, interpolation):
    init_device(device)
    shape = (shape1,) * dim
    _, grid = make_data(shape, device, dtype)
    grid.requires_grad = True
    assert gradcheck(grid_count, (grid, shape, interpolation, bound, True),
                     rtol=1., raise_exception=False)



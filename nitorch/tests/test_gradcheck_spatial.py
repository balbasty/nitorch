import torch
from torch.autograd import gradcheck
from nitorch.spatial import grid_grad, grid_pull, grid_push, grid_count
from nitorch.spatial import identity_grid, BoundType, InterpolationType
import pytest

# global parameters
dtype = torch.double        # data type (double advised to check gradients)
shape1 = 3                  # size along each dimension

# parameters
bounds = set(BoundType.__members__.values())
orders = set(InterpolationType.__members__.values())
devices = [('cpu', 1)]
if torch.backends.openmp.is_available() or torch.backends.mkl.is_available():
    print('parallel backend available')
    devices.append(('cpu', 10))
if torch.cuda.is_available():
    print('cuda backend available')
    devices.append('cuda')
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
        device = '{}:{}'.format(device, param)
    else:
        assert device == 'cpu'
        torch.set_num_threads(param)
    return torch.device(device)


# FIXME: grid_grad checks are failing
# @pytest.mark.parametrize("device", devices)
# @pytest.mark.parametrize("dim", dims)
# @pytest.mark.parametrize("bound", bounds)
# @pytest.mark.parametrize("interpolation", orders)
# def test_gradcheck_grid_grad(device, dim, bound, interpolation):
#     print('grid_grad_{}d({}, {}) on {}'.format(dim, interpolation, bound, device))
#     device = init_device(device)
#     shape = (shape1,) * dim
#     vol, grid = make_data(shape, device, dtype)
#     vol.requires_grad = True
#     grid.requires_grad = True
#     assert gradcheck(grid_grad, (vol, grid, interpolation, bound, True),
#                      rtol=1., raise_exception=False)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_grid_pull(device, dim, bound, interpolation):
    print('grid_pull_{}d({}, {}) on {}'.format(dim, interpolation, bound, device))
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_pull, (vol, grid, interpolation, bound, True),
                     rtol=1., raise_exception=False)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_grid_push(device, dim, bound, interpolation):
    print('grid_push_{}d({}, {}) on {}'.format(dim, interpolation, bound, device))
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_push, (vol, grid, shape, interpolation, bound, True),
                     rtol=1., raise_exception=False)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_grid_count(device, dim, bound, interpolation):
    print('grid_count_{}d({}, {}) on {}'.format(dim, interpolation, bound, device))
    device = init_device(device)
    shape = (shape1,) * dim
    _, grid = make_data(shape, device, dtype)
    grid.requires_grad = True
    assert gradcheck(grid_count, (grid, shape, interpolation, bound, True),
                     rtol=1., raise_exception=False)

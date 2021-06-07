import torch
from nitorch import spatial
from nitorch.core import utils, math


def augment(x, rand=0.05, fwhm=2, dim=None):
    """Add noise + smooth"""
    dim = dim or x.dim()
    if rand:
        x += rand * torch.randn_like(x)
    if fwhm:
        x = spatial.smooth(x, fwhm=2, dim=dim)
    return x


def make_cat(x):
    x[x == 0] = -1
    return x


def sigmoid(x):
    return x.neg().exp_().add_(1).reciprocal_()


def circle(shape=(64, 64), radius=None, **backend):
    """Generate a circle"""
    circle = spatial.identity_grid(shape, **backend)
    backend = dict(dtype=circle.dtype, device=circle.device)
    radius = radius or min(shape) // 2
    radius = utils.make_vector(radius, len(shape), **backend)

    circle = circle.sub_(min(shape)/2).div_(radius).square_().sum(-1) < 1
    circle = circle.to(**backend)
    return circle


def square(shape=(64, 64), radius=None, **backend):
    """Generate a square"""
    square = spatial.identity_grid(shape, **backend)
    backend = dict(dtype=square.dtype, device=square.device)
    radius = radius or min(shape) // 2
    radius = utils.make_vector(radius, len(shape), **backend)

    square = (square.sub_(min(shape)/2).div_(radius).abs_() < 1).any(-1)
    square = square.to(**backend)
    return square


def squircle(shape=(64, 64), order=4, radius=None, **backend):
    """Generate a squircle"""
    squircle = spatial.identity_grid(shape, **backend)
    backend = dict(dtype=squircle.dtype, device=squircle.device)
    radius = radius or min(shape) // 2
    radius = utils.make_vector(radius, len(shape), **backend)

    square = squircle.sub_(min(shape)/2).div_(radius).abs_().pow_(order).sum(-1) < 1
    square = square.to(**backend)
    return square


def demo_register(shape=(64, 64), cat=False, **backend):
    """Generate a simulated 2d dataset (circle and square)"""
    torch.random.manual_seed(1234)
    fixed = augment(square(shape, **backend))[None]
    moving = augment(circle(shape, **backend))[None]
    if cat:
        fixed = sigmoid(make_cat(fixed))
        moving = make_cat(moving)
    return fixed, moving


def demo_atlas(batch=32, shape=(64, 64), cat=False, **backend):
    """Generate a simulated 2d dataset
    (squircles with random radii and orders)
    """
    torch.random.manual_seed(1234)
    data = torch.empty([batch, 1, *shape], **backend)
    for b in range(batch):
        order = (torch.rand([],  **backend) + 1) * 2.5
        radius = (torch.rand([], **backend) + 0.25) * (0.25*max(shape))
        x = squircle(shape, order, radius, **backend)
        data[b, 0] = x
    data = augment(data, dim=data.dim()-2)
    if cat:
        data = sigmoid(make_cat(data))
    return data

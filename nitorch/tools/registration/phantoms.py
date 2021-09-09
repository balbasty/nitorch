"""This file implements simple numerical phantoms (circle, square, squircle,
letter C) to test registration algorithms."""

import torch
from nitorch import spatial
from nitorch.core import utils, math, py


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
    radius = radius or min(shape) / 4
    radius = utils.make_vector(radius, len(shape), **backend)

    circle = circle.sub_(min(shape)/2).div_(radius)
    circle *= circle
    circle = circle.sum(-1) < 1
    circle = circle.to(**backend)
    return circle


def square(shape=(64, 64), radius=None, **backend):
    """Generate a square"""
    square = spatial.identity_grid(shape, **backend)
    backend = dict(dtype=square.dtype, device=square.device)
    radius = radius or min(shape) / 4
    radius = utils.make_vector(radius, len(shape), **backend)

    square = (square.sub_(min(shape)/2).div_(radius).abs_() < 1).all(-1)
    square = square.to(**backend)
    return square


def squircle(shape=(64, 64), order=4, radius=None, **backend):
    """Generate a squircle"""
    squircle = spatial.identity_grid(shape, **backend)
    backend = dict(dtype=squircle.dtype, device=squircle.device)
    radius = radius or min(shape) / 4
    radius = utils.make_vector(radius, len(shape), **backend)

    square = squircle.sub_(min(shape)/2).div_(radius).abs_().pow_(order).sum(-1) < 1
    square = square.to(**backend)
    return square


def letterc(shape=(64, 64), radius=None, width=None, rotation=0, **backend):
    """Generate the letter C (optionally rotated)"""
    if len(shape) != 2:
        raise NotImplementedError('Letter C only implemented in 2d')
    radius = radius or (min(shape) / 4)
    width = width or (radius / 2)
    c = circle(shape, radius, **backend)
    backend = utils.backend(c)
    innerc = circle(shape, radius-width, **backend).bool()
    c[innerc] = 0
    hole = (slice(int((shape[0]-width)//2), int((shape[0]+width)//2)),
            slice(int(shape[0]//2), None))
    c[hole] = 0

    if rotation:
        mat = spatial.affine_matrix_classic(dim=2, rotations=[rotation*3.14/180])
        mat = mat.to(**backend)
        aff = spatial.affine_default(shape, **backend)
        mat = spatial.affine_matmul(mat, aff)
        aff = spatial.affine_inv(aff)
        mat = spatial.affine_matmul(aff, mat)
        grid = spatial.affine_grid(mat, shape)
        c = spatial.grid_pull(c, grid)
        c = (c > 0.5).to(**backend)
    return c


def demo_register(shape=(64, 64), cat=False, fixed='square', moving='circle',
                  shift=0, **backend):
    """Generate a simulated 2d dataset (circle and square)"""
    torch.random.manual_seed(1234)
    fixed = (square if fixed == 'square' else
             circle if fixed == 'circle' else
             squircle if fixed == 'squircle' else
             letterc if fixed == 'c' else
             circle)
    moving = (square if moving == 'square' else
             circle if moving == 'circle' else
             squircle if moving == 'squircle' else
             letterc if moving == 'c' else
             circle)
    fixed = augment(fixed(shape, **backend))[None]
    moving = augment(moving(shape, **backend))[None]
    if cat:
        fixed = sigmoid(make_cat(fixed))
        moving = make_cat(moving)
    shift = py.make_list(shift, len(shape))
    if shift[0]:
        moving = torch.cat([moving[:, 8:, :], moving[:, :8, :]], 1)
    if shift[1]:
        moving = torch.cat([moving[:, :, 8:], moving[:, :, :8]], 2)
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


def demo_atlas_c(batch=32, shape=(64, 64), cat=False, **backend):
    """Generate a simulated 2d dataset
    (squircles with random radii and orders)
    """
    torch.random.manual_seed(1234)
    data = torch.empty([batch, 1, *shape], **backend)
    for b in range(batch):
        radius = (torch.rand([], **backend) + 0.25) * (2/5*max(shape))
        width = (torch.rand([], **backend) + 0.5) * (radius/4)
        rotation = torch.rand([], **backend) * 360
        x = letterc(shape, radius, width=width, rotation=rotation, **backend)
        data[b, 0] = x
    data = augment(data, dim=data.dim()-2)
    if cat:
        data = sigmoid(make_cat(data))
    return data

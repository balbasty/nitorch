"""Derivatives of the B-spline basis functions of the TorchScript backend
against finite differences."""
import torch
import pytest
from nitorch._C._ts.splines import Spline


def points():
    x = torch.linspace(-4.3, 4.3, 2001, dtype=torch.double)
    # stay away from the knots (integers and half-integers)
    return x[((2 * x) - (2 * x).round()).abs() > 1e-2]


@pytest.mark.parametrize('order', range(1, 8))
def test_spline_grad_hess(order):
    x, eps, s = points(), 1e-4, Spline(order)
    grad = (s.weight(x + eps) - s.weight(x - eps)) / (2 * eps)
    hess = (s.weight(x + eps) - 2 * s.weight(x) + s.weight(x - eps)) / eps**2
    assert torch.allclose(s.grad(x), grad, rtol=0, atol=1e-7)
    assert torch.allclose(s.hess(x), hess, rtol=0, atol=1e-5)

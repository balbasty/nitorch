"""Checks of the robust (IRLS) losses used by `nitorch register`
(`nitorch.tools.registration.losses.robust`) and of their CLI aliases."""
import torch
import pytest
from nitorch.tools.registration.losses.robust import MAD, Tukey
from nitorch.cli.registration.register.parser import parser


def make_data(seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = torch.randn([1, 8, 8], dtype=torch.double, generator=g)
    mov = fix + 0.1 * torch.randn(fix.shape, dtype=torch.double, generator=g)
    mov[0, 0, 0] += 10  # outlier
    return mov, fix


@pytest.mark.parametrize('cls', [MAD, Tukey])
@pytest.mark.parametrize('lam', [1., None])
def test_robust_downweights_outliers(cls, lam):
    mov, fix = make_data()
    _, g, h = cls(lam=lam).loss_grad_hess(mov, fix)
    # effective weight = hessian (lam * w); outlier must get less than inliers
    assert h[0, 0, 0] < h[0, 1:, 1:].median()
    # gradient at the outlier must not dominate
    assert g[0, 0, 0].abs() <= g.abs().max()
    assert g[0, 0, 0].abs() < 10 * g[0, 1:, 1:].abs().median() + 1e-12


def test_mad_grad_fd():
    mov, fix = make_data()
    loss = MAD(lam=1.)
    _, g = loss.loss_grad(mov, fix)
    eps = 1e-6
    fd = torch.zeros_like(mov)
    for i in range(mov.numel()):
        m1 = mov.clone().view(-1); m1[i] += eps
        m2 = mov.clone().view(-1); m2[i] -= eps
        fd.view(-1)[i] = (loss.loss(m1.view(mov.shape), fix)
                          - loss.loss(m2.view(mov.shape), fix)) / (2 * eps)
    assert torch.allclose(g, fd, rtol=1e-4, atol=1e-8)
    # l1 gradient: sign(r) / nvox
    assert torch.allclose(g, (mov - fix).sign() / mov.numel())


def test_tukey_grad_is_psi():
    mov, fix = make_data()
    c = 4.685
    _, g = Tukey(lam=1., c=c).loss_grad(mov, fix)
    r = mov - fix
    psi = r * (1 - (r / c) ** 2).square() * (r.abs() < c)
    assert torch.allclose(g, psi / mov.numel())


BASE = ['@@fix', 'fix.nii', '@@mov', 'mov.nii', '@nonlin', 'svf']


@pytest.mark.parametrize('alias, name', [('l1', 'mad'), ('l2', 'mse'),
                                          ('mad', 'mad'), ('mse', 'mse')])
def test_register_loss_aliases(alias, name):
    options = parser.parse(['@loss', alias] + BASE)
    assert options.loss[0].name == name

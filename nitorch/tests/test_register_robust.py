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


def make_noisy(seed=0, zeros=0, outliers=True):
    g = torch.Generator().manual_seed(seed)
    fix = torch.randn([2, 16, 16], dtype=torch.double, generator=g)
    sd = torch.as_tensor([0.5, 2.], dtype=torch.double)[:, None, None]
    mov = fix + sd * torch.randn(fix.shape, dtype=torch.double, generator=g)
    if outliers:
        out = torch.rand(fix.shape, generator=g) < 0.1
        mov[out] += 20 * torch.randn(int(out.sum()), dtype=torch.double,
                                     generator=g)
    if zeros:
        # voxels with exactly zero residual (e.g., background)
        mov[:, -1, :zeros] = fix[:, -1, :zeros]
    return mov, fix


@pytest.mark.parametrize('cls', [MAD, Tukey])
@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('zeros', [0, 4])
def test_robust_estimated_lam_grad_fd(cls, joint, zeros):
    """With lam estimated at each call, the finite differences of the
    value match the gradient (the value is the IRLS objective at its
    joint optimum in (lam, w))"""
    mov, fix = make_noisy(zeros=zeros)
    loss = cls(lam=None, joint=joint, cache=False)
    _, g = loss.loss_grad(mov, fix)
    gen = torch.Generator().manual_seed(1)
    dirs = [g] + [torch.randn(mov.shape, dtype=torch.double, generator=gen)
                  for _ in range(3)]
    eps = 1e-6
    for d in dirs:
        d = d / d.norm()
        fd = (loss.loss(mov + eps * d, fix) -
              loss.loss(mov - eps * d, fix)) / (2 * eps)
        an = (g * d).sum()
        # (up to the IRLS tolerance)
        assert abs(fd - an) < 1e-3 * g.norm(), (fd, an)


@pytest.mark.parametrize('cls', [MAD, Tukey])
def test_robust_estimated_lam_value_consistent(cls):
    """The value returned with an estimated lam is the fixed-lam value at
    the final lam, plus the log-precision term"""
    mov, fix = make_noisy()
    loss = cls(lam=None, cache=True)
    ll = loss.loss(mov, fix)
    lam = loss.lam
    ll_fixed = cls(lam=lam.flatten()).loss(mov, fix)
    kappa = cls.kappa() if cls is MAD else cls.kappa(c=loss.c)
    assert torch.allclose(ll, ll_fixed - 0.5 * kappa * lam.log().sum())
    # weights are those of the final lam
    _, _, lamw = loss.irls(mov, fix, None, None, False, 2,
                           **({'c': loss.c} if cls is Tukey else {}))
    w = cls.reweight(mov, fix, lam=lam, dim=2, mask=torch.ones_like(fix),
                     **({'c': loss.c} if cls is Tukey else {}))
    assert torch.allclose(lamw, lam * w, rtol=1e-5)


@pytest.mark.parametrize('cls', [MAD, Tukey])
@pytest.mark.parametrize('joint', [False, True])
def test_robust_irls_converges(cls, joint):
    mov, fix = make_noisy()
    loss = cls(lam=None, joint=joint, cache=False)
    calls = []
    reweight = cls.reweight

    def counting(*a, **k):
        calls.append(1)
        return reweight(*a, **k)

    loss.reweight = counting
    loss.loss(mov, fix)
    # IRLS loop (at most 64) + one final reweight
    assert len(calls) < 64


def test_mad_estimated_lam_zero_residuals():
    """MAD's lam is the Laplace maximum-likelihood estimate 1/mean|r|^2,
    which stays finite when some residuals are exactly zero"""
    for zeros in (0, 4):
        mov, fix = make_noisy(zeros=zeros)
        loss = MAD(lam=None)
        ll = loss.loss(mov, fix)
        lam_ml = (mov - fix).abs().mean([-1, -2]).square().reciprocal()
        assert torch.allclose(loss.lam.flatten(), lam_ml, rtol=1e-5)
        # value = Laplace negative log-likelihood (without log 2)
        r = (mov - fix).abs()
        ref = (lam_ml.sqrt()[:, None, None] * r).mean([-1, -2]).sum() \
            - 0.5 * lam_ml.log().sum()
        assert torch.allclose(ll, ref, rtol=1e-5)


def test_tukey_estimated_lam_gaussian():
    """Without outliers, Tukey's lam is consistent with the noise precision"""
    mov, fix = make_noisy(outliers=False)
    g = torch.Generator().manual_seed(3)
    fix = torch.randn([2, 64, 64], dtype=torch.double, generator=g)
    sd = torch.as_tensor([0.5, 2.], dtype=torch.double)[:, None, None]
    mov = fix + sd * torch.randn(fix.shape, dtype=torch.double, generator=g)
    loss = Tukey(lam=None)
    loss.loss(mov, fix)
    assert torch.allclose(loss.lam.flatten(), sd.flatten() ** -2, rtol=0.05)


@pytest.mark.parametrize('lam', [1., 4.])
def test_tukey_value_is_rho(lam):
    """With a fixed lam, the value is Tukey's rho of the standardised
    residual: c^2/6 (1 - (1 - s^2/c^2)^3) if |s| < c, else c^2/6"""
    c = 4.685
    r = torch.linspace(0, 2 * c, 101, dtype=torch.double)[None, :, None]
    ll = Tukey(lam=lam, c=c).loss(r, torch.zeros_like(r))
    s2 = lam * r.square()
    rho = torch.where(s2 < c * c, c * c / 6 * (1 - (1 - s2 / (c * c)) ** 3),
                      torch.full_like(s2, c * c / 6))
    assert torch.allclose(ll, rho.mean(), rtol=1e-6)


BASE = ['@@fix', 'fix.nii', '@@mov', 'mov.nii', '@nonlin', 'svf']


@pytest.mark.parametrize('alias, name', [('l1', 'mad'), ('l2', 'mse'),
                                          ('mad', 'mad'), ('mse', 'mse')])
def test_register_loss_aliases(alias, name):
    options = parser.parse(['@loss', alias] + BASE)
    assert options.loss[0].name == name

"""Checks of the robust (IRLS) losses used by `nitorch register`
(`nitorch.tools.registration.losses.robust`) and of their CLI aliases."""
import copy
import importlib
import math
import torch
import pytest
from nitorch.tools.registration.losses.robust import MAD, Tukey, RobustLoss
from nitorch.cli.registration.register.parser import parser

robust = importlib.import_module('nitorch.tools.registration.losses.robust')


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


# ----------------------------------------------------------------------
#   Estimated lam with C channels (joint or not)
# ----------------------------------------------------------------------
LAM = torch.as_tensor([4., 0.25, 1., 9., 2.], dtype=torch.double)
NB_CHANNELS = [1, 2, 3, 5]


def make_gaussian(C, seed=0, n=256):
    """Gaussian noise with precision LAM[:C] (no outliers), 2D"""
    g = torch.Generator().manual_seed(seed)
    fix = torch.randn([C, n, n], dtype=torch.double, generator=g)
    noise = torch.randn([C, n, n], dtype=torch.double, generator=g)
    return fix + LAM[:C, None, None].rsqrt() * noise, fix


def mean_chi(C):
    """E[χ_C] = √2 Γ((C+1)/2) / Γ(C/2)"""
    return math.sqrt(2) * math.exp(math.lgamma((C + 1) / 2)
                                   - math.lgamma(C / 2))


def test_tukey_kappa_closed_form():
    """κ(C) = E[w(S²) S²] / C, S² ~ χ²(C): the closed form matches the
    numerical integral of #120 for C = 1, and a quadrature of the χ²
    density for C > 1"""
    for c in (2., 3., 4.685, 6.):
        assert robust._tukey_kappa_joint(c, 1) == pytest.approx(
            robust._tukey_kappa(c), rel=1e-12)
    c = 4.685
    x = torch.linspace(0, c * c, 200001, dtype=torch.double)[1:]
    for C in (2, 3, 5):
        logpdf = ((C / 2 - 1) * x.log() - x / 2
                  - (C / 2) * math.log(2) - math.lgamma(C / 2))
        f = x * (1 - x / (c * c)).square() * logpdf.exp() / C
        ref = torch.trapezoid(f, x).item()
        assert Tukey.kappa(c=c, nb_channels=C) == pytest.approx(ref, rel=1e-6)
    # values quoted in the docstring
    kappas = [Tukey.kappa(nb_channels=C) for C in NB_CHANNELS]
    assert kappas == pytest.approx([0.7578, 0.6853, 0.6170, 0.4929], abs=1e-4)
    # one channel (non-joint, or joint with C = 1): unchanged
    assert Tukey.kappa(c=c) == robust._tukey_kappa(c)
    assert Tukey.kappa(c=c, nb_channels=1) == robust._tukey_kappa(c)
    assert MAD.kappa(nb_channels=5) == 1.


@pytest.mark.parametrize('C', NB_CHANNELS)
@pytest.mark.parametrize('joint', [False, True])
def test_tukey_estimated_lam_gaussian_channels(joint, C):
    """On Gaussian noise, Tukey's lam matches the noise precision, also
    with joint weights (the joint squared norm is χ²(C), so κ depends on
    C). With κ(1), joint lam was 20% too large for C = 2, and diverged
    for C >= 3."""
    mov, fix = make_gaussian(C)
    loss = Tukey(lam=None, joint=joint, dim=2)
    loss.loss(mov, fix)
    assert torch.allclose(loss.lam.flatten(), LAM[:C], rtol=0.03)


@pytest.mark.parametrize('C', NB_CHANNELS)
@pytest.mark.parametrize('joint', [False, True])
def test_mad_estimated_lam_gaussian_channels(joint, C):
    """MAD's lam is the maximum-likelihood estimate of a Laplace model
    (κ = 1 for any C, see `MAD.kappa`), not a Gaussian precision. On
    Gaussian noise, it is (C / E[χ_C])² times the precision with joint
    weights, and π/2 times the precision otherwise (as before)."""
    mov, fix = make_gaussian(C)
    loss = MAD(lam=None, joint=joint, dim=2)
    loss.loss(mov, fix)
    factor = (C / mean_chi(C)) ** 2 if joint else math.pi / 2
    assert torch.allclose(loss.lam.flatten(), factor * LAM[:C], rtol=0.03)


@pytest.mark.parametrize('C', NB_CHANNELS)
@pytest.mark.parametrize('joint', [False, True])
def test_mad_estimated_lam_own_model(joint, C):
    """MAD's lam recovers the precision of its own noise model: Laplace
    noise in each channel, or (joint) the C-variate density
    ∝ exp(-‖s‖), with s = √lam r, i.e. ‖s‖ ~ Gamma(C, 1) and a uniform
    direction"""
    g = torch.Generator().manual_seed(0)
    n = 256
    fix = torch.randn([C, n, n], dtype=torch.double, generator=g)
    if joint:
        u = torch.randn([C, n, n], dtype=torch.double, generator=g)
        u /= u.norm(dim=0, keepdim=True)
        rad = torch.rand([C, n, n], dtype=torch.double, generator=g)
        rad = rad.log().sum(0, keepdim=True).neg()       # Gamma(C, 1)
        s = rad * u
    else:
        s = torch.rand([C, n, n], dtype=torch.double, generator=g).log().neg()
        s *= torch.randint(0, 2, [C, n, n], generator=g).mul(2).sub(1)
    mov = fix + LAM[:C, None, None].rsqrt() * s
    loss = MAD(lam=None, joint=joint, dim=2)
    loss.loss(mov, fix)
    assert torch.allclose(loss.lam.flatten(), LAM[:C], rtol=0.03)


@pytest.mark.parametrize('cls', [MAD, Tukey])
@pytest.mark.parametrize('C', [3, 5])
def test_robust_estimated_lam_grad_fd_joint_channels(cls, C):
    """Joint weights with C > 2 channels: the finite differences of the
    value still match the gradient.

    (Outliers are whole voxels, and only 2% of them: with c = 4.685 and
    5 joint channels, lam diverges with more than about 3% of voxels
    with zero weight, see `Tukey.kappa`.)"""
    g = torch.Generator().manual_seed(0)
    fix = torch.randn([C, 32, 32], dtype=torch.double, generator=g)
    noise = torch.randn([C, 32, 32], dtype=torch.double, generator=g)
    mov = fix + LAM[:C, None, None].rsqrt() * noise
    out = (torch.rand(fix.shape[1:], generator=g) < 0.02).expand(fix.shape)
    mov[out] += 20 * torch.randn(int(out.sum()), dtype=torch.double,
                                 generator=g)
    loss = cls(lam=None, joint=True, cache=False)
    _, grad = loss.loss_grad(mov, fix)
    assert torch.isfinite(loss._last_lam).all()
    dirs = [grad] + [torch.randn(mov.shape, dtype=torch.double, generator=g)
                     for _ in range(3)]
    eps = 1e-6
    for d in dirs:
        d = d / d.norm()
        fd = (loss.loss(mov + eps * d, fix) -
              loss.loss(mov - eps * d, fix)) / (2 * eps)
        an = (grad * d).sum()
        assert abs(fd - an) < 1e-3 * grad.norm(), (fd, an)


def test_tukey_joint_unstable_warns():
    """With too many channels for c, the Gaussian precision is not a
    stable fixed point of the IRLS: warn"""
    with pytest.warns(UserWarning, match='too small'):
        Tukey.kappa(c=4.685, nb_channels=7)


# ----------------------------------------------------------------------
#   Weighted precision for display (`register -v 3`)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('cls', [MAD, Tukey])
@pytest.mark.parametrize('joint', [False, True])
def test_weighted_lam_uses_last_lam(cls, joint):
    """`weighted_lam` reuses the lam estimated by the last evaluation:
    no IRLS loop, same output as running the IRLS again"""
    mov, fix = make_noisy()
    g = torch.Generator().manual_seed(4)
    mask = (torch.rand(fix.shape[1:], generator=g) > 0.2).double()[None]
    for cache in (True, False):
        loss = cls(lam=None, joint=joint, cache=cache)
        assert loss._last_lam is None
        loss.loss(mov, fix, mask=mask)
        lam = loss._last_lam
        assert lam is not None and (lam is loss.lam) == cache
        # what the verbose print computed before
        ref = RobustLoss.irls(copy.deepcopy(loss), mov, fix, loss.lam,
                              mask, joint, 2)[-1]
        calls = []
        irls, reweight = loss.irls, loss.reweight

        def spy_irls(*a, **k):
            calls.append('irls')
            return irls(*a, **k)

        def spy_reweight(*a, **k):
            calls.append('reweight')
            return reweight(*a, **k)

        loss.irls, loss.reweight = spy_irls, spy_reweight
        out = loss.weighted_lam(mov, fix, mask, 2)
        assert calls == ['reweight']
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5 * ref.max())
        assert loss._last_lam is lam            # state not modified
        loss.clear_state()
        assert loss._last_lam is None


@pytest.mark.parametrize('cls', [MAD, Tukey])
def test_weighted_lam_fallback(cls):
    """Without a cached estimate (or with a fixed lam), `weighted_lam`
    is the output of `irls`, and does not update the cached lam"""
    mov, fix = make_noisy()
    for lam in (None, 2.):
        loss = cls(lam=lam, joint=True)
        ref = copy.deepcopy(loss).irls(mov, fix, loss.lam, None, True, 2)[-1]
        assert torch.equal(loss.weighted_lam(mov, fix, None, 2), ref)
        assert loss.lam == lam and loss._last_lam is None


def make_register_step(step_name, joint, seed=0):
    from nitorch import spatial
    from nitorch.tools.registration import objects
    from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep
    dt = torch.double
    shape = [10, 10, 10]
    g = torch.Generator().manual_seed(seed)
    fix = torch.rand([2, *shape], dtype=dt, generator=g)
    mov = fix + 0.1 * torch.randn([2, *shape], dtype=dt, generator=g)
    aff = spatial.affine_default(shape, 1., dtype=dt)
    fix, mov = objects.Image(fix, affine=aff), objects.Image(mov, affine=aff)
    affine = nonlin = None
    if step_name in ('do_affine', 'do_affine_only'):
        affine = objects.AffineModel('rigid')
        affine.set_dat(dim=3, dtype=dt)
        affine.dat.dat.copy_(torch.as_tensor([.3, -.2, .1, .02, -.01, .03]))
    if step_name in ('do_vel', 'do_affine'):
        nonlin = objects.SVFModel(steps=8, penalty=dict(
            absolute=0, membrane=0, bending=0, lame=(0, 0), factor=0))
        nonlin.set_dat(shape, affine=aff, dtype=dt)
        vel = torch.randn([*shape, 3], dtype=dt, generator=g)
        nonlin.dat.dat.copy_(0.1 * vel)
    loss = Tukey(lam=None, joint=joint, dim=3)
    sim = objects.SumSimilarity([objects.Similarity(loss, mov, fix)])
    # (a dummy figure, so that matplotlib is not needed)
    step = PairwiseRegisterStep(sim, affine, nonlin, verbose=3,
                                figure=object())
    x = (nonlin if step_name == 'do_vel' else affine).dat.dat.clone()
    return step, loss, x


@pytest.mark.parametrize('step_name', ['do_vel', 'do_affine',
                                       'do_affine_only'])
@pytest.mark.parametrize('joint', [False, True])
def test_verbose_lam_no_extra_irls(step_name, joint, capsys):
    """The live plot of `register -v 3` shows the weighted precision
    without running an extra IRLS: it reuses the lam of the last
    evaluation, and shows the same map as before"""
    step, loss, x = make_register_step(step_name, joint)
    shown, calls, refs = [], [], []
    step.mov2fix = lambda *a, lam=None, **k: shown.append(lam)
    irls, weighted_lam = loss.irls, loss.weighted_lam

    def spy_irls(*a, **k):
        calls.append('irls')
        return irls(*a, **k)

    def spy_weighted_lam(moving, fixed, mask, dim):
        # what the verbose print computed before (on a copy, since
        # running the IRLS also updates the cached lam)
        ref = RobustLoss.irls(copy.deepcopy(loss), moving, fixed, loss.lam,
                              mask, loss.joint, dim)[-1]
        if mask is not None:
            ref = ref * mask
        refs.append(ref)
        return weighted_lam(moving, fixed, mask, dim)

    loss.irls, loss.weighted_lam = spy_irls, spy_weighted_lam
    fn = getattr(step, step_name)
    fn(x, grad=True)        # nothing cached yet: falls back to `irls`
    assert calls == ['irls', 'irls']
    del calls[:]
    fn(x, grad=True)        # same parameters: reuses the cached lam
    assert calls == ['irls']    # the loss evaluation only
    assert len(shown) == 2 and all(lam is not None for lam in shown)
    for lam, ref in zip(shown, refs):
        # (same up to the IRLS tolerance, |Δ log lam| < 1e-6; Tukey weights
        # close to zero, near |s| = c, are the most sensitive to it)
        assert torch.allclose(lam, ref, rtol=1e-5, atol=1e-5 * ref.max())



@pytest.mark.parametrize('loss_name', ['tukey', 'mse'])
def test_verbose_nonlin_same_objective(loss_name, monkeypatch, capsys):
    """A nonlinear-only registration (no affine, so that the right
    affine of the display is the identity, on 1.5 mm voxels) runs with
    any verbosity, and gives the same result as verbose=0: the display
    (`-v 3`) changes neither the objective nor the parameters (it used
    to crash, or to update the cached lam)"""
    from nitorch import spatial
    from nitorch.tools.registration import objects, losses
    from nitorch.tools.registration import pairwise_impl
    from nitorch.tools.registration.pairwise_makeobj import make_nonlin_optim
    dt = torch.double
    shape = [10, 10, 10]
    aff = spatial.affine_default(shape, 1.5, dtype=dt)
    g = torch.Generator().manual_seed(0)
    fix = torch.rand([1, *shape], dtype=dt, generator=g)
    for _ in range(2):
        for d in range(1, 4):
            fix = (fix + fix.roll(1, d) + fix.roll(-1, d)) / 3
    mov = fix.roll(1, 1) + 0.01 * torch.randn(fix.shape, dtype=dt,
                                              generator=g)
    shown = []
    monkeypatch.setattr(pairwise_impl.PairwiseRegisterStep, 'mov2fix',
                        lambda self, *a, lam=None, **k: shown.append(lam))
    out = {}
    for verbose in (0, 1, 3):
        loss = (Tukey(lam=None, dim=3) if loss_name == 'tukey' else
                losses.MSE(lam=100, dim=3))
        sim = objects.Similarity(loss, objects.Image(mov, affine=aff),
                                 objects.Image(fix, affine=aff))
        nonlin = objects.SVFModel(steps=8, penalty=dict(
            absolute=1e-4, membrane=1e-3, bending=1e-2, lame=(0, 0)))
        nonlin.set_dat(shape, affine=aff, dtype=dt)
        optim = make_nonlin_optim('gn', order=2, line_search=4, max_iter=3,
                                  nonlin=nonlin)
        reg = pairwise_impl.PairwiseRegister(
            objects.SumSimilarity([sim]), None, nonlin, optim,
            verbose=verbose, figure=object())
        ll = reg.fit()
        out[verbose] = (torch.as_tensor(ll), nonlin.parameters().clone())
    for verbose in (1, 3):
        assert torch.equal(out[verbose][0], out[0][0])
        assert torch.equal(out[verbose][1], out[0][1])
    assert out[0][1].abs().max() > 0             # it did move
    assert shown                                 # the display ran
    assert all((lam is not None) == (loss_name == 'tukey') for lam in shown)


BASE = ['@@fix', 'fix.nii', '@@mov', 'mov.nii', '@nonlin', 'svf']


@pytest.mark.parametrize('alias, name', [('l1', 'mad'), ('l2', 'mse'),
                                          ('mad', 'mad'), ('mse', 'mse')])
def test_register_loss_aliases(alias, name):
    options = parser.parse(['@loss', alias] + BASE)
    assert options.loss[0].name == name

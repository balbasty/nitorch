"""Checks of the EM mutual information loss (`@loss emmi`,
`nitorch.tools.registration.losses.emmi`).

The objective is the negative marginal log-likelihood
    L(mu) = -1/N sum_n log sum_k H[x_n, k] mu[n, k]
of the fixed labels x given the moving probabilities mu, where H is
estimated by EM. The value, gradient and Hessian must describe the
same function."""
import torch
import pytest
from nitorch.tools.registration.losses.emmi import emmi, EMMI

dt = torch.double
K, J, SHAPE = 4, 6, [8, 8, 8]


def make_data(soft_fixed=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.rand([K, *SHAPE], dtype=dt, generator=g).add_(0.1)
    mu /= mu.sum(0, keepdim=True)
    if soft_fixed:
        q = torch.rand([J, *SHAPE], dtype=dt, generator=g).add_(0.1)
        q[:K] += 3 * mu
        fix = q / q.sum(0, keepdim=True)
    else:
        lab = mu.argmax(0, keepdim=True)
        noise = torch.randint(0, J, lab.shape, generator=g)
        keep = torch.rand(lab.shape, generator=g) < 0.7
        fix = torch.where(keep, lab, noise)
    return mu, fix, g


def directions(g, n=3):
    for _ in range(n):
        d = torch.randn([K, *SHAPE], dtype=dt, generator=g)
        yield d / d.norm()


def quad(h, d):
    """d^T H d, with H stored as (K*(K+1)//2, *spatial)"""
    out = (h[:K] * d.square()).sum()
    c = K
    for k in range(K):
        for kk in range(k + 1, K):
            out += 2 * (h[c] * d[k] * d[kk]).sum()
            c += 1
    return out


@pytest.mark.parametrize('soft_fixed', [False, True])
def test_emmi_fixed_prior_derivatives(soft_fixed):
    """With H fixed (no EM iteration), gradient and Hessian are exact"""
    mu, fix, g = make_data(soft_fixed)
    _, _, _, prior = emmi(mu, fix, dim=3, return_prior=True, max_iter=64)

    def f(x):
        return emmi(x, fix, dim=3, prior=prior, max_iter=0,
                    grad=False, hess=False)

    ll, gr, h = emmi(mu, fix, dim=3, prior=prior, max_iter=0)
    assert torch.allclose(ll, f(mu))
    for d in directions(g):
        eps = 1e-6
        fd = (f(mu + eps * d) - f(mu - eps * d)) / (2 * eps)
        an = (gr * d).sum()
        assert torch.allclose(fd, an, rtol=1e-5, atol=1e-10), (fd, an)
        eps = 1e-4
        curv = (f(mu + eps * d) - 2 * f(mu) + f(mu - eps * d)) / eps ** 2
        assert torch.allclose(curv, quad(h, d), rtol=1e-4), (curv, quad(h, d))


@pytest.mark.parametrize('soft_fixed', [False, True])
def test_emmi_value_matches_grad(soft_fixed):
    """With H re-estimated at each call, the finite differences of the
    value match the gradient (H maximises the objective, so its own
    variation only contributes at second order, up to the EM tolerance)."""
    mu, fix, g = make_data(soft_fixed)
    ll, gr, h = emmi(mu, fix, dim=3, max_iter=256)

    def f(x):
        return emmi(x, fix, dim=3, max_iter=256, grad=False, hess=False)

    for d in [gr / gr.norm(), *directions(g)]:
        eps = 1e-6
        fd = (f(mu + eps * d) - f(mu - eps * d)) / (2 * eps)
        an = (gr * d).sum()
        assert abs(fd - an) < 0.03 * gr.norm(), (fd, an, gr.norm())


def test_emmi_hard_moving_is_mi():
    """For a one-hot moving map, the value is -MI of the joint histogram"""
    mu, fix, _ = make_data()
    lab = mu.argmax(0)
    onehot = torch.stack([lab == k for k in range(K)]).to(dt)
    ll = emmi(onehot, fix, dim=3, max_iter=64, grad=False, hess=False)
    pxy = torch.zeros([J, K], dtype=dt)
    pxy.index_put_((fix.flatten(), lab.flatten()),
                   torch.ones(lab.numel(), dtype=dt), accumulate=True)
    pxy /= pxy.sum()
    px, py = pxy.sum(1, keepdim=True), pxy.sum(0, keepdim=True)
    nz = pxy > 0
    mi = (pxy[nz] * (pxy / (px * py))[nz].log()).sum()
    assert torch.allclose(ll, -mi, atol=1e-6), (ll, -mi)


def test_emmi_em_stopping():
    """EM stops on a small absolute change only (with a smoothed
    histogram it is not monotonic, so a decrease is not convergence),
    and on a non-finite value."""
    from nitorch.tools.registration.losses.emmi import em_converged
    t = lambda x: torch.as_tensor(x, dtype=dt)
    assert not em_converged(t(-10.), t(-float('inf')), 1e-5, 100)
    assert not em_converged(t(-10.), t(-9.), 1e-5, 100)     # decrease
    assert not em_converged(t(-9.), t(-10.), 1e-5, 100)     # increase
    assert em_converged(t(-10.), t(-10. + 1e-4), 1e-5, 100)
    assert em_converged(t(-10. + 1e-4), t(-10.), 1e-5, 100)
    assert em_converged(t(float('nan')), t(-10.), 1e-5, 100)
    assert em_converged(t(0.), t(0.), 1e-5, 0)              # empty mask


@pytest.mark.parametrize('soft_fixed', [False, True])
def test_emmi_tolerance(soft_fixed):
    """The EM tolerance is a parameter (default 1e-5 * N)"""
    mu, fix, _ = make_data(soft_fixed)
    kw = dict(dim=3, grad=False, hess=False, fwhm=1.)
    # a huge tolerance stops after the second iteration
    assert torch.allclose(emmi(mu, fix, max_iter=32, tolerance=1e30, **kw),
                          emmi(mu, fix, max_iter=2, tolerance=0, **kw))
    assert torch.allclose(emmi(mu, fix, **kw),
                          emmi(mu, fix, tolerance=1e-5, **kw))
    loss = EMMI(dim=3, fwhm=1., cache=False, tolerance=1e30)
    assert torch.allclose(loss.loss(mu, fix),
                          emmi(mu, fix, max_iter=2, tolerance=0, **kw))


def test_emmi_class_cache_and_batch():
    """The loss object runs (with a batch dimension, as in slicewise
    losses) and its value matches the functional form"""
    mu, fix, _ = make_data()
    loss = EMMI(dim=2, cache=False)
    ll, g, h = loss.loss_grad_hess(mu.movedim(-1, 0), fix.movedim(-1, 0))
    assert torch.isfinite(ll) and torch.isfinite(g).all()
    assert g.shape == mu.movedim(-1, 0).shape
    ll2 = loss.loss(mu.movedim(-1, 0), fix.movedim(-1, 0))
    assert torch.allclose(ll, ll2)


# ----------------------------------------------------------------------
#   End-to-end: recover a known translation with `nitorch register`
# ----------------------------------------------------------------------
SHIFT = [3., -2., 1.5]


def shifted_pair(odir, shape=(24, 24, 24)):
    import os
    from nitorch import io, spatial
    grid = spatial.identity_grid(shape, dtype=dt)
    center = torch.as_tensor(shape, dtype=dt) / 2

    def blobs(shift):
        x = grid - (center + torch.as_tensor(shift, dtype=dt))
        a = (x / torch.as_tensor([5., 3.5, 4.])).square().sum(-1).neg().exp()
        b = ((x - torch.as_tensor([4., -3., 2.])) / 2.5).square().sum(-1)
        return (a + 0.7 * b.neg().exp()).float()

    affine = spatial.affine_default(shape, voxel_size=1.)
    fix, mov = os.path.join(odir, 'fix.nii.gz'), os.path.join(odir, 'mov.nii.gz')
    io.savef(blobs([0, 0, 0]), fix, affine=affine)
    io.savef(blobs(SHIFT), mov, affine=affine)
    return fix, mov


class _Stop(Exception):
    pass


@pytest.mark.parametrize('symmetric', [False, True])
def test_register_emmi_soft_quantizes_moving(tmp_path, monkeypatch, symmetric):
    """`@loss emmi` passes a soft-quantised (16-channel) moving image and a
    discretised fixed image to the loss. It used to pass the raw
    single-channel moving image, for which the loss is constant."""
    import os
    from nitorch.cli.registration.register.cli import _cli as register
    fix, mov = shifted_pair(str(tmp_path), shape=(12, 12, 12))
    seen = []

    def make_spy(grad, hess):
        def spy(self, moving, fixed, **kwargs):
            seen.append((moving.shape[0], moving.sum(0), fixed.dtype))
            if len(seen) == (2 if symmetric else 1):
                raise _Stop
            return emmi(moving, fixed, dim=3, max_iter=1, grad=grad, hess=hess)
        return spy

    monkeypatch.setattr(EMMI, 'loss', make_spy(False, False))
    monkeypatch.setattr(EMMI, 'loss_grad', make_spy(True, False))
    monkeypatch.setattr(EMMI, 'loss_grad_hess', make_spy(True, True))
    lta = os.path.join(str(tmp_path), 'emmi.lta')
    with pytest.raises(_Stop):
        register(['-v', '0', '--cpu', '@loss', 'emmi',
                  *(['--symmetric'] if symmetric else []),
                  '@@fix', fix, '-o', 'false', '-r', 'false',
                  '@@mov', mov, '-o', 'false', '-r', 'false',
                  '@affine', 'translation', '-o', lta,
                  '@@optim', 'gn', '-n', '1', '@pyramid', '--levels', '0'])
    for nchannels, total, fixdtype in seen:
        assert nchannels == 16
        assert not fixdtype.is_floating_point
        # soft bins sum to one in voxels inside the field of view
        inside = total > 0
        assert torch.allclose(total[inside], torch.ones_like(total[inside]),
                              atol=1e-3)


@pytest.mark.slow
def test_register_emmi_translation(tmp_path):
    """`nitorch register @loss emmi` recovers a known translation.

    About 15 s on an idle CPU with the TorchScript backend. With 32 bins
    for the fixed image; on smaller pairs (16^3-20^3), or with 8 moving
    bins, the optimum of the loss itself is 0.4-0.9 mm short in x."""
    import os
    from nitorch import io
    from nitorch.cli.registration.register.cli import _cli as register
    fix, mov = shifted_pair(str(tmp_path))
    lta = os.path.join(str(tmp_path), 'emmi.lta')
    register(['-v', '0', '--cpu', '@loss', 'emmi',
              '@@fix', fix, '-o', 'false', '-r', 'false', '-d', '32',
              '@@mov', mov, '-o', 'false', '-r', 'false',
              '@affine', 'translation', '-o', lta,
              '@@optim', 'gn', '-n', '50', '@pyramid', '--levels', '0'])
    t = io.transforms.loadf(lta).reshape([4, 4])[:3, -1].tolist()
    assert all(abs(a - b) < 0.5 for a, b in zip(t, SHIFT)), t

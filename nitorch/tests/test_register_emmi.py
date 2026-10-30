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

"""affine_tpm: the value is the objective whose gradient is used
(marginal log-likelihood at the final H), hard and soft images, and
fixes in fit_affine_tpm / align_tpm."""
import pytest
import torch
from nitorch import spatial
from nitorch.core import linalg, math
import nitorch.tools.registration.affine_tpm as at

dt = torch.double
pull_opt = dict(bound='replicate', extrapolate=True, interpolation=2)


# ----------------------------------------------------------------------
#   helpers
# ----------------------------------------------------------------------

def make_mu(K=3, J=12, shape=(8, 8, 8), seed=0):
    """Random TPM-like probabilities, and a hard / soft image"""
    g = torch.Generator().manual_seed(seed)
    mu = torch.rand([K, *shape], dtype=dt, generator=g).add_(0.1)
    mu /= mu.sum(0, keepdim=True)
    lab = mu.argmax(0, keepdim=True)
    x = lab * (J // K) + torch.randint(0, J // K, lab.shape, generator=g)
    noise = torch.randint(0, J, lab.shape, generator=g)
    x = torch.where(torch.rand(lab.shape, generator=g) < 0.7, x, noise)
    q = torch.rand([J, *shape], dtype=dt, generator=g).pow(4)
    q /= q.sum(0, keepdim=True)
    return mu, x[None], q[None]


def make_tpm(n):
    """Three-class blob TPM (with an off-centre blob to break symmetries)"""
    shape = [n] * 3
    g = spatial.identity_grid(shape, dtype=dt)
    c = torch.tensor([n / 2 + 0.7, n / 2 - 0.9, n / 2 + 0.3], dtype=dt)
    r = ((g - c) / torch.tensor([7.0, 5.5, 6.3], dtype=dt))
    r = r.square().sum(-1).sqrt()
    c2 = torch.tensor([n / 2 + 3.0, n / 2 + 2.0, n / 2 - 2.5], dtype=dt)
    r2 = ((g - c2) / 2.5).square().sum(-1).sqrt()
    inner = torch.sigmoid((0.55 - r) * 8) + torch.sigmoid((1 - r2) * 4)
    outer = torch.sigmoid((1.0 - r) * 8)
    p1 = inner.clamp(0, 1)
    p2 = (outer - p1).clamp(0, 1)
    p3 = (1 - p1 - p2).clamp(0, 1)
    tpm = torch.stack([p1, p2, p3]) + 1e-3
    tpm /= tpm.sum(0, keepdim=True)
    return tpm


def prefilter(tpm):
    logtpm = tpm.clamp(1e-16, 1).div(tpm.sum(0, keepdim=True))
    logtpm = logtpm.add_(1e-4).log_()
    return spatial.spline_coeff_nd(logtpm, dim=3, interpolation=2,
                                   bound='replicate')


def make_problem(n=20, shift=(3., -2., 1.5), bins=32, noise=8., seed=1):
    """Image = intensities of the most likely class of a shifted TPM"""
    shape = [n] * 3
    logtpm = prefilter(make_tpm(n))
    affine = spatial.affine_default(shape, dtype=dt)
    basis = at.make_basis('translation', 3, dtype=dt)
    prm_true = torch.as_tensor(shift, dtype=dt)
    aff = linalg.expm(prm_true, basis)
    phi = spatial.affine_grid(linalg.lmdiv(affine, aff.matmul(affine)), shape)
    mov = math.softmax(spatial.grid_pull(logtpm[None], phi, **pull_opt), dim=1)
    lab = mov[0].argmax(0)
    g = torch.Generator().manual_seed(seed)
    means = torch.tensor([100., 60., 15.], dtype=dt)
    img = means[lab] + noise * torch.randn(shape, dtype=dt, generator=g)
    img = img[None, None]
    dat = at.discretize(img, nbins=bins, mask=torch.ones_like(img))
    return dat, logtpm, affine, prm_true


def value(mu, fixed, prior=None, **kwargs):
    ll, Nm, prior = at.em_prior(mu[None], fixed, prior, **kwargs)
    return ll.sum().item(), prior


def dir_fd(f, x, d, eps):
    return (f(x + eps * d) - f(x - eps * d)) / (2 * eps)


def quad(h, d):
    """d' H d for a packed (K*(K+1)//2, ...) Hessian"""
    K = d.shape[0]
    out = (h[:K] * d.square()).sum()
    c = K
    for k in range(K):
        for kk in range(k + 1, K):
            out += 2 * (h[c] * d[k] * d[kk]).sum()
            c += 1
    return out.item()


# ----------------------------------------------------------------------
#   value / gradient / Hessian (with respect to mu)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['hard', 'soft'])
def test_value_grad_fixed_prior(kind):
    # With H fixed (max_iter=0), the value is exactly the function
    # whose gradient derivatives_intensity returns (up to the sign).
    mu, x, q = make_mu()
    fixed = x if kind == 'hard' else q
    _, H = value(mu, fixed, fwhm=1., max_iter=500, tolerance=1e-12)
    g, _ = at.derivatives_intensity(mu[None], fixed, H)
    torch.manual_seed(1)
    for _ in range(3):
        d = torch.randn_like(mu)
        d /= d.norm()
        an = (g[0] * d).sum().item()
        fd = -dir_fd(lambda m: value(m, fixed, H, max_iter=0)[0], mu, d, 1e-6)
        assert abs(fd - an) <= 1e-6 * abs(an), (fd, an)


@pytest.mark.parametrize('kind', ['hard', 'soft'])
def test_value_grad_reestimated_prior(kind):
    # With H re-estimated by EM at each evaluation, the gradient with H
    # fixed is the gradient of the profile likelihood (envelope theorem).
    # Without smoothing, this holds up to the EM tolerance.
    # With smoothing, it is only approximate (but has the right sign
    # and scale).
    mu, x, q = make_mu()
    fixed = x if kind == 'hard' else q
    torch.manual_seed(2)
    for fwhm, rtol in [(0., 1e-4), (1., 0.2)]:
        _, H = value(mu, fixed, fwhm=fwhm, max_iter=1000, tolerance=1e-12)
        g, _ = at.derivatives_intensity(mu[None], fixed, H)
        for _ in range(2):
            d = torch.randn_like(mu)
            d /= d.norm()
            an = (g[0] * d).sum().item()
            fd = -dir_fd(lambda m: value(m, fixed, H, fwhm=fwhm)[0],
                         mu, d, 1e-5)
            assert abs(fd - an) <= rtol * abs(an), (fwhm, fd, an)


@pytest.mark.parametrize('kind', ['hard', 'soft'])
def test_hessian_fd(kind):
    mu, x, q = make_mu()
    fixed = x if kind == 'hard' else q
    H = torch.rand([1, 12, 3], dtype=dt,
                   generator=torch.Generator().manual_seed(5)).add_(0.1)
    _, h = at.derivatives_intensity(mu[None], fixed, H)
    torch.manual_seed(3)
    eps = 1e-6
    for _ in range(3):
        d = torch.randn_like(mu)
        d /= d.norm()
        gp, _ = at.derivatives_intensity((mu + eps * d)[None], fixed, H)
        gm, _ = at.derivatives_intensity((mu - eps * d)[None], fixed, H)
        fd = (((gp - gm)[0] / (2 * eps)) * d).sum().item()
        an = quad(h[0], d)
        assert an > 0
        assert abs(fd - an) <= 1e-6 * abs(an), (fd, an)


def test_soft_em_matches_hard_on_one_hot():
    # A one-hot soft image is the same observation as the hard image
    mu, x, _ = make_mu()
    J = 12
    q = torch.nn.functional.one_hot(x[0, 0], J).movedim(-1, 0)[None].to(dt)
    ll_h, Nm_h, H_h = at.em_prior(mu[None], x, fwhm=1., max_iter=20)
    ll_s, Nm_s, H_s = at.em_prior(mu[None], q, fwhm=1., max_iter=20)
    assert torch.allclose(ll_h, ll_s, rtol=1e-10)
    assert torch.allclose(H_h, H_s, rtol=1e-8)


def test_value_grad_lie_parameters():
    # Same check through the warp + softmax + chain rule, for the
    # value / gradient pair used by fit_affine_tpm (H fixed).
    dat, logtpm, affine, _ = make_problem(n=12, shift=(1., -0.5, 0.5))
    basis = at.make_basis('affine', 3, dtype=dt)
    shape = dat.shape[-3:]

    def warp(prm):
        aff, gaff = linalg._expm(prm, basis, grad_X=True)
        phi = spatial.affine_grid(linalg.lmdiv(affine, aff.matmul(affine)),
                                  shape)
        mov = spatial.grid_pull(logtpm[None], phi, **pull_opt)
        return math.softmax(mov, dim=1), phi, gaff

    prm = torch.zeros(len(basis), dtype=dt)
    prm[:3] = 0.3
    mov, phi, gaff = warp(prm)
    _, _, H = at.em_prior(mov, dat, max_iter=32)
    g, h = at.derivatives_intensity(mov, dat, H)
    mov = mov.unsqueeze(-1)
    gmov = spatial.grid_grad(logtpm[None], phi, **pull_opt)
    gmov = mov * (gmov - (mov * gmov).sum(1, keepdim=True))
    gaff = linalg.lmdiv(affine, gaff.matmul(affine))
    g, h = at.chain_rule(g.sum(0), h.sum(0), gmov, gaff, maj=False)

    def f(p):
        ll, Nm, _ = at.em_prior(warp(p)[0], dat, H, max_iter=0)
        return ll.sum().item()

    torch.manual_seed(4)
    scale = torch.tensor([1.] * 3 + [0.03] * 9, dtype=dt)
    for _ in range(3):
        d = torch.randn(len(basis), dtype=dt) * scale
        an = (g * d).sum().item()
        fd = -dir_fd(f, prm, d, 1e-4)
        assert abs(fd - an) <= 1e-4 * abs(an), (fd, an)


# ----------------------------------------------------------------------
#   fit_affine_tpm / align_tpm
# ----------------------------------------------------------------------

def test_aff_matches_prm_after_line_search_failure(monkeypatch):
    # Force every line search after the first iteration to fail:
    # the returned matrix must be that of the returned parameters.
    dat, logtpm, affine, _ = make_problem(n=12, shift=(1., -0.5, 0.5))
    em_prior = at.em_prior
    ncalls = [0]

    def em_prior_worse(*args, **kwargs):
        ll, Nm, prior = em_prior(*args, **kwargs)
        ncalls[0] += 1
        if ncalls[0] > 1:
            ll = ll - 1e3 * Nm
        return ll, Nm, prior

    monkeypatch.setattr(at, 'em_prior', em_prior_worse)
    prm0 = torch.as_tensor([0.2, 0.1, -0.1], dtype=dt)
    mi, aff, prm = at.fit_affine_tpm(dat, logtpm, affine, affine,
                                     basis='translation', prm=prm0.clone(),
                                     max_line_search=2, verbose=0)
    assert ncalls[0] == 3
    basis = at.make_basis('translation', 3, dtype=dt)
    assert torch.allclose(prm, prm0)
    assert torch.allclose(aff, linalg.expm(prm, basis), atol=1e-10)


def test_flexi_uses_logtpm_and_subsampled_affine(monkeypatch, capsys):
    n = 24
    tpm = make_tpm(n)
    img = tpm[0] * 100 + tpm[1] * 50
    affine = spatial.affine_default([n] * 3, dtype=dt)
    calls = []

    def spy(dat, tpm, affine=None, affine_tpm=None, weights=None,
            basis='affine', prm=None, **kwargs):
        calls.append((tpm, affine, dat.shape[-3:]))
        F = len(at.make_basis(basis, 3))
        return torch.zeros([1], dtype=dt), torch.eye(4, dtype=dt), \
            torch.zeros([F], dtype=dt)

    monkeypatch.setattr(at, 'fit_affine_tpm', spy)
    at.align_tpm((img, affine), (tpm, affine), spacing=4, flexi=True,
                 verbose=0)
    assert not capsys.readouterr().out
    assert len(calls) > 2
    logtpm = calls[-1][0]       # used by the final (non-flexi) fit
    assert logtpm.min() < -1    # prefiltered log-probabilities
    for tpm1, affine1, shape1 in calls:
        assert torch.equal(tpm1, logtpm)
        # the orientation matrix matches the (subsampled) data
        fov = spatial.voxel_size(affine1) * torch.as_tensor(shape1, dtype=dt)
        assert torch.allclose(fov, torch.full_like(fov, n)), fov


def test_fit_translation_recovery():
    dat, logtpm, affine, prm_true = make_problem()
    mi, aff, prm = at.fit_affine_tpm(dat, logtpm, affine, affine,
                                     basis='translation', max_iter_gn=40,
                                     verbose=0)
    err = (prm - prm_true).norm().item()
    assert err <= 0.2, (prm.tolist(), prm_true.tolist())


def test_fit_translation_recovery_soft():
    dat, logtpm, affine, prm_true = make_problem()
    q = torch.nn.functional.one_hot(dat[0, 0], 32).movedim(-1, 0)[None]
    q = q.to(dt)
    mi, aff, prm = at.fit_affine_tpm(q, logtpm, affine, affine,
                                     basis='translation', max_iter_gn=40,
                                     verbose=0)
    err = (prm - prm_true).norm().item()
    assert err <= 0.2, (prm.tolist(), prm_true.tolist())


def test_get_data_trailing_singleton():
    x = torch.rand([2, 5, 6, 7, 1])
    x, w, _ = at.get_data(x, None, None, 3)
    assert x.shape == (2, 5, 6, 7)

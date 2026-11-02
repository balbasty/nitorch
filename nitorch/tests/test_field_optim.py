"""Gauss-Newton optimisers on channel-first vector fields
(`FieldCG`, `FieldRelax`) and the TOPUP code path that uses them.

`search_direction` used to raise a NameError (`dim` undefined) when
normalising the regularisation factor. The factor is now used as given
(callers normalise it consistently with their objective)."""
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import optim as optm
from nitorch.tools.registration.topup import topup_fit, topup_apply


K, SHAPE = 2, (5, 6, 7)          # 2-channel 3D field
NVOX = 5 * 6 * 7


def make_field(nb_hess, seed=0):
    """grad: (K, *spatial), hess: (K2, *spatial), positive definite"""
    g = torch.Generator().manual_seed(seed)
    grad = torch.randn([K, *SHAPE], generator=g, dtype=torch.double)
    diag = torch.rand([min(nb_hess, K), *SHAPE], generator=g,
                      dtype=torch.double) + 1
    if nb_hess <= K:
        hess = diag
    else:  # K*(K+1)//2: diagonal first, then off-diagonal
        offd = 0.1 * torch.randn([nb_hess - K, *SHAPE], generator=g,
                                 dtype=torch.double)
        hess = torch.cat([diag, offd])
    return grad, hess


@pytest.mark.parametrize('klass', [optm.FieldCG, optm.FieldRelax])
@pytest.mark.parametrize('fmg', [0, 2])
@pytest.mark.parametrize('nb_hess', [1, K, K*(K+1)//2])
@pytest.mark.parametrize('penalty', ['membrane', 'bending'])
def test_field_search_direction_runs(klass, fmg, nb_hess, penalty):
    grad, hess = make_field(nb_hess)
    opt = klass(fmg=fmg, factor=1e-1 / NVOX, **{penalty: 1}, bound='dct2')
    step = opt.search_direction(grad.clone(), hess.clone())
    assert step.shape == grad.shape
    assert torch.isfinite(step).all()
    # descent direction
    assert step.flatten().dot(grad.flatten()) < 0


@pytest.mark.parametrize('klass', [optm.FieldCG, optm.FieldRelax])
def test_field_factor_used_as_is(klass):
    """With an absolute penalty only, the system is solved in closed form:
    step = -g / (h * (1 + tiny) + factor), with no extra normalisation."""
    grad, hess = make_field(K)
    factor = 0.5
    opt = klass(factor=factor, absolute=1)
    step = opt.search_direction(grad.clone(), hess.clone())
    maj = hess.abs().max(0, keepdim=True).values
    expected = -grad / (hess + 1e-5 * maj + factor)
    assert torch.allclose(step, expected)


def test_field_marquardt_channel_first():
    """Marquardt regularisation is added to the diagonal channels
    (dim -dim-1), not to the last spatial axis."""
    grad, hess = make_field(K*(K+1)//2)
    opt = optm.FieldCG(marquardt=0.5)
    _, hess2 = opt._add_marquardt(grad, hess.clone())
    assert torch.allclose(hess2[:K], hess[:K] + 0.5)
    assert torch.equal(hess2[K:], hess[K:])


def test_field_explicit_dim_with_batch():
    """Batched fields (B, K, *spatial) need an explicit `dim`."""
    grad, hess = make_field(K)
    grad, hess = torch.stack([grad, grad]), torch.stack([hess, hess])
    opt = optm.FieldCG(factor=0.5, absolute=1, dim=3)
    step = opt.search_direction(grad.clone(), hess.clone())
    maj = hess.abs().max(1, keepdim=True).values
    expected = -grad / (hess + 1e-5 * maj + 0.5)
    assert torch.allclose(step, expected)


def test_field_lr_and_bound():
    opt = optm.FieldGaussNewton(lr=0.5, bound='dft')
    assert opt.lr == 0.5
    assert opt.bound == 'dft'
    grad, hess = make_field(K)
    step1 = optm.FieldCG(membrane=1, factor=0.1).search_direction(
        grad.clone(), hess.clone())
    step2 = optm.FieldCG(membrane=1, factor=0.1, lr=0.5).search_direction(
        grad.clone(), hess.clone())
    assert torch.allclose(step2, 0.5 * step1)


def make_topup_pair(shape=(8, 9, 10), shift=0.8):
    """Smooth blob displaced by +/- shift voxels along the last dimension"""
    x = spatial.identity_grid(shape, dtype=torch.float)
    center = torch.as_tensor(shape, dtype=torch.float).sub(1).div(2)

    def blob(s):
        c = center.clone()
        c[-1] += s
        return (-(x - c).square().div(4).sum(-1)).exp()

    return blob(shift), blob(-shift)


@pytest.mark.parametrize('penalty', ['membrane', 'bending'])
@pytest.mark.parametrize('model', ['smalldef', 'svf'])
def test_topup_fieldcg(penalty, model):
    pos, neg = make_topup_pair()
    vel = topup_fit(pos, neg, lam=1, penalty=penalty, model=model,
                    max_iter=5, verbose=False)
    assert vel.shape == pos.shape
    assert torch.isfinite(vel).all()
    assert vel.abs().max() > 0
    upos, uneg = topup_apply(pos, neg, vel, model=model)
    before = (pos - neg).square().mean()
    after = (upos - uneg).square().mean()
    assert after < before

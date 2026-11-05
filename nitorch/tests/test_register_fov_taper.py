"""Samples of a masked image outside its field of view (#114).

They used to be excluded from the loss by a hard cut: the pulled mask is
zero beyond 0.05 voxel outside the lattice (`extrapolate=False`). Each
voxel that crossed the edge of the field of view made the loss jump, and
the gradient does not see the jump, so line searches stopped at it and
first-order optimisers stalled (`test_register_affine_ogm`).

`Image.pull` now extrapolates the data and multiplies the mask by
weights that taper linearly to zero over the voxel outside the field of
view (`Image.fov_weights`), so the loss is continuous. The weights are
treated as fixed when differentiating: the gradient is that of the loss
with the mask frozen, and differs from the full derivative only through
the voxels in the one-voxel band.

`HardImage` below reproduces the previous behaviour.

The robust losses (MAD, Tukey) used to scale the residuals by the mask
inside the IRLS weights (the Laplace weights then blow up as the mask
goes to zero) and left the data term out of the masked mean. The mask
now weights each voxel's term, as in the other losses.
"""
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import (
    MSE, MAD, Tukey, CC, LCC, MI, EMMI, GMMH, LGMMH, Cat, Dice)
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep

dt = torch.double
SHAPE = (12, 12, 10)


class HardImage(objects.Image):
    """Previous behaviour: hard in-bounds cut, data not extrapolated"""
    fov_tapered = property(lambda self: False)


class FrozenMaskImage(objects.Image):
    """Image whose pulled mask is frozen at `self.frozen` (if set)"""
    frozen = None

    def pull(self, grid, dat=True, mask=False, preview=False):
        out = super().pull(grid, dat=dat, mask=mask, preview=preview)
        if mask and self.frozen is not None:
            out = list(out)
            out[int(bool(dat))] = self.frozen
            out = tuple(out)
        return out


def phantom(shape, shift=0., channels=0):
    """Smooth blob, clearly non-zero at the edge of the field of view"""
    x = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=dt) for s in shape],
                                   indexing='ij'), -1)
    c = x.new_tensor([(s - 1) / 2 for s in shape])
    c[0] += shift
    d = ((x - c) / x.new_tensor([0.45 * s for s in shape])).square().sum(-1).sqrt()
    img = torch.sigmoid(4 * (1 - d)) + 0.3 * torch.cos(x[..., 1] / 2)
    if channels:
        # soft labels
        img = torch.stack([(img - k / channels).mul(-8).square().neg()
                           for k in range(channels)]).softmax(0)
        return img
    return img[None]


def make_step(loss, klass=objects.Image, fixed=None, channels=0):
    aff = spatial.affine_default(SHAPE, 1., dtype=dt)
    ones = torch.ones([1, *SHAPE], dtype=dt)
    fix = phantom(SHAPE, channels=channels) if fixed is None else fixed
    fix = objects.Image(fix, aff, mask=ones)
    mov = klass(phantom(SHAPE, 0.7, channels=channels), aff.clone(), mask=ones)
    sim = objects.SumSimilarity([objects.Similarity(loss, mov, fix)])
    affine = objects.AffineModel('rigid', position='mov')
    affine.set_dat(dim=3, dtype=dt)
    step = PairwiseRegisterStep(sim, affine, None, verbose=0)
    return step, affine.dat.dat.clone(), mov


# a direction that pushes a face of voxels out of the field of view
# (mostly a translation along x; all parameters move, so that no axis
# samples exactly on the nodes, where linear interpolation has kinks)
DIRECTION = [1., 0.3, 0.17, 0.01, -0.015, 0.02]


def jump_ratio(step, prm, length=1.5, n=301, sub=10):
    """Continuity of the loss along `DIRECTION`, over `length` voxels.

    Find the largest change of the loss between consecutive points, and
    split that interval in `sub`: return the largest change over a
    sub-interval, relative to the change over the interval. About 1/sub
    for a continuous loss; about 1 if the loss jumps there."""
    d = prm.new_tensor(DIRECTION)
    loss = lambda t: step.do_affine_only(prm + t * d, in_line_search=True)
    ts = torch.linspace(0, length, n, dtype=dt)
    f = torch.stack([loss(t) for t in ts])
    df = (f[1:] - f[:-1]).abs()
    i = int(df.argmax())
    fs = torch.stack([loss(t) for t in torch.linspace(ts[i], ts[i+1], sub+1, dtype=dt)])
    return ((fs[1:] - fs[:-1]).abs().max() / df[i]).item()


def test_fov_weights():
    img = objects.Image(torch.zeros([1, 4, 3, 1], dtype=dt))
    g = torch.tensor([[-1.5, 1, 0], [-1., 1, 0], [-0.5, 1, 0], [-0.25, 1, 0],
                      [0., 1, 0], [3., 1, 0], [3.5, 1, 0], [3, 2.75, 0],
                      [3, 3, 0], [1, 1, 0.4], [1, 1, -0.5]], dtype=dt)
    w = img.fov_weights(g)[0]
    expected = [0, 0, 0.5, 0.75, 1, 1, 0.5, 0.25, 0, 0.6, 0.5]
    assert torch.allclose(w, w.new_tensor(expected)), w
    # only masked, non-extrapolated images are tapered
    assert not img.fov_tapered
    assert objects.Image(img.dat, mask=torch.ones_like(img.dat)).fov_tapered
    assert not objects.Image(img.dat, mask=torch.ones_like(img.dat),
                             extrapolate=True).fov_tapered


@pytest.mark.parametrize('name', ['mse', 'mse_lam', 'lcc', 'cc', 'mi'])
def test_loss_continuous_across_fov_edge(name):
    """Along a line search direction that pushes voxels out of the field
    of view, the loss does not jump. With the hard cut it does."""
    loss = dict(mse=lambda: MSE(dim=3), mse_lam=lambda: MSE(lam=1, dim=3),
                lcc=lambda: LCC(dim=3), cc=lambda: CC(dim=3),
                mi=lambda: MI(dim=3))[name]
    step, prm, _ = make_step(loss())
    assert jump_ratio(step, prm) < 0.15
    step, prm, _ = make_step(loss(), HardImage)
    assert jump_ratio(step, prm) > 0.2


def fd(step, prm, eps):
    out = torch.zeros_like(prm)
    for i in range(len(prm)):
        e = torch.zeros_like(prm)
        e[i] = eps
        out[i] = (step.do_affine_only(prm + e, in_line_search=True) -
                  step.do_affine_only(prm - e, in_line_search=True)) / (2 * eps)
    return out


@pytest.mark.parametrize('name', ['mse', 'mse_lam', 'lcc', 'cc'])
@pytest.mark.parametrize('t', [0.05, 0.33, 0.77])
def test_grad_across_fov_edge(name, t):
    """Gradient at points where a face of voxels is outside the lattice
    (in the band where the mask tapers):
    - it is close to the full derivative of the loss (4-14% off, cosine
      > 0.99): the derivative of the mask is not included, which matters
      here because a whole face (1/12 of the voxels) is in the band;
    - with fixed statistics (lam=1, LCC), it is exactly the derivative of
      the loss with the mask frozen;
    - with the hard cut, the finite difference across the cut (the x face
      0.05 voxel out) is far off."""
    loss = dict(mse=lambda: MSE(dim=3), mse_lam=lambda: MSE(lam=1, dim=3),
                lcc=lambda: LCC(dim=3), cc=lambda: CC(dim=3))[name]
    step, prm, mov = make_step(loss(), FrozenMaskImage)
    p0 = prm + t * prm.new_tensor(DIRECTION)
    _, g = step.do_affine_only(p0, grad=True, in_line_search=True)
    full = fd(step, p0, 1e-5)
    rel = ((g - full).norm() / full.norm()).item()
    cos = (g.dot(full) / g.norm() / full.norm()).item()
    assert rel < 0.2 and cos > 0.98, (rel, cos)   # 0.04-0.14, > 0.991
    if name in ('mse_lam', 'lcc'):
        # freeze the mask at p0
        aff = step.affine.exp(p0)
        aff = torch.linalg.solve(mov.affine, aff @ step.losses[0].fixed.affine)
        mov.frozen = mov.pull(spatial.affine_grid(aff, SHAPE), dat=False, mask=True)
        frozen = fd(step, p0, 1e-5)
        assert torch.allclose(g, frozen, rtol=1e-6, atol=1e-6 * frozen.abs().max()), (g, frozen)
    if t == 0.05:
        step, prm, _ = make_step(loss(), HardImage)
        p0 = prm.clone()
        p0[0] += 0.05
        _, g = step.do_affine_only(p0, grad=True, in_line_search=True)
        full = fd(step, p0, 1e-5)
        assert ((g - full).norm() / full.norm()).item() > 0.5


def losses_smoke():
    labels = (phantom(SHAPE) * 4).floor().clamp(0, 3).long()
    return dict(
        mse=(lambda: MSE(dim=3), {}),
        mad=(lambda: MAD(dim=3), {}),
        tukey=(lambda: Tukey(dim=3), {}),
        cc=(lambda: CC(dim=3), {}),
        lcc=(lambda: LCC(dim=3), {}),
        mi=(lambda: MI(dim=3), {}),
        gmm=(lambda: GMMH(dim=3), {}),
        lgmm=(lambda: LGMMH(dim=3, patch=6, max_iter=16), {}),
        emmi=(lambda: EMMI(dim=3), dict(fixed=labels, channels=3)),
        cat=(lambda: Cat(dim=3), dict(channels=3)),
        dice=(lambda: Dice(dim=3), dict(channels=3)),
    )


# Fitted by EM to a tolerance: not continuous in the transformation
# whatever the mask (they jump with the hard cut and with the taper alike)
EM_FITTED = ('gmm', 'lgmm')


@pytest.mark.parametrize('name', list(losses_smoke()))
def test_losses_with_tapered_mask(name):
    """All losses accept the tapered (fractional) mask: finite value,
    gradient and Hessian with voxels in the band, and (except for losses
    fitted by EM) no jump across the edge of the field of view."""
    loss, kw = losses_smoke()[name]
    step, prm, _ = make_step(loss(), **kw)
    p0 = prm + 0.33 * prm.new_tensor(DIRECTION)
    out = step.do_affine_only(p0, grad=True, hess=True)
    assert all(bool(torch.isfinite(x).all()) for x in out), out
    if name not in EM_FITTED:
        assert jump_ratio(step, prm, n=101) < 0.15


@pytest.mark.parametrize('joint', [False, True])
def test_robust_mask_weights_the_mean(joint):
    """Robust losses with a fractional mask m:
    F = sum(m [1/2 lam w r^2 + phi(w)]) / sum(m) - 1/2 kappa sum(log lam),
    with the IRLS weights w computed from the unweighted residuals, and a
    gradient that is the derivative of F (m and the estimates fixed)."""
    g = torch.Generator().manual_seed(3)
    fix = torch.randn([2, 7, 6, 5], dtype=dt, generator=g)
    mov = fix + 0.3 * torch.randn(fix.shape, dtype=dt, generator=g)
    m = torch.rand([1, 7, 6, 5], dtype=dt, generator=g)
    loss = Tukey(dim=3, joint=joint)
    ll, grad = loss.loss_grad(mov, fix, mask=m)
    lam = loss._last_lam
    prm = loss.reweight_params(joint, 2 if joint else 1)
    s2 = (mov - fix).square() * lam
    if joint:
        s2 = s2.sum(0, keepdim=True)
    w = loss.weights_(s2, **prm)
    kappa = loss.kappa(nb_channels=2 if joint else 1, **prm)
    F = ((0.5 * lam * w * (mov - fix).square() * m).sum()
         + (loss.dual(w, **prm) * m).sum()) / m.sum()
    F = F - 0.5 * kappa * lam.log().sum()
    assert float(ll) == pytest.approx(float(F), rel=1e-10)
    assert torch.allclose(grad, lam * w * (mov - fix) * m / m.sum())

"""Losses for sparse label maps (e.g., vessels): dot/ndot/sqz and edt.

- derivatives of `prod`, `normprod`, `squeezed_prod` and `edt`
- Hessian shapes when a (C|1, *spatial) mask is passed (as the
  registration loop does)
- `edt` in `nitorch register`: distance maps computed internally
"""
import numpy as np
import torch
import pytest

from nitorch.tools.registration.losses.prod import (
    prod, normprod, squeezed_prod)
from nitorch.tools.registration.losses.edt import edt, stack_distance

dt = torch.double


def autograd_derivatives(fn, moving, fixed, **kwargs):
    """Gradient and diagonal Hessian (by autograd) wrt `moving`"""
    x = moving.clone().requires_grad_()
    ll = fn(x, fixed, grad=False, hess=False, **kwargs)
    g, = torch.autograd.grad(ll, x, create_graph=True)
    if not g.requires_grad:  # linear loss
        return g, torch.zeros_like(g)
    h = torch.stack([
        torch.autograd.grad(g.flatten()[i], x, retain_graph=True)[0]
        .flatten()[i] for i in range(x.numel())]).reshape(x.shape)
    return g.detach(), h


def random_images(nc=1, shape=(5, 4, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    moving = torch.rand([nc, *shape], dtype=dt, generator=g)
    fixed = 3 * torch.rand([nc, *shape], dtype=dt, generator=g)
    return moving, fixed


@pytest.mark.parametrize('fn', [prod, normprod, squeezed_prod])
def test_prod_gradients(fn):
    moving, fixed = random_images()
    _, g, h = fn(moving, fixed, dim=3)
    ga, ha = autograd_derivatives(fn, moving, fixed, dim=3)
    assert torch.allclose(g, ga)
    assert (h >= 0).all()
    if fn is squeezed_prod:
        # magnitude of the (concave) true Hessian
        assert torch.allclose(h, ha.abs())
    if fn is normprod:
        # upper bound of the Gauss-Newton diagonal
        nvox_m = moving.sum()
        r = (moving * fixed).sum() / nvox_m
        assert (h >= (fixed - r).square() / nvox_m.square() - 1e-12).all()


@pytest.mark.parametrize('fn', [prod, normprod, squeezed_prod, edt])
def test_hessian_shape_with_mask(fn):
    nc = 2 if fn is edt else 1
    moving, fixed = random_images(nc)
    mask = torch.rand([1, *moving.shape[1:]], dtype=dt)
    _, g, h = fn(moving, fixed, dim=3, mask=mask)
    assert g.shape == moving.shape
    assert h.dim() == moving.dim()
    assert torch.broadcast_shapes(h.shape, moving.shape) == moving.shape


def test_normprod_perfect_fit_is_finite():
    moving = torch.zeros([1, 6, 5, 1], dtype=dt)
    moving[:, 2] = 1
    fixed = 1 - moving    # distance map: zero on the moving labels
    out = normprod(moving, fixed, dim=3)
    assert all(torch.isfinite(t).all() for t in out)
    assert out[0] == 0


@pytest.mark.parametrize('sigma', [None, 1.5])
def test_edt_derivatives(sigma):
    labels, _ = random_images(2, seed=1)
    labels = (labels > 0.7).to(dt)
    _, dist = random_images(2, seed=2)
    moving = torch.cat([torch.rand_like(labels), dist])
    fixed = torch.cat([labels, torch.rand_like(dist)])
    mask = torch.rand([1, *labels.shape[1:]], dtype=dt)
    _, g, h = edt(moving, fixed, dim=3, sigma=sigma, mask=mask)
    ga, ha = autograd_derivatives(edt, moving, fixed, dim=3,
                                  sigma=sigma, mask=mask)
    assert torch.allclose(g, ga)
    assert (h >= 0).all()
    assert (h[:2] == 0).all() and (g[:2] == 0).all()   # label channels
    if sigma is None:
        assert torch.allclose(h, ha)    # quadratic: GN is exact


def test_edt_does_not_reward_shrinking():
    # dot: removing the warped labels decreases the loss to zero
    # edt: the sum runs over the static labels, so it does not
    shape = (12, 12, 1)
    fix = torch.zeros([1, *shape], dtype=dt)
    fix[:, 3, :] = 1
    mov = torch.zeros([1, *shape], dtype=dt)
    mov[:, 6, :] = 1
    fix = stack_distance(fix, dim=3)
    mov = stack_distance(mov, dim=3)
    empty = torch.zeros_like(mov[:1])
    assert prod(empty, fix[1:], dim=3, grad=False, hess=False) == 0
    ll = edt(mov, fix, dim=3, grad=False, hess=False)
    assert torch.allclose(ll, torch.as_tensor(0.5 * 3. ** 2, dtype=dt))


def test_stack_distance():
    labels = torch.zeros([1, 7, 6, 5])
    labels[0, 2, 3, 1] = 1
    labels[0, 5, 1, 4] = 0.8
    out = stack_distance(labels, dim=3, vx=[1, 2, 1])
    assert out.shape == (2, 7, 6, 5)
    assert torch.equal(out[0], labels[0])
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, dtype=torch.float) for s in labels.shape[1:]],
        indexing='ij'), -1) * torch.as_tensor([1., 2., 1.])
    pts = grid[labels[0] > 0.5]
    ref = (grid[..., None, :] - pts).norm(dim=-1).min(-1).values
    assert torch.allclose(out[1], ref, atol=1e-4)


# ----------------------------------------------------------------------
#   edt in `nitorch register`
# ----------------------------------------------------------------------

nib = pytest.importorskip('nibabel')


def tubes(shape, shift=0):
    """A few straight tubes (as uint8 labels), translated by
    `(shift, -shift/2, shift/2)` voxels"""
    lab = np.zeros(shape, 'uint8')
    lab[4, 3:-3, 6] = 1
    lab[3:-3, 9, 9] = 1
    lab[11, 5, 3:-3] = 1
    return np.roll(lab, (shift, -(shift // 2), shift // 2), (0, 1, 2))


def write(path, dat):
    nib.save(nib.Nifti1Image(dat, np.eye(4)), str(path))
    return str(path)


def test_cli_edt_builds_stacked_images(tmp_path):
    from nitorch.cli.registration.register.parser import parser
    from nitorch.cli.registration.register.cli import build_losses
    from nitorch.tools.registration import losses as L
    from nitorch.tools.registration.pairwise_pyramid import pyramid_levels
    fix = write(tmp_path / 'fix.nii.gz', tubes([16] * 3))
    mov = write(tmp_path / 'mov.nii.gz', tubes([16] * 3, 1))
    options = parser.parse(['@loss', 'edt', '-s', '-r', '2',
                            '@@fix', fix, '@@mov', mov,
                            '@affine', '@pyramid', '-l', '0'])
    levels = list(pyramid_levels([[1.] * 3] * 2, [[16] * 3] * 2,
                                 **options.pyramid))
    sims, _ = build_losses(options, [{'fix': levels[0], 'mov': levels[1]}],
                           torch.device('cpu'))
    sims = sims[0]
    assert len(sims) == 2                       # symmetric
    loss = sims[0].loss
    assert isinstance(loss, L.EDTLoss) and loss.sigma == 2
    fixdat = sims[0].fixed.dat
    assert fixdat.shape[0] == 2
    # integer labels are binarized despite the dequantization noise
    ref = torch.as_tensor(tubes([16] * 3), dtype=fixdat.dtype)
    assert torch.equal(fixdat[0], ref)
    assert (fixdat[1][ref > 0] == 0).all() and (fixdat[1][ref == 0] >= 1).all()


def test_cli_edt_recovers_translation(tmp_path):
    from nitorch.cli.registration.register.cli import _cli
    from nitorch import io
    fix = write(tmp_path / 'fix.nii.gz', tubes([16] * 3))
    mov = write(tmp_path / 'mov.nii.gz', tubes([16] * 3, 2))
    out = str(tmp_path / 'aff.lta')
    # full pyramid: at the coarsest level, the first step used to push
    # the labels out of the field of view (zero distance = perfect fit)
    _cli(['-v', '0', '@loss', 'edt', '-s', '@@fix', fix, '@@mov', mov,
          '@affine', 'translation', '-p', 'mov', '-o', out])
    aff = io.transforms.map(out).fdata().squeeze()
    trl = aff[:3, -1]
    ref = torch.as_tensor([2., -1., 1.], dtype=trl.dtype)
    assert torch.allclose(trl, ref, atol=0.1), aff


# ----------------------------------------------------------------------
#   @@fix/@@mov -e/--edt: precomputed distance maps
# ----------------------------------------------------------------------

def scipy_edt(lab, dtype='float32'):
    ndi = pytest.importorskip('scipy.ndimage')
    return ndi.distance_transform_edt(lab == 0).astype(dtype)


def built_edt_images(tmp_path, movfile, *movargs):
    from nitorch.cli.registration.register.parser import parser
    from nitorch.cli.registration.register.cli import build_losses
    from nitorch.tools.registration.pairwise_pyramid import pyramid_levels
    fix = write(tmp_path / 'fix.nii.gz', tubes([16] * 3))
    options = parser.parse(['@loss', 'edt', '@@fix', fix,
                            '@@mov', movfile, *movargs,
                            '@affine', '@pyramid', '-l', '0'])
    levels = list(pyramid_levels([[1.] * 3] * 2, [[16] * 3] * 2,
                                 **options.pyramid))
    sims, _ = build_losses(options, [{'fix': levels[0], 'mov': levels[1]}],
                           torch.device('cpu'))
    return sims[0][0].moving.dat


@pytest.mark.parametrize('dtype', ['float32', 'uint8'])
def test_cli_edt_precomputed_matches_computed(tmp_path, dtype):
    lab = tubes([16] * 3, 1)
    dist = scipy_edt(lab)
    if dtype == 'uint8':   # loaded with dequantization noise in [x, x+1)
        dist = np.round(dist).astype(dtype)
    computed = built_edt_images(
        tmp_path, write(tmp_path / 'mov.nii.gz', lab))
    given = built_edt_images(
        tmp_path, write(tmp_path / 'movd.nii.gz', dist), '-e')
    assert given.shape == computed.shape == (2, 16, 16, 16)
    assert torch.equal(given[0], computed[0])              # labels
    if dtype == 'float32':
        assert torch.allclose(given[1], computed[1], atol=1e-4)


def test_cli_edt_precomputed_needs_zeros(tmp_path):
    dist = scipy_edt(tubes([16] * 3)) + 1
    with pytest.raises(ValueError, match='zero on the structures'):
        built_edt_images(tmp_path, write(tmp_path / 'movd.nii.gz', dist), '-e')


@pytest.mark.parametrize('args, match', [
    (['@loss', 'mse', '@@fix', 'f.nii', '@@mov', 'm.nii', '-e'],
     'only valid with `@loss edt`'),
    (['@loss', 'edt', '@@fix', 'f.nii', '-e', '-l', '@@mov', 'm.nii'],
     'exclusive'),
])
def test_cli_edt_flag_validated(args, match):
    from nitorch.core.cli import ParseError
    from nitorch.cli.registration.register.parser import parser
    from nitorch.cli.registration.register.cli import check_options
    with pytest.raises(ParseError, match=match):
        check_options(parser.parse(args + ['@affine']))


def test_cli_edt_precomputed_recovers_translation(tmp_path):
    from nitorch.cli.registration.register.cli import _cli
    from nitorch import io
    fix = write(tmp_path / 'fix.nii.gz', tubes([16] * 3))
    mov = write(tmp_path / 'movd.nii.gz', scipy_edt(tubes([16] * 3, 2)))
    out = str(tmp_path / 'aff.lta')
    _cli(['-v', '0', '@loss', 'edt', '-s', '@@fix', fix, '@@mov', mov, '-e',
          '@affine', 'translation', '-p', 'mov', '-o', out])
    trl = io.transforms.map(out).fdata().squeeze()[:3, -1]
    ref = torch.as_tensor([2., -1., 1.], dtype=trl.dtype)
    assert torch.allclose(trl, ref, atol=0.1)


# ----------------------------------------------------------------------
#   Normalization and precision
# ----------------------------------------------------------------------

@pytest.mark.parametrize('fn', [prod, squeezed_prod])
def test_prod_is_a_weighted_mean(fn):
    # mean over the (weighted) voxels: invariant to a constant mask
    moving, fixed = random_images()
    ref = fn(moving, fixed, dim=3)
    out = fn(moving, fixed, dim=3, mask=torch.full_like(moving, 0.25))
    for a, b in zip(ref, out):
        assert torch.allclose(torch.as_tensor(a), torch.as_tensor(b))


def edt_images(nlab=(3, 30), seed=0):
    """Two channels with very different numbers of labels"""
    g = torch.Generator().manual_seed(seed)
    shape = (6, 5, 4)
    labels = torch.zeros([2, *shape], dtype=dt)
    for k, n in enumerate(nlab):
        labels[k].view(-1)[torch.randperm(labels[k].numel(), generator=g)[:n]] = 1
    dist = 3 * torch.rand([2, *shape], dtype=dt, generator=g)
    moving = torch.cat([torch.rand_like(labels), dist])
    fixed = torch.cat([labels, torch.rand_like(dist)])
    return moving, fixed, labels, dist


def test_edt_mean_per_channel():
    moving, fixed, labels, dist = edt_images()
    ll = edt(moving, fixed, dim=3, grad=False, hess=False)
    ref = sum(0.5 * dist[k][labels[k] > 0].square().mean() for k in range(2))
    assert torch.allclose(ll, ref)
    # precision: scales the loss (per channel)
    lam = torch.as_tensor([2., 5.], dtype=dt)
    ll = edt(moving, fixed, dim=3, lam=lam, grad=False, hess=False)
    ref = sum(0.5 * lam[k] * dist[k][labels[k] > 0].square().mean()
              for k in range(2))
    assert torch.allclose(ll, ref)


def test_edt_lam_vx():
    from nitorch.tools.registration.losses import EDTLoss
    moving, fixed, *_ = edt_images()
    ref = edt(moving, fixed, dim=3, lam=1/0.25, grad=False, hess=False)
    ll = EDTLoss(lam='vx', voxel_size=0.5).loss(moving, fixed, dim=3)
    assert torch.allclose(ll, ref)


@pytest.mark.parametrize('sigma', [None, 1.5])
def test_edt_lam_auto(sigma):
    from nitorch.tools.registration.losses import EDTLoss
    moving, fixed, labels, dist = edt_images()
    loss = EDTLoss(lam=None, sigma=sigma, voxel_size=0.5)
    lam, _ = loss.compute_lam(moving, fixed, None, 3, sigma)
    # MAP estimate under the Gamma prior
    rsq = dist.square()
    if sigma:
        rsq = rsq / (1 + rsq / sigma ** 2)
    for k in range(2):
        n = labels[k].sum()
        ssq = rsq[k][labels[k] > 0].sum()
        s0 = (loss.prior_std * 0.5) ** 2
        assert torch.allclose(lam[k], (loss.prior_dof + n) /
                              (ssq + loss.prior_dof * s0))
    # identical label maps: finite precision
    zero = torch.cat([moving[:2], torch.zeros_like(dist)])
    lam0, _ = loss.compute_lam(zero, fixed, None, 3, sigma)
    assert torch.isfinite(lam0).all()


def test_edt_lam_auto_gradient():
    # Without prior, lam is the ML estimate: the gradient at fixed lam
    # is the full gradient of the (lam-profiled) objective
    from nitorch.tools.registration.losses import EDTLoss
    moving, fixed, *_ = edt_images()
    loss = EDTLoss(lam=None, prior_dof=0)
    _, g, _ = loss.loss_grad_hess(moving, fixed, dim=3)
    x = moving.clone().requires_grad_()
    ga, = torch.autograd.grad(loss.loss(x, fixed, dim=3), x)
    assert torch.allclose(g, ga)


def test_cli_edt_weight(tmp_path):
    from nitorch.cli.registration.register.parser import parser
    from nitorch.cli.registration.register.cli import build_losses
    from nitorch.tools.registration.pairwise_pyramid import pyramid_levels
    fix = tmp_path / 'fix.nii.gz'
    nib.save(nib.Nifti1Image(tubes([16] * 3), np.diag([.5, .5, 2., 1.])), str(fix))
    mov = write(tmp_path / 'mov.nii.gz', tubes([16] * 3, 1))
    for arg, lam in [([], 'vx'), (['-w', 'auto'], None),
                     (['-w', 'vx'], 'vx'), (['-w', '3'], 3.)]:
        options = parser.parse(['@loss', 'edt', '-s', *arg,
                                '@@fix', str(fix), '@@mov', mov,
                                '@affine', '@pyramid', '-l', '0'])
        levels = list(pyramid_levels([[1.] * 3] * 2, [[16] * 3] * 2,
                                     **options.pyramid))
        sims, _ = build_losses(options, [{'fix': levels[0], 'mov': levels[1]}],
                               torch.device('cpu'))
        fwd, bwd = sims[0]
        assert fwd.loss.lam == lam and bwd.loss.lam == lam
        # static image of each term: geometric mean of its voxel size
        assert fwd.loss.voxel_size == pytest.approx(0.5 ** (2/3) * 2 ** (1/3))
        assert bwd.loss.voxel_size == pytest.approx(1.)

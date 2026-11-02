"""Affine gradient and Hessian of `do_affine` (affine + nonlinear model).

With the affine on the moving side (`-p mov`, and the moving half of the
default symmetric position), the forward warp is

    warped(x) = mu( L(theta) phi_right(x) ),

where `phi_right` maps fixed voxels to nonlinear-field voxels (right
affine + displacement) and `L(theta)` maps field voxels to moving voxels.
The exact derivative is

    dwarped(x)/dtheta = grad_mu(warped point)' dL/dtheta [phi_right(x); 1],

i.e., image gradients and coordinates taken at the points `phi_right(x)`.
`do_affine` used to push the loss gradient through `phi_right` to the
field lattice first, and then take the image gradient and coordinates
at the field nodes: exact only if `phi_right` is the identity. In the
default symmetric position, `phi_right` always contains the fixed half
of the affine, so the gradient was 15-25% off even without any
deformation.
"""
import torch
import pytest
from nitorch import spatial
from nitorch.core import linalg
from nitorch.tools.registration import objects, utils as regutils
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep

dt = torch.double


def smooth(x, n):
    """Box-smooth a (C, *spatial) tensor with circular boundaries"""
    for _ in range(n):
        for d in range(1, x.dim()):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    return x


def make_images(shape, vx_mov=1.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 3)
    mov = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 3)
    fix = objects.Image(fix, affine=spatial.affine_default(shape, 1., dtype=dt))
    mov = objects.Image(mov, affine=spatial.affine_default(shape, vx_mov, dtype=dt))
    return fix, mov


def make_affine(pos, basis='rigid', seed=5):
    affine = objects.AffineModel(basis, position=pos)
    affine.set_dat(dim=3, dtype=dt)
    g = torch.Generator().manual_seed(seed)
    prm = torch.randn(affine.dat.dat.shape, dtype=dt, generator=g) * 0.05
    prm[:3] *= 10   # translations (mm)
    affine.dat.dat.copy_(prm)
    return affine


def make_svf(fix, lattice='same', scale=1., seed=1):
    """SVF with a zero band along its borders.

    lattice: 'same' (the fixed lattice) or 'coarse' (2 mm, shifted)
    """
    if lattice == 'same':
        shape, aff = fix.shape, fix.affine
    else:
        shape = [s // 2 + 1 for s in fix.shape]
        aff = spatial.affine_default(shape, 2., dtype=dt)
        aff[:3, -1] += 0.3
    g = torch.Generator().manual_seed(seed)
    vel = smooth(torch.randn([3, *shape], dtype=dt, generator=g), 4) * scale
    band = 2 if lattice == 'same' else 1
    window = torch.zeros(shape, dtype=dt)
    window[tuple(slice(band, -band) for _ in shape)] = 1
    vel = (vel * window).movedim(0, -1)
    nonlin = objects.SVFModel(steps=8, penalty=dict(
        absolute=0, membrane=0, bending=0, lame=(0, 0), factor=0))
    nonlin.set_dat(shape, affine=aff, dtype=dt)
    nonlin.dat.dat.copy_(vel)
    return nonlin


def make_sim(mov, fix, kind):
    sim = objects.Similarity(MSE(lam=1, dim=3), mov, fix)
    if kind == 'fwd':
        return objects.SumSimilarity([sim])
    if kind == 'bwd':
        return objects.SumSimilarity([sim.reverse()])
    return sim.symmetrize()


def fd_grad(fn, x, eps=1e-6):
    fd = torch.zeros_like(x).view(-1)
    for i in range(x.numel()):
        xp, xm = x.clone().view(-1), x.clone().view(-1)
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (fn(xp.view_as(x), in_line_search=True) -
                 fn(xm.view_as(x), in_line_search=True)) / (2 * eps)
    return fd.view_as(x)


def setup(pos, kind, lattice='same', scale=1., basis='rigid'):
    fix, mov = make_images([12, 12, 12])
    affine = make_affine(pos, basis)
    nonlin = make_svf(fix, lattice, scale)
    sim = make_sim(mov, fix, kind)
    step = PairwiseRegisterStep(sim, affine, nonlin, verbose=0)
    return step, affine, nonlin, sim


def warp(affine, nonlin, sim, logaff):
    """Reference forward pass of `do_affine` for one term"""
    pos = affine.position[0].lower()
    if sim.backward:
        pos = {'m': 'f', 'f': 'm'}.get(pos, pos)
        aff, phi = affine.iexp(logaff), nonlin.iexp()
    else:
        aff, phi = affine.exp(logaff), nonlin.exp()
    moving, fixed = sim.moving, sim.fixed
    aff_right = fixed.affine
    if pos in 'fs':
        aff_right = aff @ aff_right
    aff_right = linalg.lmdiv(nonlin.affine, aff_right)
    aff_left = nonlin.affine
    if pos in 'ms':
        aff_left = aff @ aff_left
    aff_left = linalg.lmdiv(moving.affine, aff_left)
    right = spatial.affine_grid(aff_right, fixed.shape)
    psi = regutils.smart_pull_grid(phi, right, bound='zero') + right
    psi = spatial.affine_matvec(aff_left, psi)
    return moving.pull(psi, mask=True)


def gn_hessian(affine, nonlin, sims, logaff):
    """Gauss-Newton Hessian of a sum of terms"""
    return sum(sim.factor * _gn_hessian(affine, nonlin, sim, logaff)
               for sim in sims)


def ref_loss(affine, nonlin, sims, logaff):
    loss = 0
    for sim in sims:
        warped, mask = warp(affine, nonlin, sim, logaff)
        loss += sim.factor * sim.loss.loss(warped, sim.fixed.dat,
                                           dim=3, mask=mask)
    return loss


def _gn_hessian(affine, nonlin, sim, logaff, eps=1e-6):
    """Exact Gauss-Newton Hessian sum_x h(x) a(x) a(x)', with
    a = dwarped/dtheta from central finite differences."""
    warped, mask = warp(affine, nonlin, sim, logaff)
    _, _, h = sim.loss.loss_grad_hess(warped, sim.fixed.dat, dim=3, mask=mask)
    jac = []
    for i in range(logaff.numel()):
        xp, xm = logaff.clone(), logaff.clone()
        xp[i] += eps
        xm[i] -= eps
        jac.append((warp(affine, nonlin, sim, xp)[0] -
                    warp(affine, nonlin, sim, xm)[0]) / (2 * eps))
    jac = torch.stack(jac, -1).reshape([-1, logaff.numel()])
    h = h.expand(warped.shape).reshape([-1, 1])
    return (jac * h).T @ jac


# ----------------------------------------------------------------------
#   gradient
# ----------------------------------------------------------------------

@pytest.mark.parametrize('pos', ['m', 's', 'f'])
@pytest.mark.parametrize('kind', ['fwd', 'bwd', 'sym'])
@pytest.mark.parametrize('lattice', ['same', 'coarse'])
def test_affine_grad(pos, kind, lattice):
    """Affine gradient vs central finite differences, with a deformation.

    Backward terms put the affine on the other side (`-p fix` backward
    goes through the moving-side branch). The fixed-side branch samples
    a finite-difference Jacobian of the displacement, which is not the
    derivative of its linear interpolant when the field is not sampled
    at its nodes: that branch (unchanged here) is ~1e-5 off with the
    coarse field. Everything else is exact to FD precision."""
    step, affine, *_ = setup(pos, kind, lattice, 1., 'affine')
    x = affine.dat.dat.clone()
    _, g = step.do_affine(x, grad=True, in_line_search=True)
    fd = fd_grad(step.do_affine, x)
    err = (fd - g).norm() / fd.norm()
    fixed_side_jac = (lattice == 'coarse' and kind != 'bwd' and pos == 'f')
    assert err < (1e-4 if fixed_side_jac else 1e-6), err


@pytest.mark.parametrize('pos', ['m', 's'])
@pytest.mark.parametrize('kind', ['fwd', 'sym'])
def test_affine_grad_no_deformation(pos, kind):
    """Zero displacement: in the symmetric position `phi_right` is still
    not the identity (it holds the fixed half of the affine)."""
    step, affine, *_ = setup(pos, kind, 'coarse', 0., 'rigid')
    x = affine.dat.dat.clone()
    _, g = step.do_affine(x, grad=True, in_line_search=True)
    fd = fd_grad(step.do_affine, x)
    err = (fd - g).norm() / fd.norm()
    assert err < 1e-6, err


# ----------------------------------------------------------------------
#   Hessian
# ----------------------------------------------------------------------

@pytest.mark.parametrize('pos', ['m', 'f'])
@pytest.mark.parametrize('kind', ['fwd', 'bwd', 'sym'])
@pytest.mark.parametrize('lattice', ['same', 'coarse'])
def test_affine_hess_one_side_is_gauss_newton(pos, kind, lattice):
    """Affine on one side: the Hessian is the exact Gauss-Newton Hessian
    (up to the fixed-side finite-difference Jacobian, see
    `test_affine_grad`)."""
    step, affine, nonlin, sim = setup(pos, kind, lattice, 1., 'affine')
    x = affine.dat.dat.clone()
    ll, _, h = step.do_affine(x, grad=True, hess=True, in_line_search=True)
    # the reference forward pass is the one `do_affine` evaluates
    assert torch.allclose(ll, ref_loss(affine, nonlin, sim, x))
    ref = gn_hessian(affine, nonlin, sim, x)
    fixed_side_jac = (lattice == 'coarse' and kind != 'bwd' and pos == 'f')
    tol = 1e-4 if fixed_side_jac else 1e-6
    assert (h - ref).norm() / ref.norm() < tol


@pytest.mark.parametrize('kind', ['fwd', 'bwd', 'sym'])
@pytest.mark.parametrize('lattice', ['same', 'coarse'])
def test_affine_hess_symmetric_position(kind, lattice):
    """Default position: H = 2 (H_mov + H_fix), twice the sum of the
    Gauss-Newton Hessians of the two halves. Since
    (a+b)(a+b)' <= 2(aa' + bb'), it majorises the full Gauss-Newton
    Hessian; both halves being nearly equal, it is also tight (the
    plain sum used to be about half of it)."""
    step, affine, nonlin, sim = setup('s', kind, lattice, 1., 'affine')
    x = affine.dat.dat.clone()
    ll, _, h = step.do_affine(x, grad=True, hess=True, in_line_search=True)
    assert torch.allclose(h, h.T)
    assert torch.allclose(ll, ref_loss(affine, nonlin, sim, x))
    ref = gn_hessian(affine, nonlin, sim, x)
    # generalised eigenvalues of (H, H_gn)
    L = torch.linalg.cholesky(ref)
    M = torch.linalg.solve_triangular(L, h, upper=False)
    M = torch.linalg.solve_triangular(L, M.T, upper=False)
    ev = torch.linalg.eigvalsh((M + M.T) / 2)
    assert ev.min() > 1 - 1e-6, ev     # majorises
    assert ev.max() < 1.05, ev         # tight

# grid_inv needs its own file to avoid circular imports
import torch
from nitorch.core import utils
from ._grid import (grid_push, grid_pull, grid_count, grid_jacobian,
                    identity_grid)
from ._solvers import solve_grid_fmg


# Values of the `inversion` option of nonlinear transforms
# (`spatial.shoot`, the registration models, `register --inversion`).
# 'default' keeps the historical computation; the other values are
# methods of `grid_inv`.
inversion_methods = ('default', 'push', 'newton')


def check_inversion(inversion):
    """Validate (and normalise) the value of an `inversion` option.

    Parameters
    ----------
    inversion : {'default', 'push', 'newton'} or None
        None means 'default'.

    Returns
    -------
    inversion : {'default', 'push', 'newton'}

    """
    inversion = 'default' if inversion is None else inversion
    if inversion not in inversion_methods:
        raise ValueError(f"inversion must be one of {inversion_methods} "
                         f"but got {inversion!r}")
    return inversion


def grid_inv(grid, type='grid', bound='dft',
             extrapolate=True, method='push', newton_iter=3, **prm):
    r"""Invert a dense deformation (or displacement) grid

    Notes
    -----
    The deformation/displacement grid must be expressed in
    voxels, and map from/to the same lattice.

    Let `f = id + d` be the transformation. The inverse
    is obtained as `id - (f.T @ 1 + L) \ (f.T @ d)`
    where `L` is a regulariser, `f.T @ _` is the adjoint
    operation ("push") of `f @ _` ("pull"). and `1` is an
    image of ones.

    If `method='newton'`, this "push" inverse $\psi$ is then refined by
    `newton_iter` Newton steps on $\varphi \circ \psi = id$ (with
    $\varphi = id + u$ the transform to invert):
    $\psi \gets \psi - \lambda (I + \nabla u \circ \psi)^{-1}
    (\varphi \circ \psi - id)$. Each step is safeguarded voxel-wise:
    where $\det(I + \nabla u \circ \psi) < 0.05$ (folded or nearly
    singular Jacobian) the Newton direction is replaced by a plain
    fixed-point step $\psi \gets id - u \circ \psi$, and a
    backtracking line search ($\lambda = 1, 1/2, 1/4, 1/8$) keeps, in
    each voxel, the first step that decreases the voxel's residual
    $|\varphi \circ \psi - id|$ (voxels without such a step do not
    move). The residual therefore never increases. Each step costs
    one interpolation of the Jacobian and up to four of the
    displacement (linear interpolation, same `bound`).

    The idea behind this is that `f.T @ _` is approximately
    the inverse transformation weighted by the determinant
    of the Jacobian of the tranformation so, in the (theoretical)
    continuous case, `inv(f) @ _ = f.T @ _ / f.T @ 1`.
    However, in the (real) discrete case, this leads to
    lots of holes in the inverse. The solution we use
    therefore fills these holes using a regularised
    least-squares scheme, where the regulariser penalizes
    the spatial gradients of the inverse field.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation (or displacement) grid
    type : {'grid', 'disp'}, default='grid'
        Type of deformation.
    membrane : float, default=0.1
        Regularisation
    bound : str, default='dft'
        Boundary conditions
    extrapolate : bool, default=True
        Extrapolate the transformation field when
        it is sampled out-of-bounds.
    method : {'push', 'newton'}, default='push'
        'push': push + count + regularised fill (see Notes).
        'newton': 'push', then `newton_iter` safeguarded Newton steps.
        More accurate, in particular on steep warps (large rotations or
        expansions), about 40% slower.
    newton_iter : int, default=3
        Number of Newton steps (if `method='newton'`).

    Returns
    -------
    grid_inv : (..., *spatial, dim)
        Inverse transformation (or displacement) grid

    """
    if method not in ('push', 'newton'):
        raise ValueError(f"method must be 'push' or 'newton' "
                         f"but got {method!r}")
    prm = prm or dict(membrane=0.1)

    # get shape components
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    shape = grid.shape[-(dim + 1):-1]
    batch = grid.shape[:-(dim + 1)]
    grid = grid.reshape([-1, *shape, dim])
    backend = dict(dtype=grid.dtype, device=grid.device)

    # get displacement
    identity = identity_grid(shape, **backend)
    if type == 'grid':
        disp = grid - identity
    else:
        disp = grid
        grid = disp + identity
    newton = method == 'newton' and newton_iter > 0
    fwd = disp if newton else None  # displacement to invert

    # push displacement
    push_opt = dict(bound=bound, extrapolate=extrapolate)
    disp = utils.movedim(disp, -1, 1)
    disp = grid_push(disp, grid, **push_opt)
    count = grid_count(grid, **push_opt)
    disp = utils.movedim(disp, 1, -1)
    count = utils.movedim(count, 1, -1)

    # Fill missing values using regularised least squares
    # (the multigrid solver expects a symmetric Hessian with D*(D+1)//2
    # channels: diagonal = counts, off-diagonal = 0)
    hess = count.new_zeros([*count.shape[:-1], dim * (dim + 1) // 2])
    hess[..., :dim] = count
    del count
    disp = solve_grid_fmg(hess, disp, bound=bound, **prm)
    del hess

    if newton:
        del grid
        disp = -_newton_inv(fwd, disp.neg_(), identity, newton_iter,
                            bound=bound, extrapolate=extrapolate)
        del fwd
    disp = disp.reshape([*batch, *shape, dim])

    if type == 'grid':
        return identity - disp
    else:
        return -disp


def _newton_inv(disp, idisp, identity, nb_iter=3, detmin=0.05,
                nb_halvings=3, bound='dft', extrapolate=True):
    r"""Refine an inverse displacement with safeguarded Newton steps.

    Solves $\varphi \circ \psi = id$, with $\varphi = id + u$,
    $\psi = id + w$, i.e., $r(w) = w + u \circ (id + w) = 0$, by
    $w \gets w - \lambda (I + \nabla u \circ \psi)^{-1} r(w)$.
    See `grid_inv` for the safeguards.

    Parameters
    ----------
    disp : (batch, *spatial, dim) tensor
        Displacement `u` to invert
    idisp : (batch, *spatial, dim) tensor
        Initial inverse displacement `w`
    identity : (*spatial, dim) tensor
        Identity grid
    nb_iter : int
        Number of Newton steps
    detmin : float
        Smallest Jacobian determinant for which the Newton direction
        is used (else, fixed-point direction).
    nb_halvings : int
        Number of step halvings in the line search.

    Returns
    -------
    idisp : (batch, *spatial, dim) tensor
        Refined inverse displacement

    """
    dim = disp.shape[-1]
    opt = dict(bound=bound, extrapolate=extrapolate, interpolation=1)

    def pull(field, w):
        # field(id + w), with field (batch, *spatial, channels)
        out = grid_pull(utils.movedim(field, -1, 1), identity + w, **opt)
        return utils.movedim(out, 1, -1)

    def sqnorm(r):
        return r.square().sum(-1)

    # Jacobian of phi (central finite differences), flattened
    jac = grid_jacobian(disp, type='disp', bound=bound)
    jac = jac.reshape([*jac.shape[:-2], dim * dim])

    res = idisp + pull(disp, idisp)          # phi o psi - id
    for _ in range(nb_iter):
        resnorm = sqnorm(res)

        # Newton direction (fixed-point direction where J is unsafe)
        jacw = pull(jac, idisp).reshape([*idisp.shape, dim])
        delta = _safe_solve(jacw, res, detmin)
        del jacw

        # per-voxel backtracking line search
        todo = torch.ones_like(resnorm, dtype=torch.bool)
        new_idisp, new_res = idisp, res
        lam = 1.
        for _ in range(nb_halvings + 1):
            cand = idisp - delta if lam == 1 else idisp - lam * delta
            cand_res = cand + pull(disp, cand)
            accept = todo & (sqnorm(cand_res) < resnorm)
            new_idisp = torch.where(accept[..., None], cand, new_idisp)
            new_res = torch.where(accept[..., None], cand_res, new_res)
            del cand, cand_res
            todo = todo & ~accept
            lam /= 2
            if not todo.any():
                break
        idisp, res = new_idisp, new_res
        del delta, new_idisp, new_res, todo, accept
    return idisp


def _safe_solve(jac, res, detmin):
    """Solve `jac @ x = res` voxel-wise, with `x = res` where
    `det(jac) <= detmin`. Closed form in 2D and 3D.

    jac : (..., dim, dim), res : (..., dim) -> (..., dim)
    """
    dim = res.shape[-1]
    if dim == 1:
        det = jac[..., 0, 0]
        ok = det > detmin
        x = res / torch.where(ok, det, det.new_ones([]))[..., None]
    elif dim == 2:
        a, b = jac[..., 0, 0], jac[..., 0, 1]
        c, d = jac[..., 1, 0], jac[..., 1, 1]
        det = a * d - b * c
        ok = det > detmin
        det = torch.where(ok, det, det.new_ones([]))
        r0, r1 = res[..., 0], res[..., 1]
        x = torch.stack([d * r0 - b * r1, a * r1 - c * r0], -1)
        x /= det[..., None]
    elif dim == 3:
        j = [[jac[..., i, k] for k in range(3)] for i in range(3)]
        # cofactors (adjugate = transpose of the cofactor matrix)
        c00 = j[1][1] * j[2][2] - j[1][2] * j[2][1]
        c01 = j[1][2] * j[2][0] - j[1][0] * j[2][2]
        c02 = j[1][0] * j[2][1] - j[1][1] * j[2][0]
        det = j[0][0] * c00 + j[0][1] * c01 + j[0][2] * c02
        ok = det > detmin
        det = torch.where(ok, det, det.new_ones([]))
        c10 = j[0][2] * j[2][1] - j[0][1] * j[2][2]
        c11 = j[0][0] * j[2][2] - j[0][2] * j[2][0]
        c12 = j[0][1] * j[2][0] - j[0][0] * j[2][1]
        c20 = j[0][1] * j[1][2] - j[0][2] * j[1][1]
        c21 = j[0][2] * j[1][0] - j[0][0] * j[1][2]
        c22 = j[0][0] * j[1][1] - j[0][1] * j[1][0]
        r0, r1, r2 = res[..., 0], res[..., 1], res[..., 2]
        x = torch.stack([c00 * r0 + c10 * r1 + c20 * r2,
                         c01 * r0 + c11 * r1 + c21 * r2,
                         c02 * r0 + c12 * r1 + c22 * r2], -1)
        x /= det[..., None]
    else:
        det = torch.linalg.det(jac)
        ok = det > detmin
        eye = torch.eye(dim, dtype=jac.dtype, device=jac.device)
        jac = torch.where(ok[..., None, None], jac, eye)
        x = torch.linalg.solve(jac, res[..., None])[..., 0]
    return torch.where(ok[..., None], x, res)

# grid_inv needs its own file to avoid circular imports
import torch
from nitorch.core import utils
from ._grid import (grid_push, grid_pull, grid_count, grid_jacobian,
                    identity_grid)
from ._solvers import solve_grid_fmg, solve_grid


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


# Keywords of `grid_inv(**prm)` that configure the multigrid solver
# rather than the regularisation.
_fmg_keys = ('optim', 'nb_cycles', 'nb_iter')
# Solvers of the Gauss-Newton steps (same names as in `denoise`)
newton_optims = ('fmg', 'fmg+cg', 'fmg+relax', 'cg', 'relax')


def grid_inv(grid, type='grid', bound='dft', extrapolate=True,
             method='push', newton_iter=5, newton_factor=1.,
             newton_optim='fmg', **prm):
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

    The idea behind this is that `f.T @ _` is approximately
    the inverse transformation weighted by the determinant
    of the Jacobian of the tranformation so, in the (theoretical)
    continuous case, `inv(f) @ _ = f.T @ _ / f.T @ 1`.
    However, in the (real) discrete case, this leads to
    lots of holes in the inverse. The solution we use
    therefore fills these holes using a regularised
    least-squares scheme, where the regulariser penalizes
    the spatial gradients of the inverse field.

    If `method='newton'`, this "push" inverse $\psi$ is then refined by
    `newton_iter` regularised Gauss-Newton steps on
    $\varphi \circ \psi = id$ (with $\varphi = id + u$ the transform to
    invert). Each step $\psi \gets \psi + \delta$ minimises the
    linearised residual plus a penalty on the step,
    $\frac{1}{2}\|r + J\delta\|^2 + \frac{\lambda}{2}\delta^T L \delta$,
    with $r = \varphi \circ \psi - id$, $J = I + \nabla u \circ \psi$,
    $L$ the regulariser of the push step (`membrane=0.1` by default)
    and $\lambda$ = `newton_factor`. It is one coupled solve,
    $(J^T J + \lambda L)\,\delta = -J^T r$, with the field solver of the
    push step (`solve_grid_fmg`, see `newton_optim`). Because the
    penalty is on the step, not on $\psi - id$, the iteration still
    converges to the exact inverse where $J$ is invertible. Where it is
    singular (folds, steep compressions), $J^T J$ vanishes and the step
    is filled in from the neighbours, as the holes of the push step
    are. The steps are not safeguarded: the residual decreases on the
    whole field, not necessarily in every voxel. Each step costs one
    (linear, same `bound`) interpolation of the Jacobian and one of the
    displacement, and one regularised solve.

    The inversion is **not differentiable**: backpropagating through
    it fails inside the multigrid solver (an in-place modification of
    a view). Call it on tensors that do not require grad, or under
    `torch.no_grad()`.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation (or displacement) grid
    type : {'grid', 'disp'}, default='grid'
        Type of deformation.
    membrane : float, default=0.1
        Regularisation. The other keywords of `solve_grid_fmg`
        (`absolute`, `bending`, `lame`, `factor`, `voxel_size`, `optim`,
        `nb_cycles`, `nb_iter`) are accepted too; if any keyword is
        given, `membrane` defaults to 0.
    bound : str, default='dft'
        Boundary conditions
    extrapolate : bool, default=True
        Extrapolate the transformation field when
        it is sampled out-of-bounds.
    method : {'push', 'newton'}, default='push'
        'push': push + count + regularised fill (see Notes).
        'newton': 'push', then `newton_iter` regularised Gauss-Newton
        steps. More accurate, in particular on steep warps (large
        rotations or expansions); each step costs about as much as
        the push step.
    newton_iter : int, default=5
        Number of Gauss-Newton steps (if `method='newton'`).
    newton_factor : float, default=1
        Weight of the regularisation of the Gauss-Newton steps, relative
        to that of the push step. It must be positive (the steps are
        ill-posed without regularisation).
    newton_optim : {'fmg', 'fmg+cg', 'fmg+relax', 'cg', 'relax'}, default='fmg'
        Solver of the Gauss-Newton steps: full multigrid with conjugate
        gradient ('fmg' = 'fmg+cg') or relaxation ('fmg+relax')
        smoothing (`solve_grid_fmg`, with the `nb_cycles` and `nb_iter`
        of the push step); or single-resolution conjugate gradient or
        relaxation (`solve_grid`, 16 iterations).

    Returns
    -------
    grid_inv : (..., *spatial, dim)
        Inverse transformation (or displacement) grid

    """
    if method not in ('push', 'newton'):
        raise ValueError(f"method must be 'push' or 'newton' "
                         f"but got {method!r}")
    if newton_optim not in newton_optims:
        raise ValueError(f"newton_optim must be one of {newton_optims} "
                         f"but got {newton_optim!r}")
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
        solve = _step_solver(prm, newton_factor, newton_optim, bound)
        disp = _gauss_newton_inv(fwd, disp.neg_(), identity, newton_iter,
                                 solve, bound=bound, extrapolate=extrapolate)
        disp = disp.neg_()
        del fwd
    disp = disp.reshape([*batch, *shape, dim])

    if type == 'grid':
        return identity - disp
    else:
        return -disp


def _step_solver(prm, factor, optim, bound):
    """Solver `(hess, grad) -> (hess + factor * L) \\ grad` of the
    Gauss-Newton steps, with `L` the regulariser of the push step."""
    reg = {k: v for k, v in prm.items() if k not in _fmg_keys}
    reg['factor'] = reg.get('factor', 1) * factor
    if optim.startswith('fmg'):
        fmg = {k: v for k, v in prm.items() if k in _fmg_keys}
        fmg['optim'] = 'relax' if optim == 'fmg+relax' else 'cg'

        def solve(hess, grad):
            return solve_grid_fmg(hess, grad, bound=bound, **reg, **fmg)
    else:
        def solve(hess, grad):
            return solve_grid(hess, grad, bound=bound, optim=optim, **reg)
    return solve


def _gauss_newton_inv(disp, idisp, identity, nb_iter, solve,
                      bound='dft', extrapolate=True):
    r"""Refine an inverse displacement with regularised Gauss-Newton steps.

    Solves $\varphi \circ \psi = id$, with $\varphi = id + u$,
    $\psi = id + w$, i.e., $r(w) = w + u \circ (id + w) = 0$, by
    $w \gets w - (J^T J + L)^{-1} J^T r(w)$, with
    $J = I + \nabla u \circ (id + w)$. See `grid_inv`.

    Parameters
    ----------
    disp : (batch, *spatial, dim) tensor
        Displacement `u` to invert
    idisp : (batch, *spatial, dim) tensor
        Initial inverse displacement `w` (overwritten)
    identity : (*spatial, dim) tensor
        Identity grid
    nb_iter : int
        Number of Gauss-Newton steps
    solve : callable(hess, grad) -> step
        Regularised solver of `(hess + L) step = grad`, with `hess` a
        field of symmetric matrices (D*(D+1)//2 channels).

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

    # Jacobian of phi (central finite differences), flattened
    jac = grid_jacobian(disp, type='disp', bound=bound)
    jac = jac.reshape([*jac.shape[:-2], dim * dim])

    for _ in range(nb_iter):
        res = pull(disp, idisp).add_(idisp)             # phi o psi - id
        jacw = pull(jac, idisp).reshape([*idisp.shape, dim])
        hess, grad = _gauss_newton_terms(jacw, res)
        del jacw, res
        idisp = idisp.sub_(solve(hess, grad))
        del hess, grad
    return idisp


def _gauss_newton_terms(jac, res):
    """Gauss-Newton Hessian `J.T @ J` (symmetric, D*(D+1)//2 channels:
    diagonal, then upper triangle row by row) and gradient `J.T @ r`.

    jac : (..., dim, dim), res : (..., dim)
    -> hess : (..., dim*(dim+1)//2), grad : (..., dim)
    """
    dim = res.shape[-1]
    grad = jac.transpose(-1, -2).matmul(res[..., None])[..., 0]
    hess = res.new_empty([*res.shape[:-1], dim * (dim + 1) // 2])
    for i in range(dim):
        hess[..., i] = jac[..., :, i].square().sum(-1)
    c = dim
    for i in range(dim):
        for j in range(i + 1, dim):
            hess[..., c] = (jac[..., :, i] * jac[..., :, j]).sum(-1)
            c += 1
    return hess, grad

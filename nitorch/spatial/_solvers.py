"""
This file implements solvers for spatially regularized vector fields.

The regularization can be stationary -- and can therefore be represented
by a convolution -- or be further weighted by a non-stationary weight map.
This non-stationary weight map can arise in a reweighted least squares
context.

All functions are implemented in a "field" (for vector fields) or "grid"
(for displacement fields) flavor.

- solve_field_fmg(H, g) / solve_grid_fmg(H, g)
    Solve the linear system with a full multi-grid solver.
    By default, the underlying solver uses conjugate gradient descent
    with Jacobi preconditioning ('cg'). Alternatively, a checkerboard
    Gauss-Seidel scheme ('relax') can be used. In both cases, the spatial
    Hessian is implemented using finite differences in python. Alternatively,
    'relaks' is equivalent to 'relax' but uses sparse convolutions.

- solve_field(H, g) / solve_grid(H, g)
    Solve the linear system with a "classical" single-grid solver.
    By default, the underlying solver uses conjugate gradient descent
    with Jacobi preconditioning ('cg'). Alternatively, a checkerboard
    Gauss-Seidel scheme ('relax') can be used. In both cases, the spatial
    Hessian is implemented using finite differences in python.

- solve_field_kernel(H, g) / solve_grid_kernel(H, g)
    Solve the linear system with a "classical" single-grid solver.
    By default, the underlying solver uses conjugate gradient descent
    with Jacobi preconditioning ('cg'). Alternatively, a checkerboard
    Gauss-Seidel scheme ('relax') can be used. In both cases, the spatial
    Hessian is implemented using sparse convolutions.
    These functions do not support weight maps.

- solve_field_closedform(H, g) / solve_grid_closedform(H, g)
    Solve the linear system with closed-form matrix inversions.
    These functions can only be used is no spatial regularization is used.

"""

import torch
import math
from nitorch.core import utils, py
from nitorch.core import optim as optimizers
from nitorch.core.linalg import sym_matvec, sym_solve
from ._grid import grid_pull, grid_push
from ._regularisers import (absolute, membrane, bending,
                            absolute_grid, membrane_grid, bending_grid,
                            lame_shear, lame_div,
                            absolute_diag, membrane_diag, bending_diag,
                            regulariser_grid_kernel, regulariser_kernel)
from ._spconv import spconv


# TODO:
#   - implement separable prolong/restrict (with torchscript?)


# ======================================================================
#                           Multigrid solvers
# ======================================================================


def prolong(x, shape=None, bound='dct2', order=2, dim=None):
    """Prolongation of a tensor to a finer grid.
    
    Uses the pulling operator (2nd order by default).
    Corresponds to mode "edges" in `spatial.resize`.
    
    Parameters
    ----------
    x : (..., *spatial, K) tensor
    shape : sequence of int, default=`2*spatial`
    bound : [sequence of] bound_type, default='dct2'
    order : [sequence of] int, default=1
    dim : int, default=`x.dim()-1`
    
    Returns
    -------
    y : (..., *out_spatial, K) tensor
    
    """
    backend = utils.backend(x)
    dim = dim or (x.dim() - 1)
    in_spatial = x.shape[-dim-1:-1]
    out_spatial = shape or [2*s for s in in_spatial]
    shifts = [0.5 * (inshp / outshp - 1)
              for inshp, outshp in zip(in_spatial, out_spatial)]
    grid = [torch.arange(0., outshp, **backend).mul_(inshp/outshp).add_(shift)
            for inshp, outshp, shift in zip(in_spatial, out_spatial, shifts)]
    grid = torch.stack(torch.meshgrid(*grid), dim=-1)
    x = utils.fast_movedim(x, -1, -dim-1)
    x = grid_pull(x, grid, bound=bound, interpolation=order, extrapolate=True)
    x = utils.fast_movedim(x, -dim-1, -1)
    return x
    
    
def restrict(x, shape=None, bound='dct2', order=1, dim=None):
    """Restriction of a tensor to a coarser grid.
    
    Uses the pushing operator (1st order by default).
    Corresponds to mode "edges" in `spatial.resize`.
    
    Parameters
    ----------
    x : (..., *spatial, K) tensor
    shape : sequence of int, default=`ceil(spatial/2)`
    bound : [sequence of] bound_type, default='dct2'
    order : [sequence of] int, default=1
    dim : int, default=`x.dim()-1`
    
    Returns
    -------
    y : (..., *out_spatial, K) tensor
    
    """
    backend = utils.backend(x)
    dim = dim or (x.dim() - 1)
    out_spatial = x.shape[-dim-1:-1]
    in_spatial = shape or [math.ceil(s/2) for s in out_spatial]
    shifts = [0.5 * (inshp / outshp - 1)
              for inshp, outshp in zip(in_spatial, out_spatial)]
    grid = [torch.arange(0., outshp, **backend).mul_(inshp/outshp).add_(shift)
            for inshp, outshp, shift in zip(in_spatial, out_spatial, shifts)]
    grid = torch.stack(torch.meshgrid(*grid), dim=-1)
    x = utils.fast_movedim(x, -1, -dim-1)
    x = grid_push(x, grid, in_spatial, bound=bound, interpolation=order,
                  extrapolate=True)
    x = utils.fast_movedim(x, -dim-1, -1)

    # normalize by change or resolution
    x = x.mul_(py.prod(in_spatial)/py.prod(out_spatial))
    return x


class _FMG:
    """Base class for multi-grid solvers"""

    def __init__(self, bound='dct2', nb_cycles=2, nb_iter=2, max_levels=16,
                 optim='cg', tolerance=0, stop='e', verbose=False,
                 matvec=None, matsolve=None, matdiag=None):
        self.bound = bound
        self.nb_cycles = nb_cycles
        self.nb_iter = nb_iter
        self.max_levels = max_levels
        self.verbose = verbose
        self.optim = optim
        self.tolerance = tolerance
        self.stop = stop

        # These function understand how sparse Hessians are stored.
        # By default, they suit sparse Hessian where the diagonal is
        # stored first, followed by the upper triangular part (row by row).
        # They can be overridden to handle other types of sparse Hessians
        # (e.g., Hessians with a single off-diagonal element per row, as
        #  is the case in multi-exponential fitting).
        self.matvec = matvec or sym_matvec
        self.matsolve = matsolve or sym_solve
        self.matdiag = matdiag or (lambda x, dim: x[..., :dim])

        self.hessian = self.gradient = self.weights = self.init = None
        self.pyrh = self.pyrg = self.pyrx = self.pyrw = []
        self.pyrn = self.pyrv = self.spatial = self.solvers = self.forward = []

    @property
    def dim(self):
        return len(self.spatial)

    @property
    def nb_levels(self):
        return len(self.pyrh)

    @property
    def pullopt(self):
        return dict(bound=self.bound, dim=self.dim)

    def prolong(self, x, shape):
        return prolong(x, shape, bound=self.bound, dim=self.dim)

    def restrict(self, x, shape):
        return restrict(x, shape, bound=self.bound, dim=self.dim)

    prolong_h = prolong
    prolong_g = prolong
    prolong_w = prolong
    restrict_h = restrict
    restrict_g = restrict
    restrict_w = restrict

    def allocate(self):
        # allocate tensors at each pyramid level
        if self.dim == 0:
            raise RuntimeError("Cannot allocate before `set_data`")

        self.trace(f'(fmg) allocate [0]')
        init_zero = self.init is None
        if init_zero:
            self.init = torch.zeros_like(self.gradient)

        pyrh = [self.hessian]          # hessians / matrix
        pyrg = [self.gradient]         # gradients / target
        pyrx = [self.init]             # solutions
        pyrw = [self.weights]          # voxel-wise weights
        pyrn = [self.spatial]          # shapes
        pyrv = [[1.]*self.dim]         # voxel size scaling
        for i in range(1, self.max_levels+1):
            self.trace(f'(fmg) - prolong [{i - 1} -> {i}]')

            spatial1 = [math.ceil(s/2) for s in pyrn[-1]]
            if all(s == 1 for s in spatial1):
                break
            pyrn.append(spatial1)
            pyrv.append([n0/n for n, n0 in zip(spatial1, pyrn[0])])
            pyrh.append(self.restrict_h(pyrh[-1], spatial1))
            pyrg.append(self.restrict_g(pyrg[-1], spatial1))
            if init_zero:
                pyrx.append(torch.zeros_like(pyrg[-1]))
            else:
                pyrx.append(self.restrict_g(pyrx[-1], spatial1))
            if isinstance(self.weights, dict):
                pyrw.append({key: self.restrict_w(val, spatial1)
                             if val is not None else None
                             for key, val in pyrw[-1].items()})
            elif self.weights is not None:
                pyrw.append(self.restrict_w(pyrw[-1], spatial1))
            else:
                pyrw.append(None)

        self.pyrh = pyrh
        self.pyrg = pyrg
        self.pyrx = pyrx
        self.pyrw = pyrw
        self.pyrn = pyrn
        self.pyrv = pyrv

    def trace(self, *a, **k):
        if self.verbose:
            print(*a, **k)

    def solve(self):

        g = self.pyrg       # gradient (g + Lv)
        x = self.pyrx       # solution (input/output)
        d = self.pyrn       # shape
        h = self.forward    # full hessian (H + L)
        ih = self.solvers   # inverse of the hessian:  ih(g, x) <=> x = h\g

        # Initial solution at coarsest grid
        self.trace(f'(fmg) solve [{self.nb_levels - 1}]')
        x[-1] = ih[-1](g[-1], x[-1])

        # The full multigrid (FMG) solver performs lots of  "two-grid"
        # multigrid (2MG) steps, at different scales. There are therefore
        # `len(pyrh) - 1` scales at which to work.

        for n_base in reversed(range(self.nb_levels-1)):
            self.trace(f'(fmg) prolong to level [{n_base + 1} -> {n_base}]')
            x[n_base] = self.prolong_g(x[n_base+1], d[n_base])

            for n_cycle in range(self.nb_cycles):
                self.trace(f'(fmg) - V cycle {n_cycle}')
                for n in range(n_base, self.nb_levels-1):
                    self.trace(f'(fmg) -- solve residuals [{n}]')
                    x[n] = ih[n](g[n], x[n])
                    res = h[n](x[n]).neg_().add_(g[n])
                    self.trace(f'(fmg) -- restrict residuals [{n} -> {n + 1}]')
                    g[n+1] = self.restrict_g(res, d[n+1])
                    del res
                    x[n+1].zero_()

                self.trace(f'(fmg) -- solve [{self.nb_levels - 1}]')
                x[-1] = ih[-1](g[-1], x[-1])

                for n in reversed(range(n_base, self.nb_levels-1)):
                    self.trace(f'(fmg) -- prolong residuals [{n + 1} -> {n}]')
                    x[n] += self.prolong_g(x[n+1], d[n])
                    self.trace(f'(fmg) -- solve full [{n}]')
                    x[n] = ih[n](g[n], x[n])

        x = x[0]
        self.clear_data()
        return x

    def clear_data(self):
        self.pyrh = self.pyrg = self.pyrx = self.pyrw = self.pyrn = self.pyrv = []
        self.hessian = self.gradient = self.weights = self.init = None
        if hasattr(self, '_optim'):
            self.optim = self._optim
            del self._optim
        return self


class _FieldFMG(_FMG):
    """Specialized class for vector fields"""

    def __init__(self, absolute=0, membrane=0, bending=0, factor=1,
                 voxel_size=1, **kwargs):
        super().__init__(**kwargs)

        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.factor = factor
        self.voxel_size = voxel_size

    def set_data(self, hessian, gradient, weights=None, init=None, dim=None):
        """

        Parameters
        ----------
        hessian : (..., K2, *spatial) tensor, with K2 in {1, K, K*(K+1)//2}
            Hessian // Matrix field to invert
        gradient : (..., K, *spatial) tensor
            Gradient // Target vector field
        weights : [dict of] (..., K|1, *spatial) tensor, optional
            Voxel-wise weights (for RLS).
            If a dictionary, energy-specific weights can be specified
            using fields `absolute`, `membrane`, `bending`.
        init : (..., K, *spatial) tensor, optional
            Initial solution of `H\g
        dim : int, default=`gradient.dim()-1`

        Returns
        -------
        self

        """
        dim = dim or (gradient.dim() - 1)

        if weights is not None and self.optim == 'relaks':
            self._optim = self.optim
            self.optim = 'relax'

        def prepare(weights):
            if weights is None:
                return None
            if weights.dim() < dim+1:
                weights = utils.unsqueeze(weights, 0, dim+1-weights.dim())
            return utils.fast_movedim(weights, -dim - 1, -1)

        self.spatial = gradient.shape[-dim:]
        self.hessian = prepare(hessian)
        self.gradient = prepare(gradient)
        if isinstance(weights, dict):
            self.weights = {key: prepare(weights.pop(key, None))
                            for key in ('absolute', 'membrane', 'bending')}
        else:
            self.weights = prepare(weights)
        self.init = prepare(init)

        self.allocate()
        self.make_solvers()
        return self

    def make_solvers(self):
        if self.optim == 'relaks':
            return self.make_solvers_kernel()

        # create solvers at each pyramid level

        nb_prm = self.gradient.shape[-1]
        vx0 = utils.make_vector(self.voxel_size, self.dim).tolist()
        factor = utils.make_vector(self.factor, **utils.backend(self.gradient))
        batch = self.gradient.shape[:-self.dim-1]
        is_diag = self.hessian.shape[-1] in (1, nb_prm)
        optim = getattr(optimizers, self.optim)

        def last2channel(x):
            if x is None:
                return None
            if not torch.is_tensor(x) or x.dim() == 0:
                return x
            return utils.fast_movedim(x, -1, -self.dim-1)

        def channel2last(x):
            if x is None:
                return None
            if not torch.is_tensor(x) or x.dim() <= 1:
                return x
            return utils.fast_movedim(x, -self.dim-1, -1)

        def make_solver(h, w, v):

            if isinstance(w, dict):
                wa = w['absolute']
                wm = w['membrane']
                wb = w['bending']
            else:
                wa = wb = wm = w
            has_weights = (wa is not None or wm is not None or wb is not None)

            vx = [v0*vs for v0, vs in zip(vx0, v)]
            opt = dict(bound=self.bound, voxel_size=vx, dim=self.dim)
            spatial = h.shape[-self.dim-1:-1]

            def regulariser(x):
                x = utils.fast_movedim(x, -1, -self.dim-1)
                y = torch.zeros_like(x)
                if self.absolute:
                    y.add_(absolute(x, weights=last2channel(wa)),
                           alpha=self.absolute)
                if self.membrane:
                    y.add_(membrane(x, weights=last2channel(wm), **opt),
                           alpha=self.membrane)
                if self.bending:
                    y.add_(bending(x, weights=last2channel(wb), **opt),
                           alpha=self.bending)
                y = utils.fast_movedim(y, -self.dim-1, -1)
                y = y.mul_(factor)
                return y

            # diagonal of the regulariser
            if has_weights:
                smo = h.new_zeros([*batch, *spatial, nb_prm])
            else:
                smo = h.new_zeros(nb_prm)
            if self.absolute:
                smo.add_(channel2last(absolute_diag(weights=last2channel(wa))),
                         alpha=self.absolute)
            if self.membrane:
                smo.add_(channel2last(membrane_diag(weights=last2channel(wm), **opt)),
                         alpha=self.membrane)
            if self.bending:
                smo.add_(channel2last(bending_diag(weights=last2channel(wb), **opt)),
                         alpha=self.bending)
            smo *= factor

            if is_diag:
                h_smo = h + smo
            else:
                h_smo = h.clone()
                self.matdiag(h_smo, nb_prm).add_(smo)

            def s2h(s):
                # do not slice if hessian_smo is constant across space
                if s is Ellipsis:
                    return s
                if all(sz == 1 for sz in h_smo.shape[-self.dim-1:-1]):
                    s = list(s)
                    s[-self.dim-1:-1] = [slice(None)] * self.dim
                    s = tuple(s)
                return s

            if is_diag:
                def forward(x):
                    return regulariser(x).addcmul_(x, h)
                def precond(x, s=Ellipsis):
                    return x[s] / h_smo[s2h(s)]
            else:
                def forward(x):
                    return regulariser(x).add_(self.matvec(h, x))
                def precond(x, s=Ellipsis):
                    return self.matsolve(h_smo[s2h(s)], x[s])

            prm = dict(max_iter=self.nb_iter, verbose=self.verbose > 1,
                       tolerance=self.tolerance, stop=self.stop,
                       inplace=True)
            if self.optim == 'relax':
                prm['scheme'] = (3 if bending else 'checkerboard')
            def solve(b, x0=None, it=None):
                pprm = prm
                if it is not None:
                    pprm = dict(pprm)
                    pprm['max_iter'] = it
                return optim(forward, b, x=x0, precond=precond, **pprm)

            return solve, forward

        solvers = []
        forward = []
        for h, w, v in zip(self.pyrh, self.pyrw, self.pyrv):
            solve, fwd = make_solver(h, w, v)
            solvers.append(solve)
            forward.append(fwd)
        self.solvers = solvers
        self.forward = forward

    def make_solvers_kernel(self):

        nb_prm = self.gradient.shape[-1]
        is_diag = self.hessian.shape[-1] in (1, nb_prm)
        vx0 = utils.make_vector(self.voxel_size, self.dim).tolist()
        optim = (optimizers.relax if self.optim == 'relaks' else
                 getattr(optimizers, self.optim))

        def make_solver(h, v):

            vx = [v0*vs for v0, vs in zip(vx0, v)]

            ker = regulariser_kernel(self.dim,
                                     absolute=self.absolute,
                                     membrane=self.membrane,
                                     bending=self.bending,
                                     factor=self.factor,
                                     voxel_size=vx, **utils.backend(h))

            def regulariser(x, s=Ellipsis):
                x = utils.fast_movedim(x, -1, -self.dim-1)
                if s and s is not Ellipsis:
                    if s[0] is Ellipsis:
                        s = s[1:]
                    start = [sl.start for sl in s[:self.dim]]
                    stop = [sl.stop for sl in s[:self.dim]]
                    step = [sl.step for sl in s[:self.dim]]
                else:
                    start = step = stop = None
                x = spconv(x, ker, start=start, stop=stop, step=step,
                           dim=self.dim, bound=self.bound)
                x = utils.fast_movedim(x, -self.dim-1, -1)
                return x

            # diagonal of the regulariser
            center = [s // 2 for s in ker.shape[-self.dim:]]
            smo = ker[(Ellipsis, *center)]

            if is_diag:
                hessian_smo = h + smo
            else:
                hessian_smo = h.clone()
                hessian_smo[..., :nb_prm].add_(smo)

            def s2h(s):
                # do not slice if hessian_smo is constant across space
                if s is Ellipsis:
                    return s
                if all(sz == 1 for sz in hessian_smo.shape[-self.dim-1:-1]):
                    return Ellipsis
                return s

            if is_diag:
                def forward(x, s=Ellipsis):
                    x = x[s] * h[s2h(s)] + regulariser(x, s)
                    return x
                def precond(x, s=Ellipsis):
                    x = x[s] / hessian_smo[s2h(s)]
                    return x
            else:
                def forward(x, s=Ellipsis):
                    x = sym_matvec(h[s2h(s)], x[s]) + regulariser(x, s)
                    return x
                def precond(x, s=Ellipsis):
                    x = sym_solve(hessian_smo[s2h(s)], x[s])
                    return x

            prm = dict(max_iter=self.nb_iter, verbose=self.verbose > 1,
                       tolerance=self.tolerance, stop=self.stop,
                       scheme=(3 if self.bending else
                               'checkerboard'),
                       inplace=True)

            def solve(b, x0=None, it=None):
                pprm = prm
                if it is not None:
                    pprm = dict(pprm)
                    pprm['max_iter'] = it
                return optim(forward, b, x=x0, precond=precond, **pprm)

            return solve, forward

        solvers = []
        forward = []
        for h, v in zip(self.pyrh, self.pyrv):
            solve, fwd = make_solver(h, v)
            solvers.append(solve)
            forward.append(fwd)
        self.solvers = solvers
        self.forward = forward

    def solve(self):
        dim = self.dim
        out = super().solve()
        out = utils.fast_movedim(out, -1, -dim-1)
        return out


class _GridFMG(_FMG):
    """Specialized class for displacement fields"""

    def __init__(self, absolute=0, membrane=0, bending=0, lame=0, factor=1,
                 voxel_size=1, bound='dft', **kwargs):
        super().__init__(**kwargs, bound=bound)

        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.lame = py.ensure_list(lame, 2)
        self.factor = factor
        self.voxel_size = voxel_size

    def prolong_g(self, x, shape):
        shape0 = x.shape[-self.dim-1:-1]
        scale = [s/s0 for s0, s in zip(shape0, shape)]
        scale = torch.as_tensor(scale, **utils.backend(x))
        return prolong(x, shape, bound=self.bound, dim=self.dim).mul_(scale)

    def prolong_h(self, x, shape):
        shape0 = x.shape[-self.dim-1:-1]
        scale = [s/s0 for s0, s in zip(shape0, shape)]
        scale = torch.as_tensor(scale, **utils.backend(x))
        x = prolong(x, shape, bound=self.bound, dim=self.dim)
        x[..., :self.dim].mul_(scale.square())
        c = 0
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                x[..., self.dim+c].mul_(scale[i]*scale[j])
                c = c + 1
        return x

    def restrict_g(self, x, shape):
        shape0 = x.shape[-self.dim-1:-1]
        scale = [s0/s for s0, s in zip(shape0, shape)]
        scale = torch.as_tensor(scale, **utils.backend(x))
        return restrict(x, shape, bound=self.bound, dim=self.dim).mul_(scale)

    def restrict_h(self, x, shape):
        shape0 = x.shape[-self.dim-1:-1]
        scale = [s0/s for s0, s in zip(shape0, shape)]
        scale = torch.as_tensor(scale, **utils.backend(x))
        x = restrict(x, shape, bound=self.bound, dim=self.dim)
        x[..., :self.dim].mul_(scale.square())
        c = 0
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                x[..., self.dim+c].mul_(scale[i]*scale[j])
                c = c + 1
        return x

    def set_data(self, hessian, gradient, weights=None, init=None):
        """

        Parameters
        ----------
        hessian : (..., *spatial, D*(D+1)//2) tensor
            Hessian // Matrix field to invert
        gradient : (..., *spatial, D) tensor
            Gradient // Target vector field
        weights : [dict of] (..., *spatial) tensor, optional
            Voxel-wise weights (for RLS).
        init : (..., *spatial, D) tensor, optional
            Initial solution of `H\g

        Returns
        -------
        self

        """
        dim = gradient.shape[-1]

        if weights is not None and self.optim == 'relaks':
            self._optim = self.optim
            self.optim = 'relax'

        self.spatial = gradient.shape[-dim-1:-1]
        self.hessian = hessian
        self.gradient = gradient
        if isinstance(weights, dict):
            weights = {key: weights.pop(key, None)
                       for key in ('absolute', 'membrane', 'bending')}
            self.weights = {k: v.unsqueeze(-1) if v is not None else v
                            for k, v in weights.items()}
        else:
            self.weights = weights.unsqueeze(-1) if weights is not None else None
        self.init = init

        self.allocate()
        self.make_solvers()
        return self

    def make_solvers(self):
        if self.optim == 'relaks':
            return self.make_solvers_kernel()

        # create solvers at each pyramid level

        vx0 = utils.make_vector(self.voxel_size, self.dim).tolist()
        factor = utils.make_vector(self.factor, **utils.backend(self.gradient))
        batch = self.gradient.shape[:-self.dim-1]
        optim = getattr(optimizers, self.optim)

        def make_solver(h, w, v):

            if isinstance(w, dict):
                wa = w['absolute']
                wm = w['membrane']
                wb = w['bending']
                wl = w['lame']
            else:
                wa = wb = wm = wl = w
            wl = py.ensure_list(wl, 2)
            has_weights = (wa is not None or wm is not None or wb is not None
                           or any(w is not None for w in wl))

            vx = [v0*vs for v0, vs in zip(vx0, v)]
            opt = dict(bound=self.bound, voxel_size=vx)
            spatial = h.shape[-self.dim-1:-1]

            def regulariser(x):
                y = torch.zeros_like(x)
                if self.absolute:
                    y.add_(absolute_grid(x, weights=wa, voxel_size=vx),
                           alpha=self.absolute)
                if self.membrane:
                    y.add_(membrane_grid(x, weights=wm, **opt),
                           alpha=self.membrane)
                if self.bending:
                    y.add_(bending_grid(x, weights=wb, **opt),
                           alpha=self.bending)
                if self.lame[0]:
                    y.add_(lame_div(x, weights=wl[0], **opt),
                           alpha=self.lame[0])
                if self.lame[1]:
                    y.add_(lame_shear(x, weights=wl[1], **opt),
                           alpha=self.lame[1])
                y = y.mul_(factor)
                return y

            # diagonal of the regulariser
            if has_weights:
                smo = h.new_zeros([*batch, *spatial, self.dim])
            else:
                smo = h.new_zeros(self.dim)
            if self.absolute:
                smo.add_(absolute_diag(weights=wa), alpha=self.absolute)
            if self.membrane:
                smo.add_(membrane_diag(weights=wm, **opt), alpha=self.membrane)
            if self.bending:
                smo.add_(bending_diag(weights=wb, **opt), alpha=self.bending)
            vx2 = utils.make_vector(vx, **utils.backend(smo)).square()
            smo *= vx2
            ivx2 = vx2.reciprocal_()
            if self.lame[0]:
                if wl[0] is not None:
                    smo.add_(wl[0], alpha=2 * self.lame[0])
                else:
                    smo += 2 * self.lame[0]
            if self.lame[1]:
                if wl[1] is not None:
                    smo.add_(wl[1], alpha=2 * self.lame[1] * (1 + ivx2.sum() / ivx2))
                else:
                    smo += 2 * self.lame[1] * (1 + ivx2.sum() / ivx2)
            smo *= factor

            h_smo = h.clone()
            self.matdiag(h_smo, self.dim).add_(smo)

            def s2h(s):
                # do not slice if hessian_smo is constant across space
                if s is Ellipsis:
                    return s
                if all(sz == 1 for sz in h_smo.shape[-self.dim-1:-1]):
                    s = list(s)
                    s[-self.dim-1:-1] = [slice(None)] * self.dim
                    s = tuple(s)
                return s

            def forward(x):
                return regulariser(x).add_(self.matvec(h, x))

            def precond(x, s=Ellipsis):
                return self.matsolve(h_smo[s2h(s)], x[s])

            prm = dict(max_iter=self.nb_iter, verbose=self.verbose > 1,
                       tolerance=self.tolerance, stop=self.stop, inplace=True)
            if self.optim == 'relax':
                prm['scheme'] = (3 if self.bending else 'checkerboard')

            def solve(b, x0=None):
                return optim(forward, b, x=x0, precond=precond, **prm)

            return solve, forward

        solvers = []
        forward = []
        for h, w, v in zip(self.pyrh, self.pyrw, self.pyrv):
            solve, fwd = make_solver(h, w, v)
            solvers.append(solve)
            forward.append(fwd)
        self.solvers = solvers
        self.forward = forward

    def make_solvers_kernel(self):

        vx0 = utils.make_vector(self.voxel_size, self.dim).tolist()
        factor = utils.make_vector(self.factor, **utils.backend(self.gradient))
        optim = (optimizers.relax if self.optim == 'relaks' else
                 getattr(optimizers, self.optim))

        def make_solver(h, v):

            vx = [v0*vs for v0, vs in zip(vx0, v)]
            opt = dict(bound=self.bound, voxel_size=vx)

            ker = regulariser_grid_kernel(self.dim,
                                          absolute=self.absolute,
                                          membrane=self.membrane,
                                          bending=self.bending,
                                          lame=self.lame,
                                          factor=self.factor,
                                          voxel_size=vx, **utils.backend(h))

            def regulariser(v, s=Ellipsis):
                if s and s is not Ellipsis:
                    if s[0] is Ellipsis:
                        s = s[1:]
                    start = [sl.start for sl in s[:self.dim]]
                    stop = [sl.stop for sl in s[:self.dim]]
                    step = [sl.step for sl in s[:self.dim]]
                else:
                    start = step = stop = None
                v = utils.fast_movedim(v, -1, -self.dim - 1)
                v = spconv(v, ker, start=start, stop=stop, step=step,
                           dim=self.dim, bound=self.bound)
                v = utils.fast_movedim(v, -self.dim - 1, -1)
                return v

            # diagonal of the regulariser
            smo = h.new_zeros(self.dim)
            if self.absolute:
                smo.add_(absolute_diag(), alpha=self.absolute)
            if self.membrane:
                smo.add_(membrane_diag(**opt), alpha=self.membrane)
            if self.bending:
                smo.add_(bending_diag(**opt), alpha=self.bending)
            vx2 = utils.make_vector(vx, **utils.backend(smo)).square()
            smo *= vx2
            ivx2 = vx2.reciprocal_()
            if self.lame[0]:
                smo += 2 * self.lame[0]
            if self.lame[1]:
                smo += 2 * self.lame[1] * (1 + ivx2.sum() / ivx2)
            smo *= factor

            h_smo = h.clone()
            self.matdiag(h_smo, self.dim).add_(smo)

            def forward(x):
                return regulariser(x).add_(self.matvec(h, x))

            def precond(x, s=Ellipsis):
                return self.matsolve(h_smo[s], x[s])

            prm = dict(max_iter=self.nb_iter, verbose=self.verbose > 1,
                       tolerance=self.tolerance, stop=self.stop,
                       scheme=(3 if self.bending else
                               2 if any(self.lame) else
                               'checkerboard'),
                       inplace=True)

            def solve(b, x0=None):
                return optim(forward, b, x=x0, precond=precond, **prm)

            return solve, forward

        solvers = []
        forward = []
        for h, v in zip(self.pyrh, self.pyrv):
            solve, fwd = make_solver(h, v)
            solvers.append(solve)
            forward.append(fwd)
        self.solvers = solvers
        self.forward = forward


def solve_grid_fmg(hessian, gradient, absolute=0, membrane=0, bending=0,
                   lame=0, factor=1, voxel_size=1, bound='dft', weights=None,
                   optim='cg', nb_cycles=2, nb_iter=2, tolerance=0,
                   verbose=False):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Notes
    -----
    .. This function is specialized for displacement fields
    .. This function uses preconditioned gradient descent
       (Conjugate Gradient or Gauss-Seidel) in a full multi-grid fashion.

    Inputs
    ------
    hessian : (..., *spatial, D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Regularization
    --------------
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Optimization
    ------------
    optim : {'cg', 'relax'}, default='cg'
    nb_cycles : int, default=2
    nb_iter : int, default=2
    tolerance : float, default=0
    verbose : int, default=0

    Returns
    -------
    solution : (..., *spatial, D) tensor

    """
    if not (membrane or bending or any(py.ensure_list(lame))):
        return solve_grid_closedform(hessian, gradient, weights=weights,
                                     absolute=absolute, factor=factor,
                                     voxel_size=voxel_size)

    FMG = _GridFMG(absolute=absolute, membrane=membrane, bending=bending,
                   lame=lame, factor=factor, voxel_size=voxel_size,
                   bound=bound, optim=optim, nb_cycles=nb_cycles,
                   nb_iter=nb_iter, verbose=verbose, tolerance=tolerance)
    FMG.set_data(hessian, gradient, weights=weights)
    return FMG.solve()


def solve_field_fmg(hessian, gradient, weights=None, voxel_size=1, bound='dct2',
                    absolute=0, membrane=0, bending=0, factor=1, dim=None,
                    optim='cg', nb_cycles=2, nb_iter=2, tolerance=0,
                    verbose=False, matvec=None, matsolve=None, matdiag=None):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Notes
    -----
    .. This function is specialized for multi-channel vector fields
    .. This function uses preconditioned gradient descent
       (Conjugate Gradient or Gauss-Seidel) in a full multi-grid fashion.

    Inputs
    ------
    hessian : (..., K2, *spatial) tensor, with K2 in {1, K, K*(K+1)//2}
    gradient : (..., K, *spatial) tensor
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.
    dim : int, default=gradient.dim()-1

    Regularization
    --------------
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    factor : [sequence of] float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'

    Optimization
    ------------
    optim : {'cg', 'relax'}, default='cg'
    nb_cycles : int, default=2
    nb_iter : int, default=2
    tolerance : float, default=1e-5
    verbose : int, default=0

    Returns
    -------
    solution : (..., K, *spatial) tensor

    """
    if not (membrane or bending):
        return solve_field_closedform(hessian, gradient, weights=weights,
                                      absolute=absolute, factor=factor,
                                      matsolve=matsolve, matdiag=matdiag)

    FMG = _FieldFMG(absolute=absolute, membrane=membrane, bending=bending,
                    factor=factor, voxel_size=voxel_size,
                    bound=bound, optim=optim, nb_cycles=nb_cycles,
                    nb_iter=nb_iter, verbose=verbose, tolerance=tolerance,
                    matvec=matvec, matsolve=matsolve, matdiag=matdiag)
    FMG.set_data(hessian, gradient, weights=weights, dim=dim)
    return FMG.solve()


# ======================================================================
#                       Single resolution solvers
# ======================================================================


_absolute = absolute
_membrane = membrane
_bending = bending


def solve_field(hessian, gradient, weights=None, dim=None,
                absolute=0, membrane=0, bending=0, factor=1,
                voxel_size=1, bound='dct2',
                optim='cg', max_iter=16, tolerance=1e-5, stop='e',
                verbose=False, matvec=None, matsolve=None, matdiag=None):
    """Solve a positive-definite linear system of the form (H + L)x = g

    Notes
    -----
    .. This function is specialized for multi-channel vector fields
    .. This function uses preconditioned gradient descent
       (Conjugate Gradient or Gauss-Seidel) at a single resolution.

    Inputs
    ------
    hessian : (..., K2, *spatial) tensor, with K2 in {1, K, K*(K+1)//2}
    gradient : (..., K, *spatial) tensor
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.
    dim : int, default=gradient.dim()-1

    Regularization
    --------------
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    factor : float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'

    Optimization
    ------------
    optim : {'cg', 'relax'}, default='cg'
    max_iter : int, default=16
    tolerance : float, default=1e-5
    stop : {'e', 'a'}, default='e'
    precond : callable, optional
    verbose : int, default=0

    Returns
    -------
    solution : (..., K, *spatial) tensor

    """
    if not (membrane or bending):
        return solve_field_closedform(hessian, gradient, weights=weights,
                                      absolute=absolute, factor=factor,
                                      matsolve=matsolve, matdiag=matdiag)

    matvec = matvec or sym_matvec
    matsolve = matsolve or sym_solve
    matdiag = matdiag or (lambda x, dim: x[..., :dim])

    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = dim or gradient.dim() - 1
    nb_prm = gradient.shape[-dim-1]

    voxel_size = utils.make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-dim-1] in (1, gradient.shape[-dim-1])

    factor = utils.make_vector(factor, nb_prm, **backend)
    pad_spatial = (Ellipsis,) + (None,) * dim
    factor = factor[pad_spatial]
    no_reg = not (membrane or bending)

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
    else:
        wa = wm = wb = weights
    has_weights = (wa is not None or wm is not None or wb is not None)

    def regulariser(x):
        y = torch.zeros_like(x)
        if absolute:
            y.add_(_absolute(x, weights=wa), alpha=absolute)
        if membrane:
            y.add_(_membrane(x, weights=wm, **fdopt), alpha=membrane)
        if bending:
            y.add_(_bending(x, weights=wb, **fdopt), alpha=bending)
        return y.mul_(factor)

    # diagonal of the regulariser
    if has_weights:
        smo = torch.zeros_like(gradient)
    else:
        smo = gradient.new_zeros([nb_prm] + [1]*dim)
    if absolute:
        smo.add_(absolute_diag(weights=wa), alpha=absolute)
    if membrane:
        smo.add_(membrane_diag(weights=wm, **fdopt), alpha=membrane)
    if bending:
        smo.add_(bending_diag(weights=wb, **fdopt), alpha=bending)
    smo.mul_(factor)

    if is_diag:
        hessian_smo = hessian + smo
    else:
        hessian_smo = hessian.clone()
        hessian_diag = matdiag(hessian_smo.transpose(-dim-1, -1), nb_prm).transpose(-dim-1, -1)
        hessian_diag.add_(smo)

    def s2h(s):
        # do not slice if hessian_smo is constant across space
        if s is Ellipsis:
            return s
        if all(sz == 1 for sz in hessian_smo.shape[-dim:]):
            s = list(s)
            s[-dim:] = [slice(None)] * dim
            s = tuple(s)
        return s

    def _matvec(h, x):
        h = h.transpose(-dim-1, -1)
        x = x.transpose(-dim-1, -1)
        x = matvec(h, x)
        x = x.transpose(-dim-1, -1)
        return x

    def _solve(h, x):
        h = h.transpose(-dim-1, -1)
        x = x.transpose(-dim-1, -1)
        x = matsolve(h, x)
        x = x.transpose(-dim-1, -1)
        return x

    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: _matvec(hessian, x) + regulariser(x)))
    precond = ((lambda x, s=Ellipsis: x[s] / hessian_smo[s2h(s)]) if is_diag else
               (lambda x, s=Ellipsis: _solve(hessian_smo[s2h(s)], x[s])))

    if no_reg:
        result = precond(gradient)
    else:
        prm = dict(max_iter=max_iter, verbose=verbose, stop=stop,
                   tolerance=tolerance)
        if optim == 'relax':
            prm['scheme'] = (3 if bending else 'checkerboard')
        optim = getattr(optimizers, optim)
        result = optim(forward, gradient, precond=precond, **prm)
    return result


def solve_grid(hessian, gradient, absolute=0, membrane=0, bending=0,
               lame=0, factor=1, voxel_size=1, bound='dft', weights=None,
               optim='cg', max_iter=16, tolerance=1e-5, stop='e',
               verbose=False):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Notes
    -----
    .. This function is specialized for displacement fields.
    .. This function uses preconditioned gradient descent
       (Conjugate Gradient or Gauss-Seidel) at a single resolution.

    Inputs
    ------
    hessian : (..., *spatial, D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Regularization
    --------------
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Optimization
    ------------
    optim : {'cg', 'relax'}, default='cg'
    max_iter : int, default=16
    tolerance : float, default=1e-5
    verbose : int, default=0

    Returns
    -------
    solution : (..., *spatial, D) tensor

    """
    if not (membrane or bending or any(py.ensure_list(lame))):
        return solve_grid_closedform(hessian, gradient, weights=weights,
                                     absolute=absolute, factor=factor,
                                     voxel_size=voxel_size)

    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = gradient.shape[-1]
    voxel_size = utils.make_vector(voxel_size, dim, **backend)

    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = [l*factor for l in utils.make_vector(lame, 2).tolist()]
    no_reg = not (membrane or bending or any(lame))

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
        wl = weights.get('lame', None)
    else:
        wa = wm = wb = wl = weights
    wl = py.ensure_list(wl, 2)
    has_weights = (wa is not None or wm is not None or wb is not None or
                   wl[0] is not None or wl[1] is not None)

    def regulariser(v):
        y = torch.zeros_like(v)
        if absolute:
            y.add_(absolute_grid(v, weights=wa, voxel_size=voxel_size), alpha=absolute)
        if membrane:
            y.add_(membrane_grid(v, weights=wm, **fdopt), alpha=membrane)
        if bending:
            y.add_(bending_grid(v, weights=wb, **fdopt), alpha=bending)
        if lame[0]:
            y.add_(lame_div(v, weights=wl[0], **fdopt), alpha=lame[0])
        if lame[1]:
            y.add_(lame_shear(v, weights=wl[1], **fdopt), alpha=lame[1])
        return y

    # diagonal of the regulariser
    vx2 = voxel_size.square()
    ivx2 = vx2.reciprocal()
    smo = torch.zeros_like(gradient) if has_weights else 0
    if absolute:
        if wa is not None:
            smo.add_(wa, alpha=absolute * vx2)
        else:
            smo += absolute * vx2
    if membrane:
        if wm is not None:
            smo.add_(wm, alpha=2 * membrane * ivx2.sum() * vx2)
        else:
            smo += 2 * membrane * ivx2.sum() * vx2
    if bending:
        val = torch.combinations(ivx2, r=2).prod(dim=-1).sum()
        if wb is not None:
            smo.add_(wb, alpha=(8 * val + 6 * ivx2.square().sum()) * vx2)
        else:
            smo += bending * (8 * val + 6 * ivx2.square().sum()) * vx2
    if lame[0]:
        if wl[0] is not None:
            smo.add_(wl[0], alpha=2 * lame[0])
        else:
            smo += 2 * lame[0]
    if lame[1]:
        if wl[1] is not None:
            smo.add_(wl[1], alpha=2 * lame[1] * (1 + ivx2.sum() / ivx2))
        else:
            smo += 2 * lame[1] * (1 + ivx2.sum() / ivx2)

    if smo.shape[-1] > hessian.shape[-1]:
        hessian_smo = hessian + smo
    else:
        hessian_smo = hessian.clone()
        hessian_smo[..., :dim] += smo
    is_diag = hessian_smo.shape[-1] in (1, gradient.shape[-1])

    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: sym_matvec(hessian, x) + regulariser(x)))
    precond = ((lambda x, s=Ellipsis: x[s] / hessian_smo[s]) if is_diag else
               (lambda x, s=Ellipsis: sym_solve(hessian_smo[s], x[s])))

    if no_reg:
        # no spatial regularisation: we can use a closed-form
        result = precond(gradient)
    else:
        prm = dict(max_iter=max_iter, verbose=verbose, stop=stop,
                   tolerance=tolerance)
        if optim == 'relax':
            prm['scheme'] = (3 if bending else
                             2 if any(lame) else
                             'checkerboard')
        optim = getattr(optimizers, optim)
        result = optim(forward, gradient, precond=precond, **prm)
    return result


def solve_grid_kernel(hessian, gradient, absolute=0, membrane=0, bending=0,
                      lame=0, factor=1, voxel_size=1, bound='dft',
                      optim='relax', max_iter=16, tolerance=1e-5, stop='e',
                      verbose=False):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Notes
    -----
    .. This function is specialized for displacement fields.
    .. This function uses preconditioned gradient descent
       (Conjugate Gradient or Gauss-Seidel) at a single resolution.
    .. This function implements the regularizer using a sparse convolution.

    Inputs
    ------
    hessian : (..., *spatial, D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Regularization
    --------------
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Optimization
    ------------
    optim : {'cg', 'relax'}, default='cg'
    max_iter : int, default=16
    tolerance : float, default=1e-5
    verbose : int, default=0

    Returns
    -------
    solution : (..., *spatial, D) tensor

    """
    if not (membrane or bending or any(py.ensure_list(lame))):
        return solve_grid_closedform(hessian, gradient,
                                     absolute=absolute, factor=factor,
                                     voxel_size=voxel_size)

    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = gradient.shape[-1]
    voxel_size = utils.make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    lame = py.ensure_list(lame)
    no_reg = not (membrane or bending or any(lame))
    if not no_reg:
        ker = regulariser_grid_kernel(dim, absolute=absolute, membrane=membrane,
                                      bending=bending, lame=lame, factor=factor,
                                      voxel_size=voxel_size, **backend)

    # pre-multiply by factor
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = [l*factor for l in utils.make_vector(lame, 2, **backend)]

    # regulariser
    fdopt = dict(bound=bound)
    def regulariser(v, s=Ellipsis):
        if s and s is not Ellipsis:
            if s[0] is Ellipsis:
                s = s[1:]
            start = [sl.start for sl in s[:dim]]
            stop = [sl.stop for sl in s[:dim]]
            step = [sl.step for sl in s[:dim]]
        else:
            start = step = stop = None
        if no_reg:
            v = absolute_grid(v[s], voxel_size=voxel_size)
        else:
            v = utils.fast_movedim(v, -1, -dim-1)
            v = spconv(v, ker, start=start, stop=stop, step=step, dim=dim, **fdopt)
            v = utils.fast_movedim(v, -dim-1, -1)
        return v

    # diagonal of the regulariser
    vx2 = voxel_size.square()
    ivx2 = vx2.reciprocal()
    smo = 0
    if absolute:
        smo += absolute * vx2
    if membrane:
        smo += 2 * membrane * ivx2.sum() * vx2
    if bending:
        val = torch.combinations(ivx2, r=2).prod(dim=-1).sum()
        smo += bending * (8 * val + 6 * ivx2.square().sum()) * vx2
    if lame[0]:
        smo += 2 * lame[0]
    if lame[1]:
        smo += 2 * lame[1] * (1 + ivx2.sum() / ivx2)
    hessian_smo = hessian.clone()
    hessian_smo[..., :dim] += smo

    forward = ((lambda x, s=Ellipsis: (x[s] * hessian[s]).add_(regulariser(x, s))) if is_diag else
               (lambda x, s=Ellipsis: sym_matvec(hessian[s], x[s]).add_(regulariser(x, s))))
    precond = ((lambda x, s=Ellipsis: x[s] / hessian_smo[s]) if is_diag else
               (lambda x, s=Ellipsis: sym_solve(hessian_smo[s], x[s])))

    if no_reg:
        # no spatial regularisation: we can use a closed-form
        result = precond(gradient)
    else:
        prm = dict(max_iter=max_iter, verbose=verbose, stop=stop,
                   tolerance=tolerance)
        if optim == 'relax':
            prm['scheme'] = (3 if bending else
                             2 if any(lame) else
                             'checkerboard')
            prm['mode'] = 2
        optim = getattr(optimizers, optim)
        prm['verbose'] = False
        result = optim(forward, gradient, precond=precond, **prm)
    return result


def solve_field_kernel(hessian, gradient, dim=None,
                       absolute=0, membrane=0, bending=0, factor=1,
                       voxel_size=1, bound='dct2',
                       optim='relax', max_iter=16, tolerance=1e-5, stop='e',
                       verbose=False):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Notes
    -----
    .. This function is specialized for multichannel displacement fields.
    .. This function uses preconditioned gradient descent
       (Conjugate Gradient or Gauss-Seidel) at a single resolution.
    .. This function implements the regularizer using a sparse convolution.

    Inputs
    ------
    hessian : (..., K2, *spatial) tensor, with K2 in {1, K, K*(K+1)//2}
    gradient : (..., K, *spatial) tensor
    dim : int, default=gradient.dim()-1

    Regularization
    --------------
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    factor : [sequence of] float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'

    Optimization
    ------------
    optim : {'cg', 'relax'}, default='cg'
    max_iter : int, default=16
    tolerance : float, default=1e-5
    stop : {'e', 'a'}, default='e'
    verbose : int, default=0

    Returns
    -------
    solution : (..., K, *spatial) tensor

    """
    if not (membrane or bending):
        return solve_field_closedform(hessian, gradient,
                                      absolute=absolute, factor=factor)

    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = dim or (gradient.dim() - 1)
    voxel_size = utils.make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    nb_prm = gradient.shape[-dim-1]
    no_reg = not (membrane or bending)
    if not no_reg:
        ker = regulariser_kernel(dim, absolute=absolute, membrane=membrane,
                                 bending=bending, factor=factor,
                                 voxel_size=voxel_size, **backend)

    gradient = utils.fast_movedim(gradient, -dim - 1, -1)
    hessian = utils.fast_movedim(hessian, -dim-1, -1)

    # regulariser
    def regulariser(v, s=Ellipsis):
        if s and s is not Ellipsis:
            if s[0] is Ellipsis:
                s = s[1:]
            start = [sl.start for sl in s[:dim]]
            stop = [sl.stop for sl in s[:dim]]
            step = [sl.step for sl in s[:dim]]
        else:
            start = step = stop = None
        v = utils.fast_movedim(v, -1, -dim-1)
        if no_reg:
            v = _absolute(v[s])
        else:
            v = spconv(v, ker, start=start, stop=stop, step=step, dim=dim,
                       bound=bound)
        v = utils.fast_movedim(v, -dim-1, -1)
        return v

    # diagonal of the regulariser
    center = [s//2 for s in ker.shape[-dim:]]
    smo = ker[(Ellipsis, *center)]

    if is_diag:
        hessian_smo = hessian + smo
    else:
        hessian_smo = hessian.clone()
        hessian_smo[..., :nb_prm].add_(smo)

    def s2h(s):
        # do not slice if hessian_smo is constant across space
        if s is Ellipsis:
            return s
        if all(sz == 1 for sz in hessian_smo.shape[-dim-1:-1]):
            return Ellipsis
        return s

    if is_diag:
        def forward(x, s=Ellipsis):
            x = x[s] * hessian[s2h(s)] + regulariser(x, s)
            return x
        def precond(x, s=Ellipsis):
            x = x[s] / hessian_smo[s2h(s)]
            return x
    else:
        def forward(x, s=Ellipsis):
            x = sym_matvec(hessian[s2h(s)], x[s]) + regulariser(x, s)
            return x
        def precond(x, s=Ellipsis):
            x = sym_solve(hessian_smo[s2h(s)], x[s])
            return x
    if precond is False:
        precond = lambda x, s=Ellipsis: x

    if no_reg:
        # no spatial regularisation: we can use a closed-form
        result = precond(gradient)
    else:
        prm = dict(max_iter=max_iter, verbose=verbose, stop=stop,
                   tolerance=tolerance)
        if optim == 'relax':
            prm['scheme'] = (3 if bending else
                             'checkerboard')
            prm['mode'] = 2
        optim = getattr(optimizers, optim)
        prm['verbose'] = False
        result = optim(forward, gradient, precond=precond, **prm)

    result = utils.fast_movedim(result, -1, -dim-1)
    return result


# aliases for backward compatibility
solve_grid_sym = solve_grid
solve_field_sym = solve_field


# ======================================================================
#                       Closed form solvers
# ======================================================================
# These solvers can only be used if no spatial regularization (membrane,
# bending) is used.


def solve_field_closedform(hessian, gradient, weights=None, dim=None,
                           absolute=0, factor=1, matsolve=None, matdiag=None):
    """Solve a positive-definite linear system of the form (H + I)x = g

    Notes
    -----
    .. This function is specialized for multi-channel vector fields.
    .. This function only supports non-spatial regularization.
    .. This function solves the system in closed form.

    Inputs
    ------
    hessian : (..., K2, *spatial) tensor, with K2 in {1, K, K*(K+1)//2}
    gradient : (..., K, *spatial) tensor
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.
    dim : int, default=gradient.dim()-1

    Regularization
    --------------
    absolute : float, default=0
    factor : [sequence of] float, default=1

    Returns
    -------
    solution : (..., K, *spatial) tensor

    """
    matsolve = matsolve or sym_solve
    matdiag = matdiag or (lambda x, dim: x[..., :dim])

    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = dim or gradient.dim() - 1
    nb_prm = gradient.shape[-dim-1]
    is_diag = hessian.shape[-dim-1] in (1, gradient.shape[-dim-1])

    factor = utils.make_vector(factor, nb_prm, **backend)
    pad_spatial = (Ellipsis,) + (None,) * dim
    factor = factor[pad_spatial]

    if isinstance(weights, dict):
        weights = weights.get('absolute', None)
    has_weights = weights is not None

    if absolute:
        hessian = hessian.clone()
        hessian_diag = matdiag(hessian.transpose(-dim-1, -1), nb_prm).transpose(-dim-1, -1)
        if has_weights:
            hessian_diag.addcmul_(weights, absolute*factor)
        else:
            hessian_diag.add_(absolute*factor)

    if is_diag:
        return gradient / hessian
    else:
        def solve(h, x):
            h = h.transpose(-dim-1, -1)
            x = x.transpose(-dim-1, -1)
            x = matsolve(h, x)
            x = x.transpose(-dim-1, -1)
            return x

        return solve(hessian, gradient)


def solve_grid_closedform(hessian, gradient, weights=None,
                          absolute=0, voxel_size=1, factor=1):
    """Solve a positive-definite linear system of the form (H + I)x = g

    Notes
    -----
    .. This function is specialized for multi-channel vector fields.
    .. This function only supports non-spatial regularization.
    .. This function solves the system in closed form.

    Inputs
    ------
    hessian : (..., *spatial, D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Regularization
    --------------
    absolute : float, default=0
    factor : float, default=1

    Returns
    -------
    solution : (..., *spatial, D) tensor

    """
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = gradient.shape[-1]
    nb_prm = dim
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    voxel_size = utils.make_vector(voxel_size, nb_prm, **backend)
    factor = factor * voxel_size.square()

    if isinstance(weights, dict):
        weights = weights.get('absolute', None)
    has_weights = weights is not None

    if absolute:
        hessian = hessian.clone()
        hessian_diag = utils.slice_tensor(hessian, slice(nb_prm), -dim - 1)
        if has_weights:
            hessian_diag.addcmul_(weights.unsqueeze(-1), absolute*factor)
        else:
            hessian_diag.add_(absolute*factor)

    if is_diag:
        return gradient / hessian
    else:
        return sym_solve(hessian, gradient)

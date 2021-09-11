import torch
import math
from nitorch.core import utils, py
from nitorch.core import optim as optimizers
from nitorch.core.linalg import sym_matvec, sym_solve
from ._grid import grid_pull, grid_push
from ._regularisers import (absolute, membrane, bending,
                            absolute_grid, membrane_grid, bending_grid,
                            lame_shear, lame_div,
                            absolute_diag, membrane_diag, bending_diag)


# TODO:
#   - switch out of FMG if no spatial regularization
#   - implement separable prolong/restrict (with torchscript?)


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
    x = grid_pull(x, grid, bound=bound, interpolation=order)
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
    x = grid_push(x, grid, in_spatial, bound=bound, interpolation=order)
    x = utils.fast_movedim(x, -dim-1, -1)
    return x


class _FMG:

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

        init_zero = self.init is None
        if init_zero:
            self.init = torch.zeros_like(self.gradient)

        pyrh = [self.hessian]          # hessians / matrix
        pyrg = [self.gradient]         # gradients / target
        pyrx = [self.init]             # solutions
        pyrw = [self.weights]          # voxel-wise weights
        pyrn = [self.spatial]          # shapes
        pyrv = [[1.]*self.dim]         # voxel size scaling
        for _ in range(self.max_levels):

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

    def solve(self):

        g = self.pyrg       # gradient (g + Lv)
        x = self.pyrx       # solution (input/output)
        d = self.pyrn       # shape
        h = self.forward    # full hessian (H + L)
        ih = self.solvers   # inverse of the hessian:  ih(g, x) <=> x = h\g

        # JA has precomputed gradients in b0 and also another pyramid `b`
        # which is initialized by b0 and then updated during the residuals
        # steps. I don't understand the point of keeping two separate buffers.
        # Trying with one for now:
        b = g
        # That's if we want to use two buffers:
        # b = [None] * self.nb_levels
        # b[0] = g[0]

        # Initial solution at coarsest grid
        if self.verbose > 0:
            print(f'(fmg) solve last ({self.nb_levels-1})')
        x[-1] = ih[-1](b[-1], x[-1])

        # The full multigrid (FMG) solver performs lots of  "two-grid"
        # multigrid (2MG) steps, at different scales. There are therefore
        # `len(pyrh) - 1` scales at which to work.

        for n_base in reversed(range(self.nb_levels-1)):
            if self.verbose > 0:
                print(f'(fmg) base level {n_base}')
            x[n_base] = self.prolong_g(x[n_base+1], d[n_base])
            # if n_base > 0:
            #     b[n_base] = g[n_base].clone()

            for n_cycle in range(self.nb_cycles):
                if self.verbose > 0:
                    print(f'(fmg) -- cycle {n_cycle}')
                for n in range(n_base, self.nb_levels-1):
                    if self.verbose > 0:
                        print(f'(fmg) ---- solve {n}')
                    x[n] = ih[n](b[n], x[n])
                    res = h[n](x[n]).neg_().add_(b[n])
                    if self.verbose > 0:
                        print(f'(fmg) ---- restrict {n} -> {n+1} (sub)')
                    b[n+1] = self.restrict_g(res, d[n+1])
                    del res
                    x[n+1].zero_()

                if self.verbose > 0:
                    print(f'(fmg) ---- solve last ({self.nb_levels-1})')
                x[-1] = ih[-1](b[-1], x[-1])

                for n in reversed(range(n_base, self.nb_levels-1)):
                    if self.verbose > 0:
                        print(f'(fmg) ---- prolong {n+1} -> {n} (add)')
                    x[n] += self.prolong_g(x[n+1], d[n])
                    if self.verbose > 0:
                        print(f'(fmg) ---- solve {n}')
                    x[n] = ih[n](b[n], x[n])

        x = x[0]
        self.clear_data()
        return x

    def clear_data(self):
        self.pyrh = self.pyrg = self.pyrx = self.pyrw = self.pyrn = self.pyrv = []
        self.hessian = self.gradient = self.weights = self.init = None
        return self


class _FieldFMG(_FMG):

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
            def solve(b, x0=None):
                return optim(forward, b, x0, precond=precond, **prm)

            return solve, forward

        solvers = []
        forward = []
        for h, w, v in zip(self.pyrh, self.pyrw, self.pyrv):
            solve, fwd = make_solver(h, w, v)
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
                prm['scheme'] = (3 if bending else 'checkerboard')

            def solve(b, x0=None):
                return optim(forward, b, x0, precond=precond, **prm)

            return solve, forward

        solvers = []
        forward = []
        for h, w, v in zip(self.pyrh, self.pyrw, self.pyrv):
            solve, fwd = make_solver(h, w, v)
            solvers.append(solve)
            forward.append(fwd)
        self.solvers = solvers
        self.forward = forward


def solve_grid_fmg(hessian, gradient, absolute=0, membrane=0, bending=0,
                   lame=0, factor=1, voxel_size=1, bound='dft', weights=None,
                   optim='cg', nb_cycles=2, nb_iter=2, verbose=False):
    """Solve a positive-definite linear system of the form (H + L)v = g

    This function uses preconditioned gradient descent
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
    verbose : int, default=0

    Returns
    -------
    solution : (..., *spatial, D) tensor

    """
    FMG = _GridFMG(absolute=absolute, membrane=membrane, bending=bending,
                   lame=lame, factor=factor, voxel_size=voxel_size,
                   bound=bound, optim=optim, nb_cycles=nb_cycles,
                   nb_iter=nb_iter, verbose=verbose)
    FMG.set_data(hessian, gradient, weights=weights)
    return FMG.solve()


def solve_grid(hessian, gradient, weights=None, voxel_size=1, bound='dft',
               absolute=0, membrane=0, bending=0, lame=0, factor=1,
               optim='cg', max_iter=32, stop='e', tolerance=1e-5,
               verbose=False):
    """Solve a positive-definite linear system of the form (H + L)v = g

    This function uses preconditioned gradient descent at a single resolution
    (Conjugate Gradient or Gauss-Seidel).

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
    max_iter : int, default=32
    stop : {'E', 'A'}, default='E'
    tolerance : float, default=1e-5
    verbose : int, default=0

    Returns
    -------
    solution : (..., *spatial, D) tensor

    """
    FMG = _GridFMG(absolute=absolute, membrane=membrane, bending=bending,
                   lame=lame, factor=factor, voxel_size=voxel_size,
                   bound=bound, optim=optim, nb_cycles=0, max_levels=0,
                   nb_iter=max_iter, verbose=verbose, stop=stop,
                   tolerance=tolerance)
    FMG.set_data(hessian, gradient, weights=weights)
    return FMG.solve()


def solve_field_fmg(hessian, gradient, weights=None, voxel_size=1, bound='dct2',
                    absolute=0, membrane=0, bending=0, factor=1, dim=None,
                    optim='cg', nb_cycles=2, nb_iter=2, verbose=False,
                    matvec=None, matsolve=None, matdiag=None):
    """Solve a positive-definite linear system of the form (H + L)v = g

    This function uses preconditioned gradient descent
    (Conjugate Gradient or Gauss-Seidel) in a full multi-grid fashion.

    Inputs
    ------
    hessian : (..., 1|K|K*(K+1)//2, *spatial) tensor
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
    nb_cycles : int, default=2
    nb_iter : int, default=2
    verbose : int, default=0

    Returns
    -------
    solution : (..., K, *spatial) tensor

    """
    FMG = _FieldFMG(absolute=absolute, membrane=membrane, bending=bending,
                    factor=factor, voxel_size=voxel_size,
                    bound=bound, optim=optim, nb_cycles=nb_cycles,
                    nb_iter=nb_iter, verbose=verbose,
                    matvec=matvec, matsolve=matsolve, matdiag=matdiag)
    FMG.set_data(hessian, gradient, weights=weights, dim=dim)
    return FMG.solve()


def solve_field(hessian, gradient, weights=None, voxel_size=1, bound='dct2',
                absolute=0, membrane=0, bending=0, factor=1, dim=None,
                optim='cg', max_iter=32, stop='e', tolerance=1e-5,
                verbose=False, matvec=sym_matvec, matsolve=sym_solve):
    """Solve a positive-definite linear system of the form (H + L)v = g

    This function uses preconditioned gradient descent at a single resolution
    (Conjugate Gradient or Gauss-Seidel).

    Inputs
    ------
    hessian : (..., 1|K|K*(K+1)//2, *spatial) tensor
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
    max_iter : int, default=32
    stop : {'E', 'A'}, default='E'
    tolerance : float, default=1e-5
    verbose : int, default=0

    Returns
    -------
    solution : (..., K, *spatial) tensor

    """
    FMG = _FieldFMG(absolute=absolute, membrane=membrane, bending=bending,
                    factor=factor, voxel_size=voxel_size,
                    bound=bound, optim=optim, nb_cycles=0, max_levels=0,
                    nb_iter=max_iter, verbose=verbose, stop=stop,
                    tolerance=tolerance, matvec=matvec, matsolve=matsolve)
    FMG.set_data(hessian, gradient, weights=weights, dim=dim)
    return FMG.solve()

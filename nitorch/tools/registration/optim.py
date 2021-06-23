"""This file implements generic optimizers that do not require autograd.
They take evaluated first and (optionally) second derivatives and return
a step. A few classes also require the ability to evaluate the function
(e.g., LBFGS), in which case a closure function that allows the objective
and its derivatives to be computed must be provided.

Most optimizers, however, simply use derivatives; and a wrapper class
that iterates multiple such steps is provided (IterOptim).
Available optimizers are:
- GradientDescent
- Momentum
- Nesterov
- GridCG
- GridRelax
- LBFGS
Along with helpers
- BacktrackingLineSearch
- StrongWolfe
- IterateOptim
- MultiOptim
"""

from nitorch import spatial
import torch
import copy


class Optim:
    """Base class for optimizers"""

    def __init__(self, lr=1, **opt):
        self.lr = lr
        for k, v in opt.items():
            setattr(self, k, v)

    @staticmethod
    def update(params, step):
        params.add_(step)
        return params

    def step(self, *derivatives):
        raise NotImplementedError

    requires_grad = False
    requires_hess = False


class ZerothOrder(Optim):
    requires_grad = False
    requires_hess = False


class FirstOrder(Optim):

    def __init__(self, lr=1, preconditioner=None, **opt):
        super().__init__(lr, **opt)
        self.preconditioner = preconditioner

    def precondition(self, grad):
        if callable(self.preconditioner):
            grad = self.preconditioner(grad)
        elif self.preconditioner is not None:
            # assume diagonal
            grad = grad.mul(self.preconditioner)
        return grad

    def precondition_(self, grad):
        if callable(self.preconditioner):
            grad = self.preconditioner(grad)
        elif self.preconditioner is not None:
            # assume diagonal
            grad = grad.mul_(self.preconditioner)
        return grad

    requires_grad = True
    requires_hess = False


class SecondOrder(Optim):
    requires_grad = True
    requires_hess = True


class GradientDescent(FirstOrder):
    """Gradient descent

    Δ{k+1} = - η ∇f(x{k})
    x{k+1} = x{k} + Δ{k+1}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, grad):
        grad = self.precondition_(grad.clone())
        return grad.mul_(-self.lr)


class Momentum(FirstOrder):
    """Gradient descent with momentum

    Δ{k+1} = α Δ{k} - η ∇f(x{k})
    x{k+1} = x{k} + Δ{k+1}
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('momentum', 0.9)
        super().__init__(*args, **kwargs)
        self.delta = 0
        self.n_iter = 0

    def step(self, grad):
        grad = self.precondition(grad)
        # momentum
        if torch.is_tensor(self.delta):
            self.delta.mul_(self.momentum)
            self.delta.sub_(grad, alpha=self.lr)
        else:
            self.delta = grad.mul(-self.lr)
        return self.delta


class Nesterov(FirstOrder):
    """Nesterov accelerated gradient

    Nesterov's acceleration can be seen as taking a step in the same
    direction as before (momentum) before taking a gradient descent step
    (correction):
    Δ{k+1} = α{k} Δ{k} - η ∇f(x{k} + α{k} Δ{k})
    x{k+1} = x{k} + Δ{k+1}

    We can introduce an auxiliary sequence of points at which the
    gradient is computed:
    y{k} = x{k} + α{k} Δ{k}

    We choose to follow this alternative sequence and switch half
    iterations, along the line of Sutskever and Bengio:
    Δ{k+1} = α{k} Δ{k} - η ∇f(y{k})
    y{k+1} = y{k} + α{k+1} Δ{k+1} - η ∇f(y{k})
    """

    def __init__(self, momentum=0, auto_restart=True, **kwargs):
        """

        Parameters
        ----------
        momentum : float, optional
            Momentum Factor that modulates the weight of the previous step.
            By default, the (t_k) sequence from Kim & Fessler is used.
        auto_restart : bool, default=True
            Automatically restart the state if the gradient and step
            disagree.
        lr : float, default=1
            Learning rate.
        """
        super().__init__(**kwargs)
        self.auto_restart = auto_restart
        self.delta = 0
        self.theta = 1
        self.momentum = self._momentum = momentum

    def restart(self):
        self.delta = 0
        self.theta = 1
        self._momentum = self.momentum

    def adaptive_restart(self, grad):
        if self.auto_restart and grad.flatten().dot(self.delta.flatten()) >= 0:
            # Kim & Fessler (2017) 4.2 (26)
            # gradient and step disagree
            self.restart()
            self.delta = grad.mul(-self.lr)

    def get_momentum(self):
        return self.momentum or self._momentum

    def get_relaxation(self):
        return self.relaxation or self._relaxation

    def update_momentum(self):
        theta_prev = self.theta
        theta = 0.5 * (1 + (1 + 4 * self.theta * self.theta) ** 0.5)
        self.theta = theta
        self._momentum = self.momentum or (theta_prev - 1) / theta

    def step(self, grad):
        grad = self.precondition(grad)

        # update momentum
        prev_momentum = self.get_momentum()
        self.update_momentum()

        # update delta (y{k+1} - y{k})
        if torch.is_tensor(self.delta):
            self.delta.mul_(prev_momentum)
            self.delta.sub_(grad, alpha=self.lr)
        else:
            self.delta = grad.mul(-self.lr)

        # adaptive restart
        self.adaptive_restart(grad)

        # compute step (x{k+1} - x{k})
        momentum = self.get_momentum()

        step = self.delta.mul(momentum).sub_(grad, alpha=self.lr)
        return step


class OGM(FirstOrder):
    """Optimized Gradient Method (Kim & Fessler)

    It belongs to the family of Accelerated First order Methods (AFM)
    that can be written as:
        y{k+1} = x{k} - η ∇f(x{k})
        x{k+1} = y{k+1} + a{k}(y{k+1} - y{k}) + β{k} (y{k+1} - x{k})

    Gradient descent has a{k} = β{k} = 0
    Nesterov's accelerated gradient has β{k} = 0

    Similarly to what we did with Nesterov's iteration, we rewrite
    it as a function of Δ{k+1} = y{k+1} - y{k}:
        Δ{k+1} = α{k-1} Δ{k} - η ∇f(x{k}) + β{k-1} η ∇f(x{k-1})
        x{k+1} = x{k} + α{k} Δ{k+1} - η ∇f(x{k})
    Note that we must store the last gradient on top of the previous Δ.
    """

    def __init__(self, momentum=0, relaxation=0,
                 auto_restart=True, damping=0.5, **kwargs):
        """

        Parameters
        ----------
        momentum : float, optional
            Momentum Factor that modulates the weight of the previous step.
            By default, the (t_k) sequence from Kim & Fessler is used.
        relaxation : float, optional
            Over-relaxation factor that modulates the weight of the
            previous gradient.
            By default, the (t_k) sequence from Kim & Fessler is used.
        auto_restart : bool, default=True
            Automatically restart the state if the gradient and step
            disagree.
        damping : float in (0..1), default=0.5
            Damping of the relaxation factor when consecutive gradients
            disagree (1 == no damping).
        lr : float, default=1
            Learning rate.
        """
        super().__init__(**kwargs)
        self.auto_restart = auto_restart
        self.damping = damping
        self._damping = 1
        self.delta = 0
        self.theta = 1
        self.momentum = self._momentum = momentum
        self.relaxation = self._relaxation = relaxation
        self._grad = 0

    def restart(self):
        self.delta = 0
        self.theta = 1
        self._momentum = self.momentum       # previous momentum
        self._relaxation = self.relaxation   # previous relaxation
        self._grad = 0                       # previous gradient
        self._damping = 1                    # current damping

    def adaptive_restart(self, grad):
        if self.auto_restart and grad.flatten().dot(self.delta.flatten()) >= 0:
            # Kim & Fessler (2017) 4.2 (26)
            # gradient and step disagree
            self.restart()
            self.delta = grad.mul(-self.lr)
        elif (self.damping != 1 and self._grad is not 0 and
              grad.flatten().dot(self._grad.flatten()) < 0):
            # Kim & Fessler (2017) 4.3 (27)
            # consecutive gradients disagree
            self._damping *= self.damping
        self._relaxation *= self._damping

    def get_momentum(self):
        return self.momentum or self._momentum

    def get_relaxation(self):
        return self.relaxation or self._relaxation

    def update_momentum(self):
        theta_prev = self.theta
        theta = 0.5 * (1 + (1 + 4 * self.theta * self.theta) ** 0.5)
        self.theta = theta
        self._momentum = self.momentum or (theta_prev - 1) / theta
        self._relaxation = self.relaxation or theta_prev / theta

    def step(self, grad):
        grad = self.precondition(grad)

        # update momentum
        prev_momentum = self.get_momentum()
        prev_relaxation = self.get_relaxation()
        self.update_momentum()

        # update delta (y{k+1} - y{k})
        if torch.is_tensor(self.delta):
            self.delta.mul_(prev_momentum)
            self.delta.sub_(grad, alpha=self.lr)
            self.delta.sub_(self._grad, alpha=self.lr * prev_relaxation)
        else:
            self.delta = grad.mul(-self.lr)

        # adaptive restart
        self.adaptive_restart(grad)

        # compute step (x{k+1} - x{k})
        momentum = self.get_momentum()
        relaxation = self.get_relaxation()
        step = self.delta.mul(momentum)
        step = step.sub_(grad, alpha=self.lr * (1 + relaxation))

        # save gradient
        if self._grad is 0:
            self._grad = grad.mul(self.lr)
        else:
            self._grad.copy_(grad).mul_(self.lr)
        return step


class MultiOptim(Optim):
    """A group of optimizers"""

    def __init__(self, optims, **opt):
        """

        Parameters
        ----------
        optims : list of dict or `Optim` types
            If a dict, it may have the key 'optim' to specify which
            optimizer to use. Other keys are parameters passed to the
            `Optim` constructor.
        opt : dict
            Additional parameters shared by all optimizers.
        """
        super().__init__()
        ooptims = []
        for optim in optims:
            if isinstance(optim, dict):
                ooptim = dict(opt)
                ooptim.update(optim)
                klass = ooptim.pop('optim', None)
            else:
                klass = optim
                ooptim = dict(opt)
                ooptim.pop('optim')
            if not klass:
                raise ValueError('No optimizer provided')
            ooptims.append(klass(**ooptim))
        self.optims = ooptims

    def step(self, grad):
        """Perform an optimization step.

        Parameters
        ----------
        grad : list[tensor]

        Returns
        -------
        step : list[tensor]

        """
        deltas = []
        for param, g in zip(self.optims, grad):
            deltas.append(param.step(g))
        return deltas

    @staticmethod
    def update(params, step):
        """

        Parameters
        ----------
        params : list[tensor]
        step : list[tensor]

        Returns
        -------
        params : list[tensor]

        """
        for p, s in zip(params, step):
            p.add_(s)
        return params


class GridGaussNewton(SecondOrder):
    """Base class for Gauss-Newton on displacement grids"""

    def __init__(self, max_iter=16, factor=1, voxel_size=1,
                 absolute=0, membrane=0, bending=0, lame=0, marquardt=True,
                 preconditioner=None, **kwargs):
        super().__init__(**kwargs)
        self.preconditioner = preconditioner
        self.factor = factor
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.lame = lame
        self.voxel_size = voxel_size
        self.marquardt = marquardt
        self.max_iter = max_iter

    def _get_prm(self):
        prm = dict(absolute=self.absolute,
                   membrane=self.membrane,
                   bending=self.bending,
                   lame=self.lame,
                   factor=self.factor,
                   voxel_size=self.voxel_size,
                   max_iter=self.max_iter)
        return prm

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        dim = grad.shape[-1]
        if self.marquardt is True:
            # maj = hess[..., :dim].abs()
            # if hess.shape[-1] > dim:
            #     maj.add_(hess[..., dim:].abs(), alpha=2)
            maj = hess[..., :dim].abs().max(-1, True).values
            hess[..., :dim].add(maj, alpha=tiny)
            # hess[..., :dim] += tiny
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess


class GridCG(GridGaussNewton):
    """Gauss-Newton on displacement grids using Conjugate Gradients"""

    def step(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess, 1e-5)
        prm = self._get_prm()
        step = spatial.solve_grid_sym(hess, grad, optim='cg',
                                      precond=self.preconditioner, **prm)
        step.mul_(-self.lr)
        return step


class GridRelax(GridGaussNewton):
    """Gauss-Newton on displacement grids using Relaxation"""

    def step(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        step = spatial.solve_kernel_grid_sym(hess, grad, optim='relax',
                                      precond=self.preconditioner, **prm)
        step.mul_(-self.lr)
        return step


class GridJacobi(GridGaussNewton):
    """Gauss-Newton on displacement grids using Relaxation"""

    def step(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        step = spatial.solve_grid_sym(hess, grad, optim='jacobi',
                                      precond=self.preconditioner, **prm)
        step.mul_(-self.lr)
        return step


class GridNesterov(GridGaussNewton):
    """Gauss-Newton on displacement grids using Relaxation"""

    def step(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        step = spatial.solve_grid_sym(hess, grad, optim='nesterov',
                                      precond=self.preconditioner, **prm)
        step.mul_(-self.lr)
        return step


class FieldGaussNewton(SecondOrder):
    """Base class for Gauss-Newton on vector fields"""

    def __init__(self, max_iter=16, factor=1, voxel_size=1,
                 absolute=0, membrane=0, bending=0, marquardt=True,
                 preconditioner=None, **kwargs):
        super().__init__(**kwargs)
        self.preconditioner = preconditioner
        self.factor = factor
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.voxel_size = voxel_size
        self.marquardt = marquardt
        self.max_iter = max_iter

    def _get_prm(self):
        prm = dict(absolute=self.absolute,
                   membrane=self.membrane,
                   bending=self.bending,
                   factor=self.factor,
                   voxel_size=self.voxel_size,
                   max_iter=self.max_iter)
        return prm

    def _add_marquardt(self, grad, hess):
        dim = grad.shape[-1]
        if self.marquardt is True:
            hess[..., :dim] += hess[..., :dim].abs().max(-1, True).values * 1e-5
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess


class FieldCG(FieldGaussNewton):
    """Gauss-Newton on vector fields using Conjugate Gradients"""

    def step(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        step = spatial.solve_field_sym(hess, grad, optim='cg',
                                       precond=self.preconditioner, **prm)
        step.mul_(-self.lr)
        return step


class FieldRelax(FieldGaussNewton):
    """Gauss-Newton on vector fields using Relaxation"""

    def step(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        step = spatial.solve_field_sym(hess, grad, optim='relax',
                                       precond=self.preconditioner, **prm)
        step.mul_(-self.lr)
        return step


class StrongWolfe(ZerothOrder):
    # Adapted from PyTorch

    def __init__(self, c1=0, c2=0.9, tol=1e-9, max_iter=25):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.max_iter = max_iter

    def iter(self, x0, f0, g0, a, delta, closure):
        """

        Parameters
        ----------
        f0 : previous value
        g0 : previous gradient
        delta : step direction
        closure : callable(x, grad=False), evaluate function

        Returns
        -------
        delta : step

        """

        # initialization
        ls_iter = 0
        a1 = 0
        f1 = f0
        g1 = g0
        dg0 = dg1 = delta.flatten().dot(g0.flatten())
        f, g = closure(x0.add(delta, alpha=a), grad=True)
        dd = delta.abs().max()

        def update_bracket_init(a, f, g, a1, f1, g1, dg1):
            # (a, f, g, dg)     -> new values (step, function, gradient, dot)
            # (a1, f1, g1, dg1) -> old values
            # (_, f0, _, dg0)   -> initial value
            dg = delta.flatten().dot(g.flatten())

            if f > (f0 + self.c1 * a * dg0) or (ls_iter > 1 and f >= f1):
                # Armijo condition failed=
                return -1, [a1, a], [f1, f], [g1, g], [dg1, dg]

            if abs(dg) <= -self.c2 * dg0:
                # Armijo + Wolfe succeeded
                return 0, [a, a], [f, f], [g, g], [dg, dg]

            if dg >= 0:
                # Wolfe failed + wrong direction
                return -1, [a1, a], [f1, f], [g1, g], [dg1, dg]

            # Wolfe failed + right direction
            return 1, [a1, a], [f1, f], [g1, g], [dg1, dg]

        def update_bracket(a, f, g, ba, bf, bg, bdg):
            # (a, f, g, dg)     -> new values (step, function, gradient, dot)
            # (ba, bf, bg, bdg) -> brackets (ordered [low, high])
            dg = delta.flatten().dot(g.flatten())

            if f > (f0 + self.c1 * a * dg0) or f >= bf[0]:
                # Armijo condition failed -> replace high  value
                ba = [ba[0], a]
                bf = [bf[0], f]
                bg = [bg[0], g]
                bdg = [bdg[0], dg]
                ba, bf, bg, bdg = self.sort_bracket(ba, bf, bg, bdg)
                return False, ba, bf, bg, bdg

            if abs(dg) <= -self.c2 * dg0:
                # Armijo + Wolfe succeeded
                return True, [a, a], [f, f], [g, g], [dg, dg]

            if dg * (ba[1] - ba[0]) >= 0:
                # old high becomes new low
                ba = ba[::-1]
                bf = bf[::-1]
                bg = bg[::-1]
                bdg = bdg[::-1]

            # new point becomes new low
            ba = [a, ba[1]]
            bf = [f, bf[1]]
            bg = [g, bg[1]]
            bdg = [dg, bdg[1]]
            return False, ba, bf, bg, bdg

        # We first want to find an upper bound for the step size.
        # We explore the space on the right of the initial step size
        # (a > a0) until we either find a point that enforces all Wolfe
        # conditions (we're lucky and can stop there) or one where the
        # gradient changes "sign" (i.e. the dot product of the new
        # gradient and search direction is >= 0). As long as this is not
        # true, cubic interpolation brings us to the right of the
        # rightmost point.
        while ls_iter < self.max_iter:
            side, (a1, a), (f1, f), (g1, g), (dg1, dg) \
                = update_bracket_init(a, f, g, a1, f1, g1, dg1)
            ls_iter += 1
            if side == 0:
                return a1, f1, g1
            elif side < 0:
                break
            elif abs(a1 - a) * dd < self.tol:
                break
            else:
                a_prev = a
                a = self.interpolate_bracket((a, a1), (f, f1), (dg, dg1), bound=True)
                a1 = a_prev
                f1 = f
                g1 = g
                dg1 = dg
                f, g = closure(x0.add(delta, alpha=a), grad=True)

        if ls_iter == self.max_iter:
            a1 = 0

        # We have a lower and upper bound for the step size.
        # We can now find a points that falls in the middle using
        # cubic interpolation and either find a point that enforces
        # all Wolfe conditions (we can stop there) or explore the half
        # space between the two lowest points.
        (a, a1), (f, f1), (g, g1), (dg, dg1) \
            = self.sort_bracket((a, a1), (f, f1), (g, g1), (dg, dg1))
        progress = True
        success = False
        while ls_iter < self.max_iter:
            if success or abs(a1 - a) * dd < self.tol:
                break
            new_a = self.interpolate_bracket((a, a1), (f, f1), (dg, dg1))
            ls_iter += 1
            new_a, progress = self.check_progress(new_a, (a, a1), progress)
            new_f, new_g = closure(x0.add(delta, alpha=new_a), grad=True)
            success, (a, a1), (f, f1), (g, g1), (dg, dg1) \
                = update_bracket(new_a, new_f, new_g,
                                 (a, a1), (f, f1), (g, g1), (dg, dg1))

        return a, f, g

    @staticmethod
    def sort_bracket(ba, bf, bg, bdg):
        if bf[0] > bf[-1]:
            ba = ba[::-1]
            bf = bf[::-1]
            bg = bg[::-1]
            bdg = bdg[::-1]
        return ba, bf, bg, bdg

    @staticmethod
    def check_progress(a, bracket, progress):
        # Copied from PyTorch
        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - a, a - min(bracket)) < eps:
            # interpolation close to boundary
            if not progress or a >= max(bracket) or a <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(a - max(bracket)) < abs(a - min(bracket)):
                    a = max(bracket) - eps
                else:
                    a = min(bracket) + eps
                progress = True
            else:
                progress = False
        else:
            progress = True
        return a, progress

    @staticmethod
    def interpolate_bracket(bracket_a, bracket_f, bracket_dg, bound=False):
        if bound:
            min_step = bracket_a[0] + 0.01 * (bracket_a[0] - bracket_a[1])
            max_step = bracket_a[0] * 10
            bounds = (min_step, max_step)
        else:
            bounds = None
        a = StrongWolfe.cubic_interpolate(
            bracket_a[1], bracket_f[1], bracket_dg[1],
            bracket_a[0], bracket_f[0], bracket_dg[0],
            bounds=bounds)
        return a

    @staticmethod
    def cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
        # copied from PyTorch
        # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
        # Compute bounds of interpolation area
        if bounds is not None:
            xmin_bound, xmax_bound = bounds
        else:
            xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

        # Code for most common case: cubic interpolation of 2 points
        #   w/ function and derivative values for both
        # Solution in this case (where x2 is the farthest point):
        #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
        #   d2 = sqrt(d1^2 - g1*g2);
        #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
        #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2_square = d1 ** 2 - g1 * g2
        if d2_square >= 0:
            d2 = d2_square.sqrt()
            if x1 <= x2:
                min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            else:
                min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
            return min(max(min_pos, xmin_bound), xmax_bound)
        else:
            return (xmin_bound + xmax_bound) / 2.


class LBFGS(FirstOrder):
    """Limited memory BFGS"""

    def __init__(self, max_iter=20, history=100, lr=1, wolfe=None, **kwargs):
        """

        Parameters
        ----------
        max_iter : int, default=20
            Maximum number of iterations
        history : int, default=5
            Number of past gradients stored in memory
        lr : float, default=1
            Learning rate
        wolfe : dict(c1=1e-4, c2=0.9, max_iter=25, tol=1e-9)
            Options for the Wolfe line search
        """
        super().__init__(lr=lr, **kwargs)

        self.max_iter = max_iter
        self.history = history
        if wolfe is False:
            self.wolfe = False
        else:
            self.wolfe = StrongWolfe(**(wolfe or dict()))
        self.delta = []  # s
        self.delta_grad = []  # y
        self.sy = []
        self.yy = []
        self.last_grad = 0

    def iter(self, param, closure, derivatives=False):
        """Perform multiple LBFGS iterations, including Wolfe line search

        Parameters
        ----------
        param : tensor
            Initial guess (will be modified inplace)
        closure : callable(tensor, grad=False) -> Tensor, Tensor
            Function that takes a parameter and returns the objective and
            its (optionally) gradient evaluated at that point.
        derivatives : bool, default=False
            Return latest derivatives

        Returns
        -------
        param : tensor
            Updated parameter
        grad : tensor, if `derivatives is True`

        """
        import inspect
        if 'in_line_search' in inspect.signature(closure).parameters:
            closure_line_search = lambda *a, **k: closure(*a, **k, in_line_search=True)
        else:
            closure_line_search = closure

        ll_prev = None
        for n_iter in range(1, self.max_iter+1):
            if n_iter == 1:
                ll, grad = closure(param, grad=True)
                delta = grad.neg()
                step = min(1., 1. / grad.abs().sum()) * self.lr
                # step = self.lr
            else:
                self.update_state(grad, delta)
                delta = self.step(grad, update=False)
                step = self.lr
                # if torch.dot(delta.flatten(), grad.flatten()) > -1e-9:
                #     print('delta x grad', grad.abs().max())
                #     break
            if self.wolfe is not False:
                step, ll, grad = self.wolfe.iter(
                    param, ll, grad, step, delta, closure_line_search)
                # closure(param)  # to plot stuff
            delta.mul_(step)
            param.add_(delta)
            if self.wolfe is False and n_iter != self.max_iter:
                ll, grad = closure(param, grad=True)
            # convergence
            if grad.abs().max() <= 1e-7:
                # print('grad', grad.abs().max())
                break
            if delta.abs().max() <= 1e-9:
                # print('step', delta.abs().max())
                break
            if ll_prev and abs(ll - ll_prev) < 1e-9:
                # print('ll')
                break
        return (param, grad) if derivatives else param

    def step(self, grad, update=True):
        """Compute gradient direction, without Wolfe line search"""
        alphas = []
        delta = grad.neg()
        for i in range(1, min(self.history, len(self.delta))+1):
            alpha = torch.dot(self.delta[-i].flatten(),
                              delta.flatten()) / self.sy[-i]
            alphas.append(alpha)
            delta.sub_(self.delta_grad[-i], alpha=alpha)
        alphas = list(reversed(alphas))
        if callable(self.preconditioner):
            delta = self.preconditioner(delta)
        elif self.preconditioner:
            # assume diagonal
            delta.mul_(self.preconditioner)
        else:
            gamma = self.sy[-1] / self.yy[-1]
            delta *= gamma
        for i in range(min(self.history, len(self.delta))):
            beta = torch.dot(self.delta_grad[i].flatten(),
                             delta.flatten()) / self.sy[i]
            delta.add_(self.delta[i], alpha=alphas[i] - beta)
        if update:
            self.update_state(grad, delta.mul(self.lr))
        return delta

    def update_state(self, grad, delta):
        """Update state"""
        if len(self.delta) == self.history:
            self.delta.pop(0)
            self.delta_grad.pop(0)
            self.sy.pop(0)
            self.yy.pop(0)
        self.delta_grad.append(grad - self.last_grad)
        self.last_grad = grad
        self.delta.append(delta)
        self.yy.append(torch.dot(self.delta_grad[-1].flatten(),
                                 self.delta_grad[-1].flatten()))
        self.sy.append(torch.dot(self.delta_grad[-1].flatten(),
                                 self.delta[-1].flatten()))


class IterationStep(Optim):
    """One step of an optimizer (compute gradient, compute step, update)"""

    def __init__(self, optim):
        super().__init__()
        self.optim = optim
        self._requires = None

    requires_grad = property(lambda self: self.optim.requires_grad)
    requires_hess = property(lambda self: self.optim.requires_hess)

    @property
    def preconditioner(self):
        return self.optim.preconditioner

    @preconditioner.setter
    def preconditioner(self, value):
        self.optim.preconditioner = value

    @property
    def requires(self):
        if self._requires is None:
            self._requires = dict()
            if self.requires_grad:
                self._requires['grad'] = True
            if self.requires_hess:
                self._requires['hess'] = True
        return self._requires

    def step(self, param, closure, derivatives=False):
        """

        Parameters
        ----------
        param : [list of] tensor
            Current state of the optimized parameters
        closure : callable([list of] tensor) -> tensor
            Function that takes optimized parameters as inputs and
            returns the objective function.
        derivaitves : bool, default=False
            Return most recent derivatives

        Returns
        -------
        param : [list of] tensor
            Updated parameters.
        ll : () tensor
        *derivatives : tensor, optional

        """
        return_derivatives = derivatives
        derivatives = []
        ll = closure(param, **self.requires)
        if self.requires:
            ll, *derivatives = ll
        delta = self.optim.step(*derivatives)
        param = self.optim.update(param, delta)
        if return_derivatives:
            return (param, ll, *derivatives)
        return param, ll


class BacktrackingLineSearch(IterationStep):
    """Meta-optimizer that performs a backtracking line search on the
    value of an optimizer's parameter.
    """

    def __init__(self, optim, max_iter=6, key='lr', factor=0.5,
                 store_value=False):
        """

        Parameters
        ----------
        optim : `Optim`, default=GradientDescent
            Optimizer whose parameter is line-searched
        max_iter : int, default=6
            Maximum number of searches
        key : str, default='lr'
            Name of the parameter to search.
            It should be a mutable attribute of `optim`.
        factor : float, default=0.5
            The searched parameter is multiplied by this factor after
            each failed iteration.
        store_value : bool or float, default=1.1
            Whether to store the successful parameter in `optim`.
            If False, the original value is restored at the end of the
            line search. If a float, the successful parameter is modulated
            by that value before being stored.
        """
        super().__init__(optim)
        self.key = key
        self.factor = factor
        self.max_iter = max_iter
        self.store_value = store_value

    def step(self, param, closure, derivatives=False):

        import inspect
        if 'in_line_search' in inspect.signature(closure).parameters:
            closure_line_search = lambda *a, **k: closure(*a, **k, in_line_search=True)
        else:
            closure_line_search = closure

        # compute derivatives
        return_derivatives = derivatives
        derivatives = []
        ll = closure(param, **self.requires)
        if self.requires:
            ll, *derivatives = ll

        # line search
        ok = False
        ll0 = ll
        param0 = param
        value0 = getattr(self.optim, self.key)
        value = value0
        for n_iter in range(self.max_iter):

            optim = copy.deepcopy(self.optim)
            setattr(optim, self.key, value)
            delta = optim.step(*derivatives)
            param = optim.update(param0.clone(), delta)

            ll = closure_line_search(param)
            if ll < ll0:
                ok = True
                break
            else:
                value = self.factor * value

        if ok:
            self.optim = optim
            if self.store_value:
                setattr(self.optim, self.key, value * self.store_value)
            else:
                setattr(self.optim, self.key, value0)
        else:
            param = param0
            ll = ll0
        if return_derivatives:
            return (param, ll, *derivatives)
        return param, ll


class StepSizeLineSearch(BacktrackingLineSearch):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, param, closure, derivatives=False):

        import inspect
        if 'in_line_search' in inspect.signature(closure).parameters:
            closure_line_search = lambda *a, **k: closure(*a, **k, in_line_search=True)
        else:
            closure_line_search = closure

        # compute derivatives and step
        return_derivatives = derivatives
        derivatives = []
        ll = closure(param, **self.requires)
        if self.requires:
            ll, *derivatives = ll
        delta = self.optim.step(*derivatives)

        # line search step size
        ok = False
        ll0 = ll
        param0 = param
        lr0 = self.lr
        lr = lr0
        for n_iter in range(self.max_iter):

            param = param0.add(delta, alpha=lr)

            ll = closure_line_search(param)
            if ll < ll0:
                ok = True
                break
            else:
                lr = self.factor * lr

        if ok:
            if self.store_value:
                self.lr = lr * self.store_value
            else:
                self.lr = lr0
        else:
            param = param0
            ll = ll0
        if return_derivatives:
            return (param, ll, *derivatives)
        return param, ll


class IterateOptim(Optim):
    """Wrapper that performs multiple steps of an optimizer.

    Backtracking line search can be used.
    """

    def __init__(self, optim, max_iter=20, tol=1e-9, ls=0):
        """

        Parameters
        ----------
        optim : Optim or IterationStep
        max_iter : int, default=20
        tol : float, default=1e-9
        ls : int or callable(Optim), default=0
        """
        super().__init__()
        self.optim = optim
        self.max_iter = max_iter
        self.tol = tol
        if ls:
            if isinstance(self.optim, IterationStep):
                raise ValueError('Cannot add line search to pre-defined '
                                 'IterationStep')
            if isinstance(ls, int):
                if isinstance(self.optim, SecondOrder):
                    self.optim = StepSizeLineSearch(self.optim, max_iter=ls)
                else:
                    self.optim = BacktrackingLineSearch(self.optim, max_iter=ls)
            elif callable(ls):
                self.optim = ls(self.optim)
        elif not isinstance(self.optim, IterationStep):
            self.optim = IterationStep(self.optim)

    requires_grad = property(lambda self: self.optim.requires_grad)
    requires_hess = property(lambda self: self.optim.requires_hess)

    @property
    def preconditioner(self):
        return self.optim.preconditioner

    @preconditioner.setter
    def preconditioner(self, value):
        self.optim.preconditioner = value

    def iter(self, param, closure, derivatives=False):
        """Perform multiple optimization iterations

        Parameters
        ----------
        param : tensor
            Initial guess (will be modified inplace)
        closure : callable(tensor, [grad=False], [hess=False]) -> Tensor, *Tensor
            Function that takes a parameter and returns the objective and
            its (optionally) gradient evaluated at that point.

        Returns
        -------
        param : tensor
            Updated parameter

        """
        ll_prev = float('inf')
        ll_max = 0
        for n_iter in range(1, self.max_iter+1):
            # perform step
            param, ll, *drv = self.optim.step(param, closure,
                                              derivatives=derivatives)

            # check convergence
            if abs((ll_prev-ll)/max(ll_max-ll, 1e-9)) < self.tol:
                break
            ll_prev = ll
            ll_max = max(ll, ll_max)

        if derivatives:
            return (param, *drv)
        return param

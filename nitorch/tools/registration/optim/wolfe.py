import math

from .base import FirstOrder, _wrapped_prop
from .linesearch import LineSearch
import torch


class StrongWolfe(FirstOrder):
    """Optimize along a search direction until Wolfe conditions are enforced"""

    def __init__(self, lr=1, c1=1e-4, c2=0.9, tol=1e-9, max_iter=25, **kwargs):
        """

        Parameters
        ----------
        lr : float, default=1
            Learning rate (= first step size tried)
        c1 : float, default=1e-4
            Wolfe condition on the loss
        c2 : float, default=0.9
            Wolfe condition on gradients
        tol : float, default=1e-9
            Tolerance for early stopping
        max_iter : int, default=25
            Maximum number of iterations
        """
        super().__init__(lr=lr, iter=0, **kwargs)
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.max_iter = max_iter

    def iter(self, *args, **kwargs):
        # No point iterating a Line search
        return self.step(*args, **kwargs)

    def step(self, step_size, closure, delta=None, **kwargs):
        """

        Parameters
        ----------
        step_size : float
            Previous step size (usually zero)
        closure : callable(step_size, [grad=False]) -> scalar tensor
            Function that evaluates the loss and gradient at a given step size
        delta : tensor, default=-gradient
            Search direction


        Other Parameters
        ----------------
        derivatives : bool, default=False
            Whether to return the next derivatives
        loss : tensor, optional
            Previous loss
        grad : tensor, optional
            Previous gradient

        Returns
        -------
        step_size : float, New step size
        loss : tensor, New loss
        grad : tensor, New gradient, if `derivatives`

        """
        a0 = step_size
        f0, g0 = self._get_loss(step_size, closure, **kwargs)
        if delta is None:
            delta = -g0

        if torch.is_tensor(a0):
            a0 = a0.item()
        a = self.lr

        # initialization
        ls_iter = 0
        a1, f1, g1 = a0, f0, g0
        dg0 = dg1 = delta.flatten().dot(g0.flatten())

        def update_bracket_init(a_new, f_new, dg_new, f_old):
            # Here, we're just looking for an upper bound on `a`:
            # `a_new` (step), `f_new` (func) and `dg_new` (grad) are a
            # values for a new step size to investigate. `f_old` was the
            # previous best value (with a negative gradient) .
            # - If we find a point that does not decrease the objective
            #   (Armijo failed), we know that the minimum lies between
            #   `a_old` and `a_new` (side == -1)
            # - If we find a point that does decrease the objective
            #   (Armijo success) but whose gradient is positive
            #   (dg > 0), we're also too far and know that the minimum
            #   lies between `a_old` and `a_new` (side == -1)
            # - If we find a point that does decrease the objective
            #   (Armijo success) but whose gradient is still too steep
            #   (Wolfe failed) we have to keep moving right (side == 1)
            # - Else, we found a point that enforces all conditions, we
            #   were lucky and return it (side == 0)

            if f_new > (f0 + self.c1 * a_new * dg0) or f_new >= f_old:
                # Armijo condition failed
                return -1

            if abs(dg_new) <= -self.c2 * dg0:
                # Armijo + Wolfe succeeded
                return 0

            if dg_new >= 0:
                # Wolfe failed + wrong direction
                return -1

            if not math.isfinite(f_new):
                # Same as inf
                return -1

            # Wolfe failed + right direction
            return 1

        def debug(*a, **k): pass

        def update_bracket(a, f, g, dg, ba, bf, bg, bdg):
            # (a, f, g, dg)     -> new values (step, function, gradient, dot)
            # (ba, bf, bg, bdg) -> brackets (ordered [low, high])

            if f > (f0 + self.c1 * a * dg0) or f >= bf[0]:
                # Armijo condition failed -> replace high  value
                debug('Armijo condition failed -> replace high  value')
                ba = [ba[0], a]
                bf = [bf[0], f]
                bg = [bg[0], g]
                bdg = [bdg[0], dg]
                ba, bf, bg, bdg = self.sort_bracket(ba, bf, bg, bdg)
                return False, ba, bf, bg, bdg

            if abs(dg) <= -self.c2 * dg0:
                # Armijo + Wolfe succeeded
                debug('Armijo + Wolfe succeeded')
                return True, [a, a], [f, f], [g, g], [dg, dg]

            if dg * (ba[1] - ba[0]) >= 0:
                # old low becomes new high
                debug('old low becomes new high')
                ba = ba[::-1]
                bf = bf[::-1]
                bg = bg[::-1]
                bdg = bdg[::-1]

            # new point becomes new low
            debug('new point becomes new low')
            ba = [a, ba[1]]
            bf = [f, bf[1]]
            bg = [g, bg[1]]
            bdg = [dg, bdg[1]]
            return False, ba, bf, bg, bdg

        # value at initial step size
        f, g = closure(a, grad=True)
        while not math.isfinite(f):
            a = 0.5 * a
            f, g = closure(a, grad=True)
        dg = delta.flatten().dot(g.flatten())

        # We first want to find an upper bound for the step size.
        # We explore the space on the right of the initial step size
        # (a > a0) until we either find a point that enforces all Wolfe
        # conditions (we're lucky and can stop there) or one where the
        # gradient changes "sign" (i.e. the dot product of the new
        # gradient and search direction is >= 0). As long as this is not
        # true, cubic interpolation brings us to the right of the
        # rightmost point.
        dd = delta.abs().max()
        debug('bracket')
        while ls_iter < self.max_iter:
            debug((a1, f1.item(), "\\" if dg1 < 0 else '//'),
                  (a, f.item(), "\\" if dg < 0 else '//'))
            side = update_bracket_init(a, f, dg, f1)
            ls_iter += 1
            if side == 0:
                debug('side = 0')
                return a, f, g
            elif side < 0:
                debug('side < 0')
                break
            elif abs(a1 - a) * dd < self.tol:
                debug('side ≈ 0')
                break
            else:
                debug('side > 0')
                bracket = (a1, a), (f1, f), (dg1, dg)
                a1, f1, g1, dg1 = a, f, g, dg
                a = self.interpolate_bracket(*bracket, bound=True)
                f, g = closure(a, grad=True)
                dg = delta.flatten().dot(g.flatten())

        if ls_iter == self.max_iter:
            a1 = 0

        # We have a lower and upper bound for the step size.
        # We can now find a points that falls in the middle using
        # cubic interpolation and either find a point that enforces
        # all Wolfe conditions (we can stop there) or explore the half
        # space between the two lowest points.
        debug('optimize')
        (a, a1), (f, f1), (g, g1), (dg, dg1) \
            = self.sort_bracket((a, a1), (f, f1), (g, g1), (dg, dg1))
        progress = True
        success = False
        while ls_iter < self.max_iter:
            if success or abs(a1 - a) * dd < self.tol:
                debug('converge', a1, a, dd)
                break
            if a1 < a:
                debug((a1, f1.item(), "\\" if dg1 < 0 else '//'),
                      (a, f.item(), "\\" if dg < 0 else '//'))
            else:
                debug((a, f.item(), "\\" if dg < 0 else '//'),
                      (a1, f1.item(), "\\" if dg1 < 0 else '//'))
            new_a = self.interpolate_bracket((a, a1), (f, f1), (dg, dg1))
            ls_iter += 1
            new_a, progress = self.check_progress(new_a, (a, a1), progress)
            new_f, new_g = closure(new_a, grad=True)
            new_dg = delta.flatten().dot(new_g.flatten())
            debug('new:', new_a, new_f.item(), "\\" if new_dg < 0 else '//')
            success, (a, a1), (f, f1), (g, g1), (dg, dg1) \
                = update_bracket(new_a, new_f, new_g, new_dg,
                                 (a, a1), (f, f1), (g, g1), (dg, dg1))

        if f > f0:
            return 0, f0, g0
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
            min_step = bracket_a[1] + 0.01 * (bracket_a[1] - bracket_a[0])
            max_step = bracket_a[1] * 10
            bounds = (min_step, max_step)
        else:
            bounds = None
        a = StrongWolfe.cubic_interpolate(
            bracket_a[0], bracket_f[0], bracket_dg[0],
            bracket_a[1], bracket_f[1], bracket_dg[1],
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

        if x2 < x1:
            ((x1, f1, g1), (x2, f2, g2)) = ((x2, f2, g2), (x1, f1, g1))

        # My code
        # I reimplemented it to deal with the case where both critical
        # points lie in the interval, which I take as a hint that we are in
        # a non-convex portion and we should not trust the minimum.
        # Solution based on https://math.stackexchange.com/questions/1522439/
        #
        # 18 Aug 2021: I might not need this extra check, but the pytorch
        # solution is weird so I am keeping my solution for now.
        a = g1 + g2 - 2 * (f2 - f1) / (x2 - x1)
        a /= (x1 - x2) ** 2
        b = 0.5 * (g2 - g1) / (x2 - x1) - 1.5 * a * (x1 + x2)
        c = g1 - 3 * a * x1 * x1 - 2 * x1 * b
        # d = f1 - x1 ** 3 * a - x1 ** 2 * b - x1 * c  # not used
        delta = b*b - 3 * a * c
        # import matplotlib.pyplot as plt
        # fig = plt.gcf()
        # x = torch.linspace(x1, x2, 512)
        # f = lambda x: a*x*x*x+b*x*x+c*x+d
        # plt.plot(x, f(x))
        # plt.scatter([x1, x2], [f1, f2], color='k', marker='o')
        if delta > 0:
            if a < 0:
                x_min = (- b + (delta ** 0.5)) / (3 * a)
                x_max = (- b - (delta ** 0.5)) / (3 * a)
            else:
                x_min = (- b - (delta ** 0.5)) / (3 * a)
                x_max = (- b + (delta ** 0.5)) / (3 * a)
            # plt.scatter([x_min], [f(x_min)], color='r', marker='o')
            # plt.xlim([x1-0.5*(x2-x1), x2+0.5*(x2-x1)])
            # plt.scatter([x_max], [f(x_max)], color='k', marker='o')
            # if x1 < x_min < x2 and x1 < x_max < x2:
            #     # both critical points are in the interval
            #     # we are probably in a non-convex portion
            #     return (xmin_bound + xmax_bound) / 2.
            # else:
            #     return min(max(x_min, xmin_bound), xmax_bound)
            sol = min(max(x_min, xmin_bound), xmax_bound)
        else:
            # no critical point (no or one inflexion point)
            sol = (xmin_bound + xmax_bound) / 2.
        # plt.vlines(sol, min(f(x)), max(f(x)))
        # plt.show(block=False)
        # input('')
        # plt.close(fig)

        if torch.is_tensor(sol):
            sol = sol.item()
        return sol

        # PyTorch code
        # /!\ I don't find this solution when checking with the symbolic
        # toolbox. I have reimplemented using something that passes the
        # symbolic test above.
        #
        # Code for most common case: cubic interpolation of 2 points
        #   w/ function and derivative values for both
        # Solution in this case (where x2 is the farthest point):
        #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
        #   d2 = sqrt(d1^2 - g1*g2);
        #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
        #   t_new = min(max(min_pos,xmin_bound),xmax_bound);

        # d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        # d2_square = d1 ** 2 - g1 * g2
        # if d2_square >= 0:
        #     d2 = d2_square.sqrt()
        #     if x1 <= x2:
        #         min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        #     else:
        #         min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        #     return min(max(min_pos, xmin_bound), xmax_bound)
        # else:
        #     return (xmin_bound + xmax_bound) / 2.


class StrongWolfeLineSearch(LineSearch):

    def repr_keys(self):
        keys = super().repr_keys() + ['c1', 'c2', 'max_iter']
        return keys

    def __init__(self, optim=None, lr=1, c1=0, c2=0.9, tol=1e-9, max_iter=25,
                 store_value=False, **kwargs):
        """

        Parameters
        ----------
        optim : `Optim`, optional
            Optimizer whose parameter is line-searched
        lr : float, default=1
            Learning rate (= first step size tried)
        c1 : float, default=1e-4
            Wolfe condition on the loss
        c2 : float, default=0.9
            Wolfe condition on gradients
        tol : float, default=1e-9
            Tolerance for early stopping
        max_iter : int, default=25
            Maximum number of iterations
        store_value : bool or float, default=False
            Whether to store the successful parameter in `optim`.
            If False, the original value is restored at the end of the
            line search. If a float, the successful parameter is modulated
            by that value before being stored.
        """
        self.wolfe = StrongWolfe(lr=lr, c1=c1, c2=c2, tol=tol, max_iter=max_iter)
        super().__init__(optim, **kwargs)
        self.store_value = store_value

    c1 = _wrapped_prop('c1', 'wolfe')
    c2 = _wrapped_prop('c2', 'wolfe')
    tol = _wrapped_prop('tol', 'wolfe')
    max_iter = _wrapped_prop('max_iter', 'wolfe')
    lr = _wrapped_prop('lr', 'wolfe')

    def step(self, param=None, closure=None, **kwargs):

        import inspect
        if 'in_line_search' in inspect.signature(closure).parameters:
            closure_line_search = lambda *a, **k: closure(*a, **k, in_line_search=True)
        else:
            closure_line_search = closure

        optim = kwargs.get('optim', self.optim)
        if param is None:
            param = optim.param
        if closure is None:
            closure = optim.closure

        # compute loss and derivatives
        loss, grad, *derivatives = self._get_loss(param, closure, **kwargs)
        delta = optim.search_direction(grad, *derivatives)

        def closure1d(a, **kwargs):
            return closure_line_search(param.add(delta, alpha=a), **kwargs)

        step_size, loss, grad = self.wolfe.step(0, closure1d, delta,
                                                loss=loss, grad=grad)
        delta.mul_(step_size)
        param.add_(delta)

        if self.store_value:
            self.wolfe.lr = step_size

        # for verbose
        closure(param)

        return (param, loss, grad, *derivatives)

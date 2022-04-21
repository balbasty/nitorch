from .base import ZerothOrder
from .linesearch import LineSearch
from nitorch.core import utils
import torch
import math as pymath


class Brent(ZerothOrder):
    # 1D line search

    gold = (1 + pymath.sqrt(5))/2
    igold = 1 - (pymath.sqrt(5) - 1)/2
    tiny = 1e-8

    def __init__(self, lr=1, tol=1e-9, max_iter=128):
        super().__init__(lr=lr, iter=0)
        self.tol = tol
        self.max_iter = max_iter

    def iter(self, *args, **kwargs):
        # No point iterating a Line search
        return self.step(*args, **kwargs)

    def step(self, step_size, closure, **kwargs):
        """

        Parameters
        ----------
        step_size : float
            Previous step size (usually zero)
        closure : callable(step_size, [grad=False]) -> scalar tensor
            Function that evaluates the loss and gradient at a given step size


        Other Parameters
        ----------------
        loss : tensor, optional
            Previous loss

        Returns
        -------
        step_size : float, New step size
        loss : tensor, New loss

        """
        def closure_(a):
            loss = closure(a)
            if torch.is_tensor(loss):
                loss = loss.item()
            return loss
        loss = kwargs.get('loss', closure_(step_size))
        if torch.is_tensor(loss):
            loss = loss.item()
        bracket = self.bracket(loss, closure_)
        step_size, loss = self.search_in_bracket(bracket, closure_)
        return step_size, loss

    def bracket(self, f0, closure):
        """Bracket the minimum

        Parameters
        ----------
        f0 : Initial value
        closure : callable(a) -> evaluate function at `a0 + a`

        Returns
        -------
        (a0, f0), (a1, f1), (a2, f2)
        """
        a0, a1 = 0, self.lr
        f1 = closure(a1)

        # sort such that f1 < f0
        if f1 > f0:
            a0, f0, a1, f1 = a1, f1, a0, f0

        a2 = a1 + self.gold * (a1 - a0)
        f2 = closure(a2)

        while f1 > f2:
            # fit quadratic polynomial
            a = utils.as_tensor([a0-a1, 0., a2-a1])
            quad = torch.stack([torch.ones_like(a), a, a.square()], -1)
            f = utils.as_tensor([f0, f1, f2]).unsqueeze(-1)
            quad = quad.pinverse().matmul(f).squeeze(-1)

            if quad[2] > 0:  # There is a minimum
                delta = -0.5 * quad[1] / quad[2].clamp_min(self.tiny)
                delta = delta.clamp_max_((1 + self.gold) * (a2 - a1))
                delta = delta.item()
                a = a1 + delta
            else:  # No minimum -> we must go farther than a2
                delta = self.gold * (a2 - a1)
                a = a2 + delta

            # check progress and update bracket
            # f2 < f1 < f0 so (assuming unicity) the minimum is in
            # (a1, a2) or (a2, inf)
            f = closure(a)
            if a1 < a < a2 or a2 < a < a1:
                if f1 < f < f2:
                    # minimum is in (a1, a2)
                    (a0, f0), (a1, f1), (a2, f2) = (a1, f1), (a, f), (a2, f2)
                    break
                elif f1 < f:  # implicitly: f0 < f1 < f
                    # minimum is in (a0, a)
                    (a0, f0), (a1, f1), (a2, f2) = (a0, f0), (a1, f1), (a, f)
                    break
            # shift by one point
            (a0, f0), (a1, f1), (a2, f2) = (a1, f1), (a2, f2), (a, f)

        return (a0, f0), (a1, f1), (a2, f2)

    def sort_bracket(self, bracket):
        (a0, f0), (a1, f1), (a2, f2) = bracket
        if f0 < f1 and f0 < f2:
            if f1 < f2:
                return bracket
            else:
                return (a0, f0), (a2, f2), (a1, f1)
        elif f1 < f0 and f1 < f2:
            if f0 < f2:
                return (a1, f1), (a0, f0), (a2, f2)
            else:
                return (a1, f1), (a2, f2), (a0, f0)
        else:
            if f0 < f1:
                return (a2, f2), (a0, f0), (a1, f1)
            else:
                return (a2, f2), (a1, f1), (a0, f0)

    def search_in_bracket(self, bracket, closure):
        b0, b1 = (bracket[0][0], bracket[2][0])
        if b1 < b0:
            b0, b1 = b1, b0
        # sort by values
        (a0, f0), (a1, f1), (a2, f2) = self.sort_bracket(bracket)

        d = d0 = float('inf')
        for n_iter in range(self.max_iter):

            if abs(a0 - 0.5 * (b0 + b1)) + 0.5 * (b1 - b0) <= 2 * self.tol:
                return a0, f0

            d1, d0 = d0, d

            # fit quadratic polynomial
            a = utils.as_tensor([0., a1-a0, a2-a0])
            quad = torch.stack([torch.ones_like(a), a, a.square()], -1)
            f = utils.as_tensor([f0, f1, f2]).unsqueeze(-1)
            quad = quad.pinverse().matmul(f).squeeze(-1)

            d = -0.5 * quad[1] / quad[2].clamp_min(self.tiny)
            d = d.item()
            a = a0 + d

            tiny = self.tiny * (1 + 2 * abs(a0))
            if abs(d) > abs(d1)/2 or not (b0 + tiny < a < b1 - tiny) or quad[-1] < 0:
                if a0 > 0.5 * (b0 + b1):
                    d = self.igold * (b0 - a0)
                else:
                    d = self.igold * (b1 - a0)
                a = a0 + d

            # check progress and update bracket
            f = closure(a)
            if f < f0:
                # f < f0 < f1 < f2
                b0, b1 = (b0, a0) if a < a0 else (a0, b1)
                (a0, f0), (a1, f1), (a2, f2) = (a, f), (a0, f0), (a1, f1)
            else:
                b0, b1 = (a, b1) if a < a0 else (b0, a)
                if f < f1:
                    # f0 < f < f1 < f2
                    (a0, f0), (a1, f1), (a2, f2) = (a0, f0), (a, f), (a1, f1)
                elif f < f2:
                    # f0 < f1 < f < f2
                    (a0, f0), (a1, f1), (a2, f2) = (a0, f0), (a1, f1), (a, f)

        return a0, f0


class BrentLineSearch(LineSearch):

    def repr_keys(self):
        keys = super().repr_keys() + ['c1', 'c2', 'max_iter']
        return keys

    def __init__(self, optim=None, lr=1, tol=1e-9, max_iter=128,
                 store_value=False, **kwargs):
        """

        Parameters
        ----------
        optim : `Optim`, optional
            Optimizer whose parameter is line-searched
        lr : float, default=1
            Learning rate (= first step size tried)
        tol : float, default=1e-9
            Tolerance for early stopping
        max_iter : int, default=128
            Maximum number of iterations
        store_value : bool or float, default=False
            Whether to store the successful parameter in `optim`.
            If False, the original value is restored at the end of the
            line search. If a float, the successful parameter is modulated
            by that value before being stored.
        """
        super().__init__(optim, **kwargs)
        self.brent = Brent(lr=lr, tol=tol, max_iter=max_iter)
        self.store_value = store_value

    tol = property(lambda self: self.wolfe.tol)
    max_iter = property(lambda self: self.wolfe.max_iter)
    lr = property(lambda self: self.wolfe.lr)

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
        loss, *derivatives = self._get_loss(param, closure, **kwargs)
        delta = optim.search_direction(*derivatives)

        def closure1d(a, **kwargs):
            return closure_line_search(param.add(delta, alpha=a), **kwargs)

        step_size, loss = self.brent.step(0, closure1d, loss=loss)
        delta.mul_(step_size)
        param.add_(delta)

        if self.store_value:
            self.brent.lr = step_size

        return (param, loss, *derivatives)

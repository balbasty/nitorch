from .base import ZerothOrder
from .brent import Brent
from nitorch.core import utils
import torch


class Powell(ZerothOrder):

    def __init__(self, lr=0.02, tol=1e-9, max_iter=512, sub_iter=128):
        """

        Parameters
        ----------
        lr : float, default=0.02
            Learning rate
        tol : float, default=1e-9
            Tolerance
        max_iter : int, default=512
            Maximum number of iterations
        sub_iter : int, default=128
            Maximum number of line search iterations
        """
        super().__init__(lr, tol=tol, max_iter=max_iter)
        self.brent = Brent(max_iter=sub_iter, tol=tol)
        self.delta = None

    def step(self, param=None, closure=None, **kwargs):

        if param is None:
            param = self.param
        if closure is None:
            closure = self.closure
        x = param

        import inspect
        if 'in_line_search' in inspect.signature(closure).parameters:
            closure_ls = lambda *a, **k: closure(*a, **k, in_line_search=True)
        else:
            closure_ls = closure

        if self.delta is None:
            self.delta = torch.eye(len(x), **utils.backend(x)).mul_(self.lr)

        f = kwargs.get('loss', closure(x)).item()
        x0, f0 = x.clone(), f
        i_largest_step = len(x)
        largest_step = 0
        for i in range(len(x)):
            fi = f
            a, f = self.brent.iter(x, f, self.delta[i], closure_ls)
            x.add_(self.delta[i], alpha=a)
            step = abs(f - fi)
            if step > largest_step:
                largest_step = step
                i_largest_step = i
            # for verbosity only
            closure(x)
        # repeat the same step and see if we improve
        f1 = closure(2 * x - x0).item()  # x + (x - x0)
        if f1 < f:
            delta1 = x - x0
            a, f = self.brent.iter(x, f, delta1, closure_ls)
            x.add_(delta1, alpha=a)
            self.delta[i_largest_step].copy(delta1).mul_(a)
            # for verbosity only
            closure(x)

        f = torch.as_tensor(f, **utils.backend(x))
        return x, f


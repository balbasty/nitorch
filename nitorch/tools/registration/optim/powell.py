from .base import ZerothOrder
from .brent import Brent
from .utils import get_closure_ls
from nitorch.core import utils
import torch


class Powell(ZerothOrder):

    def __init__(self, lr=0.02, tol=1e-9, max_iter=512, sub_iter=128,
                 **kwargs):
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
        super().__init__(lr, tol=tol, max_iter=max_iter, **kwargs)
        self.brent = Brent(max_iter=sub_iter, tol=tol)
        self.delta = None

    def reset_state(self):
        self.delta = None

    def step(self, param=None, closure=None, **kwargs):

        if param is None:
            param = self.param
        if closure is None:
            closure = self.closure
        closure_ls = get_closure_ls(closure)
        x = param

        # `x` may be a tensor or a wrapper that only implements
        # `len`/`add`/`add_` (e.g., `AffineModel.Parameters`), so the
        # step is accumulated in `dx` and applied once at the end.
        n = len(x)
        if self.delta is None:
            self.delta = torch.eye(n, **utils.backend(x)).mul_(self.lr)

        f = kwargs.get('loss', None)
        f = (closure(x) if f is None else f).item()
        f0 = f
        dx = torch.zeros(n, **utils.backend(x))
        i_largest_step = n
        largest_step = 0
        for i in range(n):
            fi = f
            closure_i = lambda a: closure_ls(x.add(dx.add(self.delta[i], alpha=a))).item()
            a, f = self.brent(0, closure_i, loss=f)
            if f < fi:
                dx.add_(self.delta[i], alpha=a)
                step = abs(fi - f)
                if step > largest_step:
                    largest_step = step
                    i_largest_step = i
            else:
                f = fi
        if f < f0:
            # repeat the same step and see if we improve
            f1 = closure_ls(x.add(dx, alpha=2)).item()  # x + (x - x0)
            if f1 < f:
                delta1 = dx.clone()
                closure_i = lambda a: closure_ls(x.add(dx.add(delta1, alpha=a))).item()
                a, f = self.brent(0, closure_i, loss=f)
                dx.add_(delta1, alpha=a)
                self.delta[i_largest_step].copy_(delta1).mul_(a)
        x.add_(dx)
        # for verbosity only
        closure(x)

        f = torch.as_tensor(f, **utils.backend(x))
        return x, f


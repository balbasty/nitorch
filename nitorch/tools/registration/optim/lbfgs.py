from .base import FirstOrder
from .wolfe import StrongWolfe
from .utils import get_closure_ls
import torch


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

        self.tol = kwargs.pop('tol', 1e-9)
        self.max_iter = max_iter
        self.history = history
        if wolfe is False:
            self.wolfe = False
        else:
            self.wolfe = StrongWolfe(**(wolfe or dict()))
        # state
        self._lr = self.lr
        self.delta = []  # s
        self.delta_grad = []  # y
        self.sy = []
        self.yy = []
        self.last_grad = 0

    def reset_state(self):
        # state
        self.delta = []  # s
        self.delta_grad = []  # y
        self.sy = []
        self.yy = []
        self.last_grad = 0
        self.lr = self._lr

    def step(self, param=None, closure=None, derivatives=False, **kwargs):
        if param is None:
            param = self.param
        if closure is None:
            closure = self.closure
        closure_ls = get_closure_ls(closure)

        loss, grad = self._get_loss(param, closure, **kwargs)

        if not self.sy:
            self.lr = self.lr * min(1., 1. / grad.abs().sum())
        delta = self.search_direction(grad, update=False)

        if self.wolfe:
            def closure1(x, *a, **k):
                return closure_ls(param.add(delta, alpha=x), *a, **k)
            step, loss, grad = self.wolfe(
                0, closure1, delta=delta, loss=loss, grad=grad)
            closure(param)  # to plot stuff / update loss state
            delta.mul_(step)
            self.lr *= step

        # print('up ', delta.tolist())
        self.update_state(grad, delta)
        param = self.update(delta, param)

        if not self.wolfe:
            loss, grad = closure(param, grad=True)

        return param, loss, grad

    def search_direction(self, grad, update=True):
        """Compute gradient direction, without Wolfe line search"""
        alphas = []
        delta = grad.neg()
        # print('   ', delta.tolist())
        for i in range(1, len(self.delta)+1):
            alpha = torch.dot(self.delta[-i].flatten(),
                              delta.flatten()) / self.sy[-i]
            alphas.append(alpha)
            delta.sub_(self.delta_grad[-i], alpha=alpha)
        alphas = list(reversed(alphas))
        if callable(self.preconditioner):
            delta = self.preconditioner(delta)
        elif self.preconditioner is not None:
            # assume diagonal
            delta.mul_(self.preconditioner)
        if self.sy:
            gamma = self.sy[-1] / self.yy[-1]
            delta *= gamma
        for i in range(len(self.delta)):
            beta = torch.dot(self.delta_grad[i].flatten(),
                             delta.flatten()) / self.sy[i]
            delta.add_(self.delta[i], alpha=alphas[i] - beta)
        # print('-> ', delta.tolist())
        delta = delta.mul_(self.lr)
        if update:
            self.update_state(grad, delta)
        return delta

    def update_state(self, grad, delta):
        """Update state"""
        delta_grad = grad - self.last_grad
        sy = torch.dot(delta_grad.flatten(), delta.flatten())
        if sy > 1e-10:
            yy = torch.dot(delta_grad.flatten(), delta_grad.flatten())
            if len(self.delta) == self.history:
                self.delta.pop(0)
                self.delta_grad.pop(0)
                self.sy.pop(0)
                self.yy.pop(0)
            self.delta_grad.append(delta_grad)
            self.last_grad = grad
            self.delta.append(delta)
            self.yy.append(yy)
            self.sy.append(sy)

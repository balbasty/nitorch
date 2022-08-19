import torch
import functorch
import math
from nitorch.core.linalg import sym_outer


class NonLinearLeastSquares:
    """Class that computes the gradient and (stabilized) Hessian of
    a nonlinear least-squares problem"""

    def __init__(self, model: "ForwardModel", fisher: bool = False):
        self.model = model
        self.fisher = fisher

    def loss(self, obs, *args, **kwargs):
        s = self.model.signal(*args, **kwargs)
        return 0.5 * (s - obs).square().sum()

    def grad(self, obs, *args, **kwargs):
        s = self.model.signal(*args, **kwargs)
        g = torch.stack(self.model.grad(*args, **kwargs))
        g *= (s - obs)
        s = 0.5 * (s - obs).square().sum()
        return s, g

    def hess(self, obs, *args, **kwargs):
        s = self.model.signal(*args, **kwargs)
        g = torch.stack(self.model.grad(*args, **kwargs))
        h = sym_outer(g)
        if not self.fisher:
            hh = torch.stack(self.model.hess(*args, **kwargs))
            h[..., :s.shape[-1]] += hh * (s - obs).abs()
        g *= (s - obs)
        s = 0.5 * (s - obs).square().sum()
        return s, g, h


class ForwardModel:
    """Base class for (MR, OCT, ...) forward models"""

    def signal(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, *args):
        argnums = [i for i in range(len(args)) if args[i] is not None]
        grads = [None] * len(args)

        sum_signal = lambda *a: self.signal(*a).sum()
        auto_grads = functorch.jacrev(sum_signal, argnums=argnums, in_dims=)
        auto_grads = auto_grads(*args)
        for i, grad in zip(argnums, auto_grads):
            grads[i] = grad

        return grads

    def hess(self, *args):
        args = torch.stack([a for a in args if a is not None], -1)
        argnums = [i for i in range(len(args)) if args[i] is not None]

        def sub_signal(x):
            xx = [None] * len(args)
            for i, j in zip(argnums, range(x.shape[-1])):
                xx[i] = x[..., j]
            return self.signal(*xx)

        auto_hess = functorch.vmap(functorch.hessian(sub_signal, argnums=0))
        auto_hess = auto_hess(args).abs_()
        if self.maj:
            auto_hess = auto_hess.sum(-1)
        else:
            auto_hess = auto_hess.diag(0, -1, -2)

        hess = [None] * len(args)
        for i, hess1 in zip(argnums, auto_hess):
            hess[i] = hess1

        return hess


class SpoiledGradientEcho(ForwardModel):
    """
                                     (1 - mt) * (1 - exp(-r1*tr))
        fit[0] = pd * sin(a) * ---------------------------------------
                                (1 - cos(a) * (1 - mt) * exp(-r1*tr))
        fit[te] = fit[0] * exp(-r2star * te)
    """

    def __init__(self, fa, tr, te=0, mt=False, log=True, maj=False):
        self.fa = fa
        self.tr = tr
        self.te = te
        self.mt = mt
        self.log = log
        self.maj = maj

    def exp(self, pd, r1, r2star=None, mtsat=None, transmit=None, receive=None):
        if self.log:
            pd = pd.exp()
            r1 = r1.exp()
            if r2star is not None:
                r2star = r2star.exp()
            if mtsat is not None:
                mtsat = mtsat.neg().exp_().add_(1).reciprocal_()
            if transmit is not None:
                transmit = transmit.exp()
            if receive is not None:
                receive = receive.exp()
        return pd, r1, r2star, mtsat, transmit, receive

    def signal(self, pd, r1, r2star=None, mtsat=None, transmit=None, receive=None):
        pd, r1, r2star, mtsat, transmit, receive \
            = self.exp(pd, r1, r2star, mtsat, transmit, receive)

        if receive is not None:
            pd = pd * receive
        if transmit is not None:
            fa = self.fa * transmit
            cosfa = fa.cos()
            sinfa = fa.sin()
        else:
            cosfa = math.cos(self.fa)
            sinfa = math.sin(self.fa)

        imtsat = 1 - mtsat if self.mt else 1
        e1 = (-self.tr * r1).exp()
        s = pd * sinfa * imtsat * (1 - e1) / (1 - cosfa * imtsat * e1)
        if self.te and r2star is not None:
            s = s * (-self.te * r2star).exp()
        return s

    def grad(self, pd, r1, r2star=None, mtsat=None, transmit=None, receive=None):
        return super().grad(pd, r1, r2star, mtsat, transmit, receive)

    def hess(self, pd, r1, r2star=None, mtsat=None, transmit=None, receive=None):
        return super().hess(pd, r1, r2star, mtsat, transmit, receive)
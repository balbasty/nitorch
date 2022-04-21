from .base import FirstOrder
import torch


class GradientDescent(FirstOrder):
    """Gradient descent

    Δ{k+1} = - η ∇f(x{k})
    x{k+1} = x{k} + Δ{k+1}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search_direction(self, grad):
        grad = self.precondition_(grad.clone())
        return grad.mul_(-self.lr)


class ConjugateGradientDescent(FirstOrder):
    """Conjugate Gradient descent

    Δ{k+1} = - ∇f(x{k})
    s{k+1} = β{k+1} s{k+1} + Δ{k+1}
    x{k+1} = x{k} + η s{k+1}

    Fletcher-Reeves:    β{k+1} = (Δ{k+1}'Δ{k+1}) / (Δ{k}'Δ{k})
    Polak–Ribière:      β{k+1} = (Δ{k+1}'(Δ{k+1} - Δ{k})) / (Δ{k}'Δ{k})
    Hestenes-Stiefel:   β{k+1} = (Δ{k+1}'(Δ{k+1} - Δ{k})) / (-s{k}'(Δ{k+1} - Δ{k}))
    Dai–Yuan:           β{k+1} = (Δ{k+1}'Δ{k+1}) / (-s{k}'(Δ{k+1} - Δ{k}))
    """

    def __init__(self, *args, **kwargs):
        beta = kwargs.pop('beta', 'pr').lower()
        super().__init__(*args, **kwargs)
        self.delta = 0  # previous search direction
        self.grad = 0   # previous (negative of) gradient
        beta = {'fr': 'fletcher_reeves', 'pr': 'polak_ribiere',
                'hs': 'hestenes_stiefel', 'dy': 'dai_yuan'}.get(beta, beta)
        self.beta = getattr(self, beta)

    def reset_state(self):
        self.delta = 0
        self.grad = 0

    def fletcher_reeves(self, grad):
        gg0 = self.grad.flatten().dot(self.grad.flatten())
        gg = grad.flatten().dot(grad.flatten())
        return gg / gg0

    def polak_ribiere(self, grad):
        gg0 = self.grad.flatten().dot(self.grad.flatten())
        gg = grad.flatten().dot(grad.flatten() - self.grad.flatten())
        return gg / gg0

    def hestenes_stiefel(self, grad):
        diff = grad - self.grad
        gg0 = -self.delta.flatten().dot(diff.flatten())
        gg = grad.flatten().dot(diff.flatten())
        return gg / gg0

    def dai_yuan(self, grad):
        diff = grad - self.grad
        gg0 = -self.delta.flatten().dot(diff.flatten())
        gg = grad.flatten().dot(grad.flatten())
        return gg / gg0

    def search_direction(self, grad):
        grad = self.precondition_(grad.clone()).neg_()
        if not torch.is_tensor(self.delta):
            self.delta = grad
        else:
            beta = self.beta(grad).clamp_min_(0)
            self.delta.mul_(beta).add_(grad)
        self.grad = grad
        delta = self.delta
        if self.lr != 1:
            delta = self.lr * delta
        return delta

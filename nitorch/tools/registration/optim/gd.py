from .base import FirstOrder


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

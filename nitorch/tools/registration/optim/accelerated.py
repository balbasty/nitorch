from . import FirstOrder
import torch


class AcceleratedFirstOrder(FirstOrder):
    """Base class for accelerated methods.

    Accelerated methods store the previous step taken and use it
    when computing the next step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 0

    def update_lr(self, lr):
        if torch.is_tensor(self.delta):
            self.delta.mul_(lr / self.lr)
        self.lr = lr

    def reset_state(self):
        self.delta = 0


class Momentum(AcceleratedFirstOrder):
    """Gradient descent with momentum

    Δ{k+1} = α Δ{k} - η ∇f(x{k})
    x{k+1} = x{k} + Δ{k+1}
    """

    def __init__(self, *args, **kwargs):
        momentum = kwargs.pop('momentum', 0.9)
        super().__init__(*args, **kwargs)
        self.n_iter = 0
        self.momentum = momentum

    def search_direction(self, grad):
        grad = self.precondition(grad)
        # momentum
        if torch.is_tensor(self.delta):
            self.delta.mul_(self.momentum)
            self.delta.sub_(grad, alpha=self.lr)
        else:
            self.delta = grad.mul(-self.lr)
        return self.delta.clone()


class Nesterov(AcceleratedFirstOrder):
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

    def __init__(self, *args, momentum=0, auto_restart=True, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.auto_restart = auto_restart
        self.theta = 1
        self.momentum = self._momentum = momentum

    def reset_state(self):
        self.restart()

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

    def update_momentum(self):
        theta_prev = self.theta
        theta = 0.5 * (1 + (1 + 4 * self.theta * self.theta) ** 0.5)
        self.theta = theta
        self._momentum = self.momentum or (theta_prev - 1) / theta

    def search_direction(self, grad):
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


class OGM(AcceleratedFirstOrder):
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
        self.theta = 1
        self.momentum = self._momentum = momentum
        self.relaxation = self._relaxation = relaxation
        self._grad = 0

    def reset_state(self):
        self.restart()

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
        elif (self.damping != 1 and torch.is_tensor(self._grad) and
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

    def search_direction(self, grad):
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
        if not torch.is_tensor(self._grad):
            self._grad = grad.mul(self.lr)
        else:
            self._grad.copy_(grad).mul_(self.lr)
        return step

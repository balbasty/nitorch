import torch
from torch.optim.optimizer import Optimizer


class OGM(Optimizer):
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

    def __init__(self, params, lr=1e-3, momentum=0, relaxation=0,
                 dampening=0.5, weight_decay=0, precond=None,
                 adaptive_restart=False):
        """

        Parameters
        ----------
        lr : float, default=1e-3
            Learning rate
        momentum : float, optional
            Momentum Factor that modulates the weight of the previous step.
            By default, the (t_k) sequence from Kim & Fessler is used.
        relaxation : float, optional
            Over-relaxation factor that modulates the weight of the
            previous gradient.
            By default, the (t_k) sequence from Kim & Fessler is used.
        dampening : float in (0..1), default=0.5
            Damping of the relaxation factor when consecutive gradients
            disagree (1 == no damping).
        lr : float, default=1
            Learning rate.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if relaxation < 0.0:
            raise ValueError("Invalid relaxation value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, relaxation=relaxation,
                        dampening=dampening, weight_decay=weight_decay,
                        precond=precond, adaptive_restart=adaptive_restart)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            relaxation = group['relaxation']
            lr = group['lr']
            precond = group['precond']
            adaptive_restart = group['adaptive_restart']

            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)
                    if callable(precond):
                        grad = precond(grad)
                    elif precond is not None:
                        grad = grad * precond

                    state = self.state[p]

                    # update parameters
                    theta_prev = state.get('theta', 1)
                    theta = 0.5 * (1 + (1 + 4 * theta_prev * theta_prev) ** 0.5)
                    prev_momentum = momentum or state.get('momentum', momentum)
                    _momentum = momentum or (theta_prev - 1) / theta
                    prev_relaxation = relaxation or state.get('relaxation', relaxation)
                    _relaxation = relaxation or theta_prev / theta
                    _dampening = state.get('dampening', dampening)

                    # update delta (y{k+1} - y{k})
                    delta = state.get('delta_buffer', 0)
                    grad_prev = state.get('grad_buffer', 0)

                    if torch.is_tensor(delta):
                        delta.mul_(prev_momentum)
                        delta.sub_(grad, alpha=lr)
                        delta.sub_(grad_prev, alpha=lr * prev_relaxation)
                    else:
                        delta = grad.mul(-lr)

                    if adaptive_restart and grad.flatten().dot(delta.flatten()) >= 0:
                        # Kim & Fessler (2017) 4.2 (26)
                        # gradient and step disagree
                        theta = 1
                        _momentum = momentum
                        _relaxation = relaxation
                        grad_prev = 0
                        _dampening = 1
                        delta = grad.mul(-lr)
                    elif (dampening != 1 and torch.is_tensor(grad_prev) and
                          grad.flatten().dot(grad_prev.flatten()) < 0):
                        # Kim & Fessler (2017) 4.3 (27)
                        # consecutive gradients disagree
                        _dampening *= dampening
                    _relaxation *= _dampening

                    # update step (x{k+1} - x{k})
                    step = delta.mul(_momentum)
                    step = step.sub_(grad, alpha=lr * (1 + _relaxation))
                    p.add_(step)

                    # save buffers
                    state['relaxation'] = _relaxation
                    state['momentum'] = _momentum
                    state['theta'] = theta
                    state['dampening'] = _dampening
                    state['delta_buffer'] = delta
                    if torch.is_tensor(grad_prev):
                        grad = grad_prev.copy_(grad).mul_(lr)
                    else:
                        grad = grad.mul(lr)
                    state['grad_buffer'] = grad

        return loss

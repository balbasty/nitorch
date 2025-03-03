"""This file implements generic optimizers that do not require autograd.
They take evaluated first and (optionally) second derivatives and return
a step. A few classes also require the ability to evaluate the function
(e.g., LBFGS), in which case a closure function that allows the objective
and its derivatives to be computed must be provided.

Most optimizers, however, simply use derivatives; and a wrapper class
that iterates multiple such steps is provided (IterOptim).
Available optimizers are:
- Powell
- GradientDescent
- Momentum
- Nesterov
- OGM
- GaussNewton
- GridCG
- GridRelax
- LBFGS
Along with helpers
- StepSizeLineSearch
- BacktrackingLineSearch
- StrongWolfeLineSearch
- BrentLineSearch
- IterateOptim
- IterateOptimInterleaved
"""


def _make_search(search, second_order=False, **opt):
    from .linesearch import make_search
    return make_search(search, second_order=second_order, **opt)


class Optim:
    """Base class for optimizers"""

    def repr_keys(self):
        return ['lr']

    def __init__(self, lr=1, param=None, closure=None,
                 search=None, iter=None, **opt):
        """

        Parameters
        ----------
        lr : float, default=1
        param : tensor, optional
        closure : callable(x, [grad=True], [hess=True]) -> scalar tensor
        search : LineSearch or int, optional
        iter : OptimIterator or int, optional
        opt
        """
        self.lr = lr
        self.param = param
        self.closure = closure
        opt_iter = ('max_iter', 'tol', 'stop')
        opt_iter = {k: opt[k] for k in opt_iter if k in opt}
        if iter is None and not isinstance(self, OptimWrapper):
            iter = OptimIterator(**opt_iter)
        self._set_iter(iter)
        opt_ls = ('max_ls',)
        opt_ls = {k: opt[k] for k in opt_ls if k in opt}
        self._set_search(search, **opt_ls)

    @property
    def iter(self):
        return self._iter or self.step

    def _set_iter(self, x):
        self._iter = x
        if self._iter:
            self._iter.optim = self

    @iter.setter
    def iter(self, x):
        self._set_iter(x)

    @property
    def search(self):
        return self._search

    def _set_search(self, x, **opt):
        self._search = _make_search(x, **opt, second_order=self.requires_hess)
        if self._search:
            self._search.optim = self

    @search.setter
    def search(self, x):
        self._set_search(x)

    def __call__(self, *args, **kwargs):
        return self.iter(*args, **kwargs)

    def reset_state(self):
        pass

    def update(self, step, param=None):
        """Apply a step to the current parameters"""
        if param is None:
            param = self.param
        param.add_(step)
        return param

    def search_direction(self, *derivatives):
        raise NotImplementedError

    def _get_loss(self, param, closure, **kwargs):
        loss = kwargs.get('loss', None)
        if any(self.requires[key] and kwargs.get(key, None) is None
               for key in self.requires):
            loss, *derivatives = closure(param, **self.requires)
        else:
            derivatives = [kwargs[key] for key in ('grad', 'hess')
                           if self.requires.get(key, False)]
        if loss is None:
            loss = closure(param)
        return (loss, *derivatives)

    def step(self, param=None, closure=None, **kwargs):
        """

        Parameters
        ----------
        param : [list of] tensor, default=self.param
            Current state of the optimized parameters
        closure : callable([list of] tensor) -> tensor, default=self.closure
            Function that takes optimized parameters as inputs and
            returns the objective function.

        Returns
        -------
        param : [list of] tensor
            Updated parameters.
        loss : () tensor
            Objective function.
        *derivatives : tensor
            Derivatives, if required.

        """
        if self.search:
            return self.search.step(param, closure, **kwargs)

        if param is None:
            param = self.param
        if closure is None:
            closure = self.closure
        loss, *derivatives = self._get_loss(param, closure, **kwargs)
        delta = self.search_direction(*derivatives)
        param = self.update(delta, param)
        loss = closure(param, **self.requires)
        if any(self.requires.values()):
            loss, *derivatives = loss
        return (param, loss, *derivatives)

    requires_grad = False
    requires_hess = False

    @property
    def requires(self):
        d = {}
        if self.requires_grad:
            d['grad'] = True
        if self.requires_hess:
            d['hess'] = True
        return d

    def _repr_simple(self):
        s = [f'{key}={getattr(self, key)}' for key in self.repr_keys()]
        if self._search and not isinstance(self, OptimWrapper):
            s += [f'ls={self.search._repr_simple()}']
        if self._iter and not isinstance(self, OptimWrapper):
            s += [f'iter={self.iter._repr_simple()}']
        s = ', '.join(s)
        return f'{type(self).__name__}({s})'

    def _repr(self):
        return self._repr_simple()

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return repr(self)


def _wrapped_prop(key, sub='optim'):
    def get(self):
        sub_ = getattr(self, sub)
        if isinstance(sub_, (list, tuple)):
            return [getattr(opt, key) for opt in sub_]
        elif sub_:
            return getattr(sub_, key)
        else:
            return None
    def set(self, value):
        sub_ = getattr(self, sub)
        if isinstance(sub_, (list, tuple)):
            for opt in sub_:
                setattr(opt, key, value)
        elif sub_:
            setattr(sub_, key, value)
    return property(get, set)


class OptimWrapper(Optim):
    """Utilities for Optimizers that wrap another optimizer"""

    def __init__(self, optim=None, *args, **kwargs):
        self.optim = optim
        super().__init__(*args, **kwargs)

    requires_grad = _wrapped_prop('requires_grad')
    requires_hess = _wrapped_prop('requires_hess')
    requires = _wrapped_prop('requires')
    param = _wrapped_prop('param')
    closure = _wrapped_prop('closure')

    def reset_state(self):
        if self.optim:
            self.optim.reset_state()

    def update(self, *a, **k):
        optim = k.pop('optim', self.optim)
        return optim.update(*a, **k)

    def search_direction(self, *a, **k):
        optim = k.pop('optim', self.optim)
        return optim.search_direction(*a, **k)

    def step(self, *a, **k):
        optim = k.pop('optim', self.optim)
        return optim.step(*a, **k)

    def _repr(self):
        s = self._repr_simple()
        if self.optim is not None:
            s += ' o [' + repr(self.optim) + ']'
        return s


class OptimIterator(OptimWrapper):
    """Wrapper that performs multiple steps of an optimizer.

    Backtracking line search can be used.
    """

    def __init__(self, optim=None, max_iter=20, tol=1e-9, stop='diff',
                 keep_state=True, **kwargs):
        """

        Parameters
        ----------
        optim : Optim
            Optimizer
        max_iter : int, default=20
            Maximum number of iterations
        tol : float, default=1e-9
            Tolerance for early stopping
        stop : {'diff', 'gain', 'grad'}, default='diff'
            Stopping criterion. Can be a combination.

        """
        super().__init__(optim, **kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self.stop = stop
        self.keep_state = keep_state
        self.loss_max = -float('inf')
        self.loss_prev = float('inf')

    def __call__(self, *args, **kwargs):
        return self.iter(*args, **kwargs)

    def derivatives_to_dict(self, *derivatives):
        d = dict()
        for drv, name in zip(derivatives, ('loss', 'grad', 'hess')):
            d[name] = drv
        return d

    def step(self, param=None, closure=None, **kwargs):
        """Perform multiple optimization iterations

        Parameters
        ----------
        param : tensor
            Initial guess (will be modified inplace)
        closure : callable(tensor, [grad=False], [hess=False]) -> Tensor, *Tensor
            Function that takes a parameter and returns the objective and
            its (optionally) gradient evaluated at that point.

        Returns
        -------
        param : tensor
            Updated parameter
        loss : tensor
            Updated loss
        *derivatives : tensor, if `derivatives`
            derivatives

        """
        if param is None:
            param = self.optim.param
        if closure is None:
            closure = self.optim.closure

        optim = kwargs.get('optim', self.optim)
        loss, *derivatives = self._get_loss(param, closure, **kwargs)

        loss_prev = loss_max = loss
        for n_iter in range(1, self.max_iter+1):

            # perform step
            step = optim.step if optim.iter is self else optim.iter
            param, loss, *derivatives = step(
                param, closure,
                **self.derivatives_to_dict(loss, *derivatives))

            # check convergence
            stop = abs(loss_prev-loss)
            if self.stop == 'gain':
                denom = abs(loss_max-loss)
                denom = max(max(denom, 1e-9 * abs(loss_max)), 1e-9)
                stop = stop / denom
            if stop < self.tol:
                break
            loss_prev, loss_max = loss, max(loss, loss_max)

        return (param, loss, *derivatives)

    def repr_keys(self):
        return ['max_iter', 'tol', 'stop']


class InterleavedOptimIterator(OptimIterator):
    """Interleave optimizers (for block coordinate descent)"""

    def __init__(self, optim=None, max_iter=20, tol=1e-9, stop='gain', **kwargs):
        """

        Parameters
        ----------
        optim : list[IterateOptim]
            Optimizer for each variable
        max_iter : int, default=20
            Maximum number of outer loops
        tol : float, default=1e-9
            Outer loop tolerance for early stopping

        """
        super().__init__(list(optim), max_iter, tol, stop, **kwargs)

    def __getitem__(self, item):
        return self.optim[item]

    def __iter__(self):
        for optim in self.optim:
            yield optim

    def __len__(self):
        return len(self.optim)

    def reset_state(self):
        for optim in self.optim:
            optim.reset_state()

    def step(self, param=None, closure=None, **kwargs):
        if param is None:
            param = self.param
        if closure is None:
            closure = self.closure
        optim = kwargs.get('optim', self.optim)
        loss_prev, loss_max = float('inf'), -float('inf')
        outputs = [(p,) for p in param]
        for n_iter in range(1, self.max_iter+1):
            outputs0, outputs = outputs, []
            for opt, cls, out in zip(optim, closure, outputs0):
                step = opt.step if opt.iter is self else opt.iter
                out = step(out[0], cls, **self.derivatives_to_dict(*out[1:]))
                outputs.append(out)

            # check convergence
            loss = outputs[-1][1]
            stop = abs(loss_prev-loss)
            if self.stop == 'gain':
                denom = max(abs(loss_max-loss), 1e-9)
                stop /= denom
            if stop < self.tol:
                break
            loss_prev = loss
            loss_max = max(loss, loss_max)

        return list(zip(*outputs))

    def _repr(self):
        s = self._repr_simple()
        sep = (len(s) + 4) * ' ' + ',\n'
        if self.optim is not None:
            s += ' o [' + sep.join(map(repr, self.optim)) + ']'
        return s


class ZerothOrder(Optim):
    requires_grad = False
    requires_hess = False


class FirstOrder(Optim):

    def __init__(self, lr=1, preconditioner=None, **opt):
        super().__init__(lr, **opt)
        self.preconditioner = preconditioner

    def precondition(self, grad):
        if callable(self.preconditioner):
            grad = self.preconditioner(grad)
        elif self.preconditioner is not None:
            # assume diagonal
            grad = grad.mul(self.preconditioner)
        return grad

    def precondition_(self, grad):
        if callable(self.preconditioner):
            grad = self.preconditioner(grad)
        elif self.preconditioner is not None:
            # assume diagonal
            grad = grad.mul_(self.preconditioner)
        return grad

    requires_grad = True
    requires_hess = False


class SecondOrder(Optim):
    requires_grad = True
    requires_hess = True


class MultiOptim(Optim):
    """A group of optimizers"""

    def __init__(self, optims, **opt):
        """

        Parameters
        ----------
        optims : list of dict or `Optim` types
            If a dict, it may have the key 'optim' to specify which
            optimizer to use. Other keys are parameters passed to the
            `Optim` constructor.
        opt : dict
            Additional parameters shared by all optimizers.
        """
        super().__init__()
        ooptims = []
        for optim in optims:
            if isinstance(optim, dict):
                ooptim = dict(opt)
                ooptim.update(optim)
                klass = ooptim.pop('optim', None)
            else:
                klass = optim
                ooptim = dict(opt)
                ooptim.pop('optim')
            if not klass:
                raise ValueError('No optimizer provided')
            ooptims.append(klass(**ooptim))
        self.optims = ooptims

    def search_direction(self, grad):
        """Perform an optimization step.

        Parameters
        ----------
        grad : list[tensor]

        Returns
        -------
        step : list[tensor]

        """
        deltas = []
        for param, g in zip(self.optims, grad):
            deltas.append(param.search_direction(g))
        return deltas

    def update(self, step, param=None):
        """

        Parameters
        ----------
        step : list[tensor]
        param : list[tensor]

        Returns
        -------
        params : list[tensor]

        """
        for p, s in zip(param, step):
            p.add_(s)
        return param

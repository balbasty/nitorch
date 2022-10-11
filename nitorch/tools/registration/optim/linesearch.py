from .base import OptimWrapper
import copy


def make_search(search, second_order=False, **opt):
    if isinstance(search, int):
        opt['max_ls'] = search
        search = None
    if not search:
        if not opt.get('max_ls', 0):
            return None
        elif second_order:
            return BacktrackingLineSearch(**opt)
        else:
            return StepSizeLineSearch(**opt)
    elif search == 'wolfe':
        from .wolfe import StrongWolfeLineSearch
        return StrongWolfeLineSearch(**opt)
    elif search == 'brent':
        from .brent import BrentLineSearch
        return BrentLineSearch(**opt)
    if isinstance(search, str):
        raise ValueError(f'Unknown line search "{search}"')
    return search


class LineSearch(OptimWrapper):
    """Base class for line searches.
    Line searches are heuristics that are often called after the search
    direction is found. Some line search only look at the loss value
    (e.g., backtracking line searches generally try to find the largest
    step size --up to a limit -- that improves the loss) while other look at
    the gradient as well (e.g. a Wolfe line search looks for a step size
    that brings the loss in a "nice" region).
    """
    def __init__(self, *args, **kwargs):
        kwargs['max_ls'] = 0
        super().__init__(*args, **kwargs)


class NoLineSearch(LineSearch):
    pass


class BacktrackingLineSearch(LineSearch):
    """Meta-optimizer that performs a backtracking line search on the
    value of an optimizer's parameter. The search direction is recomputed
    for each new parameter value, so this is not purely a "line" search.
    But it can be seen as a 1D optimizer.
    """

    def repr_keys(self):
        keys = super().repr_keys() + ['max_iter']
        if self.key != 'lr':
            keys += ['key', 'factor']
        return keys

    def __init__(self, optim=None, max_iter=6, key='lr', factor=0.5,
                 store_value=False, **kwargs):
        """

        Parameters
        ----------
        optim : `Optim`, optional
            Optimizer whose parameter is line-searched
        max_iter : int, default=6
            Maximum number of searches
        key : str, default='lr'
            Name of the parameter to search.
            It should be a mutable attribute of `optim`.
        factor : float, default=0.5
            The searched parameter is multiplied by this factor after
            each failed iteration.
        store_value : bool or float, default=False
            Whether to store the successful parameter in `optim`.
            If False, the original value is restored at the end of the
            line search. If a float, the successful parameter is modulated
            by that value before being stored.
        """
        super().__init__(optim, **kwargs)
        self.key = key
        self.factor = factor
        self.max_iter = max_iter
        self.store_value = store_value

    def step(self, param=None, closure=None, **kwargs):

        import inspect
        if 'in_line_search' in inspect.signature(closure).parameters:
            closure_line_search = lambda *a, **k: closure(*a, **k, in_line_search=True)
        else:
            closure_line_search = closure

        if param is None:
            param = self.param
        if closure is None:
            closure = self.closure
        optim = kwargs.get('optim', self.optim)

        # compute loss and derivatives
        loss, *derivatives = self._get_loss(param, closure, **kwargs)

        # line search
        success = False
        loss0, param0, optim0 = loss, param, optim
        value0 = getattr(self.optim, self.key)
        value = value0
        for n_iter in range(self.max_iter):

            # Clone the optimizer in case step changes its state
            optim = copy.deepcopy(optim0)
            setattr(optim, self.key, value)

            # Perform step
            delta = optim.search_direction(*derivatives)
            param = optim.update(param0.clone(), delta)
            loss = closure_line_search(param)

            if loss < loss0:
                success = True
                break
            else:
                value = self.factor * value

        if success:
            if self.store_value:
                setattr(self.optim, self.key, value * self.store_value)
            else:
                setattr(self.optim, self.key, value0)
            param = param0.copy_(param)
            if optim0 is self.optim:
                self.optim = optim
            print('success')
            loss, *derivatives = closure(param, **self.requires)
        else:
            print('failure')
            param, loss = param0, loss0
        return (param, loss, *derivatives)


class StepSizeLineSearch(BacktrackingLineSearch):
    """Backtracking line search that only look at the step size.
    The search direction is fixed a priori.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, param=None, closure=None, **kwargs):

        import inspect
        if 'in_line_search' in inspect.signature(closure).parameters:
            closure_line_search = lambda *a, **k: closure(*a, **k, in_line_search=True)
        else:
            closure_line_search = closure

        if param is None:
            param = self.param
        if closure is None:
            closure = self.closure
        optim = kwargs.get('optim', self.optim)

        # compute loss and derivatives
        loss, *derivatives = self._get_loss(param, closure, **kwargs)
        delta = optim.search_direction(*derivatives)

        # line search step size
        success = False
        loss0, param0, lr0 = loss, param, optim.lr
        lr = lr0
        for n_iter in range(self.max_iter):

            param = param0.add(delta, alpha=lr)
            loss = closure_line_search(param.add(delta, alpha=lr))

            if loss < loss0:
                success = True
                break
            else:
                lr = self.factor * lr

        if success:
            if self.store_value:
                self.lr = lr * self.store_value
            param = param0.copy_(param)
            loss, *derivatives = closure(param, **self.requires)
        else:
            param, loss = param0, loss0
        return (param, loss, *derivatives)


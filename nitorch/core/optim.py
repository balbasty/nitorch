# -*- coding: utf-8 -*-
"""Optimisers."""

import torch

# TODO:
# . Implement CG/RELAX as autograd functions.
#   > Since A is positive definite, inv(A)' = inv(A') = inv(A)
#   > The gradient of (A\b).dot(x) w.r.t. b is A\x, which can be solved
#     (again) by CG


def cg(A, b, x=None, precond=lambda y: y, max_iter=None,
       tolerance=1e-5, verbose=False, sum_dtype=torch.float64,
       inplace=True, stop='residuals'):
    """ Solve A*x = b by the conjugate gradient method.

        The method of conjugate gradients solves linear systems of
        the form A*x = b, where A is positive-definite.

    Parameters
    ----------
    A : tensor or callable
        Linear operator.
        If a function: should take a tensor with the same shape as `b` and
        return a tensor with the same shape as `b`.
    b : tensor
        Right hand side 'vector'.
    x : tensor, optional, default=0
        Initial guess.
    precond : callable, default=identity
        Preconditioner.
    max_iter : int, default=len(b)*10
        Maximum number of iteration.
    tolerance : float, default=1e-5
        Tolerance for early-stopping.
    verbose : bool, default = False
        Write something at each iteration.
    sum_dtype : torch.dtype, default=float64
        Choose `float32` for speed or `float64` for precision.
    inplace : bool, default=True
        Perform computations inplace (saves performance but overrides
        `x` and may break the computational graph).
    stop : {'residuals', 'norm'}, default='residuals'
        What stopping criteria to use.
        The squared residuals ('residuals') are not motonically decreasing,
        whilst the norm induced by the A-weighted scalar product ('norm') is.
        If monotinicity is important then select 'norm'. However, this requires
        an additional evaluation of A.

    Returns
    -------
    x : tensor
        Solution of the linear system.

    Note
    ----
    In practice, if A is provided as a function, b and x do not need
    to be vector-shaped.

    Example
    -------
    ```python
    >>> # Let's solve Ax = b using both regular inversion and CG
    >>> import torch
    >>> from nitorch.core.optim import cg
    >>> from timeit import default_timer as timer
    >>> # Simulate A and b
    >>> N = 100
    >>> A = torch.randn(N, N)
    >>> b = torch.randn(N, 1)
    >>> # Make A symmetric and pos. def.
    >>> U, S, _ = torch.svd(A)
    >>> A = U.matmul((S + S.max()).diag().matmul(U.t()))
    >>> # Solve by inversion
    >>> t0 = timer()
    >>> x1 = torch.solve(b, A)[0]
    >>> print('A.inv*b | elapsed time: {:0.4f} seconds'.format(timer() - t0))
    >>> # Solve by CG
    >>> t0 = timer()
    >>> x2 = cg(A, b, verbose=True, sum_dtype=torch.float32)
    >>> print('cg(A, b) | elapsed time: {:0.4f} seconds'.format(timer() - t0))
    >>> # Inspect errors
    >>> e1 = torch.sqrt(torch.sum((x1 - x2) ** 2))
    >>> print(e1)
    >>> e2 = torch.sqrt(torch.sum((b - A.matmul(x2)) ** 2))
    >>> print(e2)
    ```

    """
    # Format arguments
    device = b.device
    dtype = b.dtype
    if max_iter is None:
        max_iter = len(b) * 10
    if x is None:
        x = torch.zeros_like(b)
    elif not inplace:
        x = x.clone()

    # Create functor if A is a tensor
    if isinstance(A, torch.Tensor):
        A_tensor = A
        def A_function(x):
            return A_tensor.mm(x)
        A = A_function

    # Initialisation
    r = b - A(x)  # Residual: b - A*x
    z = precond(r)  # Preconditioned residual
    rz = torch.sum(r * z, dtype=sum_dtype)  # Inner product of r and z
    p = z.clone()  # Initial conjugate directions p
    beta = torch.tensor(0, dtype=dtype, device=device)  # Initial step size
    
    if tolerance or verbose:
        if stop == 'residuals':
            obj0 = torch.sqrt(rz)
        else:
            obj0 = A(x) - 2 * b
            obj0 = torch.sum(0.5 * x * obj0, dtype=sum_dtype)
        if verbose:
            s = '{:' + str(len(str(max_iter+1))) + '} | {} = {:12.6g}'
            print(s.format(0, stop, obj0))
        obj = torch.zeros(max_iter+1, dtype=sum_dtype, device=device)
        obj[0] = obj0

    # Run algorithm
    for n_iter in range(1, max_iter+1):
        # Calculate conjugate directions P (descent direction)
        p *= beta
        p += z
        # Find the step size of the conj. gradient descent
        Ap = A(p)
        alpha = rz / torch.sum(p * Ap, dtype=sum_dtype)
        # Perform conj. gradient descent, obtaining updated X and R, using the
        # calculated P and alpha
        x += alpha * p
        r -= alpha * Ap
        # Update preconditioned residual
        z = precond(r)
        # Finds the step size for updating P
        rz0 = rz
        rz = torch.sum(r * z, dtype=sum_dtype)
        beta = rz / rz0
        
        # Check convergence
        if tolerance or verbose:
            if stop == 'residuals':
                obj1 = torch.sqrt(rz)
            else:
                obj1 = A(x) - 2 * b
                obj1 = torch.sum(0.5 * x * obj1, dtype=sum_dtype)
            obj[n_iter] = obj1
            gain = get_gain(obj[:n_iter + 1], monotonicity='decreasing')
            if verbose:
                s = '{:' + str(len(str(max_iter+1))) + '} | {} = {:12.6g} | gain = {:12.6g}'
                print(s.format(n_iter, stop, obj[n_iter], gain))
            if gain.abs() < tolerance:
                break

    return x


def get_gain(obj, monotonicity='increasing'):
    """ Compute gain of some objective function.

    Args:
        obj (torch.tensor): Vector of values (e.g., loss).
        direction (string, optional): Monotonicity of values ('increasing'/'decreasing'),
            defaults to 'increasing'.

    Returns:
        gain (torch.tensor): Computed gain.

    """
    if len(obj) <= 1:
        return torch.tensor(float('inf'), dtype=obj.dtype, device=obj.device)
    if monotonicity == 'increasing':
        gain = (obj[-1] - obj[-2])
    elif monotonicity == 'decreasing':
        gain = (obj[-2] - obj[-1])
    else:
        raise ValueError('Undefined monotonicity')
    gain = gain / (torch.max(obj) - torch.min(obj))
    return gain


def plot_convergence(vals, fig_ax=None, fig_num=1, fig_title='Model convergence',
                     xlab='', ylab='', legend=None):
    """ Plots an algorithm's convergence (e.g. negative log-likelihood, lower bound).

    Allows for real-time plotting if giving returned fig_ax objects as input.

    Args:
        vals (torch.tensor): Values to be plotted (N,) | (N, C).
        fig_ax ([matplotlib.figure, matplotlib.axes])
        fig_num (int, optional): Figure number to plot to, defaults to 1.
        fig_title (str, optional): Figure title, defaults to 'Model convergence'.
        xlab (str, optional): x-label, defaults to ''.
        ylab (str, optional): y-label, defaults to ''.
        legend (list(str)): Figure legend, list with C strings. Defaults to None.

    Returns:
        fig_ax ([matplotlib.figure, matplotlib.axes])

    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if fig_ax is None:
        fig, ax = plt.subplots(1, 2, num=fig_num)
        fig_ax = [fig ,ax]
        plt.ion()
        fig.show()

    # Get figure and axis objects
    fig = fig_ax[0]
    ax = fig_ax[1]

    vals = vals.detach().cpu()  # To CPU

    ax[0].clear()
    x = torch.arange(0, len(vals)) + 1
    ax[0].plot(x, vals)
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel(ylab)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid()

    ax[1].clear()
    x = torch.arange(0, len(vals)) + 1
    lines = ax[1].plot(x[-3:, ...], vals[-3:, ...])
    ax[1].set_xlabel(xlab)
    ax[1].set_ylabel(ylab)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid()

    if legend is not None:
        # Add legend
        plt.legend(lines, legend)

    fig.suptitle(fig_title)
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig_ax


def relax(A, b, iE=lambda x: x, x0=0, max_iter=10):
    pass

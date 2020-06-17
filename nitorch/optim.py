# -*- coding: utf-8 -*-
"""Optimisers."""

import torch

# TODO:
# . Implement CG/RELAX as autograd functions.
#   > Since A is positive definite, inv(A)' = inv(A') = inv(A)
#   > The gradient of (A\b).dot(x) w.r.t. b is A\x, which can be solved
#     (again) by CG
# . Add tolerance argument, in which case the objective function (the A-norm
#   of the residuals) is tracked


def cg(A, b, x=None, precond=lambda y: y, maxiter=None,
        tolerance=1e-5, verbose=False, sum_dtype=torch.float64):
    """ Solve A*x = b by the conjugate gradient method.

        The method of conjugate gradients solves linear systems of
        the form A*x = b, where A is positive-definite.

    Args:
        A (torch.tensor()): Linear operator (M, N), as a function handle.
        b (torch.tensor()): Vector of right hand side arguments (N, 1).
        x (torch.tensor(), optional): The initial guess, defaults to all zero vector (M, 1).
        precond (torch.tensor(), optional): Preconditioner, defaults to lambda x: x.
        maxiter (int, optional): Defaults to len(b)*10.
        tolerance (float, optional): Defaults to 1e-5.
        verbose (bool, optional): Defaults to False.
        sum_dtype (torch.dtype): Defaults to torch.float64.

    Returns:
        x (torch.tensor()): Solution of the linear system (M, 1).

    Example:
        >>> # Let's solve Ax = b using both regular inversion and CG
        >>> import torch
        >>> from nitorch.optim import cg
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
        >>> x2 = cg(lambda x: A.matmul(x), b, verbose=True, sum_dtype=torch.float32)
        >>> print('cg(A, b) | elapsed time: {:0.4f} seconds'.format(timer() - t0))
        >>> # Inspect errors
        >>> e1 = torch.sqrt(torch.sum((x1 - x2) ** 2))
        >>> print(e1)
        >>> e2 = torch.sqrt(torch.sum((b - A.matmul(x2)) ** 2))
        >>> print(e2)

    """
    # Format arguments
    device = b.device
    dtype = b.dtype
    if maxiter is None:
        maxiter = len(b) * 10
    if x is None:
        x = torch.zeros(b.shape, dtype=dtype, device=device)

    # Initialisation
    r = b - A(x)  # Residual: b - A*x
    z = precond(r)  # Preconditioned residual
    rz = torch.sum(r * z, dtype=sum_dtype)  # Inner product of r and z
    p = z  # Initial conjugate directions p
    beta = torch.tensor(0, dtype=dtype, device=device)  # Initial step size

    # Run algorithm
    for iter in range(maxiter):
        # Calculate conjugate directions P (descent direction)
        p = z + beta * p
        # Find the step size of the conj. gradient descent
        Ap = A(p)
        alpha = rz / torch.sum(p * Ap, dtype=sum_dtype)
        # Perform conj. gradient descent, obtaining updated X and R, using the
        # calculated P and alpha
        x = x + alpha * p
        r = r - alpha * Ap
        # Update preconditioned residual
        z = precond(r)
        # Finds the step size for updating P
        rz0 = rz
        rz = torch.sum(r * z, dtype=sum_dtype)
        if verbose:
            s = '{:' + str(len(str(maxiter))) + '} - sqrt(rtr)={:0.6f}'
            print(s.format(iter + 1, torch.sqrt(rz)))
        if torch.sqrt(rz) < tolerance:
            break
        beta = rz / rz0

    return x


def get_gain(obj, iter, monotonicity='increasing'):
    """ Compute gain of some objective function.

    Args:
        obj (torch.tensor()): Vector of values (e.g., log-likelihoods computations).
        iter (int): Iteration number.
        direction (string, optional): Monotonicity of values ('increasing'/'decreasing'),
            defaults to 'increasing'.

    Returns:
        gain (torch.tensor()): Computed gain.

    """
    if iter == 0:
        return torch.tensor(float('inf'), dtype=obj.dtype, device=obj.device)
    if monotonicity == 'increasing':
        gain = (obj[iter] - obj[iter - 1])
    elif monotonicity == 'decreasing':
        gain = (obj[iter - 1] - obj[iter])
    else:
        raise ValueError('Undefined monotonicity')
    gain = gain / (torch.max(obj[1:iter + 1]) - torch.min(obj[1:iter + 1]))
    return gain


def plot_convergence(vals, fig_ax=None, fig_num=1, fig_title='Model convergence',
                     xlab='', ylab=''):
    """ Plots an algorithm's convergence (e.g. negative log-likelihood, lower bound).

    Allows for real-time plotting if giving returned fig_ax objects as input.

    Args:
        vals (torch.tensor): Vector of values to be plotted.
        fig_ax ([matplotlib.figure, matplotlib.axes])
        fig_num (int, optional): Figure number to plot to, defaults to 1.
        fig_title (str, optional): Figure title, defaults to 'Model convergence'.
        xlab (str, optional): x-label, defaults to ''.
        ylab (str, optional): y-label, defaults to ''.

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

    vals = vals.cpu()  # To CPU

    ax[0].clear()
    x = torch.arange(0, len(vals)) + 1
    ax[0].plot(x, vals)
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel(ylab)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid()

    ax[1].clear()
    x = torch.arange(0, len(vals)) + 1
    ax[1].plot(x[-3:], vals[-3:], 'r')
    ax[1].set_xlabel(xlab)
    ax[1].set_ylabel(ylab)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid()

    fig.suptitle(fig_title)
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig_ax


def relax(A, b, iE=lambda x: x, x0=0, iter=10):
    pass

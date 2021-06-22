# -*- coding: utf-8 -*-
"""Optimisers."""

import torch
import itertools

# TODO:
# . Implement CG/RELAX as autograd functions.
#   > Since A is positive definite, inv(A)' = inv(A') = inv(A)
#   > The gradient of (A\b).dot(x) w.r.t. b is A\x, which can be solved
#     (again) by CG


def cg(A, b, x=None, precond=lambda y: y, max_iter=None,
       tolerance=1e-5, verbose=False, sum_dtype=torch.float64,
       inplace=True, stop='E'):
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
    stop : {'E', 'A'}, default='E'
        What (squared) norm to use a stopping criterion:
            - 'E': (squared) Euclidean norm of the residuals
            - 'A': (squared) A-norm of the error
                   <=> A^{-1}-norm of the residuals
        The Euclidean norm is not motonically decreasing,
        whilst the norm induced by the A-weighted scalar product ('norm') is.
        If monotonicity is important then select 'norm'. However, this requires
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
    max_iter = max_iter or len(b) * 10
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
    r = b - A(x)    # Residual: b - A*x
    z = precond(r)  # Preconditioned residual
    rz = torch.sum(r * z, dtype=sum_dtype)  # Inner product of r and z
    p = z.clone()   # Initial conjugate directions p
    
    if tolerance or verbose:
        if stop == 'residual':
            stop = 'e'
        elif stop == 'norm':
            stop = 'a'
        stop = stop[0].lower()
        if stop == 'e':
            obj0 = torch.sqrt(rz)
        else:
            obj0 = A(x).sub_(2*b).mul_(x)
            obj0 = 0.5 * torch.sum(obj0, dtype=sum_dtype)
        if verbose:
            s = '{:' + str(len(str(max_iter+1))) + '} | {} = {:12.6g}'
            print(s.format(0, stop, obj0))
        obj = obj0.new_zeros(max_iter+1)
        obj[0] = obj0

    # Run algorithm
    for n_iter in range(1, max_iter+1):
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
            if stop == 'e':
                obj1 = torch.sqrt(rz)
            else:
                obj1 = A(x).sub_(2*b).mul_(x)
                obj1 = 0.5 * torch.sum(obj1, dtype=sum_dtype)
            obj[n_iter] = obj1
            gain = get_gain(obj[:n_iter + 1], monotonicity='decreasing')
            if verbose:
                s = '{:' + str(len(str(max_iter+1))) + '} | {} = {:12.6g} | gain = {:12.6g}'
                print(s.format(n_iter, stop, obj[n_iter], gain))
            if gain.abs() < tolerance:
                break

        # Calculate conjugate directions P (descent direction)
        p *= beta
        p += z

    return x


def jacobi(A, b, x=None, precond=lambda y: y, max_iter=None,
           tolerance=1e-5, verbose=False, sum_dtype=torch.float64,
           inplace=True, stop='residuals'):
    """Solve `A*x = b` by the Jacobi method.

    The Jacobi method solves linear systems of the form `A*x = b`,
    where A is diagonal dominant.

    The Jacobi method performs updates of the form `x += D\(b - A*x)`,
    where `D` is the diagonal of `A`.

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
    precond : tensor or callable, default=identity
        Inverse of the diagonal of `A`.
        If a function, should take a tensor with the same shape as `b` and
        divide it by the diagonal of `A`.
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
        If monotonicity is important then select 'norm'. However, this requires
        an additional evaluation of A.

    Returns
    -------
    x : tensor
        Solution of the linear system.

    Note
    ----
    In practice, if A is provided as a function, b and x do not need
    to be vector-shaped.

    """
    # Format arguments
    max_iter = max_iter or len(b) * 10
    if x is None:
        x = torch.zeros_like(b)
    elif not inplace:
        x = x.clone()

    # Create functor if A is a tensor
    if isinstance(A, torch.Tensor):
        A_tensor = A
        A = lambda x: A_tensor.mm(x)

    # Create functor if D is a tensor
    if isinstance(precond, torch.Tensor):
        D_tensor = precond
        precond = lambda x: x * D_tensor

    # Initialisation
    r = b - A(x)

    if tolerance or verbose:
        if stop == 'residual':
            stop = 'e'
        elif stop == 'norm':
            stop = 'a'
        stop = stop[0].lower()
        if stop == 'e':
            obj0 = torch.sqrt(r.square().sum(dtype=sum_dtype))
        else:
            obj0 = A(x).sub_(2 * b).mul_(x)
            obj0 = 0.5 * torch.sum(obj0, dtype=sum_dtype)
        if verbose:
            s = '{:' + str(len(str(max_iter + 1))) + '} | {} = {:12.6g}'
            print(s.format(0, stop, obj0))
        obj = obj0.new_zeros(max_iter + 1)
        obj[0] = obj0

    # Run algorithm
    for n_iter in range(1, max_iter + 1):

        x += 0.5 * precond(r)
        r = b - A(x)

        # Check convergence
        if tolerance or verbose:
            if stop == 'e':
                obj1 = torch.sqrt(r.square().sum(dtype=sum_dtype))
            else:
                obj1 = A(x).sub_(2 * b).mul_(x)
                obj1 = 0.5 * torch.sum(obj1, dtype=sum_dtype)
            obj[n_iter] = obj1
            gain = get_gain(obj[:n_iter + 1], monotonicity='decreasing')
            if verbose:
                width = str(len(str(max_iter + 1)))
                s = '{:' + width + '} | {} = {:12.6g} | gain = {:12.6g}'
                print(s.format(n_iter, stop, obj[n_iter], gain))
            if gain.abs() < tolerance:
                break

    return x


def relax_slicers(shape, scheme='checkerboard'):

    # We use strides to extract subvolumes whose voxels belong to a
    # single group.
    #
    # We have to be careful: with circular boundary conditions,
    # the first and last indices of a line can belong to the same
    # group while they should not. Since we don't know which
    # boundary conditions are used, we are extra careful and treat
    # the last few lines independently.
    #
    # Therefore, there are `bandwidth**dim x 2**dim` subvolumes, where
    # 2**dim corresponds to the quadrants. Although some quadrants can be
    # dropped if the shape along a dimension is a multiple of the bandwidth.
    #
    # Since strides cannot define all possible patterns (e.g. checkerboard)
    # some groups are formed of multiple subvolumes.

    dim = len(shape)
    checkerboard = isinstance(scheme, str) and scheme[0] == 'c'
    bandwidth = 2 if checkerboard else scheme
    size_last = tuple(int(s % bandwidth) for s in shape)
    # size_last = [0] * dim
    offsets = list(itertools.product(range(bandwidth), repeat=dim))
    quadrants = list(itertools.product([True, False], repeat=dim))
    slicers = []
    for offset in offsets:
        for quadrant in quadrants:
            if any(o >= l for o, l, q in zip(offset, size_last, quadrant)
                   if not q):
                continue
            slicer = tuple(slice(int(o) if q else int(s-l+o),       # start
                                 (int(-l) or None) if q else None,  # stop
                                 bandwidth)                         # stride
                           for o, s, l, q in zip(offset, shape, size_last, quadrant))
            slicers.append(slicer)
    # add vector dimension
    slicers = [(*slicer, slice(None)) for slicer in slicers]
    if checkerboard:
        # gather odd/even slicers
        slicers = [[slicer for slicer in slicers
                    if sum(s.start for s in slicer[1:-1]) % 2],
                   [slicer for slicer in slicers
                    if not sum(s.start for s in slicer[1:-1]) % 2]]
    else:
        slicers = [[slicer] for slicer in slicers]
    return slicers


def relax(A, b, precond, x=None, scheme='checkerboard', max_iter=None,
          dim=None, tolerance=1e-5, verbose=False, sum_dtype=torch.float64,
          inplace=True, stop='E', mode=1):
    """Solve `A*x = b` by block-relaxation (e.g., checkerboard Gauss-Seidel).

    The Gauss-Seidel method solves linear systems of the form `A*x = b`,
    where A is either positive definite or diagonal dominant.

    This method performs updates of the form `x[block] += (P\(b - A*x))[block]`,
    where P is a suitable pre-conditioner, and elements within a block
    are independent of each other conditioned on the elements outside
    of the block.

    Parameters
    ----------
    A : callable(tensor)
        Linear operator.
        Should take as inputs a tensor with the same shape as `b`
        and return a tensor with the same shape as `b`.
    b : (..., *spatial, K) tensor
        Right hand side 'vector'.
        Note that it needs a tensor dimension on the right.
    precond : callable(tensor, slicer)
        Preconditioner.
        Should take as inputs a tensor with the same shape as `b` and
        a tuple of slices, and return a tensor with the same shape as
        `b[slicer]`.
    x : (..., *spatial, K) tensor, optional, default=0
        Initial guess.
    scheme : 'checkerboard' or int, default='checkerboard'
        Scheme used to define conditionally independent blocks.
         - 'checkerboard' uses a "black and white" scheme where all
           elements that do not share a "face" belong to the same block.
        - `int` uses strides to define blocks. In effect, elements must
           have an l1 distance larger than this number to belong to
           the same block.
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
    stop : {'E', 'A'}, default='E'
        What (squared) norm to use a stopping criterion:
            - 'E': (squared) Euclidean norm of the residuals
            - 'A': (squared) A-norm of the error
                   <=> A^{-1}-norm of the residuals

    Returns
    -------
    x : tensor
        Solution of the linear system.

    """

    # Format arguments
    dim = dim or (b.dim() - 1)
    max_iter = max_iter or len(b) * 10
    if x is None:
        x = torch.zeros_like(b)
    elif not inplace:
        x = x.clone()
    shape = x.shape[-dim-1:-1]

    r = b - A(x)

    if tolerance or verbose:
        if stop == 'residual':
            stop = 'e'
        elif stop == 'norm':
            stop = 'a'
        stop = stop[0].lower()
        if stop == 'e':
            obj0 = r.square().sum(dtype=sum_dtype).sqrt()
        else:
            obj0 = A(x).sub_(2 * b).mul_(x)
            obj0 = 0.5 * torch.sum(obj0, dtype=sum_dtype)
        if verbose:
            s = '{:' + str(len(str(max_iter + 1))) + '} | {} = {:12.6g}'
            print(s.format(0, stop, obj0))
        obj = torch.zeros(max_iter + 1, dtype=sum_dtype, device=obj0.device)
        obj[0] = obj0

    slicers = relax_slicers(shape, scheme)
    for n_iter in range(1, max_iter+1):

        for group in slicers:

            if mode == 1:
                r.copy_(b).sub_(A(x))
                for slicer in group:
                    # compute residuals
                    # apply preconditioner in selected voxels and update
                    x[slicer] += precond(r, slicer)
            else:
                for slicer in group:
                    r[slicer].copy_(b[slicer]).sub_(A(x, slicer))
                for slicer in group:
                    x[slicer] += precond(r, slicer)

        # Check convergence
        if tolerance or verbose:
            if stop == 'e':
                obj1 = r.square().sum(dtype=sum_dtype).sqrt()
            else:
                obj1 = A(x).sub_(2 * b).mul_(x)
                obj1 = 0.5 * torch.sum(obj1, dtype=sum_dtype)
            obj[n_iter] = obj1
            gain = get_gain(obj[:n_iter + 1], monotonicity='decreasing')
            if verbose:
                width = str(len(str(max_iter + 1)))
                s = '{:' + width + '} | {} = {:12.6g} | gain = {:12.6g}'
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


# def relax(A, b, iE=lambda x: x, x0=0, max_iter=10):
#     pass

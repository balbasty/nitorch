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


def cg(A, b, precond=lambda x: x, x=0, iter=10):
    """Conjugate-Gradient solver.

    The method of conjugate gradients solves linear systems of the form
    A*x = b, where A is positive-definite.


    Args:
        A (array_like,function)
        b (array_like,function)
        precond (array_like,function, optional): Preconditioner.
            Defaults to lambda x: x.
        x (array_like, optional): Initial guess. Defaults to 0.
        iter (int, optional): Nunber of iterations. Defaults to 10.

    Returns:
        x (torch.tensor): Solution of the linear system.

    """

    # Format arguments
    x = torch.as_tensor(x)
    if not callable(A):
        A = torch.as_tensor(A)
        def A(x): return A.matmul(x)
    if not callable(precond):
        precond = torch.as_tensor(precond)
        def precond(x): precond.matmul(x)

    # Initialisation
    r = b - A(x)                              # Residual: b - A*x
    z = precond(r)                            # Preconditioned residual

    rz = r.flatten().matmul(z.flatten())      # Inner product of r and z
    p = z                                     # Initial conjugate directions p
    beta = 0.                                 # Initial step size

    # Run algorithm
    for j in range(iter):
        # Calculate conjugate directions P (descent direction)
        p = z + beta*p

        # Find the step size of the conj. gradient descent
        Ap = A(p)
        alpha = rz / p.flatten().matmul(Ap.flatten())

        # Perform conj. gradient descent, obtaining updated X and R, using the
        # calculated P and alpha
        x = x + alpha * p
        r = r - alpha * Ap

        # Update preconditioned residual
        z = precond(r)

        # Finds the step size for updating P
        rz0 = rz
        rz = r.flatten().matmul(z.flatten())
        beta = rz / rz0

    return x


def relax(A, b, iE=lambda x: x, x0=0, iter=10):
    pass

"""Principal matrix square root (pure PyTorch, batched)."""
import torch


def sqrtm(mat, tol=None, max_iter=100):
    """Batched principal matrix square root.

    Uses the scaled product form of the Denman-Beavers iteration
    (Higham, "Functions of Matrices", 2008, eq. 6.17-6.29), computed
    in double precision. It converges to the principal square root
    whenever `mat` has no eigenvalue on the closed negative real axis,
    which is exactly when a real principal square root exists.

    If the last row of a matrix is exactly `[0, ..., 0, 1]` (homogeneous
    affine), the last row of its square root is set to exactly
    `[0, ..., 0, 1]` as well.

    Parameters
    ----------
    mat : (..., N, N) tensor
        Input matrix or batch of matrices
    tol : float, optional
        Convergence tolerance (default: 1e-14 * N)
    max_iter : int, default=100
        Maximum number of iterations

    Returns
    -------
    sqrtmat : (..., N, N) tensor
        Principal square root, with the input dtype and device.

    Raises
    ------
    ValueError
        If a matrix has an eigenvalue on the closed negative real axis
        (e.g. det <= 0, such as a reflection), so that no real principal
        square root exists, or if the iteration does not converge.

    """
    mat = torch.as_tensor(mat)
    if mat.dim() < 2 or mat.shape[-1] != mat.shape[-2]:
        raise ValueError(f'Expected a (batch of) square matrices but got '
                         f'shape {tuple(mat.shape)}')
    dtype, device = mat.dtype, mat.device
    if mat.is_complex():
        raise ValueError('sqrtm only supports real matrices')
    shape = mat.shape
    n = shape[-1]
    if tol is None:
        tol = 1e-14 * n
    A = mat.reshape([-1, n, n]).to(torch.float64)

    if not torch.isfinite(A).all():
        raise ValueError('sqrtm: input contains non-finite values')

    # --- check that a real principal square root exists ---------------
    eig = torch.linalg.eigvals(A)
    scale = A.abs().amax(dim=(-1, -2), keepdim=True).squeeze(-1).clamp_min(1e-300)
    on_neg_axis = ((eig.imag.abs() <= 1e-10 * scale) &
                   (eig.real <= 1e-10 * scale))
    if on_neg_axis.any():
        bad = eig[on_neg_axis.any(-1)][0]
        raise ValueError(
            'sqrtm: matrix has eigenvalue(s) on the closed negative real '
            'axis (e.g. det <= 0 / reflection, or a 180 degree rotation), '
            'so it has no real principal square root. '
            f'Eigenvalues: {bad.tolist()}')

    # --- scaled product-form Denman-Beavers ---------------------------
    eye = torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)
    M = A.clone()
    Y = A.clone()
    converged = False
    for it in range(max_iter):
        Minv = torch.linalg.inv(M)
        # determinant scaling (only while far from convergence)
        delta = (M - eye).abs().amax(dim=(-1, -2))
        if (delta > 1e-2).any():
            mu = torch.linalg.det(M).abs().pow(-1 / (2 * n))
            mu = torch.where(delta > 1e-2, mu, torch.ones_like(mu))
            mu = mu[:, None, None]
        else:
            mu = torch.ones_like(delta)[:, None, None]
        mu2 = mu * mu
        Y = Y.matmul(eye + Minv / mu2).mul_(mu / 2)
        M = (eye + (mu2 * M + Minv / mu2) / 2) / 2
        if (M - eye).abs().amax() <= tol:
            converged = True
            break

    if not converged or not torch.isfinite(Y).all():
        raise ValueError(f'sqrtm: Denman-Beavers iteration did not converge '
                         f'in {max_iter} iterations')

    # --- Newton refinement (small matrices) --------------------------
    # Solve X E + E X = A - X^2 via its Kronecker form and update
    # X <- X + E. One or two steps bring the residual to ~machine eps.
    if n <= 16:
        eye_n = torch.eye(n, dtype=A.dtype, device=A.device)
        for _ in range(2):
            R = A - Y.matmul(Y)
            K = (torch.kron(Y, eye_n[None]) +
                 torch.kron(eye_n[None], Y.transpose(-1, -2).contiguous()))
            try:
                E = torch.linalg.solve(K, R.reshape([-1, n * n, 1]))
            except RuntimeError:
                break
            Ynew = Y + E.reshape([-1, n, n])
            if not torch.isfinite(Ynew).all():
                break
            Y = Ynew

    # --- keep homogeneous affines exactly homogeneous -----------------
    lastrow = torch.zeros(n, dtype=A.dtype, device=A.device)
    lastrow[-1] = 1
    homog = (A[:, -1, :] == lastrow).all(-1)
    if homog.any():
        Y[homog, -1, :] = lastrow

    # --- sanity check on the residual ---------------------------------
    res = (Y.matmul(Y) - A).abs().amax(dim=(-1, -2))
    nrm = A.abs().amax(dim=(-1, -2)).clamp_min(1e-300)
    if (res > 1e-8 * nrm).any():
        raise ValueError('sqrtm: inaccurate square root '
                         f'(relative residual {(res / nrm).max().item():.3g})')

    return Y.reshape(shape).to(dtype=dtype, device=device)

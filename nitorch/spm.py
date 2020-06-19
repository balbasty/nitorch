# -*- coding: utf-8 -*-
""" Various SPM functions:
    . def2sparse (spm_def2sparse, 'Longitudinal' toolbox)
    . dexpm (spm_dexpm, 'Longitudinal' toolbox)
    . imatrix (spm_imatrix)
    . matrix (spm_matrix)
    . mean_matrix (spm_meanm, 'Longitudinal' toolbox)
    . mean_space (Ashburner bonus)
    . noise_estimate (spm_noise_estimate, 'Longitudinal' toolbox)

    Mostly authored by John Ashburner as part of the SPM software:
    . (fil.ion.ucl.ac.uk/spm/software/spm12)
"""


import nibabel as nib
from nitorch.mathfun import expm, logm
from nitorch.mixtures import GMM
from nitorch.mixtures import RMM
from nitorch.spatial import voxsize
import math
import torch


__all__ = ['affine', 'affine_basis', 'def2sparse', 'dexpm', 'identity',
           'imatrix', 'matrix', 'mean_matrix', 'mean_space', 'noise_estimate']


def affine(dm, mat, dtype=torch.float32, device='cpu'):
    """ Generate an affine warp on a lattice defined by dm and mat.

    Args:
        dm (torch.Size): Image dimensions (X, Y, Z).
        mat (torch.Tensor): Affine transform.
        dtype (torch.dtype, optional): Defaults to torch.float32.
        device (string, optional): Defaults to 'cpu'.

    Returns:
        a (torch.Tensor): Affine warp (1, X, Y, Z, 3).

    """
    mat = mat.type(dtype)
    a = identity(dm, dtype=dtype, device=device)
    a = torch.reshape(a, (dm[0]*dm[1]*dm[2], 3))
    a = torch.matmul(a, torch.t(mat[0:3, 0:3])) + torch.t(mat[0:3, 3])
    a = torch.reshape(a, (dm[0], dm[1], dm[2], 3))
    if dm[0] == 1:
        a[:, :, :, 0] = 0
    a = a[None, ...]
    return a


def affine_basis(basis='SE', dim=3, dtype=torch.float64, device='cpu'):
    """ Generate a basis for the Lie algebra of affine matrices.

    Args:
        basis (string, optional): Name of the group that must be encoded:
            . 'T': Translation
            . 'SO': Special orthogonal (= translation + rotation)
            . 'SE': Special Euclidean (= translation + rotation + scale)
            Defaults to 'SE'.
        dim (int, optional): Basis dimensions: 0, 2, 3. Defaults to 3.
        dtype (torch.dtype, optional): Data type. Defaults to float64.
        device (str, optional): Device. Defaults to cpu.

    Returns:
        B (torch.tensor): Affine basis (4, 4, num_basis)

    """
    if dim == 0:
        B = torch.zeros((4, 4, 0), device=device, dtype=dtype)
    elif dim == 2:
        if basis == 'T':
            B = torch.zeros((4, 4, 2), device=device, dtype=dtype)
            B[0, 3, 0] = 1
            B[1, 3, 1] = 1
        elif basis == 'SO':
            B = torch.zeros((4, 4, 1), device=device, dtype=dtype)
            B[0, 1, 0] = 1
            B[1, 0, 0] = -1
        elif basis == 'SE':
            B = torch.zeros((4, 4, 3), device=device, dtype=dtype)
            B[0, 3, 0] = 1
            B[1, 3, 1] = 1
            B[0, 1, 2] = 1
            B[1, 0, 2] = -1
        else:
            raise ValueError('Unknown group!')
    else:
        if basis == 'T':
            B = torch.zeros((4, 4, 3), device=device, dtype=dtype)
            B[0, 3, 0] = 1
            B[1, 3, 1] = 1
            B[2, 3, 2] = 1
        elif basis == 'SO':
            B = torch.zeros((4, 4, 3), device=device, dtype=dtype)
            B[0, 1, 0] = 1
            B[1, 0, 0] = -1
            B[0, 2, 1] = 1
            B[2, 0, 1] = -1
            B[1, 2, 2] = 1
            B[2, 1, 2] = -1
        elif basis == 'SE':
            B = torch.zeros((4, 4, 6), device=device, dtype=dtype)
            B[0, 3, 0] = 1
            B[1, 3, 1] = 1
            B[2, 3, 2] = 1
            B[0, 1, 3] = 1
            B[1, 0, 3] = -1
            B[0, 2, 4] = 1
            B[2, 0, 4] = -1
            B[1, 2, 5] = 1
            B[2, 1, 5] = -1
        else:
            raise ValueError('Unknown group!')

    return B


def def2sparse(phi, dm_in, nn=False):
    """ Generate a sparse matrix Phi, which encodes the deformation phi.
        This function assumes zero boundary conditions
        (i.e., Dirichlet with zero at the boundary).

    Args:
        phi (torch.Tensor): Deformation (X1, Y1, Z1, 3).
        dm_in (torch.Size): Size of input FOV (X0, Y0, Z0).
        nn (bool, optional): Encode nearest neighbour interpolation, else trilinear.
            Defaults to True.

    Returns:
        Phi (torch.sparse.FloatTensor): Sparse matrix encoding the deformation.

    See also:
        Deformation Phi is applied to a 3D volume x as:
        x1 = torch.reshape(torch.matmul(Phi, x.flatten()), phi.shape[:3])

        Implementation of John Ashburner's SPM12 function spm_def2sparse.

    """
    # Allocate on GPU or CPU
    device = phi.device
    # Get data type
    dtype = phi.dtype

    # Output dimensions
    dim_out = phi.shape
    dim_out = dim_out[:3]

    # The sparse matrix will be of size ni x nj
    ni = dim_out[0] * dim_out[1] * dim_out[2]
    nj = dm_in[0] * dm_in[1] * dm_in[2]
    size = torch.Size([ni, nj], device=device)  # The (implicit) size of the sparse matrix

    if nn:
        """
        Encode nearest neighbour interpolation
        """

        # Round values in deformation (now integers)
        phi = torch.floor(phi + 0.5)
        phi = phi.type(torch.int64)

        # Flatten rounded
        rphi0 = phi[:, :, :, 0].flatten()
        rphi1 = phi[:, :, :, 1].flatten()
        rphi2 = phi[:, :, :, 2].flatten()

        # Get valid indices in output image
        i = torch.where((rphi0 >= 0) & (rphi0 < dm_in[0]) &
                        (rphi1 >= 0) & (rphi1 < dm_in[1]) &
                        (rphi2 >= 0) & (rphi2 < dm_in[2]))[0]  # OBS: torch.where returns tuple of tensors, [0] to get tensor from tuple
        # Get corresponding indices in input image
        j = rphi2[i] + dm_in[2] * (rphi1[i] + dm_in[1] * rphi0[i])

        # The weight in the sparse matrix are all one for nearest neighbour interpolation
        w = torch.ones(len(j), device=device, dtype=dtype)

        # Construct the sparse matrix
        Phi = torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)
    else:
        """
        Encode trilinear interpolation
        """

        # Flatten phi
        phi0 = phi[:, :, :, 0].flatten()
        phi1 = phi[:, :, :, 1].flatten()
        phi2 = phi[:, :, :, 2].flatten()

        # Floor values in deformation (now integers)
        phi = torch.floor(phi)
        phi = phi.type(torch.int64)

        # Flatten floored
        fphi0 = phi[:, :, :, 0].flatten()
        fphi1 = phi[:, :, :, 1].flatten()
        fphi2 = phi[:, :, :, 2].flatten()

        """
        Corner 1 - C000
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= 0) & (fphi0 < dm_in[0]) &
                        (fphi1 >= 0) & (fphi1 < dm_in[1]) &
                        (fphi2 >= 0) & (fphi2 < dm_in[2]))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + dm_in[2] * (fphi1[i] + dm_in[1] * fphi0[i])

        # Compute interpolation weights
        w = (1 - phi0[i] + fphi0[i]) * (1 - phi1[i] + fphi1[i]) * (1 - phi2[i] + fphi2[i])

        # Initialise the sparse matrix
        Phi = torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

        """
        Corner 2 - C100
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= -1) & (fphi0 < (dm_in[0] - 1)) &
                        (fphi1 >= 0) & (fphi1 < dm_in[1]) &
                        (fphi2 >= 0) & (fphi2 < dm_in[2]))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + dm_in[2] * (fphi1[i] + dm_in[1] * fphi0[i] + 1)

        # Compute interpolation weights
        w = (phi0[i] - fphi0[i]) * (1 - phi1[i] + fphi1[i]) * (1 - phi2[i] + fphi2[i])

        # Add to the sparse matrix
        Phi = Phi + torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

        """
        Corner 3 -  C010
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= 0) & (fphi0 < dm_in[0]) &
                        (fphi1 >= -1) & (fphi1 < (dm_in[1] - 1)) &
                        (fphi2 >= 0) & (fphi2 < dm_in[2]))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + dm_in[2] * (fphi1[i] + 1 + dm_in[1] * fphi0[i])

        # Compute interpolation weights
        w = (1 - phi0[i] + fphi0[i]) * (phi1[i] - fphi1[i]) * (1 - phi2[i] + fphi2[i])

        # Add to the sparse matrix
        Phi = Phi + torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

        """
        Corner 4 -  C110
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= -1) & (fphi0 < (dm_in[0] - 1)) &
                        (fphi1 >= -1) & (fphi1 < (dm_in[1] - 1)) &
                        (fphi2 >= 0) & (fphi2 < dm_in[2]))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + dm_in[2] * (fphi1[i] + 1 + dm_in[1] * fphi0[i] + 1)

        # Compute interpolation weights
        w = (phi0[i] - fphi0[i]) * (phi1[i] - fphi1[i]) * (1 - phi2[i] + fphi2[i])

        # Add to the sparse matrix
        Phi = Phi + torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

        """
        Corner 5 -  C001
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= 0) & (fphi0 < dm_in[0]) &
                        (fphi1 >= 0) & (fphi1 < dm_in[1]) &
                        (fphi2 >= -1) & (fphi2 < (dm_in[2] - 1)))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + 1 + dm_in[2] * (fphi1[i] + dm_in[1] * fphi0[i])

        # Compute interpolation weights
        w = (1 - phi0[i] + fphi0[i]) * (1 - phi1[i] + fphi1[i]) * (phi2[i] - fphi2[i])

        # Add to the sparse matrix
        Phi = Phi + torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

        """
        Corner 6 -  C101
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= -1) & (fphi0 < (dm_in[0] - 1)) &
                        (fphi1 >= 0) & (fphi1 < dm_in[1]) &
                        (fphi2 >= -1) & (fphi2 < (dm_in[2] - 1)))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + 1 + dm_in[2] * (fphi1[i] + dm_in[1] * fphi0[i] + 1)

        # Compute interpolation weights
        w = (phi0[i] - fphi0[i]) * (1 - phi1[i] + fphi1[i]) * (phi2[i] - fphi2[i])

        # Add to the sparse matrix
        Phi = Phi + torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

        """
        Corner 7 -  C011
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= 0) & (fphi0 < dm_in[0]) &
                        (fphi1 >= -1) & (fphi1 < (dm_in[1] - 1)) &
                        (fphi2 >= -1) & (fphi2 < (dm_in[2] - 1)))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + 1 + dm_in[2] * (fphi1[i] + 1 + dm_in[1] * fphi0[i])

        # Compute interpolation weights
        w = (1 - phi0[i] + fphi0[i]) * (phi1[i] - fphi1[i]) * (phi2[i] - fphi2[i])

        # Add to the sparse matrix
        Phi = Phi + torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

        """
        Corner 8 -  C111
        """

        # Get valid indices in output image
        i = torch.where((fphi0 >= -1) & (fphi0 < (dm_in[0] - 1)) &
                        (fphi1 >= -1) & (fphi1 < (dm_in[1] - 1)) &
                        (fphi2 >= -1) & (fphi2 < (dm_in[2] - 1)))[0]
        # Get corresponding indices in input image
        j = fphi2[i] + 1 + dm_in[2] * (fphi1[i] + 1 + dm_in[1] * fphi0[i] + 1)

        # Compute interpolation weights
        w = (phi0[i] - fphi0[i]) * (phi1[i] - fphi1[i]) * (phi2[i] - fphi2[i])

        # Add to the sparse matrix
        Phi = Phi + torch.sparse.FloatTensor(torch.stack((i, j), 0), w, size)

    return Phi


def dexpm(A, dA=None, max_iter=10000):
    """ Differentiate a matrix exponential.

    Args:
        A (torch.tensor): Lie algebra (num_basis,).
        dA (torch.tensor, optional): Basis function to differentiate
            with respect to (4, 4, num_basis). Defaults to None.
        max_iter (int, optional): Max number of iterations, defaults to 10,000.

    Returns:
        E (torch.tensor): expm(A) (4, 4).
        dE (torch.tensor): Derivative of expm(A) (4, 4, num_basis).

    Authors:
        John Ashburner, as part of the SPM12 software.

    """
    device = A.device
    dtype = A.dtype
    num_basis = A.numel()
    if dA is None:
        dA = torch.zeros(num_basis, num_basis, device=device, dtype=dtype)
        dA = dA[..., None]
    else:
        p = A.flatten()
        A = torch.zeros(4, 4, device=device, dtype=dtype)
        for m in range(dA.shape[2]):
            A += p[m]*dA[..., m]
    An = A.clone()
    E = torch.eye(4, device=device, dtype=dtype) + A

    dAn = dA.clone()
    dE = dA.clone()
    for n_iter in range(2, max_iter):
        for m in range(dA.shape[2]):
            dAn[..., m] = (dAn[..., m].mm(A) + An.mm(dA[..., m]))/n_iter
        dE += dAn
        An = An.mm(A)/n_iter
        E += An
        if torch.sum(An**2) < A.numel()*1e-32:
            break

    return E, dE


def identity(dm, dtype=torch.float32, device='cpu'):
    """ Generate the identity warp on a lattice defined by dm.

    Args:
        dm (torch.Size): Defines the size of the output lattice (X, Y, Z).
        dtype (torch.dtype, optional): Defaults to torch.float32.
        device (string, optional): Defaults to 'cpu'.

    Returns:
        i (torch.Tensor): Identity warp (X, Y, Z, 3).

    """
    i = torch.zeros((dm[0], dm[1], dm[2], 3), dtype=dtype, device=device)
    i[:, :, :, 0], i[:, :, :, 1], i[:, :, :, 2] = \
        torch.meshgrid([torch.arange(0, dm[0], dtype=dtype, device=device),
                        torch.arange(0, dm[1], dtype=dtype, device=device),
                        torch.arange(0, dm[2], dtype=dtype, device=device)])
    return i


def imatrix(M):
    """ Return the parameters for creating an affine transformation matrix.

    Args:
        mat (torch.tensor): Affine transformation matrix (4, 4).

    Returns:
        P (torch.tensor): Affine parameters (<=12).

    Authors:
        John Ashburner & Stefan Kiebel, as part of the SPM12 software.

    """
    device = M.device
    dtype = M.dtype
    one = torch.tensor(1.0, device=device, dtype=dtype)
    pi = torch.tensor(math.pi, device=device, dtype=dtype)
    # Translations and Zooms
    R = M[:-1, :-1]
    C = torch.cholesky(R.t().mm(R))
    C = C.t()
    d = torch.diag(C)
    P = torch.tensor([M[0, 3], M[1, 3], M[2, 3], 0, 0, 0, d[0], d[1], d[2], 0, 0, 0],
                     device=device, dtype=dtype)
    if R.det() < 0:  # Fix for -ve determinants
        P[6] = -P[6]
    # Shears
    C = C.solve(torch.diag(torch.diag(C)))[0]
    P[9] = C[0, 1]
    P[10] = C[0, 2]
    P[11] = C[1, 2]
    R0 = matrix(torch.tensor([0, 0, 0, 0, 0, 0, P[6], P[7], P[8], P[9], P[10], P[11]],
                             device=device, dtype=dtype))
    R0 = R0[:-1, :-1]
    R1 = R.mm(R0.inverse())  # This just leaves rotations in matrix R1
    # Correct rounding errors
    rang = lambda x: torch.min(torch.max(x, -one), one)
    P[4] = torch.asin(rang(R1[0, 2]))
    if (torch.abs(P[4]) - pi/2)**2 < 1e-9:
        P[3] = 0
        P[5] = torch.atan2(-rang(R1[1, 0]), rang(-R1[2, 0]/R1[0, 2]))
    else:
        c = torch.cos(P[4])
        P[3] = torch.atan2(rang(R1[1, 2]/c), rang(R1[2, 2]/c))
        P[5] = torch.atan2(rang(R1[0, 1]/c), rang(R1[0, 0]/c))

    return P


def matrix(P):
    """ Return an affine transformation matrix.

    Args:
        P (torch.tensor): Affine parameters (<=12).

    Returns:
        mat (torch.tensor): Affine transformation matrix (4, 4).

    Authors:
        John Ashburner & Stefan Kiebel, as part of the SPM12 software.

    """
    device = P.device
    dtype = P.dtype
    if len(P) == 3:  # Special case: translation only
        A = torch.eye(4, device=device, dtype=dtype)
        A[:-1, -1] = P
        return A
    # Pad P with 'null' parameters
    q = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                     device=device, dtype=dtype)
    P = torch.cat((P, q[len(P):]))
    # Translations
    T = torch.tensor([[1, 0, 0, P[0]],
                      [0, 1, 0, P[1]],
                      [0, 0, 1, P[2]],
                      [0, 0, 0, 1]], device=device, dtype=dtype)
    # Rotations
    c = lambda x: torch.cos(x)  # cos
    s = lambda x: torch.sin(x)  # sin
    R0 = torch.tensor([[1, 0, 0, 0],
                       [0, c(P[3]), s(P[3]), 0],
                       [0, -s(P[3]), c(P[3]), 0],
                       [0, 0, 0, 1]], device=device, dtype=dtype)
    R1 = torch.tensor([[c(P[4]), 0, s(P[4]), 0],
                       [0, 1, 0, 0],
                       [-s(P[4]), 0, c(P[4]), 0],
                       [0, 0, 0, 1]], device=device, dtype=dtype)
    R2 = torch.tensor([[c(P[5]), s(P[5]), 0, 0],
                       [-s(P[5]), c(P[5]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], device=device, dtype=dtype)
    R = R0.mm(R1.mm(R2))
    # Scales
    Z = torch.tensor([[P[6], 0,    0,    0],
                      [0,    P[7], 0,    0],
                      [0,    0,    P[8], 0],
                      [0,    0,    0,    1]], device=device, dtype=dtype)
    # Shears
    S = torch.tensor([[1, P[9], P[10], 0],
                      [0, 1,    P[11], 0],
                      [0, 0,    1,     0],
                      [0, 0,    0,     1]], device=device, dtype=dtype)

    return T.mm(R.mm(Z.mm(S)))


def mean_matrix(Mat):
    """ Compute barycentre of matrix exponentials.

    Args:
        Mat (torch.tensor): N subjects' orientation matrices (4, 4, N).

    Returns:
        mat (torch.tensor): The resulting mean (4, 4).

    Authors:
        John Ashburner, as part of the SPM12 software.

    """
    device = Mat.device
    dtype = Mat.dtype
    N = Mat.shape[2]
    mat = torch.eye(4, device=device, dtype=dtype)
    for n_iter in range(1024):
        S = torch.zeros((4, 4), device=device, dtype=dtype)
        for n in range(N):
            L = logm(Mat[..., n].solve(mat)[0])
            S += L
        S /= N
        mat = mat.mm(expm(S))
        if torch.sum(S**2) < 1e-20:
            break
    return mat


def mean_space(Mat, Dim, vx=None, mod_prct=0):
    """ Compute a (mean) model space from individual spaces.

    Args:
        Mat (torch.tensor): N subjects' orientation matrices (4, 4, N).
        Dim (torch.tensor): N subjects' dimensions (3, N).
        vx (torch.tensor|tuple|float, optional): Voxel size (3,), defaults to None (estimate from input).
        mod_prct (float, optional): Percentage to either crop or pad mean space.
            A negative value crops and a positive value pads. Should be a value
            between 0 and 1. Defaults to 0.

    Returns:
        dim (torch.tensor): Mean dimensions (3,).
        mat (torch.tensor): Mean orientation matrix (4, 4).
        vx (torch.tensor): Mean voxel size (3,).

    Authors:
        John Ashburner, as part of the SPM12 software.

    """
    device = Mat.device
    dtype0 = Mat.dtype
    dtype = torch.float64
    N = Mat.shape[2]  # Number of subjects
    inf = float('inf')
    one = torch.tensor(1.0, device=device, dtype=dtype)
    if vx is None:
        vx = torch.tensor([inf, inf, inf], device=device, dtype=dtype)
    if isinstance(vx, float) or isinstance(vx, int):
        vx = (vx,)*3
    if isinstance(vx, tuple) and len(vx) == 3:
        vx = torch.tensor([vx[0], vx[1], vx[2]], device=device, dtype=dtype)
    # To float64
    Mat = Mat.type(dtype)
    Dim = Dim.type(dtype)
    # Get affine basis
    basis = 'SE'
    dim = 3 if Dim[2, 0] > 1 else 2
    B = affine_basis(basis, dim, device=device, dtype=dtype)
    """ Find combination of 90 degree rotations and flips that brings all
        the matrices closest to axial
    """
    Mat0 = Mat.clone()
    pmatrix = torch.tensor([[0, 1, 2],
                            [1, 0, 2],
                            [2, 0, 1],
                            [2, 1, 0],
                            [0, 2, 1],
                            [1, 2, 0]], device=device)

    for n in range(N):  # Loop over subjects
        vx1 = voxsize(Mat[..., n])
        R = Mat[..., n].mm(
            torch.diag(torch.cat((vx1, one[..., None]))).inverse())[:-1, :-1]
        minss = inf
        minR = torch.eye(3, dtype=dtype, device=device)
        for i in range(6):  # Permute (= 'rotate + flip') axes
            R1 = torch.zeros((3, 3), dtype=dtype, device=device)
            R1[pmatrix[i, 0], 0] = 1
            R1[pmatrix[i, 1], 1] = 1
            R1[pmatrix[i, 2], 2] = 1
            for j in range(8):  # Mirror (= 'flip') axes
                fd = [(j & 1)*2 - 1, (j & 2) - 1, (j & 4)/2 - 1]
                F = torch.diag(torch.tensor(fd, dtype=dtype, device=device))
                R2 = F.mm(R1)
                ss = torch.sum((R.mm(R2.inverse()) -
                                torch.eye(3, dtype=dtype, device=device))**2)
                if ss < minss:
                    minss = ss
                    minR = R2
        rdim = torch.abs(minR.mm(Dim[..., n][..., None]))
        R2 = minR.inverse()
        R22 = R2.mm(0.5*(torch.sum(R2, dim=0, keepdim=True).t() - 1)*(rdim + 1))
        minR = torch.cat((R2, R22), dim=1)
        minR = torch.cat((minR, torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)[None, ...]), dim=0)
        Mat[..., n] = Mat[..., n].mm(minR)
    """ Average of the matrices in Mat
    """
    mat = mean_matrix(Mat)
    """ If average involves shears, then find the closest matrix that does not
        require them.
    """
    C_ix = torch.tensor([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
                        device=device)  # column-major ordering from (4, 4) tensor
    p = imatrix(mat)
    if torch.sum(p[[9, 10, 11]]**2) > 1e-8:
        B2 = torch.zeros((4, 4, 3), device=device, dtype=dtype)
        B2[0, 0, 0] = 1
        B2[1, 1, 1] = 1
        B2[2, 2, 2] = 1

        p = torch.zeros(9, device=device, dtype=dtype)
        for n_iter in range(10000):
            R, dR = dexpm(p[[0, 1, 2, 3, 4, 5]], B)  # Rotations + Translations
            Z, dZ = dexpm(p[[6, 7, 8]], B2)  # Zooms

            M = R.mm(Z)
            dM = torch.zeros((4, 4, 9), device=device, dtype=dtype)
            for n in range(6):
                dM[..., n] = dR[..., n].mm(Z)
            for n in range(3):
                dM[..., 6 + n] = R.mm(dZ[..., n])
            dM = dM.reshape((16, 9))
            d = M.flatten() - mat.flatten()
            gr = dM.t().mm(d[..., None])
            Hes = dM.t().mm(dM)
            p = p - gr.solve(Hes)[0][:, 0]
            if torch.sum(gr**2) < 1e-8:
                break
        mat = M.clone()

    # Set required voxel size
    vx_out = vx.clone()
    vx = voxsize(mat)
    vx_out[~torch.isfinite(vx_out)] = vx[~torch.isfinite(vx_out)]
    mat = mat.mm(torch.cat((vx_out/vx, one[..., None])).diag())
    vx = voxsize(mat)
    # Ensure that the FoV covers all images, with a few voxels to spare
    mn =  torch.tensor([inf, inf, inf], device=device, dtype=dtype)
    mx = -torch.tensor([inf, inf, inf], device=device, dtype=dtype)
    for n in range(N):
        dm = Dim[..., n]
        corners = torch.tensor([[1, dm[0], 1, dm[0], 1, dm[0], 1, dm[0]],
                                [1, 1, dm[1], dm[1], 1, 1, dm[1], dm[1]],
                                [1, 1, 1, 1, dm[2], dm[2], dm[2], dm[2]],
                                [1, 1, 1, 1, 1, 1, 1, 1]],
                               device=device, dtype=dtype)
        M = Mat0[..., n].solve(mat)[0]
        vx1 = M[:-1,:].mm(corners)
        mx = torch.max(mx, torch.max(vx1, dim=1)[0])
        mn = torch.min(mn, torch.min(vx1, dim=1)[0])
    mx = torch.ceil(mx)
    mn = torch.floor(mn)
    # Either pad or crop mean space
    off = mod_prct*(mx - mn)
    if Dim[2, 0] == 1: off = torch.tensor([0, 0, 0], device=device, dtype=dtype)
    dim = (mx - mn + (2*off + 1))
    mat = mat.mm(torch.tensor([[1, 0, 0, mn[0] - (off[0] + 1)],
                               [0, 1, 0, mn[1] - (off[1] + 1)],
                               [0, 0, 1, mn[2] - (off[2] + 1)],
                               [0, 0, 0, 1]], device=device, dtype=dtype))
    # To input data type
    Mat = Mat.type(dtype0)
    Dim = Dim.type(dtype0)
    vx = vx.type(dtype0)
    return dim, mat, vx


def noise_estimate(pth_nii, show_fit=False, fig_num=1, num_class=2,
                   mu_noise=None, max_iter=10000, verbose=0):
    """ Estimate noise from a nifti image by fitting either a GMM or an RMM to the
        image's intensity histogram.

    Args:
        pth_nii (string): Path to nifti file.
        show_fit (bool, optional): Defaults to False.
        fig_num (bool, optional): Defaults to 1.
        num_class (int, optional): Number of mixture classes (only for GMM).
            Defaults to 2.
        mu_noise (float, optional): Mean of noise class, defaults to None,
            in which case the class with the smallest sd is assumed the noise
            class.
        max_iter (int, optional) Maxmimum number of algorithm iterations.
                Defaults to 10000.
        verbose (int, optional) Display progress. Defaults to 0.
            0: None.
            1: Print summary when finished.
            2: 1 + Log-likelihood plot.
            3: 1 + 2 + print convergence.

    Returns:
        sd_noise (torch.Tensor): Standard deviation of background class.
        sd_not_noise (torch.Tensor): Standard deviation of foreground class.
        mu_noise (torch.Tensor): Mean of background class.
        mu_not_noise (torch.Tensor): Mean of foreground class.

    """
    if isinstance(pth_nii, torch.Tensor):
        X = pth_nii
    else:  # Load data from nifti
        nii = nib.load(pth_nii)
        X = torch.tensor(nii.get_fdata())
        X = X.flatten()
    device = X.device
    X = X.double()

    # Mask
    X = X[(X != 0) & (torch.isfinite(X))]

    # Bin and make x grid
    mn = torch.min(X).int()
    mx = torch.max(X).int()
    bins = mx - mn
    W = torch.histc(X, bins=bins + 1, min=mn, max=mx)
    X = torch.arange(mn, mx + 1, device=device).double()

    if mn < 0:  # Make GMM model
        model = GMM(num_class=num_class)
    else:  # Make RMM model
        model = RMM(num_class=2)

    # Fit GMM using Numpy
    model.fit(X, W=W, verbose=verbose, max_iter=max_iter, show_fit=show_fit)

    # Get means and mixing proportions
    mu, _ = model.get_means_variances()
    mu = mu.squeeze()
    mp = model.mp
    if mn < 0:  # GMM
        sd = torch.sqrt(model.Cov).squeeze()
    else:  # RMM
        sd = model.sig

    # Get std and mean of noise class
    if mu_noise:
        # Closest to mu_bg
        _, ix_noise = torch.min(torch.abs(mu - mu_noise), dim=0)
        mu_noise = mu[ix_noise]
        sd_noise = sd[ix_noise]
    else:
        # With smallest sd
        sd_noise, ix_noise = torch.min(sd, dim=0)
        mu_noise = mu[ix_noise]
    # Get std and mean of other classes (means and sds weighted by mps)
    rng = torch.arange(0, num_class, device=device)
    rng = torch.cat([rng[0:ix_noise], rng[ix_noise + 1:]])
    mu1 = mu[rng]
    sd1 = sd[rng]
    w = mp[rng]
    w = w / torch.sum(w)
    mu_not_noise = sum(w * mu1)
    sd_not_noise = sum(w * sd1)

    return sd_noise, sd_not_noise, mu_noise, mu_not_noise
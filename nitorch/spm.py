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
from nitorch.mixtures import GMM
from nitorch.mixtures import RMM
import numpy as np
from numpy.linalg import cholesky as chol
from numpy.linalg import det
from scipy.linalg import logm, expm
from scipy.linalg import inv
import torch


__all__ = ['affine', 'def2sparse', 'dexpm', 'imatrix', 'matrix',
           'mean_matrix', 'mean_space', 'noise_estimate']


def affine(dm, mat, dtype=torch.float32, device='cpu'):
    """ Generate an affine warp on a lattice defined by dm and mat.

    Args:
        dm (torch.Size): Image dimensions (dm[0], dm[1], dm[2]).
        mat (torch.Tensor): Affine transform.
        dtype (torch.dtype, optional): Defaults to torch.float32.
        device (string, optional): Defaults to 'cpu'.

    Returns:
        y (torch.Tensor): Affine warp (1, dm[0], dm[1], dm[2], 3).

    """
    mat = mat.type(dtype)
    y = identity(dm, dtype=dtype, device=device)
    y = torch.reshape(y, (dm[0]*dm[1]*dm[2], 3))
    y = torch.matmul(y, torch.t(mat[0:3, 0:3])) + torch.t(mat[0:3, 3])
    y = torch.reshape(y, (dm[0], dm[1], dm[2], 3))
    if dm[0] == 1:
        y[:, :, :, 0] = 0
    y = y[None, ...]
    return y


def affine_basis(basis='SE', dim=3):
    """ Generate a basis for the Lie algebra of affine matrices.

    Args:
        basis (string, optional): Name of the group that must be encoded:
            . 'T': Translation
            . 'SO': Special orthogonal (= translation + rotation)
            . 'SE': Special Euclidean (= translation + rotation + scale)
            Defaults to 'SE'.
        dim (int, optional): Basis dimensions: 0, 2, 3. Defaults to 3.

    Returns:
        B (np.array(double)): Affine basis (4, 4, N)

    """
    if dim == 0:
        B = np.zeros((4, 4, 0))
    elif dim == 2:
        if basis == 'T':
            B = np.zeros((4, 4, 2))
            B[0, 3, 0] = 1
            B[1, 3, 1] = 1
        elif basis == 'SO':
            B = np.zeros((4, 4, 1))
            B[0, 1, 0] = 1
            B[1, 0, 0] = -1
        elif basis == 'SE':
            B = np.zeros((4, 4, 3))
            B[0, 3, 0] = 1
            B[1, 3, 1] = 1
            B[0, 1, 2] = 1
            B[1, 0, 2] = -1
        else:
            raise ValueError('Unknown group!')
    else:
        if basis == 'T':
            B = np.zeros((4, 4, 3))
            B[0, 3, 0] = 1
            B[1, 3, 1] = 1
            B[2, 3, 2] = 1
        elif basis == 'SO':
            B = np.zeros((4, 4, 3))
            B[0, 1, 0] = 1
            B[1, 0, 0] = -1
            B[0, 2, 1] = 1
            B[2, 0, 1] = -1
            B[1, 2, 2] = 1
            B[2, 1, 2] = -1
        elif basis == 'SE':
            B = np.zeros((4, 4, 6))
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
        phi (torch.Tensor): Deformation (dm_out[0], dm_out[1], dm_out[2], 3).
        dm_in (torch.Size): Size of input FOV (dm_in[0], dm_in[1], dm_in[2]).
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


def dexpm(A, dA=None):
    """ Differentiate a matrix exponential.

    Args:
        A (np.array()): Lie algebra (1, B).
        dA (np.array(), optional): Basis function to differentiate
            with respect to (4, 4, B). Defaults to None.

    Returns:
        E (np.array()): expm(A) (4, 4).
        dE (np.array()): Derivative of expm(A) (4, 4, B).

    Authors:
        John Ashburner, as part of the SPM12 software.

    """
    if dA is None:
        dA = np.zeros((1, len(A), 0))
    else:
        p = A.flatten(order='F')
        A = np.zeros((4, 4))
        for m in range(dA.shape[2]):
            A += p[m]*dA[..., m]
    An = np.copy(A)
    E = np.eye(4) + A

    dAn = np.copy(dA)
    dE = np.copy(dA)
    for k in range(2, 10001):
        for m in range(dA.shape[2]):
            dAn[..., m] = (np.matmul(dAn[..., m], A) + np.matmul(An, dA[..., m]))/k
        dummy = 1
        dE += dAn
        An = np.matmul(An, A)/k
        E += An
        if np.sum(An**2) < A.size*1e-32:
            break

    return E, dE


def identity(dm, dtype=torch.float32, device='cpu'):
    """ Generate the identity warp on a lattice defined by dm.

    Args:
        dm (torch.Size): Defines the size of the output lattice (dm[0], dm[1], dm[2]).
        dtype (torch.dtype, optional): Defaults to torch.float32.
        device (string, optional): Defaults to 'cpu'.

    Returns:
        id_grid (torch.Tensor): Identity warp (dm[0], dm[1], dm[2], 3).

    """
    id_grid = torch.zeros((dm[0], dm[1], dm[2], 3), dtype=dtype, device=device)
    id_grid[:, :, :, 2], id_grid[:, :, :, 1], id_grid[:, :, :, 0] = \
        torch.meshgrid([torch.arange(0, dm[0], dtype=dtype, device=device),
                        torch.arange(0, dm[1], dtype=dtype, device=device),
                        torch.arange(0, dm[2], dtype=dtype, device=device)])
    return id_grid


def imatrix(M):
    """ Return the parameters for creating an affine transformation matrix.

    Args:
        mat (np.array()): Affine transformation matrix (4, 4).

    Returns:
        P (np.array()): Affine parameters (<=12).

    Authors:
        John Ashburner & Stefan Kiebel, as part of the SPM12 software.

    """
    # Translations and Zooms
    R = M[:-1, :-1]
    C = chol(np.matmul(R.transpose(), R))
    C = C.transpose()
    d = np.diag(C)
    P = np.array([M[0, 3], M[1, 3], M[2, 3], 0, 0, 0, d[0], d[1], d[2], 0, 0, 0])
    if det(R) < 0:  P[6] = -P[6]  # Fix for -ve determinants
    # Shears
    C = np.matmul(inv(np.diag(np.diag(C))), C)
    P[9] = C[0, 1]
    P[10] = C[0, 2]
    P[11] = C[1, 2]
    R0 = matrix(np.array([0, 0, 0, 0, 0, 0, P[6], P[7], P[8], P[9], P[10], P[11]]))
    R0 = R0[:-1, :-1]
    R1 = np.matmul(R, inv(R0))  # This just leaves rotations in matrix R1
    # Correct rounding errors
    rang = lambda x: np.minimum(np.maximum(x, -1), 1)
    P[4] = np.arcsin(rang(R1[0, 2]))
    if (np.abs(P[4]) - np.pi/2)**2 < 1e-9:
        P[3] = 0
        P[5] = np.arctan2(-rang(R1[1, 0]), rang(-R1[2, 0]/R1[0, 2]))
    else:
        c = np.cos(P[4])
        P[3] = np.arctan2(rang(R1[1, 2]/c), rang(R1[2, 2]/c))
        P[5] = np.arctan2(rang(R1[0, 1]/c), rang(R1[0, 0]/c))

    return P


def matrix(P):
    """ Return an affine transformation matrix.

    Args:
        P (np.array()): Affine parameters (<=12).

    Returns:
        mat (np.array()): Affine transformation matrix (4, 4).

    Authors:
        John Ashburner & Stefan Kiebel, as part of the SPM12 software.

    """
    if len(P) == 3:  # Special case: translation only
        A = np.eye(4)
        A[:-1, -1] = P
        return A
    # Pad P with 'null' parameters
    q = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    P = np.concatenate((P, q[len(P):]))
    # Translations
    T = np.array([[1, 0, 0, P[0]],
                  [0, 1, 0, P[1]],
                  [0, 0, 1, P[2]],
                  [0, 0, 0, 1]])
    # Rotations
    c = lambda x: np.cos(x)
    s = lambda x: np.sin(x)
    R0 = np.array([[1, 0, 0, 0],
                   [0, c(P[3]), s(P[3]), 0],
                   [0, -s(P[3]), c(P[3]), 0],
                   [0, 0, 0, 1]])
    R1 = np.array([[c(P[4]), 0, s(P[4]), 0],
                   [0, 1, 0, 0],
                   [-s(P[4]), 0, c(P[4]), 0],
                   [0, 0, 0, 1]])
    R2 = np.array([[c(P[5]), s(P[5]), 0, 0],
                   [-s(P[5]), c(P[5]), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    R = np.matmul(R0, np.matmul(R1, R2))
    # Scales
    Z = np.array([[P[6], 0,    0,    0],
                  [0,    P[7], 0,    0],
                  [0,    0,    P[8], 0],
                  [0,    0,    0,    1]])
    # Shears
    S = np.array([[1, P[9], P[10], 0],
                  [0, 1,    P[11], 0],
                  [0, 0,    1,     0],
                  [0, 0,    0,     1]])

    return np.matmul(T, np.matmul(R, np.matmul(Z, S)))


def mean_matrix(Mat):
    """ Compute barycentre of matrix exponentials.

    Args:
        Mat (numpy.array()): N subjects' orientation matrices (4, 4, N).

    Returns:
        mat (numpy.array()): The resulting mean (4, 4).

    Authors:
        John Ashburner, as part of the SPM12 software.

    """
    N = Mat.shape[2]
    mat = np.eye(4)
    for iter in range(1024):
        S = np.zeros((4, 4))
        for n in range(N):
            L = np.real(logm(np.matmul(inv(mat), Mat[..., n])))
            S = S + L
        S /= N
        mat = np.matmul(mat, expm(S))
        if np.sum(S**2) < 1e-20:
            break
    return mat


def mean_space(Mat, Dim, vx=None, mod_prct=0):
    """ Compute the (mean) model space from individual spaces.

    Args:
        Mat (numpy.array()): N subjects' orientation matrices (4, 4, N).
        Dim (numpy.array()): N subjects' dimensions (3, N).
        vx (numpy.array(), optional): Voxel size (3). Defaults to all np.inf.
        mod_prct (float, optional): Percentage to either crop or pad mean space.
            A negative value crops and a positive value pads. Should be a value
            between 0 and 1. Defaults to 0.

    Returns:
        dim (numpy.array()): Mean dimensions (3).
        mat (numpy.array()): Mean orientation matrix (4, 4).
        vx (numpy.array(), optional): Mean voxel size (3).

    Authors:
        John Ashburner, as part of the SPM12 software.

    """
    if vx is None:
        vx = np.array([np.inf, np.inf, np.inf])
    if type(vx) is float or type(vx) is int:
        vx = (vx,)*3
    if type(vx) is tuple and len(vx) == 3:
        vx = np.array([vx[0], vx[1], vx[2]])

    N = Mat.shape[2]  # Number of subjects
    # Get affine basis
    basis = 'SE'
    dim = 3 if Dim[2, 0] > 1 else 2
    B = affine_basis(basis, dim)
    """ Find combination of 90 degree rotations and flips that brings all
        the matrices closest to axial
    """
    Mat0 = np.copy(Mat)
    pmatrix = np.array([[0, 1, 2],
                        [1, 0, 2],
                        [2, 0, 1],
                        [2, 1, 0],
                        [0, 2, 1],
                        [1, 2, 0]])
    is_flipped = det(Mat[:3, :3, 0]) < 0  # For retaining flips

    for n in range(N):  # Loop over subjects
        vx1 = _voxsize(Mat[..., n])
        R = np.matmul(Mat[..., n], inv(np.diag([vx1[0], vx1[1], vx1[2], 1])))[:-1, :-1]
        minss = np.inf
        minR = np.eye(3)
        for i in range(6):  # Permute (= 'rotate + flip') axes
            R1 = np.zeros((3, 3))
            R1[pmatrix[i, 0], 0] = 1
            R1[pmatrix[i, 1], 1] = 1
            R1[pmatrix[i, 2], 2] = 1
            for j in range(8):  # Mirror (= 'flip') axes
                F = np.diag([np.bitwise_and(j, 1)*2 - 1,
                             np.bitwise_and(j, 2) - 1,
                             np.bitwise_and(j, 4)/2 - 1])
                R2 = np.matmul(F, R1)
                ss = np.sum(np.sum((np.matmul(R, inv(R2)) - np.eye(3))**2))
                if ss < minss:
                    minss = ss
                    minR = R2
        rdim = np.abs(np.matmul(minR, Dim[..., n][..., None]))
        R2 = inv(minR)
        R22 = np.matmul(R2, 0.5*(np.sum(R2, axis=0, keepdims=True).transpose() - 1)*(rdim + 1))
        minR = np.concatenate((R2, R22), axis=1)
        minR = np.concatenate((minR, np.array([0, 0, 0, 1])[None, ...]), axis=0)
        Mat[..., n] = np.matmul(Mat[..., n], minR)
    """ Average of the matrices in Mat
    """
    mat = mean_matrix(Mat)
    """ If average involves shears, then find the closest matrix that does not
        require them.
    """
    p = imatrix(mat)
    if np.sum(p[[9, 10, 11]]**2) > 1e-8:
        B2 = np.zeros((4, 4, 3))
        B2[0, 0, 0] = 1
        B2[1, 1, 1] = 1
        B2[2, 2, 2] = 1

        p = np.zeros(9)
        for it in range(10000):
            R, dR = dexpm(p[[0, 1, 2, 3, 4, 5]], B)  # Rotations + Translations
            Z, dZ = dexpm(p[[6, 7, 8]], B2)  # Zooms

            M = np.matmul(R, Z)
            dM = np.zeros((4, 4, 9))
            for n in range(6):
                dM[..., n] = np.matmul(dR[..., n], Z)
            for n in range(3):
                dM[..., 6 + n] = np.matmul(R, dZ[..., n])
            dM = np.reshape(dM, (16, 9))

            d = M.flatten(order='C') - mat.flatten(order='C')
            gr = np.matmul(dM.transpose(), d)
            Hes = np.matmul(dM.transpose(), dM)
            p = p - np.matmul(inv(Hes), gr)
            if np.sum(gr**2) < 1e-8:
                break
        mat = np.copy(M)

    # Set required voxel size
    vx_out = np.copy(vx)
    vx = _voxsize(mat)
    vx_out[~np.isfinite(vx_out)] = vx[~np.isfinite(vx_out)]
    # ensure isotropic
    # vx_out = np.mean(vx_out).repeat(3)
    # vx_out = np.round(vx_out, 2)
    mat = np.matmul(mat, np.diag([vx_out[0]/vx[0], vx_out[1]/vx[1], vx_out[2]/vx[2], 1]))
    vx = _voxsize(mat)
    # Ensure that the FoV covers all images, with a few voxels to spare
    mn =  np.array([np.inf, np.inf, np.inf])
    mx = -np.array([np.inf, np.inf, np.inf])
    for n in range(N):
        dm = Dim[..., n]
        corners = np.array([[1, dm[0], 1, dm[0], 1, dm[0], 1, dm[0]],
                            [1, 1, dm[1], dm[1], 1, 1, dm[1], dm[1]],
                            [1, 1, 1, 1, dm[2], dm[2], dm[2], dm[2]],
                            [1, 1, 1, 1, 1, 1, 1, 1]])
        M = np.matmul(inv(mat), Mat0[..., n])
        vx1 = np.matmul(M[:-1,:], corners)
        mx = np.maximum(mx, np.max(vx1, axis=1))
        mn = np.minimum(mn, np.min(vx1, axis=1))
    mx = np.ceil(mx)
    mn = np.floor(mn)
    # Either pad or crop mean space
    off = mod_prct*(mx - mn)
    if Dim[2, 0] == 1: off = np.array([0, 0, 0])
    dim = (mx - mn + (2*off + 1))
    mat = np.matmul(mat,
        np.array([[1, 0, 0, mn[0] - (off[0] + 1)],
                  [0, 1, 0, mn[1] - (off[1] + 1)],
                  [0, 0, 1, mn[2] - (off[2] + 1)],
                  [0, 0, 0, 1]]))

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
    if type(pth_nii) is torch.Tensor:
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
    model.fit(X, W=W, verbose=verbose, max_iter=max_iter)

    if show_fit:  # Plot fit
        model.plot_fit(X, fig_num=fig_num, W=W, suptitle='Histogram fit')

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


def _voxsize(mat):
    """ Compute voxel size from affine matrix.

    Args:
        mat (np.array()): Affine matrix (4, 4).

    Returns:
        vx (np.array()): Voxel size (3).

    """
    return np.sqrt(np.sum(mat[:-1, :-1]**2, axis=0))



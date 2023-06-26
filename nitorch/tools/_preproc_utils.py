"""Pre-processing utility functions.

"""
import os
import torch
from nitorch.spatial import (affine_grid, affine_basis, affine_matrix_classic,
                             grid_pull, voxel_size)
from nitorch.io import map, savef
from nitorch.core.py import file_mod
from nitorch.core.constants import pi
from nitorch.core.linalg import (meanm, _expm, lmdiv, cholesky)


def _format_input(img, device='cpu', rand=False, cutoff=None):
    """Format preprocessing input data.
    """
    if isinstance(img, str):
        img = [img]
    if isinstance(img, list) and isinstance(img[0], torch.Tensor):
        img = [img]
    file = []
    dat = []
    mat = []
    for n in range(len(img)):
        if isinstance(img[n], str):
            # Input are nitorch.io compatible paths
            file.append(map(img[n]))
            dat.append(file[n].fdata(dtype=torch.float32, device=device,
                                     rand=rand, cutoff=cutoff))
            mat.append(file[n].affine.to(dtype=torch.float64, device=device))
        else:
            # Input are tensors (clone so to not modify input data)
            file.append(None)
            dat.append(img[n][0].to(dtype=torch.float32, device=device, copy=True))
            mat.append(img[n][1].to(dtype=torch.float64, device=device, copy=True))

    return dat, mat, file


def _process_reg(dat, mat, mat_a, mat_fix, dim_fix, write):
    """Process registration results.
    """
    N = len(dat)
    dim = len(dim_fix)
    rdat = torch.zeros((N, ) + dim_fix,
                       dtype=dat[0].dtype, device=dat[0].device)
    for n in range(N):  # loop over input images
        if torch.all(mat_a[n] - torch.eye(dim + 1, device=mat_a[n].device) == 0):
            rdat[n] = dat[n]
        else:
            mat_r = lmdiv(mat[n], mat_a[n].mm(mat_fix))
            rdat[n] = _reslice_dat_3d(dat[n], mat_r, dim_fix)
        if write == 'reslice':
            dat[n] = rdat[n]
            mat[n] = mat_fix
        elif write == 'affine':
            mat[n] = lmdiv(mat_a[n], mat[n])
    # Write output to disk?
    if write in ['reslice', 'affine']:
        write = True
    else:
        write = False

    return dat, mat, write, rdat


def _write_output(dat, mat, file=None, prefix='', odir=None, nam=None):
    """Write preprocessed output to disk.
    """
    if odir is not None:
        os.makedirs(odir, exist_ok=True)
    pth = []
    for n in range(len(dat)):
        if file[n] is not None:
            filename = file[n].filename()
        else:
            filename = 'nitorch_file.nii.gz'
        pth.append(file_mod(filename,
            odir=odir, prefix=prefix, nam=nam))
        savef(dat[n], pth[n], like=file[n], affine=mat[n])
    if len(dat) == 1:
        pth = pth[0]

    return pth


def _msk_fov(dat, mat, mat0, dim0):
    """Mask field-of-view (FOV) of image data according to other image's
    FOV.

    Parameters
    ----------
    dat : (X, Y, Z), tensor
        Image data.
    mat : (4, 4), tensor
        Image's affine.
    mat0 : (4, 4), tensor
        Other image's affine.
    dim0 : (3, ), list/tuple
        Other image's dimensions.

    Returns
    -------
    dat : (X, Y, Z), tensor
        Masked image data.

    """
    dim = dat.shape
    M = lmdiv(mat0, mat)  # mat0\mat1
    grid = affine_grid(M, dim)
    msk = (grid[..., 0] >= 1) & (grid[..., 0] <= dim0[0]) & \
          (grid[..., 1] >= 1) & (grid[..., 1] <= dim0[1]) & \
          (grid[..., 2] >= 1) & (grid[..., 2] <= dim0[2])
    dat[~msk] = 0

    return dat


def _get_corners_3d(dim, o=[0] * 6):
    """Get eight corners of 3D image volume.

    Parameters
    ----------
    dim : (3,), list or tuple
        Image dimensions.
    o : (6, ), default=[0] * 6
        Offsets.

    Returns
    -------
    c : (8, 4), tensor_like[torch.float64]
        Corners of volume.

    """
    # Get corners
    c = torch.tensor(
        [[     1,      1,      1, 1],
         [     1,      1, dim[2], 1],
         [     1, dim[1],      1, 1],
         [     1, dim[1], dim[2], 1],
         [dim[0],      1,      1, 1],
         [dim[0],      1, dim[2], 1],
         [dim[0], dim[1],      1, 1],
         [dim[0], dim[1], dim[2], 1]])
    # Include offset
    # Plane 1
    c[0, 0] = c[0, 0] + o[0]
    c[1, 0] = c[1, 0] + o[0]
    c[2, 0] = c[2, 0] + o[0]
    c[3, 0] = c[3, 0] + o[0]
    # Plane 2
    c[4, 0] = c[4, 0] - o[1]
    c[5, 0] = c[5, 0] - o[1]
    c[6, 0] = c[6, 0] - o[1]
    c[7, 0] = c[7, 0] - o[1]
    # Plane 3
    c[0, 1] = c[0, 1] + o[2]
    c[1, 1] = c[1, 1] + o[2]
    c[4, 1] = c[4, 1] + o[2]
    c[5, 1] = c[5, 1] + o[2]
    # Plane 4
    c[2, 1] = c[2, 1] - o[3]
    c[3, 1] = c[3, 1] - o[3]
    c[6, 1] = c[6, 1] - o[3]
    c[7, 1] = c[7, 1] - o[3]
    # Plane 5
    c[0, 2] = c[0, 2] + o[4]
    c[2, 2] = c[2, 2] + o[4]
    c[4, 2] = c[4, 2] + o[4]
    c[6, 2] = c[6, 2] + o[4]
    # Plane 6
    c[1, 2] = c[1, 2] - o[5]
    c[3, 2] = c[3, 2] - o[5]
    c[5, 2] = c[5, 2] - o[5]
    c[7, 2] = c[7, 2] - o[5]

    return c


def _reslice_dat_3d(dat, affine, dim_out, interpolation='linear',
                    bound='zero', extrapolate=False):
    """Reslice 3D image data.

    Parameters
    ----------
    dat : (Xi, Yi, Zi), tensor_like
        Input image data.
    affine : (4, 4), tensor_like
        Affine transformation that maps from voxels in output image to
        voxels in input image.
    dim_out : (Xo, Yo, Zo), list or tuple
        Output image dimensions.
    interpolation : str, default='linear'
        Interpolation order.
    bound : str, default='zero'
        Boundary condition.
    extrapolate : bool, default=False
        Extrapolate out-of-bounds data.

    Returns
    -------
    dat : (dim_out), tensor_like
        Resliced image data.

    """
    grid = affine_grid(affine, dim_out).type(dat.dtype)
    grid = grid[None, ...]
    dat = dat[None, None, ...]
    dat = grid_pull(dat, grid,
        bound=bound, interpolation=interpolation, extrapolate=extrapolate)
    dat = dat[0, 0, ...]

    return dat


def _mean_space(Mat, Dim, vx=None):
    """Compute a (mean) model space from individual spaces.

    Args:
        Mat (torch.tensor): N subjects' orientation matrices (N, 4, 4).
        Dim (torch.tensor): N subjects' dimensions (N, 3).
        vx (torch.tensor|tuple|float, optional): Voxel size (3,), defaults to None (estimate from input).

    Returns:
        mat (torch.tensor): Mean orientation matrix (4, 4).
        dim (torch.tensor): Mean dimensions (3,).
        vx (torch.tensor): Mean voxel size (3,).

    Authors:
        John Ashburner, as part of the SPM12 software.

    """
    device = Mat.device
    dtype = Mat.dtype
    N = Mat.shape[0]  # Number of subjects
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
    dim = 3 if Dim.shape[1] > 2 else 2
    B = affine_basis(basis, dim, device=device, dtype=dtype)

    # Find combination of 90 degree rotations and flips that brings all
    # the matrices closest to axial
    Mat0 = Mat.clone()
    pmatrix = torch.tensor([[0, 1, 2],
                            [1, 0, 2],
                            [2, 0, 1],
                            [2, 1, 0],
                            [0, 2, 1],
                            [1, 2, 0]], device=device)

    for n in range(N):  # Loop over subjects
        vx1 = voxel_size(Mat[n, ...])
        R = Mat[n, ...].mm(
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
        rdim = torch.abs(minR.mm(Dim[n, ...][..., None]-1))
        R2 = minR.inverse()
        R22 = R2.mm(
            (torch.div(torch.sum(R2, dim=0, keepdim=True).t(), 2, rounding_mode='floor') - 1)*rdim            
        )
        minR = torch.cat((R2, R22), dim=1)
        minR = torch.cat((minR, torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)[None, ...]), dim=0)
        Mat[n, ...] = Mat[n, ...].mm(minR)

    # Average of the matrices in Mat
    mat = meanm(Mat)

    # If average involves shears, then find the closest matrix that does not
    # require them.
    C_ix = torch.tensor([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
                        device=device)  # column-major ordering from (4, 4) tensor
    p = _imatrix(mat)
    if torch.sum(p[[9, 10, 11]]**2) > 1e-8:
        B2 = torch.zeros((3, 4, 4), device=device, dtype=dtype)
        B2[0, 0, 0] = 1
        B2[1, 1, 1] = 1
        B2[2, 2, 2] = 1

        p = torch.zeros(9, device=device, dtype=dtype)
        for n_iter in range(10000):
            # Rotations + Translations
            R, dR = _expm(p[[0, 1, 2, 3, 4, 5]], B, grad_X=True)
            # Zooms
            Z, dZ = _expm(p[[6, 7, 8]], B2, grad_X=True)

            M = R.mm(Z)
            dM = torch.zeros((4, 4, 9), device=device, dtype=dtype)
            for n in range(6):
                dM[..., n] = dR[n, ...].mm(Z)
            for n in range(3):
                dM[..., 6 + n] = R.mm(dZ[n, ...])
            dM = dM.reshape((16, 9))
            d = M.flatten() - mat.flatten()
            gr = dM.t().mm(d[..., None])
            Hes = dM.t().mm(dM)
            p = p - lmdiv(Hes, gr)[:, 0]
            if torch.sum(gr**2) < 1e-8:
                break
        mat = M.clone()

    # Set required voxel size
    vx_out = vx.clone()
    vx = voxel_size(mat)
    vx_out[~torch.isfinite(vx_out)] = vx[~torch.isfinite(vx_out)]
    mat = mat.mm(torch.cat((vx_out/vx, one[..., None])).diag())
    vx = voxel_size(mat)

    # Ensure that the FoV covers all images, with a few voxels to spare
    mn_all = torch.zeros([3, N], device=device, dtype=dtype)
    mx_all = torch.zeros([3, N], device=device, dtype=dtype)
    for n in range(N):
        dm = Dim[n, ...]
        corners = torch.tensor([[1, dm[0], 1, dm[0], 1, dm[0], 1, dm[0]],
                                [1, 1, dm[1], dm[1], 1, 1, dm[1], dm[1]],
                                [1, 1, 1, 1, dm[2], dm[2], dm[2], dm[2]],
                                [1, 1, 1, 1, 1, 1, 1, 1]],
                               device=device, dtype=dtype)
        M = lmdiv(mat, Mat0[n])
        vx1 = M[:-1, :].mm(corners)
        mx_all[..., n] = torch.max(vx1, dim=1)[0]
        mn_all[..., n] = torch.min(vx1, dim=1)[0]
    mx = mx_all.max(dim=1)[0]
    mn = mn_all.min(dim=1)[0]
    mx = torch.ceil(mx)
    mn = torch.floor(mn)

    # Make output dimensions and orientation matrix
    dim = mx - mn + 1  # Output dimensions
    off = torch.tensor([0, 0, 0], device=device, dtype=dtype)
    mat = mat.mm(torch.tensor([[1, 0, 0, mn[0] - (off[0] + 1)],
                               [0, 1, 0, mn[1] - (off[1] + 1)],
                               [0, 0, 1, mn[2] - (off[2] + 1)],
                               [0, 0, 0, 1]], device=device, dtype=dtype))

    return mat, dim, vx


def _imatrix(M):
    """Return the parameters for creating an affine transformation matrix.

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
    # Translations and Zooms
    R = M[:-1, :-1]
    C = cholesky(R.t().mm(R))
    C = C.t()
    d = torch.diag(C)
    P = torch.tensor([M[0, 3], M[1, 3], M[2, 3], 0, 0, 0, d[0], d[1], d[2], 0, 0, 0],
                     device=device, dtype=dtype)
    if R.det() < 0:  # Fix for -ve determinants
        P[6] = -P[6]
    # Shears
    C = lmdiv(torch.diag(torch.diag(C)), C)
    P[9] = C[0, 1]
    P[10] = C[0, 2]
    P[11] = C[1, 2]
    R0 = affine_matrix_classic(
        torch.tensor([0, 0, 0, 0, 0, 0, P[6], P[7], P[8], P[9], P[10],
                      P[11]])).to(device)
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

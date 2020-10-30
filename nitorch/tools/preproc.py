"""Various functions for pre-processing of neuroimaging data.

All these functions work directly with the path(s) to the image data.
The input data is not overwritten by default, but copies are made with
function specific prefix. However, by setting the prefix to an empty string,
the original input data is modified.

"""


import math
import nibabel as nib
from pathlib import Path
import os
import torch
from torch.nn import functional as F
from ..core.kernels import smooth
from ..core.utils import pad
from ..spatial import voxel_size, grid_pull
from .spm import identity, matrix
from .spm import affine as apply_affine


def load_3d(img, samp=0, truncate=False, fwhm=0.0, mx_out=None, device='cpu', dtype=torch.float32,
           do_mask=True, do_smooth=True):
    """Load image volume (3D) for subsequent image processing.

    Parameters
    ----------
    img : str | (X, Y, Z), tensor_like | ((X, Y, Z), tensor_like, (4 4), tensor_like)
        Can be:
        - Path to nibabel compatible file
        - 3D image volume as tensor
        - Tuple with two tensors: 3D image volume and affine matrix

    samp : int, default=0
        Sub-sampling of image data.

    truncate : bool, default=False
        Truncate image data based on percentiles.

    fwhm : float, default=0.0
        Smoothness estimate for computing a fudge factor.

    mx_out : float, default=None
        Max output value. If None, uses max in input image.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    dtype : torch.dtype, default=torch.float64
        Data type of function output.

    do_mask : bool, default=True
        Set not finite, zeros and minimum values to NaN.

    do_smooth : bool, default=True
        If sub-sampling (samp > 0), smooth data a bit.

    Returns
    ----------
    dat : (dim_out), tensor_like[dtype]
        Preprocessed image data.

    affine : (4, 4), tensor_like[torch.float64]
        Image affine transformation.

    grid : (dim_out, 3), tensor_like[dtype]
        Grid with voxel locations in input image (possibly sampled).

    mask : (dim_out), tensor_like[bool]
        Image mask.

    ff : tensor_like[dtype]
        Fudge Factor to (approximately) account for
        non-independence of voxels.

    """
    device = _get_device(device)

    # Get image data
    if isinstance(img, str):
        # Input is filepath
        nii = nib.load(img, mmap=False)
        dat = torch.tensor(nii.get_fdata(), device=device, dtype=dtype)
        affine = torch.from_numpy(nii.affine).type(torch.float64).to(device)
        dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        if nii.get_data_dtype() in dtypes:
            scrand = nii.dataobj.slope
        else:
            scrand = 0;
    elif isinstance(img, tuple):
        # Input is tuple with two tensors: image data and affine tensors
        dat = img[0]
        affine = img[1]
        scrand = 1  # As we don't know the original data type, we add random noise
    elif isinstance(img, torch.Tensor):
        dat = img
        affine = _get_origin_mat(dat.shape, device=device, dtype=dtype)
        scrand = 1  # As we don't know the original data type, we add random noise
    else:
        raise AttributeError('Input error!')

    # Sanity check
    dim = dat.shape
    if len(dim) != 3:
        raise ValueError('Input image is {}D, should be 3D!'.format(len(dim)))
    dim = dim[:3]
    vx = voxel_size(affine)

    # Get sampling grid
    samp = torch.tensor((samp,) * 3, device=device, dtype=torch.float64)
    ones = torch.ones(3, device=device, dtype=torch.float64)
    sk = torch.max(ones , torch.round(samp*ones/vx))
    grid = identity(dim, step=sk.cpu().int().tolist(), device=device, dtype=dtype)

    # Compute fudge factor
    lntwo = torch.tensor(2, dtype=dtype, device=device).log()
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    s = (torch.tensor(fwhm, dtype=dtype, device=device) + vx.mean())/torch.sqrt(8 * lntwo)
    s = s.expand((3,))
    ff = (4*pi*(s/vx/sk)**2 + 1).prod().sqrt()

    if samp[0] > 0:
        # Subsample image data with nearest neighbour interpolation
        dat = grid_pull(dat[None, None, ...], grid[None, ...],
                        bound='zero', interpolation='nearest')[0, 0, ...]
    # Get image dimensions
    dim = tuple(dat.shape[:3])

    # Get mask
    mask = (torch.isfinite(dat)) & (dat != 0) & (dat > dat.min())

    if truncate:
        # For truncating based on percentiles
        mn_out = 0
        if mx_out is None:
            mx_out = dat.max()
        else:
            mx_out = torch.tensor(mx_out, dtype=dtype, device=device)

        mn = torch.tensor([dat.min(), 1], dtype=dtype, device=device)[None, ...]
        mx = torch.tensor([mx_out, 1], dtype=dtype, device=device)[None, ...]
        sf = torch.cat((mn, mx), dim=0)
        sf = torch.tensor([1, 4000], dtype=dtype, device=device)[..., None].solve(sf)[0].squeeze()

        p = dat[mask]
        p = (p*sf[0] + sf[1]).round().int()
        h = torch.bincount(p, weights=None, minlength=4000 - 1)
        h = h.cumsum(0) / h.sum().type(dtype)

        mn = ((h <= 0.0005).sum(dim=0) - sf[1]) / sf[0]
        mx = ((h <= 0.9995).sum(dim=0) - sf[1]) / sf[0]

        mn = torch.tensor([mn, 1], dtype=dtype, device=device)[None, ...]
        mx = torch.tensor([mx, 1], dtype=dtype, device=device)[None, ...]
        sf = torch.cat((mn, mx), dim=0)
        sf = torch.tensor([mn_out, mx_out], dtype=dtype, device=device)[..., None].solve(sf)[0].squeeze()

    # Possibly add some random noise
    if scrand:
        torch.manual_seed(0)
    scrand = torch.tensor(scrand, dtype=dtype, device=device)
    dat = dat + torch.rand_like(dat) * scrand - scrand / 2

    if truncate:
        # Truncate
        dat = dat*sf[0] + sf[1]

    if samp[0] > 0 and do_smooth:  # Smooth data a bit
        # Make smoothing kernel
        fwhm_smo = torch.sqrt(torch.max(samp ** 2 - vx ** 2, torch.zeros(3, device=device, dtype=torch.float64))) / vx
        smo = smooth(('gauss',) * 3, fwhm=fwhm_smo, device=dat.device, dtype=dat.dtype, sep=False)
        # Padding amount for subsequent convolution
        p = (torch.tensor(smo.shape[2:]) - 1) // 2
        p = tuple(p.int().cpu().tolist())
        # Smooth deformation with Gaussian kernel (by convolving)
        dat = pad(dat, p, side='both')
        dat = F.conv3d(dat[None, None, ...], smo)[0, 0, ...]

    if truncate:
        dat = dat.round()
        dat = dat.clamp_min(mn_out).clamp_max(mx_out)

    if do_mask:
        # Mask data
        dat[~mask] = float('NaN')

    return dat, affine, grid, mask, ff


def modify_affine(pth, M, prefix='ma_', dir_out=None):
    """Modify affine in image header.

    Note that if the data is compressed, or the prefix option is non-empty, new data
    will be written to disk. Otherwise, the affine in the header will be modified in-place.

    Parameters
    ----------
    pth : str
        Path to nibabel compatible file.

    M : (4, 4), tensor_like
        New affine matrix.

    prefix : str, default='ma_'
        Filename prefix of modified image. If empty string, overwrites input file.

    dir_out : str, default=None
        Output directory of modified image.

    """
    nii = nib.load(pth)
    _, ext = os.path.splitext(pth)
    # To CPU
    M = M.cpu()
    if ext in ['.gz', '.bz2'] or prefix or dir_out:
        # Image is compressed, or prefix given, or dir_out given -> overwrite original file
        pth = write_img(pth, prefix=prefix, affine=M, dir_out=dir_out)
    else:
        # Modify affine in-place
        with nib.openers.Opener(pth, 'r+b', keep_open=True) as f:
            f.seek(0)
            nii.header.set_sform(M)
            nii.header.set_qform(M)
            nii.header.set_slope_inter(nii._data.slope, nii._data.inter)
            nii.header.write_to(f)

    return pth


def reset_origin(pth, vx=None, prefix='ro_', device='cpu', interpolation='linear', bound='zero'):
    """Reset affine matrix.

    OBS: Reslices image data.

    Parameters
    ----------
    pth : str
        Path to nibabel compatible file.

    vx : int or tuple[int], default=None
        Voxel size of resliced image. If None, uses same as input.

    prefix : str, default='ro_'
        Filename prefix of resliced image. If empty string, overwrites input file.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    write : bool, default=True
        Write resliced image to disk.

    interpolation : str, default='linear'
        Interpolation order.

    bound : str, default='zero'
        Boundary condition.

    Returns
    -------
    pth_out : str
        Path to reset image.

    """
    device = _get_device(device)
    # Reslice image data to world FOV
    pth, dat, M0 = reslice2world(pth, vx=vx, prefix=prefix, device=device, write=False,
                                 interpolation=interpolation, bound=bound)
    # Compute new, reset, affine matrix
    dim = torch.tensor(dat.shape, dtype=torch.float64)
    vx = voxel_size(M0)
    if M0[:3, :3].det() < 0:
        vx[0] = - vx[0]
    affine = _get_origin_mat(dim, vx, dtype=torch.float64)
    # Write reset image
    pth_out = write_img(pth, dat, affine, prefix=prefix)

    return pth_out


def reslice_dat(dat, affine, dim_out, interpolation='linear', bound='zero', extrapolate=False):
    """ Reslice image data.

    Parameters
    ----------
    dat : (Xi, Yi, Zi), tensor_like[dat.dtype] or array_like[dat.dtype]
        Input image data.

    affine : (4, 4), tensor_like[torch.float64]
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
    dat : (dim_out), tensor_like[dat.dtype]
        Output resliced image data.

    """
    dim_in = dat.shape
    if not isinstance(dat, torch.Tensor):
        dat = torch.from_numpy(dat)
    grid = apply_affine(dim_out, affine, device=dat.device, dtype=dat.dtype)
    dat = grid_pull(dat, grid, bound=bound, interpolation=interpolation, extrapolate=extrapolate)

    return dat[0, 0, ...]


def reslice2world(pth, vx=None, prefix='rw_', device='cpu', write=True, interpolation='linear', bound='zero'):
    """Reslice image data to world field of view.

    Parameters
    ----------
    pth : str
        Path to nibabel compatible file.

    vx : int or tuple[int], default=None
        Voxel size of resliced image. If None, uses same as input.

    prefix : str, default='rw_'
        Filename prefix of resliced image. If empty string, overwrites input file.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    write : bool, default=True
        Write resliced image to disk.

    interpolation : str, default='linear'
        Interpolation order.

    bound : str, default='zero'
        Boundary condition.

    Returns
    -------
    pth_out : str
        Path to resliced image.

    dat : (X, Y, Z), tensor_like[]
        Resliced image data.

    M_out : (4, 4), tensor_like[torch.float64]
        Affine transformation of resliced image.

    """
    device = _get_device(device)

    # Read file
    img = nib.load(pth)
    dim_in = img.shape
    M_in = torch.from_numpy(img.affine).type(torch.float64)
    # Get output voxel size
    if vx is not None:
        if not isinstance(vx, tuple):
            vx_out = (vx,) * 3
        vx_out = torch.tensor(vx_out, dtype=torch.float64)
    else:
        vx_out = voxel_size(M_in)

    # Get corners
    c = _get_corners(dim_in)
    c = c.t()
    # Corners in world space
    c_world = M_in[:3, :4].mm(c)
    # c_world[0, :] = - c_world[0, :]
    # Get bounding box
    mx = c_world.max(dim=1)[0].round()
    mn = c_world.min(dim=1)[0].round()
    # Compute output affine
    M_out = matrix(mn).mm(torch.diag(torch.cat((vx_out, torch.ones(1)))).mm(matrix(-1*torch.ones(3, dtype=torch.float64))))
    # Comput output image dimensions
    dim_out = M_out.inverse().mm(torch.cat((mx, torch.ones(1)))[:, None]).ceil()
    dim_out = dim_out[:3].squeeze()
    dim_out = dim_out.int().tolist()
    # I = torch.diag(torch.ones(4, dtype=torch.float64))
    # I[0, 0] = -I[0, 0]
    # M_out = I.mm(M_out)
    # Compute mapping from output to input
    M = M_out.solve(M_in)[0]
    # Reslice image data
    dat = torch.tensor(img.get_fdata(), device=device)
    dat = reslice_dat(dat[None, None, ...], M.to(device), dim_out, interpolation=interpolation, bound=bound)

    pth_out = pth  # Image out same as input
    if write:
        # Write resliced image
        pth_out = write_img(pth, dat, M_out, prefix=prefix)

    return pth_out, dat, M_out


def write_img(pth, dat=None, affine=None, prefix='', dir_out=None):
    """Write image to disk.

    Parameters
    ----------
    pth : str
        Path to nibabel compatible file.

    dat : (dim), tensor_like[], default=None
        Input image data. If None, uses input image data.

    affine : (4, 4), tensor_like[torch.float64], default=None
        Affine transformation. If None, uses input image affine.

    prefix : str, default=''
        Filename prefix of output image.

    dir_out : str, default=None
        Full path to directory where to write image.

    Returns
    -------
    pth_out : str
        Path to output image.

    """
    # Read file
    img = nib.load(pth)
    # Write resliced image
    header = img.header.copy()
    # Depending on input, get image data and/or affine from input
    if dat is None:
        dat = img.get_fdata()
    elif isinstance(dat, torch.Tensor):
        dat = dat.cpu()
    if affine is None:
        affine = img.affine
    elif isinstance(affine, torch.Tensor):
        affine = affine.cpu()
    # Set affine
    header.set_qform(affine)
    header.set_sform(affine)
    # Create nibabel nifti1
    img = nib.nifti1.Nifti1Image(dat, None, header=header)
    # Overwrite or create new?
    if dir_out is None:
        dir_out, fnam = os.path.split(pth)
    else:
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        _, fnam = os.path.split(pth)
    pth_out = os.path.join(dir_out, prefix + fnam)
    # Write to disk
    nib.save(img, pth_out)

    return pth_out


def _get_corners(*args):
    """Get eight corners of image volume.

    Signature
    ---------
    _get_corners(pth)
    _get_corners(dim)

    Parameters
    ----------
    pth : str
        Path to nibabel compatible file.

    dim : (3,), list or tuple
        Image dimensions.

    Returns
    -------
    c : (8, 4), tensor_like[torch.float64]
        Corners of volume.

    """
    if isinstance(args[0], str):
        img = nib.load(args[0])
        dim = img.shape
    else:
        dim = args[0]
    # Get corners
    c = torch.tensor([[     1,      1,      1, 1],
                      [     1,      1, dim[2], 1],
                      [     1, dim[1],      1, 1],
                      [     1, dim[1], dim[2], 1],
                      [dim[0],      1,      1, 1],
                      [dim[0],      1, dim[2], 1],
                      [dim[0], dim[1],      1, 1],
                      [dim[0], dim[1], dim[2], 1]], dtype=torch.float64)

    return c


def _get_device(device='cpu'):
    """Get PyTorch device.

    Parameters
    ----------
    device : torch.device or str, default='cpu'
        PyTorch device type.

    Returns
    -------
    device : torch.device or str, default='cpu'
        PyTorch device type.

    """
    return torch.device(device if torch.cuda.is_available() else 'cpu')


def _get_origin_mat(dim, vx=None, dtype=torch.float64, device='cpu'):
    """Get nitorch compatible affine matrix from dimensions and voxel size.

    Parameters
    ----------
    dim : (3,), tuple, list, tensor_like
        Output image dimensions.

    vx : number, tuple, list, tensor_like, default=None
        Voxel size of resliced image. If None, uses 1 mm isotropic.

    dtype : torch.dtype, default=torch.float64
        Data type of function output.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    Returns
    -------
    mat : (4, 4), tensor_like
        Affine matrix.

    """
    if not isinstance(dim, torch.Tensor):
        dim = torch.tensor(dim, dtype=dtype, device=device)
    if vx is None:
        vx = (1,) * 3
    if not isinstance(vx, tuple) or not isinstance(vx, list):
        vx = (vx)
    if not isinstance(vx, torch.Tensor):
        vx = torch.tensor(vx, dtype=dtype, device=device)
    orig = (dim + 1) / 2
    off = -vx * orig
    mat = torch.tensor([[vx[0], 0, 0, off[0]],
                        [0, vx[1], 0, off[1]],
                        [0, 0, vx[2], off[2]],
                        [0, 0, 0, 1]], dtype=dtype, device=device)

    return mat
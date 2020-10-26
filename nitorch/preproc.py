"""Various functions for pre-processing of neuroimaging data.

All these functions work directly with the path(s) to the image data.
The input data is not overwritten by default, but copies are made with
function specific prefix. However, by setting the prefix to an empty string,
the origian input data is modified.

TODO
-------
* Must be possible to change affine or data w/o writing new image to disk?

"""


import math
import nibabel as nib
import os
import torch
from torch.nn import functional as F
from .kernels import smooth
from .spatial import voxsize, grid_pull
from .spm import identity, matrix
from .spm import affine as apply_affine
from .utils import pad


__all__ = ['load_3d', 'reslice2world', 'reset_origin', 'write_img']


def load_3d(pth_in, samp=0, truncate=False, fwhm=0.0, mx_out=None, device='cpu', dtype=torch.float32,
           do_mask=True):
    """Load image data for subsequent image processing.

    Parameters
    ----------
    pth_in : str
        Path to nibabel compatible file.

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

    do_mask : bool, default=True
        Set not finite, zeros and minimum values to NaN.

    do_smooth : bool, default=False
        Smooth data a bit.

    """
    device = _get_device(device)

    # Get image parameters
    img = nib.load(pth_in)
    affine = torch.from_numpy(img.affine).type(torch.float64).to(device)
    dim = img.shape
    if len(dim) != 3:
        raise ValueError('Input image is {}D, should be 3D!'.format(len(dim)))
    dim = dim[:3]
    vx = voxsize(affine)

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

    # Load image
    dat = torch.tensor(img.get_fdata(), device=device, dtype=dtype)
    if samp[0] > 0:
        # Subsample image data with nearest neighbour interpolation
        dat = grid_pull(dat[None, None, ...], grid[None, ...],
                        bound='zero', interpolation='nearest')[0, 0, ...]
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

    # Check if data is integer, in which case add some random noise
    dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
    if img.get_data_dtype() in dtypes:
        scrand = img.dataobj.slope
        torch.manual_seed(0)
    else:
        scrand = 0;
    scrand = torch.tensor(scrand, dtype=dtype, device=device)
    dat = dat + torch.rand_like(dat) * scrand - scrand / 2

    if truncate:
        # Truncate
        dat = dat*sf[0] + sf[1]

    if samp[0] > 0:  # Smooth data a bit
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


def reset_origin(pth_in, vx=None, prefix='o', device='cpu', interpolation='linear', bound='zero'):
    """Reset affine matrix.

    OBS: Reslices image data.

    Parameters
    ----------
    pth_in : str
        Path to nibabel compatible file.

    vx : int or tuple[int], default=None
        Voxel size of resliced image. If None, uses same as input.

    prefix : str, default='r'
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
    pth_in, dat, M0 = reslice2world(pth_in, vx=vx, prefix=prefix, device=device, write=False,
                                 interpolation=interpolation, bound=bound)
    # Compute new, reset, affine matrix
    dim = torch.tensor(dat.shape, dtype=torch.float64)
    vx = voxsize(M0)
    if M0[:3, :3].det() < 0:
        vx[0] = - vx[0]
    orig = (dim[:3] + 1) / 2
    off = -vx * orig
    affine = torch.tensor([[vx[0], 0, 0, off[0]],
                           [0, vx[1], 0, off[1]],
                           [0, 0, vx[2], off[2]],
                           [0, 0, 0, 1]], dtype=torch.float64)
    # Write reset image
    pth_out = write_img(pth_in, dat, affine, prefix=prefix)

    return pth_out


def reslice2world(pth_in, vx=None, prefix='r', device='cpu', write=True, interpolation='linear', bound='zero'):
    """Reslice image data to world field of view.

    Parameters
    ----------
    pth_in : str
        Path to nibabel compatible file.

    vx : int or tuple[int], default=None
        Voxel size of resliced image. If None, uses same as input.

    prefix : str, default='r'
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
    img = nib.load(pth_in)
    dim_in = img.shape
    M_in = torch.from_numpy(img.affine).type(torch.float64)
    # Get output voxel size
    if vx is not None:
        if not isinstance(vx, tuple):
            vx_out = (vx,) * 3
        vx_out = torch.tensor(vx_out, dtype=torch.float64)
    else:
        vx_out = voxsize(M_in)

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
    dat = _reslice_dat(dat[None, None, ...], M.to(device), dim_out, interpolation=interpolation, bound=bound)

    pth_out = pth_in  # Image out same as input
    if write:
        # Write resliced image
        pth_out = write_img(pth_in, dat, M_out, prefix=prefix)

    return pth_out, dat, M_out


def write_img(pth_in, dat=None, affine=None, prefix='', pth_out=None):
    """Write image to disk.

    Parameters
    ----------
    pth_in : str
        Path to nibabel compatible file.

    dat : (dim), tensor_like[], default=None
        Input image data. If None, uses input image data.

    affine : (4, 4), tensor_like[torch.float64], default=None
        Affine transformation. If None, uses input image affine.

    prefix : str, default=''
        Filename prefix of resliced image. If empty string, overwrites input file.

    pth_out : str, defult=None
        Path to output image. If None, uses input image path.

    Returns
    -------
    pth_out : str
        Path to output image.

    """
    # Read file
    img = nib.load(pth_in)
    # Write resliced image
    header = img.header.copy()
    # Depending on input, get image data and/or affine from input
    if dat is None:
        dat = img.get_fdata()
    else:
        dat = dat.cpu()
    if affine is None:
        affine = img.affine
    # Set affine
    header.set_qform(affine)
    header.set_sform(affine)
    # Create nibabel nifti1
    img = nib.nifti1.Nifti1Image(dat, None, header=header)
    # Overwrite or create prefixed?
    if pth_out is None:
        pth_out = pth_in
    if prefix == '':
        os.remove(pth_out)
    else:
        dir, fnam = os.path.split(pth_out)
        pth_out = os.path.join(dir, prefix + fnam)
    # Write to disk
    nib.save(img, pth_out)

    return pth_out


def _get_corners(*args):
    """Get eight corners of image volume.

    Signature
    ---------
    _get_corners(pth_in)
    _get_corners(dim)

    Parameters
    ----------
    pth_in : str
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


def _reslice_dat(dat, affine, dim_out, interpolation='linear', bound='zero'):
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

    Returns
    -------
    dat : (dim_out), tensor_like[dat.dtype]
        Output resliced image data.

    """
    dim_in = dat.shape
    if not isinstance(dat, torch.Tensor):
        dat = torch.from_numpy(dat)
    grid = apply_affine(dim_out, affine, device=dat.device, dtype=dat.dtype)
    dat = grid_pull(dat, grid, bound=bound, interpolation=interpolation)

    return dat[0, 0, ...]
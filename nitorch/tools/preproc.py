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
from ..core.pyutils import get_pckg_data
from ..core.utils import pad
from ..plot.volumes import show_slices
from ..spatial import (affine_default, affine_grid, voxel_size, grid_pull)
from .spm import (identity, matrix)


def atlas_crop(dat, mat_in, mni_align=False, fov='full'):
    """Crop an image to the NITorch atlas field-of-view.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensorlike
        Input image.
    mat_in : (4, 4) tensorlike
        Input affine matrix.
    mni_align : bool, default=False
        Do alignment to MNI space.
    fov : str, default='full'
        Output field-of-view (FOV):
        * 'full' : Full atlas FOV.
        * 'brain' : Brain FOV.

    Returns
    ----------
    dat : (X1, Y1, Z1) tensorlike
        Cropped image.
    mat : (4, 4) tensorlike
        Cropped affine matrix.
    dim : (3, ) tuple
        Cropped dimensions.

    """
    device = dat.device
    dtype = dat.dtype
    mat_in = mat_in.type(dtype)
    if mni_align:
        # TODO: Align to MNI
        raise ValueError('Not implemented!')
    offset = ((0, ) * 3, (0, ) * 3)
    if fov == 'brain':
        # Gives a good fit around the brain of the atlas
        offset = ((10, 55, 120), (10, 40, 60))
    # Get atlas information
    nii = nib.load(get_pckg_data('atlas_t1'))
    mat_mu = torch.tensor(nii.affine).type(dtype).to(device)
    dim_mu = torch.tensor(nii.shape).type(dtype).to(device)
    # Get atlas corners in image space
    mat = mat_mu.solve(mat_in)[0]
    c = get_corners(dim_mu, offset, dtype=dtype, device=device)
    c = mat[:3, ...].mm(c.t())
    # Make bounding-box
    mn = torch.min(c, dim=1)[0]
    mx = torch.max(c, dim=1)[0]
    bb = torch.stack((mn, mx))
    # Extract sub-volume
    dat, mat, dim = subvol(dat, mat_in, bb)

    return dat, mat, dim


def get_corners(dim, o=((0,)*3, (0,)*3), dtype=torch.float64, device='cpu'):
    """Get eight corners of 3D image volume.

    Parameters
    ----------
    dim : (3,), list or tuple
        Image dimensions.
    o : (2, 3), default=((0,)*3, (0,)*3)
        Add offset to corners.
    dtype : torch.dtype, default=torch.float64
        Data type of function output.
    device : torch.device or str, default='cpu'
        PyTorch device type.

    Returns
    -------
    c : (8, 4), tensor_like[torch.float64]
        Corners of volume.

    """
    c = torch.tensor([[     1 + o[0][0],      1 + o[0][1],      1 + o[0][2], 1],
                      [     1 + o[0][0],      1 + o[0][1], dim[2] - o[1][2], 1],
                      [     1 + o[0][0], dim[1] - o[1][1],      1 + o[0][2], 1],
                      [     1 + o[0][0], dim[1] - o[1][1], dim[2] - o[1][2], 1],
                      [dim[0] - o[1][0],      1 + o[0][1],      1 + o[0][2], 1],
                      [dim[0] - o[1][0],      1 + o[0][1], dim[2] - o[1][2], 1],
                      [dim[0] - o[1][0], dim[1] - o[1][1],      1 + o[0][2], 1],
                      [dim[0] - o[1][0], dim[1] - o[1][1], dim[2] - o[1][2], 1]],
                     device=device, dtype=dtype)

    return c


def load_3d(img, samp=0, rescale=False, fwhm=0.0, mn_out=0, mx_out=511,
            device='cpu', dtype=torch.float32, do_smooth=True, raw=False):
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

    rescale : bool, default=False
        Truncate image data based on percentiles.

    fwhm : float, default=0.0
        Smoothness estimate for computing a fudge factor.

    mn_out : float, default=0
        Minimum intensity in returned image data.

    mx_out : float, default=511
        Maximum intensity in returned image data.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    dtype : torch.dtype, default=torch.float64
        Data type of function output.

    do_smooth : bool, default=True
        If sub-sampling (samp > 0), smooth data a bit.

    raw : bool, default=False
        Do no processing, just return raw data.

    Returns
    ----------
    dat : (dim_out), tensor_like[dtype]
        Preprocessed image data.

    affine : (4, 4), tensor_like[torch.float64]
        Image affine transformation.

    grid : (dim_out, 3), tensor_like[dtype]
        Grid with voxel locations in input image (possibly sampled).

    ff : tensor_like[dtype]
        Fudge Factor to (approximately) account for
        non-independence of voxels.

    """
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
    elif isinstance(img, (list, tuple)):
        # Input is tuple with two tensors: image data and affine tensors
        dat = img[0]
        affine = img[1]
        scrand = 1  # As we don't know the original data type, we add random noise
    elif isinstance(img, torch.Tensor):
        dat = img
        affine = affine_default(dat.shape, device=device, dtype=dtype)
        affine = torch.cat((affine, torch.tensor([0, 0, 0, 1],
            dtype=affine.dtype, device=affine.device)[None, ...]))  # Add a row to make (4, 4)
        scrand = 1  # As we don't know the original data type, we add random noise
    else:
        raise ValueError('Input error!')
    if raw:
        # Do no processing
        grid = None
        ff = torch.tensor(0, dtype=dtype, device=device)
        return dat, affine, grid, ff

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

    # Mask
    dat[~torch.isfinite(dat) | (dat == 0) | (dat == dat.min()) | (dat == dat.max())] = 0

    import time;

    if rescale:
        # Rescale so that image intensities are between 1 and nh (so we can make a histogram)
        nh = 4000  # Number of histogram bins
        mn_scl = torch.tensor([dat.min(), 1], dtype=dtype, device=device)[None, ...]
        mx_scl = torch.tensor([dat.max(), 1], dtype=dtype, device=device)[None, ...]
        sf = torch.cat((mn_scl, mx_scl), dim=0)
        sf = torch.tensor([1, nh], dtype=dtype, device=device)[..., None].solve(sf)[0].squeeze()
        p = dat[(dat != 0)]
        p = (p*sf[0] + sf[1]).round().long()
        # Make histogram and find percentiles
        h = torch.zeros(nh + 1, device=device, dtype=p.dtype)
        h.put_(p, torch.ones(1, dtype=p.dtype, device=device).expand_as(p), accumulate=True)
        h = h.type(dtype)
        h = h.cumsum(0)/h.sum()
        mn_scl = ((h <= 0.0005).sum(dim=0) - sf[1]) / sf[0]
        mx_scl = ((h <= 0.9999).sum(dim=0) - sf[1]) / sf[0]
        # Make scaling to set image intensities between mn_out and mx_out
        mn = torch.tensor([mn_scl, 1], dtype=dtype, device=device)[None, ...]
        mx = torch.tensor([mx_scl, 1], dtype=dtype, device=device)[None, ...]
        sf = torch.cat((mn, mx), dim=0)
        sf = torch.tensor([mn_out, mx_out], dtype=dtype, device=device)[..., None].solve(sf)[0].squeeze()

        if scrand:
            # Add some random noise
            torch.manual_seed(0)
            scrand = torch.tensor(scrand, dtype=dtype, device=device)
            dat = dat + torch.rand_like(dat) * scrand - scrand / 2

    if rescale:
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

    if rescale:
        dat = dat.clamp_min(mn_out).clamp_max(mx_out)

    return dat, affine, grid, ff


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


def reset_origin(img, vx=None, prefix='ro_', device='cpu', write=True,
                 interpolation='linear', bound='zero'):
    """Reset affine matrix.

    OBS: Reslices image data.

    Parameters
    ----------
    img : str
        Path to nibabel compatible file.
          (tensor_like, tensor_like)
        Image data and affine matrix.

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

    dat : (X, Y, Z), tensor_like[]
        Resliced image data.

    affine : (4, 4), tensor_like[torch.float64]
        Affine transformation of resliced image.

    """
    # Reslice image data to world FOV
    pth, dat, M0 = reslice2world(img, vx=vx, prefix=prefix, device=device, write=False,
                                 interpolation=interpolation, bound=bound)
    # Compute new, reset, affine matrix
    dim = dat.shape
    vx = voxel_size(M0).cpu().tolist()
    if M0[:3, :3].det() < 0:
        vx[0] = - vx[0]
    affine = affine_default(dim, vx, dtype=torch.float64, device=device)
    affine = torch.cat((affine, torch.tensor([0, 0, 0, 1],
        dtype=affine.dtype, device=affine.device)[None, ...]))  # Add a row to make (4, 4)
    pth_out = None
    if write and pth is not None:
        # Write reset image
        pth_out = write_img(pth, dat, affine, prefix=prefix)

    return pth_out, dat, affine


def reslice_dat(dat, affine, dim_out, interpolation='linear',
                bound='zero', extrapolate=False):
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
    grid = affine_grid(affine, dim_out)
    dat = grid_pull(dat, grid[None, ...], bound=bound, interpolation=interpolation, extrapolate=extrapolate)

    return dat[0, 0, ...]


def reslice2world(img, vx=None, prefix='rw_', device='cpu',
                  write=True, interpolation='linear', bound='zero'):
    """Reslice image data to world field of view.

    Parameters
    ----------
    img : str
        Path to nibabel compatible file.
          (tensor_like, tensor_like)
        Image data and affine matrix.

    vx : int or tuple[int], default=None
        Voxel size of resliced image. If None, uses same as input.

    prefix : str, default='rw_'
        Filename prefix of resliced image. If empty string, overwrites input file.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    write : bool, default=True
        Write resliced image to disk (only if input was given as nibabel path).

    interpolation : str, default='linear'
        Interpolation order.

    bound : str, default='zero'
        Boundary condition.

    Returns
    -------
    pth_out : str
        Path to resliced image (only if input was given as nibabel path).

    dat : (X, Y, Z), tensor_like[]
        Resliced image data.

    M_out : (4, 4), tensor_like[torch.float64]
        Affine transformation of resliced image.

    """
    if isinstance(img, str):
        # Read file
        img = nib.load(img)
        dat = torch.tensor(img.get_fdata(), device=device, dtype=torch.float32)
        M_in = torch.from_numpy(img.affine).type(torch.float64)
        pth_out = pth  # Image out same as input
    else:
        # Image given as tensors
        dat = img[0]
        M_in = img[1].type(torch.float64)
        pth_out = None
        write = False
    dim_in = dat.shape
    # Get output voxel size
    if vx is not None:
        if not isinstance(vx, tuple):
            vx_out = (vx,) * 3
        vx_out = torch.tensor(vx_out, dtype=torch.float64)
    else:
        vx_out = voxel_size(M_in)
    # Get corners
    c = get_corners(dim_in).to(device)
    c = c.t()
    # Corners in world space
    c_world = M_in[:3, :4].mm(c)
    c_world[0, :] = - c_world[0, :]
    # Get bounding box
    mx = c_world.max(dim=1)[0].round()
    mn = c_world.min(dim=1)[0].round()
    # Compute output affine
    M_out = matrix(mn).mm(
        torch.diag(torch.cat((vx_out, torch.ones(1, dtype=torch.float64, device=device)))).mm(matrix(-1*torch.ones(3, dtype=torch.float64, device=device))))
    # Comput output image dimensions
    dim_out = M_out.inverse().mm(torch.cat((mx, torch.ones(1, dtype=torch.float64, device=device)))[:, None]).ceil()
    dim_out = dim_out[:3].squeeze()
    dim_out = dim_out.int().tolist()
    I = torch.diag(torch.ones(4, dtype=torch.float64, device=device))
    I[0, 0] = -I[0, 0]
    M_out = I.mm(M_out)
    # Compute mapping from output to input
    M = M_out.solve(M_in)[0]
    # Reslice image data
    dat = reslice_dat(dat[None, None, ...],
        M.type(dat.dtype), dim_out, interpolation=interpolation, bound=bound)
    if write:
        # Write resliced image
        pth_out = write_img(pth, dat, M_out, prefix=prefix)

    return pth_out, dat, M_out


def subvol(dat, mat, bb):
    """Extract a sub-volume.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensorlike
        Image volume.
    mat : (4, 4) tensorlike
        Image affine matrix.
    bb : (2, 3) tensorlike
        Bounding box.

    Returns
    ----------
    dat : (X1, Y1, Z1) tensorlike
        Image sub-volume.
    mat : (4, 4) tensorlike
        Sub-volume affine matrix.
    dim : (3, ) tuple
        Sub-volume dimensions.

    """
    # show_slices(dat, fig_num=1)  # For debugging
    device = dat.device
    dtype = dat.dtype
    mat = mat.type(dtype)
    bb = bb.type(dtype)
    # Sanity check
    bb = bb.round()
    bb = bb.sort(dim=0)[0]
    bb[0, ...] = torch.max(bb[0, ...],
        torch.ones(3, device=device, dtype=dtype))
    bb[1, ...] = torch.min(bb[1, ...],
        torch.tensor(dat.shape, device=device, dtype=dtype))
    # Output dimensions
    dim = bb[1, ...] - bb[0, ...] + 1
    dim = dim.cpu().int().tolist()
    # Bounding-box affine
    mat_bb = matrix(bb[0, ...] - 1)
    # Output data
    dat = reslice_dat(dat[None, None, ...], mat_bb, dim,
        interpolation='nearest', bound='zero', extrapolate=False)
    # Output affine
    mat = mat.mm(mat_bb)
    # show_slices(dat, fig_num=2)  # For debugging

    return dat, mat, dim


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


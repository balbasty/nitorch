__all__ = ['preproc_image', 'map_image', 'load_image', 'prepare_pyramid_levels',
           'rescale_image', 'discretize_image', 'soft_quantize_image']

from nitorch import io
from nitorch.core.py import make_list, flatten
from nitorch.core import dtypes, utils
from nitorch import spatial
from . import pairwise_pyramid as pyrutils
import torch
import math as pymath


def preproc_image(input, mask=None, label=False, missing=0,
                  world=None, affine=None, rescale=.95,
                  pad=None, bound='zero', fwhm=None, channels=None,
                  dim=None, device=None, **kwargs):
    """Load an image and preprocess it as required

    Parameters
    ----------
    input : (C, *spatial) Tensor or MappedTensor or [sequence of] str
        Input image.
        Either a filename, or a mapped tensor, or a pre-loaded tensor
    mask : (C|1, *spatial) Tensor  or MappedTensor or str
        Input mask of voxels to include in the loss.
        Either a filename, or a mapped tensor, or a pre-loaded tensor
    label : bool, default=False
        Input image contains hard labels
    missing : [list of] int, default=0
        Values to consider as missing values in the input volume and mask out
    world : (D+1, D+1) tensor or str, optional
        Voxel-to-world matrix that takes precedence over whatever is
        read from disk.
    affine : [sequence of] (D+1, D+1) tensor or str, optional
        A series of affine transforms that should be applied to the image.
    rescale : float or (float, float), default=(0, 0.95)
        Rescale intensities so that these quantiles map to (0, 1)
    pad : [sequence of] int or float, optional
        Pad the volume by this amount.
        If last element is "mm", values are in mm and converted to voxels.
    bound : [sequence of] str
        Boundary conditions
    fwhm : [sequence of] float
        Smooth the volume with a Gaussian kernel of that FWHM.
        If last element is "mm", values are in mm and converted to voxels.
    channels : [sequence of] int or range or slice
        Channels to load
    dim : int, optional
        Number of spatial dimensions
    device : torch.device
        Move data to this device

    Returns
    -------
    dat : (C, *spatial) tensor
        Loaded and preprocessed data
    mask : (C|1, *spatial) tensor
        Mask of voxels to include
    affine : (D+1, D+1) tensor
        Orientation matrix

    """
    dat, mask0, affine0 = load_image(input, dim=dim, device=device,
                                     label=label, missing=missing,
                                     channels=channels)

    # if not torch.is_tensor(input):
    #     dat, mask0, affine0 = load_image(input, dim=dim, device=device,
    #                                      label=label, missing=missing,
    #                                      channels=channels)
    # else:
    #     dat = input
    #     if channels is not None:
    #         channels = make_list(channels)
    #         channels = [
    #             list(c) if isinstance(c, range) else
    #             list(range(len(dat)))[c] if isinstance(c, slice) else
    #             c for c in channels
    #         ]
    #         if not all([isinstance(c, int) for c in channels]):
    #             raise ValueError('Channel list should be a list of integers')
    #     dat = dat[channels]
    #     mask0 = torch.isfinite(dat)
    #     dat = dat.masked_fill(~mask0, 0)
    #     affine0 = spatial.affine_default(dat.shape[1:])

    dim = dat.dim() - 1

    # load user-defined mask
    if mask is not None:
        mask1 = mask0
        mask0, _, _ = load_image(mask, dim=dim, device=device, missing=None)
        if mask0.shape[-dim:] != dat.shape[-dim:]:
            raise ValueError('Mask should have the same shape as the image. '
                             f'Got {mask0.shape[-dim:]} and {dat.shape[-dim:]}')
        if mask1 is not None:
            mask0 = mask0 * mask1
        del mask1

    # overwrite orientation matrix
    if world is not None:
        if isinstance(world, str):
            affine0 = io.transforms.map(world).fdata().squeeze()
        else:
            affine0 = world

    # apply input affines
    if not torch.is_tensor(affine) and not affine:
        affine = []
    affine = make_list(affine)
    for transform in affine:
        if isinstance(transform, str):
            transform = io.transforms.map(transform).fdata().squeeze()
        affine0 = spatial.affine_lmdiv(transform.to(affine0), affine0)

    # rescale intensities
    rescale = make_list(rescale)
    if not label and any(rescale):
        dat = rescale_image(dat, rescale)

    # pad image
    if pad:
        pad = make_list(pad)
        if isinstance(pad[-1], str):
            *pad, unit = pad
        else:
            unit = 'vox'
        pad = make_list(pad, dim)
        if unit == 'mm':
            voxel_size = spatial.voxel_size(affine)
            pad = torch.as_tensor(pad, **utils.backend(voxel_size))
            pad = pad / voxel_size
            pad = pad.floor().int().tolist()
        elif unit in ('%', 'pct'):
            pad = [int(pymath.ceil(p * s / 100)) for p, s in zip(pad, dat.shape[1:])]
        else:
            pad = [int(p) for p in pad]
        if any(pad):
            affine0, _ = spatial.affine_pad(affine0, dat.shape[-dim:], pad,
                                            side='both')
            dat = utils.pad(dat, pad, side='both', mode=bound)
            if mask0 is not None:
                mask0 = utils.pad(mask0, pad, side='both', mode=bound)

    # smooth image
    if fwhm:
        fwhm = make_list(fwhm)
        if isinstance(fwhm[-1], str):
            *fwhm, unit = fwhm
        else:
            unit = 'vox'
        if unit == 'mm':
            voxel_size = spatial.voxel_size(affine)
            fwhm = torch.as_tensor(fwhm, **utils.backend(voxel_size))
            fwhm = fwhm / voxel_size
        dat = spatial.smooth(dat, dim=dim, fwhm=fwhm, bound=bound)

    return dat, mask0, affine0


def prepare_pyramid_levels(images, levels, dim=None, **opt):
    """
    For each loss, compute the pyramid levels of `fix` and `mov` that
    must be computed.

    Parameters
    ----------
    images : [list of] str or MappedArray or tensor or (tensor, tensor)
        images
    dim : int, optional
        Number of spatial dimensions
    min_size, max_size : int, optional
        Minimum and maximum acceptable size, in voxels
    min_vx, max_vx : float
        Minimum and maximum acceptable voxel size, in mm

    Returns
    -------
    levels : [list of] list[int]
        Level to compute in the fixed and moving image for each global
        pyramid level

    """
    if not images:
        return [] if isinstance(images, (list, tuple)) else None

    if not levels:
        if isinstance(images, (list, tuple)):
            return [None] * len(images)
        else:
            return None

    vxs = []
    shapes = []
    for image in images:
        if torch.is_tensor(image):
            dat = image
            dim = dim or dat.ndim
            if dat.ndim == dim:
                dat = dat[None]
            affine = spatial.affine_default(dat.shape[-dim:])
        elif isinstance(image, (list, tuple)):
            dat, affine = image
        elif not isinstance(image, io.MappedArray):
            dat, affine = map_image(image, dim=dim)
        else:
            affine = image.affine
            dat = image
        if affine.dim() > 2:
            affine = affine[0]
        dim = dat.ndim - 1
        vx = spatial.voxel_size(affine).tolist()
        shape = dat.shape[-dim:]
        vxs.append(vx)
        shapes.append(shape)

    return pyrutils.pyramid_levels(vxs, shapes, levels, **opt)


def map_image(fnames, dim=None, channels=None):
    """Map an ND image from disk

    Parameters
    ----------
    fnames : [sequence of] str
        Input filenames. If multiple filenames are provided, they
        should map to volumes tht have the same spatial shape (and
        same orientation matrix) and will be considered as different
        channels.
    dim : int, optional
        Number of spatial dimensions.
        By default, try to guess form the orientation matrix.

    Returns
    -------
    image : (C, *spatial) MappedTensor
        A MappedTensor with channels in the first dimension
    affine: (D+1, D+1) tensor
        Orientation matrix
    """
    fnames = make_list(fnames)
    affine = None
    imgs = []
    for fname in fnames:
        img = io.map(fname)
        if affine is None:
            affine = img.affine
        if dim is None:
            dim = img.affine.shape[-1] - 1
        while len(img.shape) > dim and img.shape[dim] == 1:
            img = img.squeeze(dim)
        if img.dim > dim:
            img = img.movedim(-1, 0)
        else:
            img = img[None]
        img = img.unsqueeze(-1, dim + 1 - img.dim)
        if img.dim > dim + 1:
            raise ValueError(f'Don\'t know how to deal with an image of '
                             f'shape {tuple(img.shape)}')
        imgs.append(img)
        del img
    imgs = io.cat(imgs, dim=0)

    # select a subset of channels
    if channels is not None:
        channels = make_list(channels)
        channels = [
            list(c) if isinstance(c, range) else
            list(range(len(imgs)))[c] if isinstance(c, slice) else
            c for c in channels
        ]
        channels = flatten(channels)
        if not all([isinstance(c, int) for c in channels]):
            raise ValueError(
                'Channel list should be a list of integers but received:',
                channels
            )
        imgs = io.stack([imgs[c] for c in channels])

    return imgs, affine


def load_image(input, dim=None, device=None, label=False, missing=0,
               channels=None):
    """
    Load a N-D image from disk

    Parameters
    ----------
    input : (C, *spatial) Tensor or MappedTensor or [sequence of] str
        Either a filename, or a mapped tensor, or a pre-loaded tensor
    dim : int, optional
        Number of spatial dimensions.
        By default, try to guess form the orientation matrix.
    device : torch.device, default='cpu'
        Device on which to load the data
    label : bool, default=False
        Whether the tensor contains a volume of hard labels
    missing : [sequence of] int
        Values to consider missing and that will be masked out.

    Returns
    -------
    dat : (C, *spatial) tensor
        Image
    mask : (C, *spatial) tensor
        Mask of voxels to include
    affine: (D+1, D+1) tensor
        Orientation matrix
    """
    if not torch.is_tensor(input):
        dat, affine = map_image(input, dim, channels=channels)
    else:
        dat, affine = input, spatial.affine_default(input.shape[1:])

        if channels is not None:
            channels = make_list(channels)
            channels = [
                list(c) if isinstance(c, range) else
                list(range(len(dat)))[c] if isinstance(c, slice) else
                c for c in channels
            ]
            if not all([isinstance(c, int) for c in channels]):
                raise ValueError('Channel list should be a list of integers')
            dat = dat[channels]

    if label:
        dtype = dat.dtype
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        dtype = dtypes.as_torch(dtype, upcast=True)
        if torch.is_tensor(dat):
            dat0 = dat[0]
        else:
            dat0 = dat.data(device=device, dtype=dtype)[0]  # assume single channel
        if label is True:
            label = dat0.unique(sorted=True)
            label = label[label != 0].tolist()
        dat = torch.zeros([len(label), *dat0.shape], device=device)
        for i, l in enumerate(label):
            dat[i] = dat0 == l
        mask = None
    else:
        if torch.is_tensor(dat):
            if missing is not None:
                mask = utils.isin(dat, make_list(missing))
                dat = dat.to(torch.float32).masked_fill(mask, float('nan'))
        else:
            dat = dat.fdata(device=device, rand=True, missing=missing)
        mask = torch.isfinite(dat)
        mask = mask.all(dim=0)
        dat.masked_fill_(~mask, 0)
    affine = affine.to(dat.device, torch.float32)
    return dat, mask, affine


def rescale_image(dat, quantiles=0.95):
    """Rescale an image (in-place) between (0, 1) based on two quantiles

    Parameters
    ----------
    dat : (C, *shape) tensor
        Input image
    quantiles : float or (float, float)

    Returns
    -------

    """
    dim = dat.dim() - 1
    quantiles = utils.make_vector(quantiles).tolist()
    if len(quantiles) == 0:
        mn = 0
        mx = 95
    elif len(quantiles) == 1:
        mn = 0
        mx = quantiles[0]
    else:
        mn, mx = quantiles
    mx = mx / 100
    mn, mx = utils.quantile(dat, (mn, mx), dim=range(-dim, 0),
                            keepdim=True, bins=1024).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn)
    return dat


def discretize_image(dat, nbins=256):
    """Discretize an image into a number of bins

    Parameters
    ----------
    dat : (C, *spatial) tensor[float, double]
        Input image
    nbins : int, default=256
        Number of bins

    Returns
    -------
    lab : (C, *spatial) tensor[long]
        Discretized image
    """
    dim = dat.dim() - 1
    mn, mx = utils.quantile(dat, (0.0005, 0.9995), dim=range(-dim, 0), keepdim=True).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn).clamp_(0, 1).mul_(nbins-1)
    dat = dat.long()
    return dat


def soft_quantize_image(dat, nbins=16):
    """Discretize an image into a number of soft bins using a Gaussian window

    Parameters
    ----------
    dat : (1, *spatial) tensor
        Input image
    nbins : int, default=16
        Number of bins

    Returns
    -------
    soft : (nbins, *spatial) tensor
        Soft-quantized image
    """
    dim = dat.dim() - 1
    dat = dat[0]
    mn, mx = utils.quantile(dat, (0.0005, 0.9995), dim=range(-dim, 0), keepdim=True).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn).clamp_(0, 1).mul_(nbins)
    centers = torch.linspace(0, nbins, nbins+1, **utils.backend(dat))
    centers = (centers[1:] + centers[:-1]) / 2
    centers = centers.flip(0)
    centers = centers[(Ellipsis,) + (None,) * dim]
    dat = (centers - dat).square().mul_(-2.355**2).exp_()
    dat /= dat.sum(0, keepdims=True)
    return dat

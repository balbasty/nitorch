"""Task-specific augmentation.
"""
import torch
from nitorch.spatial import grid_pull
from nitorch.core.constants import eps
from nitorch.plot import show_slices
from .field import BiasFieldTransform
from .spatial import DiffeoSample


# Parameters for various augmentation techniques
augment_params = {'inu': {'amplitude': 0.25, 'fwhm': 15.0},
                  'warp': {'amplitude': 2.0, 'fwhm': 15.0},
                  'noise': {'std_prct': 0.025}}


def seg_augmentation(tag, image, ground_truth=None, vx=None):
    """Augmentation methods for segmentation network, with parameters that
    should, hopefully, work well by default.

    OBS: Grount truth input only required when doing warping augmentation.

    Parameters
    -------
    tag : str
        Augmentation method:
        'warp-img-img' : Nonlinear warp of input image and output image
        'warp-img-seg' : Nonlinear warp of input image and output segmentation
        'warp-seg-img' : Nonlinear warp of input segmentation and output image
        'warp-seg-seg' : Nonlinear warp of input segmentation and output segmentation
        'noise' : Additive gaussian noise to image
        'inu' : Multiplicative intensity non-uniformity (INU) to image
    image : (batch, input_channels, *spatial) tensor
        Input image
    ground_truth : (batch, output_classes[+1], *spatial) tensor, optional
        Ground truth segmentation, used by the loss function.
        Its data type should be integer if it contains hard labels,
        and floating point if it contains soft segmentations.
    vx : [ndim, ] sequence, optional
        Image voxel size (in mm), defaults to 1 mm isotropic.

    Returns
    -------
    image : (batch, input_channels, *spatial) tensor
        Augmented input image.
    ground_truth : (batch, output_classes[+1], *spatial) tensor, optional
        Augmented ground truth segmentation.

    """
    # sanity check
    valid_tags = ['warp-img-img', 'warp-img-seg', 'warp-seg-img', 'warp-seg-seg', 'noise', 'inu']
    if tag not in valid_tags:
        raise ValueError('Undefined tag {:}, need to be one of {:}'.format(tag, valid_tags))
    nbatch = image.shape[0]
    nchan = image.shape[1]
    dim = tuple(image.shape[2:])
    ndim = len(dim)
    nvox = int(torch.as_tensor(image.shape[2:]).prod())
    # voxel size
    if vx is None:
        vx = (1.0, ) * ndim
    vx = torch.as_tensor(vx, device=image.device, dtype=image.dtype)
    vx = vx.clamp_min(1.0)
    # Augmentation method
    if 'warp' in tag:
        # Nonlinear warp
        # Parameters
        amplitude = augment_params['warp']['amplitude']
        fwhm = (augment_params['warp']['fwhm'], ) * ndim
        fwhm = [f / v for f, v in zip(fwhm, vx)]  # modulate FWHM with voxel size
        # Instantiate augmenter
        aug = DiffeoSample(amplitude=amplitude, fwhm=fwhm, bound='zero',
                           device=image.device, dtype=image.dtype)
        # Get random grid
        grid = aug(batch=nbatch, shape=dim, dim=ndim)
        # Warp
        if tag == 'warp-img-img':
            image = warp_img(image, grid)
            if ground_truth is not None:
                ground_truth = warp_img(ground_truth, grid)
        elif tag == 'warp-img-seg':
            image = warp_img(image, grid)
            if ground_truth is not None:
                ground_truth = warp_seg(ground_truth, grid)
        elif tag == 'warp-seg-img':
            image = warp_seg(image, grid)
            if ground_truth is not None:
                ground_truth = warp_img(ground_truth, grid)
        elif tag == 'warp-seg-seg':
            image = warp_seg(image, grid)
            if ground_truth is not None:
                ground_truth = warp_seg(ground_truth, grid)
        else:
            raise ValueError('')
    elif tag == 'noise':
        # Additive gaussian noise to image
        # Parameter
        std_prct = augment_params['noise']['std_prct']  # percentage of max intensity of batch and channel
        # Get max intensity in for each batch and channel
        mx = image.reshape((nbatch, nchan, nvox)).max(dim=-1, keepdim=True)[0]
        # Add 'lost' dimensions
        for d in range(ndim - 1):
            mx = mx.unsqueeze(-1)
        # Add noise to image
        image += std_prct*mx*torch.randn_like(image)
    elif tag == 'inu':
        # Multiplicative intensity non-uniformity (INU) to image
        # Parameters
        amplitude = augment_params['inu']['amplitude']
        fwhm = (augment_params['inu']['fwhm'],) * ndim
        fwhm = [f/v for f, v in zip(fwhm, vx)]  # modulate FWHM with voxel size
        # Instantiate augmenter
        aug = BiasFieldTransform(amplitude=amplitude, fwhm=fwhm, mean=0.0,
                                 device=image.device, dtype=image.dtype)
        # Augment image
        image = aug(image)

    return image, ground_truth


def warp_img(img, grid):
    """Warp image according to grid.
    """
    img = grid_pull(img, grid, bound='dct2', extrapolate=True,
                    interpolation=1)

    return img


def warp_seg(seg, grid):
    """Warp segmentation according to grid.
    """
    ndim = len(seg.shape[2:])
    dtype_seg = seg.dtype
    if dtype_seg not in (torch.half, torch.float, torch.double):
        # hard labels to one-hot labels
        M = torch.eye(len(seg.unique()), device=seg.device,
                      dtype=torch.float32)
        seg = M[seg.type(torch.int64)]
        if ndim == 3:
            seg = seg.permute((0, 5, 2, 3, 4, 1))
        else:
            seg = seg.permute((0, 4, 2, 3, 1))
        seg = seg.squeeze(-1)
    # warp
    seg = grid_pull(seg, grid, bound='dct2', extrapolate=True,
                    interpolation=1)
    if dtype_seg not in (torch.half, torch.float, torch.double):
        # one-hot labels to hard labels
        seg = seg.argmax(dim=1, keepdim=True).type(dtype_seg)
    else:
        # normalise one-hot labels
        seg = seg / (seg.sum(dim=1, keepdim=True) + eps())

    return seg

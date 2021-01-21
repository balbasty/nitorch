"""Task-specific augmentation.
"""
import torch
from ._field import (BiasFieldTransform, RandomFieldSample)
from ._spatial import DeformedSample
from ...spatial import grid_pull


def seg_augmentation(tag, image, ground_truth):
    """Augmentation methods for segmentation network, with parameters that
    should, hopefully, work well.

    Parameters
    -------
    tag : str
        Augmentation method:
        'warp' : Nonlinear warp of image and ground_truth
        'noise' : Additive gaussian noise to image
        'inu' : Multiplicative intensity non-uniformity (INU) to image
    image : (batch, input_channels, *spatial) tensor
        Input image
    ground_truth : (batch, output_classes[+1], *spatial) tensor, optional
        Ground truth segmentation, used by the loss function.
        Its data type should be integer if it contains hard labels,
        and floating point if it contains soft segmentations.

    Returns
    -------
    image : (batch, input_channels, *spatial) tensor
        Augmented input image.
    ground_truth : (batch, output_classes[+1], *spatial) tensor, optional
        Augmented ground truth segmentation.

    """
    nbatch = image.shape[0]
    nchan = image.shape[1]
    ndim = len(image.shape[2:])
    nvox = int(torch.as_tensor(image.shape[2:]).prod())
    # Augmentation method
    if tag == 'warp':
        # Nonlinear warp of image and ground_truth
        # Parameters
        amplitude = 1.0
        fwhm = 3.0
        # Instantiate augmenter
        aug = DeformedSample(vel_amplitude=amplitude, vel_fwhm=fwhm,
                             translation=False, rotation=False, zoom=False, shear=False,
                             device=image.device, dtype=image.dtype, image_bound='zero')
        # Augment image (and get grid)
        image, grid = aug(image)
        # Augment labels, with the same grid
        dtype_gt = ground_truth.dtype
        if dtype_gt not in (torch.half, torch.float, torch.double):
            # Hard labels to one-hot labels
            M = torch.eye(len(ground_truth.unique()), device=ground_truth.device, dtype=torch.float32)
            ground_truth = M[ground_truth.type(torch.int64)]
            if ndim == 3:
                ground_truth = ground_truth.permute((0, 5, 2, 3, 4, 1))
            else:
                ground_truth = ground_truth.permute((0, 4, 2, 3, 1))
            ground_truth = ground_truth.squeeze(-1)
        # warp labels
        ground_truth = grid_pull(ground_truth, grid, bound='zero', extrapolate=True, interpolation=1)
        if dtype_gt not in (torch.half, torch.float, torch.double):
            # One-hot labels to hard labels
            ground_truth = ground_truth.argmax(dim=1, keepdim=True).type(dtype_gt)
    elif tag == 'noise':
        # Additive gaussian noise to image
        # Parameter
        sd_prct = 0.02  # percentage of max intensity of batch and channel
        # Get max intensity in for each batch and channel
        mx = image.reshape((nbatch, nchan, nvox)).max(dim=-1, keepdim=True)[0]
        # Add 'lost' dimensions
        for d in range(ndim - 1):
            mx = mx.unsqueeze(-1)
        # Add noise to image
        image += sd_prct*mx*torch.randn_like(image)        
    elif tag == 'inu':
        # Multiplicative intensity non-uniformity (INU) to image
        # Parameters
        amplitude = 0.5
        fwhm = 60
        # Instantiate augmenter
        aug = BiasFieldTransform(amplitude=amplitude, fwhm=fwhm, mean=0.0,
                                 device=image.device, dtype=image.dtype)
        # Augment image
        image = aug(image)
    else:
        raise ValueError('Undefined tag {:}'.format(tag))

    return image, ground_truth


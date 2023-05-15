import torch
from nitorch import spatial
from nitorch.core import utils, math, py, constants, linalg
from nitorch.core.utils import _hist_to_quantile
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot', _as=True)
mcolors = try_import('matplotlib.colors', _as=True)
from warnings import warn


def _get_colormap_cat(colormap, nb_classes, dtype=None, device=None):
    if colormap is None:
        if not plt:
            raise ImportError('Matplotlib not available')
        if nb_classes <= 10:
            colormap = plt.get_cmap('tab10')
        elif nb_classes <= 20:
            colormap = plt.get_cmap('tab20')
        else:
            warn('More than 20 classes: multiple classes will share'
                 'the same color.')
            colormap = plt.get_cmap('tab20')
    elif isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if isinstance(colormap, mcolors.Colormap):
        n = nb_classes
        colormap = [colormap(i/(n-1))[:3] for i in range(n)]
    colormap = torch.as_tensor(colormap, dtype=dtype, device=device)
    return colormap


def _get_colormap_intensity(colormap, n=256, dtype=None, device=None):
    if colormap is None:
        if not plt:
            raise ImportError('Matplotlib not available')
        colormap = plt.get_cmap('gray')
    elif isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if isinstance(colormap, mcolors.Colormap):
        colormap = [colormap(i/(n-1))[:3] for i in range(n)]
    colormap = torch.as_tensor(colormap, dtype=dtype, device=device)
    return colormap


def _get_colormap_depth(colormap, n=256, dtype=None, device=None):
    if colormap is None:
        if not plt:
            raise ImportError('Matplotlib not available')
        colormap = plt.get_cmap('rainbow')
    elif plt and isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if mcolors and isinstance(colormap, mcolors.Colormap):
        colormap = [colormap(i/(n-1))[:3] for i in range(n)]
    else:
        raise ImportError('Matplotlib not available')
    colormap = torch.as_tensor(colormap, dtype=dtype, device=device)
    return colormap


def disp_to_rgb(image, vmax=None, scale=1, amplitude='value'):
    """Convert a displacement field to an RGB image.

    Directions are mapped to a Hue value from the HSV color space.
    Opposite directions therefore have opposite colors in the hue
    circle:
        * red   <-> cyan
        * green <-> magenta
        * blue  <-> yellow

    The image should have been sliced already (i.e., it should have
    exactly two spatial dimensions).

    Parameters
    ----------
    image : (*batch, K, H, W)
        A (batch of) 2D image, with displacements along the 'K' dimension.
    vmax : tensor_like, optional
        Absolute displacement value corresponding to the maximum intensity.
        By default, the maximum absolute value is used.
        Must be broadcastable to `batch`.
    scale : tensor_like, optional
        A scaling factor to apply in each dimension (e.g., voxel size)
    amplitude : {'value', 'saturation'}, default='value'
        Whether to use the value or saturation component of the HSV
        color space to map amplitudes.
        If 'value', 0 maps to black // If 'saturation', 0 maps to white

    Returns
    -------
    image : (*batch, H, W, 3)
        A (batch of) RGB image.

    """

    image = torch.as_tensor(image).detach()
    backend = utils.backend(image)
    *batch, dim, height, width = image.shape
    shape = (height, width)

    # define color space
    if dim > 3:
        raise ValueError('This function only works for K <= 3')
    hue_pos = [[0, 0, 1], [0, 1, 0], [0, 0, 1]]
    hue_neg = [[1, 1, 0], [1, 0, 1], [1, 1, 0]]
    hue_pos = hue_pos[:dim]
    hue_neg = hue_neg[:dim]
    hue_pos = torch.as_tensor(hue_pos, dtype=image.dtype, device=image.device)
    hue_neg = torch.as_tensor(hue_neg, dtype=image.dtype, device=image.device)

    # rescale intensities
    scale = utils.make_vector(scale, dim, **backend)[..., None, None]
    image = image * scale
    vmax = vmax or math.max(image.square().sum(-3).sqrt(), dim=[-1, -2])
    vmax = torch.as_tensor(vmax, **backend).clone()
    vmax[vmax == 0] = 1
    image = image / vmax[..., None, None, None]

    # convert
    image = image[..., None]
    cimage = image.new_zeros([*batch, *shape, 3])
    for d in range(dim):
        cimage += (image[..., d, :, :, :] > 0) * image[..., d, :, :, :].abs() * hue_pos[d]
        cimage += (image[..., d, :, :, :] < 0) * image[..., d, :, :, :].abs() * hue_neg[d]
    if amplitude[0] == 's':
        cimage.neg_().add_(1)

    cimage.clamp_(0, 1)  # for numerical errors
    return cimage


def prob_to_rgb(image, implicit=False, colormap=None):
    """Convert soft probabilities to an RGB image.

    Parameters
    ----------
    image : (*batch, K, H, W)
        A (batch of) 2D image, with categories along the 'K' dimension.
    implicit : bool, default=False
        Whether the background class is implicit.
        Else, the first class is assumed to be the background class.
    colormap : (K, 3) tensor or str, optional
        A colormap or the name of a matplotlib colormap.

    Returns
    -------
    image : (*batch, H, W, 3)
        A (batch of) RGB image.

    """

    if not implicit:
        image = image[..., 1:, :, :]

    *batch, nb_classes, height, width = image.shape
    shape = (height, width)
    colormap = _get_colormap_cat(colormap, nb_classes, image.dtype, image.device)

    cimage = image.new_zeros([*batch, *shape, 3])
    for i in range(nb_classes):
        cimage += image[..., i, :, :, None] * colormap[i % len(colormap)]

    return cimage.clamp_(0, 1)


def depth_to_rgb(image, colormap=None):
    """Convert soft probabilities to an RGB image.

    Parameters
    ----------
    image : (*batch, D, H, W)
        A (batch of) 3D image, with depth along the 'D' dimension.
    colormap : (D, 3) tensor or str, optional
        A colormap or the name of a matplotlib colormap.

    Returns
    -------
    image : (*batch, H, W, 3)
        A (batch of) RGB image.

    """

    *batch, depth, height, width = image.shape
    colormap = _get_colormap_depth(colormap, depth, image.dtype, image.device)

    image = utils.movedim(image, -3, -1)
    cimage = linalg.dot(image.unsqueeze(-2), colormap.T)
    cimage /= image.sum(-1, keepdim=True)
    cimage *= image.max(-1, keepdim=True).values

    return cimage.clamp_(0, 1)


def intensity_to_rgb(image, min=None, max=None, colormap='gray', n=256, eq=False):
    """Colormap an intensity image

    Parameters
    ----------
    image : (*batch, H, W) tensor
        A (batch of) 2d image
    min : tensor_like, optional
        Minimum value. Should be broadcastable to batch.
        Default: min of image for each batch element.
    max : tensor_like, optional
        Maximum value. Should be broadcastable to batch.
        Default: max of image for each batch element.
    colormap : str or (K, 3) tensor, default='gray'
        A colormap or the name of a matplotlib colormap.
    n : int, default=256
        Number of color levels to use.
    eq : bool or {'linear', 'quadratic', 'log', None}, default=None
        Apply histogram equalization.
        If 'quadratic' or 'log', the histogram of the transformed signal
        is equalized.

    Returns
    -------
    rgb : (*batch, H, W, 3) tensor
        A (batch of) of RGB image.

    """
    image = torch.as_tensor(image).detach()
    image = intensity_preproc(image, min=min, max=max, eq=eq)

    # map
    colormap = _get_colormap_intensity(colormap, n, image.dtype, image.device)
    shape = image.shape
    image = image.mul_(n-1).clamp_(0, n-1)
    image = image.reshape([1, -1, 1])
    colormap = colormap.T.reshape([1, 3, -1])
    image = spatial.grid_pull(colormap, image)
    image = image.reshape([3, *shape])
    image = utils.movedim(image, 0, -1)

    return image


def intensity_preproc(*images, min=None, max=None, eq=None):
    """(Joint) rescaling and intensity equalizing.

    Parameters
    ----------
    *images : (*batch, H, W) tensor
        Input (batch of) 2d images.
        All batch shapes should be broadcastable together.
    min : tensor_like, optional
        Minimum value. Should be broadcastable to batch.
        Default: 5th percentile of each batch element.
    max : tensor_like, optional
        Maximum value. Should be broadcastable to batch.
        Default: 95th percentile of each batch element.
    eq : {'linear', 'quadratic', 'log', None} or float, default=None
        Apply histogram equalization.
        If 'quadratic' or 'log', the histogram of the transformed signal
        is equalized.
        If float, the signal is taken to that power before being equalized.

    Returns
    -------
    *images : (*batch, H, W) tensor
        Preprocessed images.
        Intensities are scaled within [0, 1].

    """

    if len(images) == 1:
        images = [utils.to_max_backend(*images)]
    else:
        images = utils.to_max_backend(*images)
    backend = utils.backend(images[0])
    eps = constants.eps(images[0].dtype)

    # rescale min/max
    min = py.make_list(min, len(images))
    max = py.make_list(max, len(images))
    min = [utils.quantile(image, 0.05, bins=2048, dim=[-1, -2], keepdim=True)
           if mn is None else torch.as_tensor(mn, **backend)[None, None]
           for image, mn in zip(images, min)]
    min, *othermin = min
    for mn in othermin:
        min = torch.min(min, mn)
    del othermin
    max = [utils.quantile(image, 0.95, bins=2048, dim=[-1, -2], keepdim=True)
           if mx is None else torch.as_tensor(mx, **backend)[None, None]
           for image, mx in zip(images, max)]
    max, *othermax = max
    for mx in othermax:
        max = torch.maximum(max, mx)
    del othermax
    images = [torch.maximum(torch.minimum(image, max), min)
              for image in images]
    images = [image.mul_(1 / (max - min + eps)).add_(1 / (1 - max / min))
              for image in images]

    if not eq:
        return tuple(images) if len(images) > 1 else images[0]

    # reshape and concatenate
    batch = utils.expanded_shape(*[image.shape[:-2] for image in images])
    images = [image.expand([*batch, *image.shape[-2:]]) for image in images]
    shapes = [image.shape[-2:] for image in images]
    chunks = [py.prod(s) for s in shapes]
    images = [image.reshape([*batch, c]) for image, c in zip(images, chunks)]
    images = torch.cat(images, dim=-1)

    if eq is True:
        eq = 'linear'
    if not isinstance(eq, str):
        if eq >= 0:
            images = images.pow(eq)
        else:
            images = images.clamp_min_(constants.eps(images.dtype)).pow(eq)
    elif eq.startswith('q'):
        images = images.square()
    elif eq.startswith('log'):
        images = images.clamp_min_(constants.eps(images.dtype)).log()

    images = histeq(images, dim=-1)

    if not (isinstance(eq, str) and eq.startswith('lin')):
        # rescale min/max
        images -= math.min(images, dim=-1, keepdim=True)
        images /= math.max(images, dim=-1, keepdim=True)

    images = images.split(chunks, dim=-1)
    images = [image.reshape(*batch, *s) for image, s in zip(images, shapes)]

    return tuple(images) if len(images) > 1 else images[0]


def histeq(x, n=1024, dim=None):
    """Histogram equalization

    Notes
    -----
    .. The minimum and maximum values of the input tensor are preserved.
    .. A piecewise linear transform is applied so that the output
       quantiles match those of a "template" histogram.
    .. By default, the template histogram is flat.

    Parameters
    ----------
    x : tensor
        Input image
    n : int or tensor
        Number of bins or target histogram
    dim : [sequence of] int, optional
        Dimensions along which to compute the histogram. Default: all.

    Returns
    -------
    x : tensor
        Transformed image

    """
    x = torch.as_tensor(x)

    # compute target cumulative histogram
    if torch.is_tensor(n):
        other_hist = n
        n = len(other_hist)
    else:
        other_hist = x.new_full([n], 1/n)
    other_hist += constants.eps(other_hist.dtype)
    other_hist = other_hist.cumsum(-1) / other_hist.sum(-1, keepdim=True)
    other_hist[..., -1] = 1

    # compute cumulative histogram
    min = math.min(x, dim=dim)
    max = math.max(x, dim=dim)
    batch_shape = min.shape
    hist = utils.histc(x, n, dim=dim, min=min, max=max)
    hist += constants.eps(hist.dtype)
    hist = hist.cumsum(-1) / hist.sum(-1, keepdim=True)
    hist[..., -1] = 1

    # match histograms
    hist = hist.reshape([-1])
    shift = _hist_to_quantile(other_hist[None], hist)
    shift = shift.reshape([-1, n])
    shift /= n

    # reshape
    shift = shift.reshape([*batch_shape, n])

    # interpolate and apply shift
    eps = constants.eps(x.dtype)
    grid = x.clone()
    grid = grid.mul_(n / (max - min + eps)).add_(n / (1 - max / min)).sub_(1)
    grid = grid.flatten()[:, None, None]
    shift = spatial.grid_pull(shift.reshape([-1, 1, n]), grid,
                              bound='zero', extrapolate=True)
    shift = shift.reshape(x.shape)
    x = (x - min) * shift + min

    return x


def set_alpha(x, alpha=0.5):
    """Set the transparancy channel of an RGB image

    Parameters
    ----------
    x : (*batch, H, W, 3|4) tensor
        Input RGB images
    alpha : float or (*batch, H, W) tensor
        Values should be in 0..1
        Transparency value (0 = hidden, 1 = visible)

    Returns
    -------
    x : (*batch, H, W, 4) tensor
        Preprocessed images.
        Intensities are scaled within [0, 1].

    """
    if x.shape[-1] == 3:
        x0, x = x, x.new_empty([*x.shape[:-1], 4])
        x[..., :3] = x0
    else:
        x = x.clone()
    x[..., -1] = alpha
    x[..., -1].clamp_(0, 1)
    return x


def stack_layers(layers):
    """Stack layers of images

    Parameters
    ----------
    layers : sequence of (*batch, H, W, 3|4) tensor
        Input RGB images

    Returns
    -------
    x : tensor (*batch, H, W, 3) tensor
        Merged image

    """
    layers, layers0 = [], layers
    x = 0
    for layer in layers0:
        if layer.shape[-1] == 3:
            layer = set_alpha(layer, 1)
        alpha = layer[..., -1:]
        layer = layer[..., :-1]
        x = (1 - alpha) * x + alpha * layer
    return x

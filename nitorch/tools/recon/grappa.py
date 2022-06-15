import torch
from nitorch.core import py, utils, linalg


def recon(kspace, calib=None, kernel_size=5, ndim=None, lam=0.01, inplace=False,
          kernels=None):
    """GRAPPA reconstruction

    Parameters
    ----------
    kspace : ([*batch], coils, *freq)
        Accelerated k-space
    calib : ([*batch], coils, freq)
        Fully-sampled calibration region
    kernel_size : sequence[int], default=5
        GRAPPA kernel size
    ndim : int, default=kspace.dim()-1
        k-space dimension
    lam : float, default=0.01
        Tikhonov regularization
    inplace : bool, default=False
        Fill-in kspace in-place

    Returns
    -------
    kspace : ([*batch], coils, *freq)
        Filled k-space

    """
    ndim = ndim or (kspace.dim() - 1)
    kernel_size = py.make_list(kernel_size, ndim)
    mask = get_sampling_mask(kspace, ndim)
    code = get_pattern_codes(mask, kernel_size)
    return_kernels = kernels is True
    if not isinstance(kernels, dict):
        kernels = kernel_fit(calib, kernel_size, code.unique(), lam)
    kspace = kernel_apply(kspace, code, kernel_size, kernels, inplace=inplace)
    return (kspace, kernels) if return_kernels else kspace


def get_sampling_mask(kspace, ndim=None):
    """Compute the sampling mask from k-space

    Parameters
    ----------
    kspace : ([*batch], coils, *freq)
    ndim: int, default=kspace.dim()-1

    Returns
    -------
    mask : (*freq) tensor[bool]

    """
    ndim = ndim or kspace.dim() - 1
    kspace = kspace != 0
    kspace = kspace.reshape([-1, *kspace.shape[-ndim:]])
    kspace = kspace.any(0)
    return kspace


def get_pattern_codes(sampling_mask, kernel_size):
    """Compute the pattern's code about each voxel

    Parameters
    ----------
    sampling_mask : (*freq) tensor[bool]
    kernel_size : [sequence of] int

    Returns
    -------
    pattern_mask : (*freq) tensor[long]

    """
    ndim = sampling_mask.dim()
    kernel_size = py.make_list(kernel_size, ndim)
    sampling_mask = sampling_mask.long()
    sampling_mask = utils.pad(sampling_mask, [(k-1)//2 for k in kernel_size],
                              side='both')
    sampling_mask = utils.unfold(sampling_mask, kernel_size, stride=1)
    return pattern_to_code(sampling_mask, ndim)


def cartesian_sampling_mask(acceleration, shape, **backend):
    """Build a cartesian sampling mask

    Parameters
    ----------
    acceleration : sequence[int]
    shape : sequence[int]
    **backend

    Returns
    -------
    mask : (*shape) tensor[bool]

    """
    backend.setdefault('dtype', torch.bool)
    ndim = len(shape)
    acceleration = py.make_list(acceleration, ndim)
    sampling_mask = torch.zeros(shape, **backend)
    slicer = tuple(slice(None, None, a) for a in acceleration)
    sampling_mask[(Ellipsis, *slicer)] = True
    return sampling_mask


def pattern_to_code(pattern, ndim=None):
    """Convert sampling patterns to unique codes

    Parameters
    ----------
    pattern : ([*batch], *kernel_size) tensor[bool]
    ndim : int, default=mask.dim()

    Returns
    -------
    code : ([*batch]) tensor[long]

    """
    ndim = ndim or pattern.dim()
    kernel_size = pattern.shape[-ndim:]
    code = 2 ** torch.arange(kernel_size.numel())
    code = code.reshape(kernel_size)
    dims = (list(range(-ndim, 0)),) * 2
    pattern_code = torch.tensordot(pattern.to(code), code, dims)
    return pattern_code


def code_to_pattern(code, kernel_size, **backend):
    """Convert a unique code to a sampling pattern

    Parameters
    ----------
    code : int or ([*batch]) tensor[int]
    kernel_size : sequence of int

    Returns
    -------
    pattern : ([*batch], *kernel_size) tensor[bool]

    """
    backend.setdefault('dtype', torch.bool)
    if torch.is_tensor(code):
        backend.setdefault('device', code.device)
    kernel_size = py.make_list(kernel_size)

    def make_pattern(code):
        pattern = torch.zeros(kernel_size, **backend)
        pattern_flat = pattern.flatten()
        for i in range(len(pattern_flat)):
            pattern_flat[i] = bool((code >> i) & 1)
        return pattern

    if torch.is_tensor(code):
        pattern = code.new_zeros([*code.shape, *kernel_size], **backend)
        for code1 in code.unique():
            pattern[code == code1] = make_pattern(code1)
    else:
        pattern = make_pattern(code)
    return pattern


def code_has_center(code, kernel_size):
    """Return True if the pattern corresponding to code has the center sampled

    Parameters
    ----------
    code : int or tensor[int]
    kernel_size : sequence[int]

    Returns
    -------
    mask : bool or tensor[bool]

    """
    kernel_size = py.make_list(kernel_size)
    code_center = torch.arange(py.prod(kernel_size)).reshape(kernel_size)
    center = [(k-1)//2 for k in kernel_size]
    code_center = code_center[tuple(center)]
    code = (code >> code_center) & 1
    return code.bool() if torch.is_tensor(code) else bool(code)


def kernel_fit(calib, kernel_size, patterns, lam=0.01):
    """Compute GRAPPA kernels

    All batch elements should have the same sampling pattern

    Parameters
    ----------
    calib : ([*batch], coils, *freq)
        Fully-sampled calibration data
    kernel_size : sequence[int]
        GRAPPA kernel size
    patterns : (N,) tensor[int]
        Code of patterns for which to learn a kernel.
        See `pattern_to_code`.
    lam : float, default=0.01
        Tikhonov regularization

    Returns
    -------
    kernels : dict of int -> ([*batch], coils, coils, nb_elem) tensor
        GRAPPA kernels

    """
    kernel_size = py.make_list(kernel_size)
    ndim = len(kernel_size)
    coils, *freq = calib.shape[-ndim-1:]
    batch = calib.shape[:-ndim-1]

    # find all possible patterns
    patterns = utils.as_tensor(patterns, device=calib.device)
    if patterns.dtype is torch.bool:
        patterns = pattern_to_code(patterns, ndim)
    patterns = patterns.flatten()

    # learn one kernel for each pattern
    calib = utils.unfold(calib, kernel_size, collapse=True)   # [*B, C, N, *K]
    calib = utils.movedim(calib, -ndim-1, -ndim-2)            # [*B, N, C, *K]

    def t(x):
        return x.transpose(-1, -2)

    def conjt(x):
        return t(x).conj()

    def diag(x):
        return x.diagonal(0, -1, -2)

    kernels = {}
    center = [(k-1)//2 for k in kernel_size]
    center = (Ellipsis, *center)
    for pattern_code in patterns:
        if code_has_center(pattern_code, kernel_size):
            continue
        pattern = code_to_pattern(pattern_code, kernel_size, device=calib.device)
        pattern_size = pattern.sum()
        if pattern_size == 0:
            continue

        calib_target = calib[center]                        # [*B, N, C]
        calib_source = calib[..., pattern]                  # [*B, N, C, P]
        calib_size = calib_target.shape[-2]
        flat_shape = [*batch, calib_size, pattern_size * coils]
        calib_source = calib_source.reshape(flat_shape)     # [*B, N, C*P]
        # solve
        H = conjt(calib_source).matmul(calib_source)             # [*B, C*P, C*P]
        diag(H).add_(lam * diag(H).abs().max(-1, keepdim=True).values)
        diag(H).add_(lam)
        g = conjt(calib_source).matmul(calib_target)             # [*B, C*P, C]
        k = linalg.lmdiv(H, g).transpose(-1, -2)                 # [*B, C, C*P]
        k = k.reshape([*batch, coils, coils, pattern_size])      # [*B, C, C, P]
        kernels[pattern_code.item()] = k

    return kernels


def kernel_apply(kspace, patterns, kernel_size, kernels, inplace=False):
    """Apply a GRAPPA kernel to an accelerated k-space

    All batch elements should have the same sampling pattern

    Parameters
    ----------
    kspace : ([*batch], coils, *freq)
        Accelerated k-space
    patterns : (*freq) tensor[long]
        Code of sampling pattern about each k-space location
    kernel_size : sequence of int
        GRAPPA kernel size
    kernels : dict of int -> ([*batch], coils, coils, nb_elem) tensor
        Dictionary of GRAPPA kernels (keys are pattern codes)

    Returns
    -------
    kspace : ([*batch], coils, *freq)

    """
    ndim = patterns.dim()
    coils, *freq = kspace.shape[-ndim-1:]
    batch = kspace.shape[:-ndim-1]
    kernel_size = py.make_list(kernel_size, ndim)

    kspace_out = kspace
    if not inplace:
        kspace_out = kspace_out.clone()
    kspace = utils.pad(kspace, [(k-1)//2 for k in kernel_size], side='both')
    kspace = utils.unfold(kspace, kernel_size, stride=1)

    def t(x):
        return x.transpose(-1, -2)

    for code, kernel in kernels.items():
        kernel = kernels[code]
        pattern = code_to_pattern(code, kernel_size, device=kspace.device)
        pattern_size = pattern.sum()
        mask = patterns == code
        kspace1 = kspace[..., mask, :, :][..., pattern]
        kspace1 = kspace1.transpose(-2, -3) \
                         .reshape([*batch, -1, coils * pattern_size])
        kernel = kernel.reshape([*batch, coils, coils * pattern_size])
        kspace1 = t(kspace1.matmul(t(kernel)))
        kspace_out[..., mask] = kspace1

    return kspace_out


from warnings import warn
from nitorch.core import utils, py, dtypes
import torch
from ..optionals import numpy as np


def astype(dat, dtype, casting='unsafe'):
    """Casting (without rescaling)

    See `np.ndarray.astype`.

    ..warning:: The output `dtype` must exist in `dat`'s framework.

    Parameters
    ----------
    dat : tensor or ndarray
    dtype : dtype
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}

    Returns
    -------
    dat : tensor or ndarray

    Raises
    ------
    TypeError

    """
    if torch.is_tensor(dat):
        return _torch_astype(dat, dtype, casting=casting)
    else:
        dtype = dtypes.dtype(dtype).numpy
        return dat.astype(dtype, casting=casting)


def _torch_astype(dat, dtype, casting='unsafe'):
    """Equivalent to `np.astype` but for torch tensors

    Parameters
    ----------
    dat : torch.tensor
    dtype : dtype
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}

    Returns
    -------
    dat : torch.as_tensor

    """
    def error(indtype, outdtype):
        raise TypeError("Cannot cast tensor from dtype('{}') to "
                        "dtype('{}}') according to the rule '{}'."
                        .format(indtype, outdtype, casting))

    if not torch.is_tensor(dat):
        raise TypeError('Expected torch tensor but got {}'.format(type(dat)))
    outdtype = dtypes.dtype(dtype)
    if outdtype.torch is None:
        raise TypeError('Data type {} does not exist in pytorch'.format(dtype))
    indtype = dtypes.dtype(dat.dtype)

    casting = casting.lower()
    if ((casting == 'no' and indtype != outdtype) or
            (casting == 'equiv' and not dtypes.equivalent(indtype, outdtype)) or
            (casting == 'safe' and not indtype <= outdtype) or
            (casting == 'same_kind' and not dtypes.same_kind(indtype, outdtype))):
        error(indtype, outdtype)
    return dat.to(outdtype.torch)


def min(dat, dim=None):
    """Minimum element (across an axis) for tensors and arrays"""
    if torch.is_tensor(dat):
        return dat.min() if dim is None else dat.min(dim=dim).values
    else:
        return dat.min(axis=dim)


def max(dat, dim=None):
    """Maximum element (across an axis) for tensors and arrays"""
    if torch.is_tensor(dat):
        return dat.max() if dim is None else dat.max(dim=dim).values
    else:
        return dat.max(axis=dim)


def cat(dats, dim=None):
    if any(torch.is_tensor(dat) for dat in dats):
        dtype = utils.max_dtype(dats)
        device = utils.max_device(dats)
        dats = [torch.as_tensor(dat, dtype=dtype, device=device)
                for dat in dats]
        return torch.cat(dats, dim=dim)
    else:
        return np.concatenate(dats, axis=dim)
    

def writeable(dat):
    """Check if a tensor or array is writeable"""
    if torch.is_tensor(dat):
        return True
    else:
        return dat.flags.writeable


def copy(dat):
    """Duplicate a tensor or array"""
    if torch.is_tensor(dat):
        return dat.clone()
    else:
        return np.copy(dat)


def cast(dat, dtype, casting='unsafe', with_scale=False):
    """Cast an array to a given type.

    Parameters
    ----------
    dat : tensor or ndarray
        Input array
    dtype : dtype
        Output data type (should have the proper on-disk byte order)
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe',
               'rescale', 'rescale_zero}, default='unsafe'
        Casting method:
        * 'rescale' makes sure that the dynamic range in the array
          matches the range of the output data type
        * 'rescale_zero' does the same, but keeps the mapping `0 -> 0`
          intact.
        * all other options are implemented in numpy. See `np.can_cast`.
    with_scale : bool, default=False
        Return the scaling applied, if any.

    Returns
    -------
    dat : tensor or ndarray

    """
    scale = 1.
    indtype = dtypes.dtype(dat.dtype)
    outdtype = dtypes.dtype(dtype)

    if casting.startswith('rescale') and not outdtype.is_floating_point:
        # rescale
        # TODO: I am using float64 as an intermediate to cast
        #       Maybe I can do things in a nicer / more robust way
        minval = astype(min(dat), dtypes.float64)
        maxval = astype(max(dat), dtypes.float64)
        if not dtypes.equivalent(indtype, dtypes.float64):
            dat = dat.astype(np.float64)
        if not writeable(dat):
            dat = copy(dat)
        if casting == 'rescale':
            scale = (1 - minval / maxval) / (1 - outdtype.min / outdtype.max)
            offset = (outdtype.max - outdtype.min) / (maxval - minval)
            dat *= scale
            dat += offset
        else:
            assert casting == 'rescale_zero'
            if minval < 0 and not outdtype.is_signed:
                warn("Converting negative values to an unsigned datatype")
            scale = min(abs(outdtype.max / maxval) if maxval else float('inf'),
                        abs(outdtype.min / minval) if minval else float('inf'))
            dat *= scale
        indtype = dtypes.dtype(dat.dtype)
        casting = 'unsafe'

    # unsafe cast
    if indtype != outdtype:
        dat = astype(dat, outdtype, casting=casting)

    return (dat, scale) if with_scale else dat


def cutoff(dat, cutoff, dim=None):
    """Clip data when outside of a range defined by percentiles

    Parameters
    ----------
    dat : np.ndarray
        Input data
    cutoff : max or (min, max)
        Percentile cutoffs (in [0, 1])
    dim : int, optional
        Dimension(s) along which to compute percentiles

    Returns
    -------
    dat : np.ndarray
        Clipped data

    """
    if cutoff is None:
        return dat
    cutoff = sorted([100 * val for val in py.make_sequence(cutoff)])
    if len(cutoff) > 2:
        raise ValueError('Maximum to percentiles (min, max) should'
                         ' be provided. Got {}.'.format(len(cutoff)))
    pct = np.nanpercentile(dat, cutoff, axis=dim, keepdims=True)
    if len(pct) == 1:
        dat = np.clip(dat, a_min=None, a_max=pct[0])
    else:
        dat = np.clip(dat, a_min=pct[0], a_max=pct[1])
    return dat


def addnoise(dat, amplitude=1, seed=0):
    """Add uniform noise in [0, amplitude)

    Parameters
    ----------
    dat : tensor or ndarray
        Input array
    amplitude : float, default=1
        Noise amplitude

    Returns
    -------
    dat : tensor or ndarray
        Input array + noise

    """
    if torch.is_tensor(dat):
        return _torch_addnoise(dat, amplitude, seed)
    else:
        return _np_addnoise(dat, amplitude, seed)


def _np_addnoise(dat, amplitude=1, seed=0):
    # make sure the sampling dtype has native byte order
    tmpdtype = dat.dtype
    if not tmpdtype.isnative:
        tmpdtype = tmpdtype.newbyteorder()

    rng = np.random.default_rng(seed=seed)
    noise = np.empty_like(dat, dtype=tmpdtype)
    noise = rng.random(size=dat.shape, dtype=tmpdtype, out=noise)
    if amplitude != 1:
        noise *= amplitude
    dat += noise

    return dat


def _torch_addnoise(dat, amplitude=1, seed=0):
    with torch.random.fork_rng(dat.device):
        torch.random.manual_seed(seed)
        noise = torch.rand_like(dat)
    if amplitude != 1:
        noise *= amplitude
    dat += noise

    return dat


def _torch_cutoff(dat, cutoff=(0.0005, 0.9995), nh=4000):
    """Torch clip data when outside of a range, defined by percentiles.

     Parameters
    ----------
    dat : tensor
        Input data.
    cutoff : (min, max), default=(0.0005, 0.9995)
        Percentile cutoffs.
    nh : int, default=4000
        Number of histogram bins.

    Returns
    -------
    dat : tensor
        Clipped data.

    """
    device = dat.device
    dtype = dat.dtype
    # Rescale intensities between 1 and nh
    mn = torch.tensor([dat.min(), 1],
        dtype=dtype, device=device)[None, ...]
    mx = torch.tensor([dat.max(), 1],
        dtype=dtype, device=device)[None, ...]
    mc = torch.cat((mn, mx), dim=0)
    mc = torch.tensor([1, nh],
        dtype=dtype, device=device)[..., None].solve(mc)[0].squeeze()
    p = dat[(dat != 0)]
    p = (p * mc[0] + mc[1])
    # Make histogram
    p = p.round().long()
    h = torch.zeros(nh + 1, device=device, dtype=p.dtype)
    h.put_(p, torch.ones(1, dtype=p.dtype, device=device).
        expand_as(p), accumulate=True)
    h = h.type(dtype)
    h = h.cumsum(0)/h.sum()
    # Find percentiles
    mn_out = ((h <= cutoff[0]).sum(dim=0) - mc[1]) / mc[0]
    mx_out = ((h <= cutoff[1]).sum(dim=0) - mc[1]) / mc[0]

    return torch.clamp(dat, min=mn_out, max=mx_out)

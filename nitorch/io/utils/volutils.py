from warnings import warn
from nitorch.core import utils, py, dtypes
import torch
from ..optionals import numpy as np


_min = min


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
    """Concatenate tensors or arrays along an axis"""
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


def cast(dat, dtype, casting='unsafe', returns='dat', indtype=None, mask=None):
    """Cast an array to a given type.

    Parameters
    ----------
    dat : tensor or ndarray
        Input array
    dtype : dtype
        Output data type (should have the proper on-disk byte order)
    indtype : dtype, default=dat.dtype
        Original input dtype
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe',
               'rescale', 'rescale_zero'}, default='unsafe'
        Casting method:
        * 'rescale' makes sure that the dynamic range in the array
          matches the range of the output data type
        * 'rescale_zero' does the same, but keeps the mapping `0 -> 0`
          intact.
        * all other options are implemented in numpy. See `np.can_cast`.
    returns : [combination of] {'dat', 'scale', 'offset'}, default='dat'
        Return the scaling/offset applied, if any.
    mask : tensor or ndarray, optional
        Mask of voxels to use to compute min/max

    Returns
    -------
    dat : tensor or ndarray, if 'dat' in `returns`
    scale : float, if 'scale' in `returns`
    offset : float, if 'offset' in `returns`

    """
    scale = 1.
    offset = 0.
    indtype = dtypes.dtype(indtype or dat.dtype)
    outdtype = dtypes.dtype(dtype)

    if mask is None:
        mask = (Ellipsis,)

    if casting.startswith('rescale') and not outdtype.is_floating_point:
        # rescale
        # TODO: I am using float64 as an intermediate to cast
        #       Maybe I can do things in a nicer / more robust way
        minval = astype(min(dat[mask]), dtypes.float64)
        maxval = astype(max(dat[mask]), dtypes.float64)
        if 'dat' in returns:
            if not dtypes.equivalent(indtype, dtypes.float64):
                dat = dat.astype(np.float64)
            if not writeable(dat):
                dat = copy(dat)
        if casting == 'rescale':
            if maxval != minval:
                scale = (outdtype.max - outdtype.min) / (maxval - minval)
                offset = outdtype.max - maxval * (outdtype.max - outdtype.min) / (maxval - minval)
            else:
                scale = 1.0
                offset = -minval
            if 'dat' in returns:
                dat *= scale
                dat += offset
        else:
            assert casting == 'rescale_zero'
            if minval < 0 and not outdtype.is_signed:
                warn("Converting negative values to an unsigned datatype")
            scale = _min(abs(outdtype.max / maxval) if maxval else float('inf'),
                         abs(outdtype.min / minval) if minval else float('inf'))
            if 'dat' in returns:
                dat *= scale
        indtype = dtypes.dtype(dat.dtype)
        casting = 'unsafe'

    # unsafe cast
    if 'dat' in returns and indtype != outdtype:
        dat = astype(dat, outdtype, casting=casting)

    output = []
    for component in returns.split('+'):
        if component == 'dat':
            output.append(dat)
        elif component == 'scale':
            output.append(scale)
        elif component == 'offset':
            output.append(offset)
        else:
            output.append(None)
    return tuple(output) if len(output) > 1 else output[0] if output else None


def cutoff(dat, cutoff, dim=None):
    """Clip data when outside of a range defined by percentiles

    Parameters
    ----------
    dat : tensor or ndarray
        Input data
    cutoff : max or (min, max)
        Percentile cutoffs (in [0, 1])
    dim : int, optional
        Dimension(s) along which to compute percentiles

    Returns
    -------
    dat : tensor or ndarray
        Clipped data

    """
    if cutoff is None:
        return dat
    cutoff = sorted([100 * val for val in py.make_sequence(cutoff)])
    if len(cutoff) > 2:
        raise ValueError('Maximum to percentiles (min, max) should'
                         ' be provided. Got {}.'.format(len(cutoff)))
    if torch.is_tensor(dat):
        dat_for_quantile = dat
        if not dat.dtype.is_floating_point:
            dat_for_quantile = dat_for_quantile.float()

        cutoff = [val/100 for val in cutoff]
        pct = utils.quantile(dat_for_quantile, cutoff, bins=1024)

        if len(pct) == 1:
            mn, mx = None, pct[0]
        else:
            mn, mx = pct[0], pct[1]
        if not dat.dtype.is_floating_point:
            mx = mx.ceil()
            if mn is not None:
                mn = mn.floor()

        mx = mx.to(dat.dtype)
        if mn is not None:
            mn = mn.to(dat.dtype)
        dat.clamp_(mn, mx)
        return dat
    else:
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
    with torch.random.fork_rng([] if dat.device.type == 'cpu' else [dat.device]):
        torch.random.manual_seed(seed)
        noise = torch.rand_like(dat)
    if amplitude != 1:
        noise *= amplitude
    dat += noise

    return dat


def missing(dat, missing):
    """Return a mask of missing data"""
    missing = py.ensure_list(missing)
    if torch.is_tensor(dat):
        mask = utils.isin(dat, missing)
    else:
        mask = np.isin(dat, missing)
    return mask

import numpy as np
from warnings import warn
from nitorch.core import pyutils
from . import dtype as cast_dtype


def cast(dat, dtype, casting='unsafe', with_scale=False):
    """Cast an array to a given type.

    Parameters
    ----------
    dat : np.ndarray
        Input array
    dtype : np.dtype
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
    dat : np.ndarray[dtype]

    """
    scale = 1.
    info = cast_dtype.info(dtype)
    if casting.startswith('rescale') and not info['is_floating_point']:
        # rescale
        # TODO: I am using float64 as an intermediate to cast
        #       Maybe I can do things in a nicer / more robust way
        minval = dat.min().astype(np.float64)
        maxval = dat.max().astype(np.float64)
        if dat.dtype != np.float64:
            dat = dat.astype(np.float64)
        if not dat.flags.writeable:
            dat = np.copy(dat)
        if casting == 'rescale':
            scale = (1 - minval / maxval) / (1 - info['min'] / info['max'])
            offset = (info['max'] - info['min']) / (maxval - minval)
            dat *= scale
            dat += offset
        else:
            assert casting == 'rescale_zero'
            if minval < 0 and not info['is_signed']:
                warn("Converting negative values to an unsigned datatype")
            scale = min(abs(info['max'] / maxval) if maxval else float('inf'),
                        abs(info['min'] / minval) if minval else float('inf'))
            dat *= scale
        casting = 'unsafe'

    # unsafe cast
    if dat.dtype != dtype:
        dat = dat.astype(dtype, casting=casting)

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
    cutoff = sorted([100 * val for val in pyutils.make_sequence(cutoff)])
    if len(cutoff) > 2:
        raise ValueError('Maximum to percentiles (min, max) should'
                         ' be provided. Got {}.'.format(len(cutoff)))
    pct = np.nanpercentile(dat, cutoff, axis=dim, keepdims=True)
    if len(pct) == 1:
        dat = np.clip(dat, a_min=None, a_max=pct[0])
    else:
        dat = np.clip(dat, a_min=pct[0], a_max=pct[1])
    return dat

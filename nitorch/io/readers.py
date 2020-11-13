"""Register readers"""
import os


reader_classes = []


def map(file_like, permission='r+', keep_open=True):
    """Map a data file

    Parameters
    ----------
    file_like : str or file object
        Input file
    permission : {'r', 'r+'}, default='r+'
        File permission: 'r' means read-only while 'r+' means read and
        write. 'r+' is necessary for partial writing into the file.
    keep_open : bool, default=True
        Keep file open. Can be more efficient if multiple reads (or
        writes) are expected.

    Returns
    -------
    dat : MappedArray
        A `MappedArray` instance. Data can then be loaded in memory
        by calling `dat.data()` or `dat.fdata()`.

    """
    remaining_classes = []

    if isinstance(file_like, (str, os.PathLike)):
        # first guess based on file extension
        base, ext = os.path.splitext(file_like)
        if ext.lower() == '.gz':
            base, ext = os.path.splitext(base)

        for klass in reader_classes:
            if ext.lower() in klass.possible_extensions():
                try:
                    obj = klass(file_like, permission, keep_open)
                except klass.FailedReadError:
                    pass
            else:
                remaining_classes.append(klass)

    # second guess
    for klass in remaining_classes:
        try:
            obj = klass(file_like, permission, keep_open)
        except klass.FailedReadError:
            pass

    raise ValueError('Could not read {}'.format(file_like))


def read(file_like, *args, attributes=None, **kwargs):
    """Read a data file and load it in memory.

    Parameters
    ----------
    file_like : str or file object
        Path to file or file object (with methods `seek`, `read`)
    dtype : type or torch.dtype or np.dtype, optional
        Output data type. By default, keep the on-disk data type.
    device : torch.device, default='cpu'
        Output device.
    rand : bool, default=False
        If the on-disk dtype is not floating point, sample noise
        in the uncertainty interval.
    cutoff : float or (float, float), default=(0, 1)
        Percentile cutoff. If only one value is provided, it is
        assumed to relate to the upper percentile.
    dim : int or list[int], optional
        Dimensions along which to compute percentiles.
        By default, they are computed on the flattened array.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'rescale'}, default='unsafe'
        Controls what kind of data casting may occur:
            * 'no': the data types should not be cast at all.
            * 'equiv': only byte-order changes are allowed.
            * 'safe': only casts which can preserve values are allowed.
            * 'same_kind': only safe casts or casts within a kind,
              like float64 to float32, are allowed.
            * 'unsafe': any data conversions may be done.
            * 'rescale': the input data is rescaled to match the dynamic
              range of the output type. The minimum value in the data
              is mapped to the minimum value of the data type and the
              maximum value in the data is mapped to the maximum value
              of the data type.
            * 'rescale_zero': the input data is rescaled to match the
              dynamic range of the output type, but ensuring that
              zero maps to zero.
              > If the data is signed and cast to a signed datatype,
                zero maps to zero, and the scaling is chosen so that
                both the maximum and minimum value in the data fit
                in the output dynamic range.
              > If the data is signed and cast to an unsigned datatype,
                negative values "wrap around" (as with an unsafe cast).
              > If the data is unsigned and cast to a signed datatype,
                values are kept positive (the negative range is unused).
    numpy : bool, default=False
        Return a numpy array rather than a torch tensor.
    attributes : list[str]
        List of attributes to return as well.
        See `MappedArray` for the possible attributes.

    Returns
    -------
    dat : array or tensor
        The array loaded in memory
    attributes : dict, if attributes is not None
        Dictionary of attributes loaded as well

    """
    dat = map(file_like, permission='r', keep_open=False)
    attributes = {getattr(dat, key) for key in attributes}
    dat = dat.data(*args, **kwargs)
    if attributes:
        return dat, attributes
    else:
        return dat


def fread(file_like, *args, attributes=None, **kwargs):
    """Read a data file and load it -- scaled -- in memory.

    This function differs from `read` in several ways:
        * The output data type should be a floating point type.
        * If an affine scaling (slope, intercept) is defined in the
          file, it is applied to the data.
        * the default output data type is `torch.get_default_dtype()`.

    Parameters
    ----------
    file_like : str or file object
        Path to file or file object (with methods `seek`, `read`)
    dtype : dtype_like, optional
        Output data type. By default, use `torch.get_default_dtype()`.
        Should be a floating point type.
    device : torch.device, default='cpu'
        Output device.
    rand : bool, default=False
        If the on-disk dtype is not floating point, sample noise
        in the uncertainty interval.
    cutoff : float or (float, float), default=(0, 1)
        Percentile cutoff. If only one value is provided, it is
        assumed to relate to the upper percentile.
    dim : int or list[int], optional
        Dimensions along which to compute percentiles.
        By default, they are computed on the flattened array.
    numpy : bool, default=False
        Return a numpy array rather than a torch tensor.
    attributes : list[str]
        List of attributes to return as well.
        See `MappedArray` for the possible attributes.

    Returns
    -------
    dat : array or tensor
        The array loaded in memory
    attributes : dict, if attributes is not None
        Dictionary of attributes loaded as well

    """
    dat = map(file_like, permission='r', keep_open=False)
    attributes = {getattr(dat, key) for key in attributes}
    dat = dat.fdata(*args, **kwargs)
    if attributes:
        return dat, attributes
    else:
        return dat


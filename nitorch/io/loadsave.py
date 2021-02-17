"""Functional API to load and save arrays."""
import os
from .mapping import MappedFile
from .readers import reader_classes as all_reader_classes
from .writers import writer_classes as all_writer_classes

_DEBUG = False


def _trace(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)


def _unroll(seq):
    """Unroll nested sequences of stuff and return a flattened list"""
    out = []
    if isinstance(seq, (list, tuple)):
        for elem in seq:
            out.extend(_unroll(elem))
    else:
        out.append(seq)
    return out


def map(file_like, permission='r', keep_open=True, reader_classes=None):
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
    reader_classes : sequence of MappedFile, optional
        MappedFile classes that can be used to map the file.

    Returns
    -------
    dat : MappedFile
        A `MappedFile` instance. Data can then be loaded in memory
        by calling `dat.data()` or `dat.fdata()`.

    """
    reader_classes = reader_classes or all_reader_classes
    reader_classes = _unroll(reader_classes)
    untried_classes = []
    unchecked_classes = []

    if isinstance(file_like, MappedFile):
        # nothing to do
        return file_like

    if isinstance(file_like, (str, os.PathLike)):
        # first guess based on file extension
        # -> if check fails, cancel the read and retry later if needed
        base, ext = os.path.splitext(file_like)
        if ext.lower() == '.gz':
            base, ext = os.path.splitext(base)

        for klass in reader_classes:
            if ext.lower() in klass.possible_extensions():
                if klass.sniff(file_like):
                    try:
                        _trace('try', klass.__name__, end=' ')
                        out = klass(file_like, permission, keep_open)
                        _trace('-> success')
                        return out
                    except klass.FailedReadError:
                        _trace('-> failed')
                        pass
                else:
                    untried_classes.append(klass)
            else:
                unchecked_classes.append(klass)

    # second guess based on checks
    for klass in unchecked_classes:
        if klass.sniff(file_like):
            try:
                _trace('try', klass.__name__, end=' ')
                out = klass(file_like, permission, keep_open)
                _trace('-> success')
                return out
            except klass.FailedReadError:
                _trace('-> failed')
                pass
        else:
            untried_classes.append(klass)

    # third guess: all remaining classes
    for klass in untried_classes:
        try:
            _trace('try', klass.__name__, end=' ')
            out = klass(file_like, permission, keep_open)
            _trace('-> success')
            return out
        except klass.FailedReadError:
            _trace('-> failed')
            pass

    raise ValueError('Could not read {}'.format(file_like))


def load(file_like, *args, attributes=None, reader_classes=None, **kwargs):
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
        See `MappedFile` for the possible attributes.
    reader_classes : sequence of MappedFile, optional
        MappedFile classes that can be used to map the file.

    Returns
    -------
    dat : array or tensor
        The array loaded in memory
    attributes : dict, if attributes is not None
        Dictionary of attributes loaded as well

    """
    file = map(file_like, permission='r', keep_open=False,
               reader_classes=reader_classes)
    dat = file.data(*args, **kwargs)
    if attributes:
        attributes = {getattr(file, key) for key in attributes}
        return dat, attributes
    else:
        return dat


def loadf(file_like, *args, attributes=None, reader_classes=None, **kwargs):
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
        See `MappedFile` for the possible attributes.
    reader_classes : sequence of MappedFile, optional
        MappedFile classes that can be used to map the file.

    Returns
    -------
    dat : array or tensor
        The array loaded in memory
    attributes : dict, if attributes is not None
        Dictionary of attributes loaded as well

    """
    file = map(file_like, permission='r', keep_open=False,
               reader_classes=reader_classes)
    dat = file.fdata(*args, **kwargs)
    if attributes:
        attributes = {getattr(file, key) for key in attributes}
        return dat, attributes
    else:
        return dat


def save(dat, file_like, like=None, casting='unsafe', writer_classes=None,
         **metadata):
    """Write an array to disk.

    This function makes educated choices for the file format and
    its metadata based on the file extension, the data type and the
    other options provided.

    Parameters
    ----------
    dat : tensor or array or MappedFile
        Data to write
    file_like : str or file object
        Path to file or file object (with methods `seek`, `read`).
        If the extension is known, it gets priority over `like` when
        choosing the output format.
    like : file or MappedFile
        An array on-disk that should be used as a template for the new
        file. Its metadata/layout/etc will be mimicked as much as possible.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'rescale'}, default='unsafe'
        Controls what kind of data casting may occur.
        See `MappedFile.set_data`
    metadata : dict
        Metadata to store on disk. Values provided there will have
        priority over `like`.
    writer_classes : sequence of MappedFile, optional
        MappedFile classes that can be used to write the file.

    Returns
    -------
    dat : array or tensor
        The array loaded in memory
    attributes : dict, if attributes is not None
        Dictionary of attributes loaded as well

    """
    if like is not None and not isinstance(like, MappedFile):
        like = map(like)

    writer_classes = writer_classes or all_writer_classes
    writer_classes = _unroll(writer_classes)
    remaining_classes = []

    if isinstance(file_like, (str, os.PathLike)):
        # first guess based on file extension
        base, ext = os.path.splitext(file_like)
        if ext.lower() == '.gz':
            base, ext = os.path.splitext(base)

        for klass in writer_classes:
            if ext.lower() in klass.possible_extensions():
                try:
                    return klass.save_new(dat, file_like, like, casting, **metadata)
                except klass.FailedWriteError:
                    pass
            else:
                remaining_classes.append(klass)

    # second guess based on `like` object
    if like is not None and type(like) in remaining_classes:
        klass = type(like)
        try:
            return klass.save_new(dat, file_like, like, casting, **metadata)
        except klass.FailedWriteError:
            remaining_classes = [k for k in remaining_classes
                                 if k is not klass]

    # third guess: try everything that's left
    for klass in remaining_classes:
        try:
            return klass.save_new(dat, file_like, like, casting, **metadata)
        except klass.FailedWriteError:
            pass

    # failed
    raise ValueError('Could not write {}'.format(file_like))


def savef(dat, file_like, like=None, writer_classes=None, **metadata):
    """Write a scaled array to disk.

    This function makes educated choices for the file format and
    its metadata based on the file extension, the data type and the
    other options provided.

    The input data type must be a floating point type.

    Parameters
    ----------
    dat : tensor or array or MappedFile
        Data to write
    file_like : str or file object
        Path to file or file object (with methods `seek`, `read`).
        If the extension is known, it gets priority over `like` when
        choosing the output format.
    like : file or MappedFile
        An array on-disk that should be used as a template for the new
        file. Its metadata/layout/etc will be mimicked as much as possible.
    metadata : dict
        Metadata to store on disk. Values provided there will have
        priority over `like`.
    writer_classes : sequence of MappedFile, optional
        MappedFile classes that can be used to write the file.

    Returns
    -------
    dat : array or tensor
        The array loaded in memory
    attributes : dict, if attributes is not None
        Dictionary of attributes loaded as well

    """
    if like is not None and not isinstance(like, MappedFile):
        like = map(like)

    writer_classes = writer_classes or all_writer_classes
    writer_classes = _unroll(writer_classes)
    remaining_classes = []

    if isinstance(file_like, (str, os.PathLike)):
        # first guess based on file extension
        base, ext = os.path.splitext(file_like)
        if ext.lower() == '.gz':
            base, ext = os.path.splitext(base)

        for klass in writer_classes:
            if ext.lower() in klass.possible_extensions():
                try:
                    return klass.savef_new(dat, file_like, like, **metadata)
                except klass.FailedWriteError:
                    pass
            else:
                remaining_classes.append(klass)

    # second guess based on `like` object
    if like is not None and type(like) in remaining_classes:
        klass = type(like)
        try:
            return klass.savef_new(dat, file_like, like, **metadata)
        except klass.FailedWriteError:
            remaining_classes = [k for k in remaining_classes
                                 if k is not klass]

    # third guess: try everything that's left
    for klass in remaining_classes:
        try:
            return klass.savef_new(dat, file_like, like, **metadata)
        except klass.FailedWriteError:
            pass

    # failed
    raise ValueError('Could not write {}'.format(file_like))

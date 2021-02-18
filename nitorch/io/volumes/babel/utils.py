"""Additional utilities for nibabel."""
import numpy as np

from nibabel.fileslice import (
    _NullLock, threshold_heuristic, is_fancy, fill_slicer,
    _positive_slice, canonical_slicers, reduce, operator, read_segments,
    predict_shape, slicers2segments)


def full_heuristic(*args, **kwargs):
    """Heuristic that always read the full volume"""
    return threshold_heuristic(*args, **kwargs, skip_thresh=0)


def write_segments(fileobj, segments, dat, lock=None):
    """Write chunks of `dat` into `fileobj` at locations described in `segments`

    Parameters
    ----------
    fileobj : file-like object
        Implements `seek` and `write`
    segments : list
        list of 2 sequences where sequences are (offset, length), giving
        absolute file offset in bytes and number of bytes to write.
    dat : byte
        Byte array to write. Its total length should be equal to the
        sum of all segment lengths.
    lock : {None, threading.Lock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``write``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.
    """
    # Make a lock-like thing to make the code below a bit nicer
    if lock is None:
        lock = _NullLock()

    if len(segments) == 0:
        return
    if len(segments) == 1:
        offset, length = segments[0]
        with lock:
            fileobj.seek(offset)
            nb_written = fileobj.write(dat)
        if nb_written != length:
            raise ValueError('Expected to write {} bytes but wrote {}.'
                             .format(length, nb_written))
        return
    # More than one segment
    dat_offset = 0
    for offset, length in segments:
        with lock:
            fileobj.seek(offset)
            nb_written = fileobj.write(dat[dat_offset:dat_offset+length])
        dat_offset += length
        if nb_written != length:
            raise ValueError('Expected to write {} bytes but wrote {}.'
                             .format(length, nb_written))


def writeslice(dat, fileobj, sliceobj, shape, dtype, offset=0, order='C',
              heuristic=threshold_heuristic, lock=None):
    """ Write a data slice in `fileobj` using `sliceobj` slicer and array definitions

    `fileobj` contains the contiguous binary data for an array ``A`` of shape,
    dtype, memory layout `shape`, `dtype`, `order`, with the binary data
    starting at file offset `offset`.

    Our job is to write the array `dat` into the slice ``A[sliceobj]``
    in the most efficient way in terms of memory and time.

    Sometimes it will be quicker to read a larger chunk of memory, write
    into it in memory and write it back to disk, because it will save
    time we might lose doing short seeks on `fileobj`. Call these
    alternatives: (read + write); and skip.  This routine guesses when to
    (read + write) or skip using the callable `heuristic`, with a default
    using a hard threshold for the memory gap large enough to prefer a skip.

    Currently, we use the same heuristic for writing as the one used for
    reading. It might not be optimal, as triggering a 'read + write'
    involves more operations than a 'read + discard'.

    Parameters
    ----------
    fileobj : file-like object
        file-like object, opened for reading and writing in binary mode.
        Implements ``read``, ``write`` and ``seek``.
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``.
    shape : sequence
        shape of the full array inside `fileobj`.
    dtype : dtype specifier
        dtype (or input to ``numpy.dtype``) of the array inside `fileobj`.
    offset : int, optional
        offset of array data within `fileobj`
    order : {'C', 'F'}, optional
        memory layout of array in `fileobj`.
    heuristic : callable, optional
        function taking slice object, axis length, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer` and see :func:`threshold_heuristic` for an
        example.
    lock : {None, threading.Lock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``read``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.

    Returns
    -------
    sliced_arr : array
        Array in `fileobj` as sliced with `sliceobj`
    """
    if is_fancy(sliceobj):
        raise ValueError("Cannot handle fancy indexing")
    dtype = np.dtype(dtype)
    itemsize = int(dtype.itemsize)
    pre_slicers, segments, sub_slicers, sub_shape = calc_slicedefs_write(
        sliceobj, shape, itemsize, offset, order, heuristic)
    dat = dat[pre_slicers]
    if not all(sub_slicer == slice(None) for sub_slicer in sub_slicers):
        # read-and-write mode
        # it is faster to read a bigger block, write in that block
        # and write it back to disk than writing lots of small chunks
        n_bytes = reduce(operator.mul, sub_shape, 1) * itemsize
        bytes = read_segments(fileobj, segments, n_bytes, lock)
        block = np.ndarray(sub_shape, dtype, buffer=bytes, order=order)
        block[sub_slicers] = dat
        dat = block
    dat = dat.tobytes(order='C')
    write_segments(fileobj, segments, dat, lock)
    return


def calc_slicedefs_write(sliceobj, in_shape, itemsize, offset, order,
                         heuristic=threshold_heuristic):
    """ Return parameters for slicing an array into `sliceobj`

    Calculate the best combination of skips / (read + write) to use for
    write the data to disk / memory, then generate corresponding
    `segments`, the disk offsets and lengths to write in memory.  If we
    have chosen some (read + write) optimization, then we need to
    write sub-slices into bigger chunks using `sub_slicers`.

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    in_shape : sequence
        shape of underlying array to be sliced
    itemsize : int
        element size in array (in bytes)
    offset : int
        offset of array data in underlying file or memory buffer
    order : {'C', 'F'}
        memory layout of underlying array
    heuristic : callable, optional
        function taking slice object, dim_len, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer` and :func:`threshold_heuristic`

    Returns
    -------
    pre_slicers : tuple[index_like]
        Slicers to apply to the data to write before anything else.
        It removes new axis and makes strides positive.
    segments : tuple[(int, int)]
        List of segments defined by an offset and a length.
        Each segment correspond to one chunk of data to write (and
        eventually read) on disk.
    sub_slicers : tuple[index_like]
        This one implements the `read + write` vs `skip` strategy.
        Slicer used to write the chunk of data obtained from `pre_slicer`
        into a bigger chunk read from disk (`read + write` strategy).
        If `sub_slicers` is only made of full slices, no need to read
        a bigger chunk (`skip` strategy)
    sub_shape : tuple[int]
        Predicted shape of each segment
    """
    if order not in "CF":
        raise ValueError("order should be one of 'CF'")
    sliceobj = canonical_slicers(sliceobj, in_shape)
    # order fastest changing first (record reordering)
    if order == 'C':
        sliceobj = sliceobj[::-1]
        in_shape = in_shape[::-1]
    # Analyze sliceobj for new read_slicers and fixup post_slicers
    # read_slicers are the virtual slices; we don't slice with these, but use
    # the slice definitions to read the relevant memory from disk
    pre_slicers, write_slicers, sub_slicers = optimize_write_slicers(
        sliceobj, in_shape, itemsize, heuristic)
    # work out segments corresponding to write_slicers
    segments = slicers2segments(write_slicers, in_shape, offset, itemsize)
    sub_shape = predict_shape(write_slicers, in_shape)
    # If reordered, order shape, post_slicers
    if order == 'C':
        sub_shape = sub_shape[::-1]
        sub_slicers = sub_slicers[::-1]
        pre_slicers = pre_slicers[::-1]
    return tuple(pre_slicers), tuple(segments), tuple(sub_slicers), sub_shape


def optimize_write_slicers(sliceobj, in_shape, itemsize, heuristic):
    """ Calculates slices to write disk

    Parameters
    ----------
    sliceobj : tuple[index_like]
        something that can be used to slice an array as in ``arr[sliceobj]``.
        Can be assumed to be canonical in the sense of ``canonical_slicers``
    in_shape : sequence
        shape of underlying array to be sliced.  Array for `in_shape` assumed
        to be already in 'F' order. Reorder shape / sliceobj for slicing a 'C'
        array before passing to this function.
    itemsize : int
        element size in array (bytes)
    heuristic : callable
        function taking slice object, axis length, and stride length as
        arguments, returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer`; see :func:`threshold_heuristic` for an
        example.

    Returns
    -------
    pre_slicers : tuple[index_like]
        Any slicing to be applied to the array before writing.
        (discard any ``newaxis``, invert negative strides)
    write_slicers : tuple[index_like]
        Slicers that corresponds to the chunk of data actually written
        (and eventually read beforehand) to disk.
    sub_slicers : tuple[index_like]
        Slicers into the chunk of data described by `write_slicers`.
        If it is only made of full slices, it is not used and data is
        directly written to disk using `write_slicers` (skip strategy).
        Else, `write_slicers` is used to read (then write) a bigger
        chunk and `sub_slicers` is used to write data into this bigger
        chunk.
    """
    pre_slicers = []
    sub_slicers = []
    write_slicers = []
    real_no = 0
    stride = itemsize
    all_full = True
    for slicer in sliceobj:
        if slicer is None:
            pre_slicers.append(0)
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_last = real_no == len(in_shape)
        # make modified sliceobj (to_read, post_slice)
        pre_slicer, write_slicer, sub_slicer = optimize_write_slicer(
            slicer, dim_len, all_full, is_last, stride, heuristic)
        pre_slicers.append(pre_slicer)
        sub_slicers.append(sub_slicer)
        write_slicers.append(write_slicer)
        all_full = all_full and write_slicer == slice(None)
        stride *= dim_len
    return tuple(pre_slicers), tuple(write_slicers), tuple(sub_slicers)


def optimize_write_slicer(slicer, dim_len, all_full, is_slowest, stride,
                          heuristic=threshold_heuristic):
    """ Return maybe modified slice and post-slice slicing for `slicer`

    Parameters
    ----------
    write_slice : slice or int
        Index along a single axis
    dim_len : int
        length of axis along which to slice
    all_full : bool
        Whether dimensions up until now have been full (all elements)
    is_slowest : bool
        Whether this dimension is the slowest changing in memory / on disk
    stride : int
        size of one step along this axis
    heuristic : callable, optional
        function taking slice object, dim_len, stride length as arguments,
        returning one of 'full', 'contiguous', None. See
        :func:`threshold_heuristic` for an example.

    Returns
    -------
    pre_slice : slice or int
        slice to be applied before the array is written.
    write_slice : slice or int
        slice of data to write (or read-and-write).
        `write_slice` must always have positive ``step`` (because we don't
        want to go backwards in the buffer / file)
    sub_slice : slice
        slice used to write the current data-block into the larger
        read-and-written data block.

    Notes
    -----
    This is the heart of the algorithm for making segments from slice objects.

    A contiguous slice is a slice with ``slice.step in (1, -1)``

    A full slice is a continuous slice returning all elements.

    The main question we have to ask is whether we should split
    `write_slice` into (`write_slice`, `sub_slice`) to prefer a large
    read+write over many small writes. We apply a heuristic `heuristic`
    to decide whether to do this, and adapt `write_slice` and `sub_slice`
    accordingly.

    Otherwise we return `write_slice` almost unaltered. We simply split
    is into (`pre_slice`, `write_slice`) to ensure that the strides
    we use are positive.

    """
    # int or slice as input?
    try:  # if int - we drop a dim (no append)
        slicer = int(slicer)  # casts float to int as well
    except TypeError:  # slice
        # Deal with full cases first
        if slicer == slice(None):
            return slicer, slicer, slice(None)
        slicer = fill_slicer(slicer, dim_len)
        # actually equivalent to slice(None)
        if slicer == slice(0, dim_len, 1):
            return slice(None), slice(None), slice(None)
        # full, but reversed
        if slicer == slice(dim_len - 1, None, -1):
            return slice(None, None, -1), slice(None), slice(None)
        # Not full, mabye continuous
        is_int = False
    else:  # int
        if slicer < 0:  # make negative offsets positive
            slicer = dim_len + slicer
        is_int = True
    if all_full:
        action = heuristic(slicer, dim_len, stride)
        # Check return values (we may be using a custom function)
        if action not in ('full', 'contiguous', None):
            raise ValueError('Unexpected return %s from heuristic' % action)
        if is_int and action == 'contiguous':
            raise ValueError("int index cannot be contiguous")
        # If this is the slowest changing dimension, never upgrade None or
        # contiguous beyond contiguous (we've already covered the already-full
        # case)
        if is_slowest and action == 'full':
            action = None if is_int else 'contiguous'
        if action == 'full':
            # read a bigger block and write into it using `slicer`
            return slice(None), slice(None), slicer
        elif action == 'contiguous':  # Cannot be int
            # If this is already contiguous, default None behavior handles it
            step = slicer.step
            if step not in (-1, 1):
                if step < 0:
                    slicer = _positive_slice(slicer)
                return (slice(None, None, -1 if step < 0 else 1),
                        slice(slicer.start, slicer.stop, 1),
                        slice(None, None, slicer.step))
    # We only need to be positive
    if is_int:
        return None, slicer, slice(None)
    if slicer.step > 0:
        return slice(None), slicer, slice(None)
    return slice(None, None, -1), _positive_slice(slicer), slice(None)

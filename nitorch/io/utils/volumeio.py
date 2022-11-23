"""
Utilities to read and write (partially) into on-disk contiguous arrays
"""
# =============================================================================
# Many of these functions are either copy-pasted or adapted from NiBabel,
# which is covered by the MIT license.
#
# The MIT License
#
# Copyright (c) 2009-2019 Matthew Brett <matthew.brett@gmail.com>
# Copyright (c) 2010-2013 Stephan Gerhard <git@unidesign.ch>
# Copyright (c) 2006-2014 Michael Hanke <michael.hanke@gmail.com>
# Copyright (c) 2011 Christian Haselgrove <christian.haselgrove@umassmed.edu>
# Copyright (c) 2010-2011 Jarrod Millman <jarrod.millman@gmail.com>
# Copyright (c) 2011-2019 Yaroslav Halchenko <debian@onerussian.com>
# Copyright (c) 2015-2019 Chris Markiewicz <effigies@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# =============================================================================

import numpy as np
from functools import reduce
import operator
from .indexing import (
    is_fancy, simplify_slice, neg2pos, fill_slice, expand_index, guess_shape,
    slice_length, invert_slice,
)

# Threshold for memory gap above which we always skip, to save memory
# This value came from trying various values and looking at the timing with
# ``bench_fileslice``
SKIP_THRESH = 2 ** 8


class NullLock(object):
    """Dummy RLock"""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def prod(x):
    # Use reduce and mul to work around numpy integer overflow
    return reduce(operator.mul, x)


def threshold_heuristic(slicer, dim_len, stride, skip_thresh=SKIP_THRESH):
    """Whether to force full axis read or contiguous read of stepped slice

    Allows :func:`fileslice` to sometimes read memory that it will throw away
    in order to get maximum speed.  In other words, trade memory for fewer disk
    reads.

    Parameters
    ----------
    slicer : slice object, or int
        If slice, can be assumed to be full as in ``fill_slicer``
    dim_len : int
        length of axis being sliced
    stride : int
        memory distance between elements on this axis
    skip_thresh : int, optional
        Memory gap threshold in bytes above which to prefer skipping memory
        rather than reading it and later discarding.

    Returns
    -------
    action : {'full', 'contiguous', None}
        Gives the suggested optimization for reading the data

        * 'full' - read whole axis
        * 'contiguous' - read all elements between start and stop
        * None - read only memory needed for output
    """
    if isinstance(slicer, int):
        gap_size = (dim_len - 1) * stride
        return 'full' if gap_size <= skip_thresh else None
    slicer = fill_slice(neg2pos(slicer, dim_len), dim_len)
    step_size = abs(slicer.step) * stride
    if step_size > skip_thresh:
        return None  # Prefer skip
    # At least contiguous - also full?
    slicer = neg2pos(slicer, dim_len)
    start, stop = slicer.start, slicer.stop
    if start is None:
        start = 0
    if stop is None:
        stop = dim_len
    read_len = stop - start
    gap_size = (dim_len - read_len) * stride
    return 'full' if gap_size <= skip_thresh else 'contiguous'


def full_heuristic(*args, **kwargs):
    """Heuristic that always read the full volume"""
    return threshold_heuristic(*args, **kwargs, skip_thresh=0)


def read_array(fileobj, shape, dtype, offset=0, order='F'):
    """Get array from file with specified shape, dtype and file offset

    Parameters
    ----------
    fileobj : file-like
        open file-like object implementing at least read() and seek()
    shape : tuple[int]
        sequence specifying output array shape
    dtype : numpy dtype
        fully specified numpy dtype, including correct endianness
    offset : int, optional
        offset in bytes into `infile` to start reading array data. Default is 0
    order : {'F', 'C'} string
        order in which to read data.  Default is 'F' (fortran order).

    Returns
    -------
    arr : array-like
        array like object that can be sliced, containing data

    """
    dtype = np.dtype(dtype)
    # Get file-like object from Opener instance
    fileobj = getattr(fileobj, 'fobj', fileobj)
    if len(shape) == 0:
        return np.array([], dtype=dtype)
    n_bytes = prod(shape) * dtype.itemsize
    if n_bytes == 0:
        return np.array([], dtype=dtype)
    # Read data from file
    fileobj.seek(offset)
    buffer = np.empty(prod(shape), dtype=dtype).view('uint8').data
    n_read = fileobj.readinto(buffer)
    if n_bytes != n_read:
        raise IOError(f'Expected {n_bytes} bytes, got {n_read} bytes '
                      f'from {getattr(fileobj, "name", "object")}\n'
                      ' - could the file be damaged?')
    arr = np.ndarray(shape, dtype, buffer=buffer, order=order)
    arr.flags.writeable = True
    return arr


def write_array(dat, fileobj, offset=0, order='F', cast=None):
    """Helper function for writing arrays to file objects

    Parameters
    ----------
    dat : array-like
        array or array-like to write.
    fileobj : file-like
        file-like object implementing ``write`` method.
    offset : None or int, optional
        offset into fileobj at which to start writing data.
        Default is 0. None means start at current file position
    order : {'F', 'C'}, optional
        memory order to write array.  Default is 'F'
    cast : callable
        function to apply before writing
    """
    if order not in 'FC':
        raise ValueError('Order should be one of F or C')
    dat = np.squeeze(np.asanyarray(dat))
    if dat.ndim < 2:
        # Trick to allow loop over rows for 1D arrays
        dat = np.atleast_2d(dat)
    elif order == 'F':
        dat = dat.T
    if offset is not None:
        seek_tell(fileobj, offset)
    for dslice in dat:
        # cycle over first dimension to save memory
        if cast:
            dslice = cast(dslice)
        if not dslice.data.c_contiguous:
            dslice = np.copy(dslice, order='C')
        dslice = dslice.reshape([-1]).view('uint8').data
        fileobj.write(dslice)


def read_subarray(fileobj, sliceobj, shape, dtype, offset=0, order='C',
                  heuristic=threshold_heuristic, lock=None):
    """ Slice array in `fileobj` using `sliceobj` slicer and array definitions

    `fileobj` contains the contiguous binary data for an array ``A`` of shape,
    dtype, memory layout `shape`, `dtype`, `order`, with the binary data
    starting at file offset `offset`.

    Our job is to return the sliced array ``A[sliceobj]`` in the most efficient
    way in terms of memory and time.

    Sometimes it will be quicker to read memory that we will later throw away,
    to save time we might lose doing short seeks on `fileobj`.  Call these
    alternatives: (read + discard); and skip.  This routine guesses when to
    (read+discard) or skip using the callable `heuristic`, with a default using
    a hard threshold for the memory gap large enough to prefer a skip.

    Parameters
    ----------
    fileobj : file-like object
        file-like object, opened for reading in binary mode. Implements
        ``read`` and ``seek``.
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``.
    shape : sequence
        shape of full array inside `fileobj`.
    dtype : dtype specifier
        dtype of array inside `fileobj`, or input to ``numpy.dtype`` to specify
        array dtype.
    offset : int, optional
        offset of array data within `fileobj`
    order : {'C', 'F'}, optional
        memory layout of array in `fileobj`.
    heuristic : callable, optional
        function taking slice object, axis length, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer` and see :func:`threshold_heuristic` for an
        example.
    lock : {None, Lock, RLock, lock-like} optional
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
    segments, sliced_shape, post_slicers = calc_slicedefs_read(
        sliceobj, shape, itemsize, offset, order, heuristic)
    n_bytes = reduce(operator.mul, sliced_shape, 1) * itemsize
    arr_data = read_segments(fileobj, segments, n_bytes, lock)
    sliced = np.ndarray(sliced_shape, dtype, buffer=arr_data, order=order)
    return sliced[post_slicers]


def write_subarray(dat, fileobj, sliceobj, shape, dtype, offset=0, order='C',
                   heuristic=threshold_heuristic, lock=None):
    """Write a data slice in `fileobj` using `sliceobj` slicer and array definitions

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
    lock : {None, Lock, RLock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``read``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.

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
        block = np.ndarray(sub_shape, dtype, order=order)
        readinto_segments(fileobj, segments, block, lock)
        block[sub_slicers] = dat
        dat = block
    if order == 'F':
        dat = dat.T
    if not dat.data.c_contiguous:
        dat = np.copy(dat, order=order)
    dat = dat.reshape([-1]).view('uint8').data
    write_segments(dat, fileobj, segments, lock)
    return


def readinto_segments(fileobj, segments, buffer, lock=None):
    """ Read `n_bytes` byte data implied by `segments` from `fileobj`

    Parameters
    ----------
    fileobj : file-like object
        Implements `seek` and `read`
    segments : sequence
        list of 2 sequences where sequences are (offset, length), giving
        absolute file offset in *elements* and number of *elements* to read
    buffer : memoryview
    lock : {None, Lock, RLock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``read``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.

    Returns
    -------
    buffer : buffer object
        object implementing buffer protocol, such as byte string or ndarray or
        mmap or ctypes ``c_char_array``
    """
    # Make a lock-like thing to make the code below a bit nicer
    lock = lock or NullLock()

    if len(segments) == 0:
        if len(buffer) != 0:
            raise ValueError("No segments, but non-empty buffer")
        return buffer
    if len(segments) == 1:
        offset, length = segments[0]
        with lock:
            fileobj.seek(offset)
            fileobj.readinto(buffer)
        return buffer
    # More than one segment
    viewbytes = memoryview(buffer)
    dat_offset = 0
    for offset, length in segments:
        with lock:
            fileobj.seek(offset)
            fileobj.readinto(viewbytes[dat_offset:dat_offset+length])
        dat_offset += length
    return buffer


def read_segments(fileobj, segments, nelem, dtype, lock=None):
    """ Read `n_bytes` byte data implied by `segments` from `fileobj`

    Parameters
    ----------
    fileobj : file-like object
        Implements `seek` and `read`
    segments : sequence
        list of 2 sequences where sequences are (offset, length), giving
        absolute file offset in bytes and number of bytes to read
    nelem : int
        total number of *elements* that will be read
    dtype : np.dtype
        data type of array to be read
    lock : {None, Lock, RLock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``read``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.

    Returns
    -------
    buffer : array[dtype]
    """
    # Make a lock-like thing to make the code below a bit nicer
    lock = lock or NullLock()
    dat = np.empty(nelem, dtype=dtype)
    buffer = dat.view('uint8').data
    readinto_segments(fileobj, segments, buffer, lock)
    return dat


def write_segments(dat, fileobj, segments, lock=None):
    """Write chunks of `dat` into `fileobj` at locations described in `segments`

    Parameters
    ----------
    dat : memoryview
        Array to write. Its total length should be equal to the
        sum of all segment lengths.
    fileobj : file-like object
        Implements `seek` and `write`
    segments : list
        list of 2 sequences where sequences are (offset, length), giving
        absolute file offset in *elements* and number of *elements* to write.
    lock : {None, Lock, RLock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``write``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.
    """
    # Make a lock-like thing to make the code below a bit nicer
    lock = lock or NullLock()
    dat = memoryview(dat)

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


def slicers2segments(read_slicers, in_shape, offset, itemsize):
    """ Get segments from `read_slicers` given `in_shape` and memory steps

    Parameters
    ----------
    read_slicers : object
        something that can be used to slice an array as in ``arr[sliceobj]``
        Slice objects can by be assumed canonical as in ``canonical_slicers``,
        and positive as in ``_positive_slice``
    in_shape : sequence
        shape of underlying array on disk before reading
    offset : int
        offset of array data in underlying file or memory buffer
    itemsize : int
        element size in array (in bytes)

    Returns
    -------
    segments : list
        list of 2 element lists where lists are [offset, length], giving
        absolute memory offset in bytes and number of bytes to read
    """
    all_full = True
    all_segments = [[offset, itemsize]]
    stride = itemsize
    real_no = 0
    for read_slicer in read_slicers:
        if read_slicer is None:
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_int = isinstance(read_slicer, int)
        if not is_int:  # slicer is (now) a slice
            # make slice full (it will always be positive)
            read_slicer = fill_slice(read_slicer, dim_len)
            slice_len = slice_length(read_slicer, dim_len)
        is_full = read_slicer == slice(0, dim_len, 1)
        is_contiguous = not is_int and read_slicer.step == 1
        if all_full and is_contiguous:  # full or contiguous
            if read_slicer.start != 0:
                all_segments[0][0] += stride * read_slicer.start
            all_segments[0][1] *= slice_len
        else:  # Previous or current stuff is not contiguous
            if is_int:
                for segment in all_segments:
                    segment[0] += stride * read_slicer
            else:  # slice object
                segments = all_segments
                all_segments = []
                for i in range(read_slicer.start,
                               read_slicer.stop,
                               read_slicer.step):
                    for s in segments:
                        all_segments.append([s[0] + stride * i, s[1]])
        all_full = all_full and is_full
        stride *= dim_len
    return all_segments


def calc_slicedefs_read(sliceobj, in_shape, itemsize, offset, order,
                        heuristic=threshold_heuristic):
    """ Return parameters for slicing array with `sliceobj` given memory layout

    Calculate the best combination of skips / (read + discard) to use for
    reading the data from disk / memory, then generate corresponding
    `segments`, the disk offsets and read lengths to read the memory.  If we
    have chosen some (read + discard) optimization, then we need to discard the
    surplus values from the read array using `post_slicers`, a slicing tuple
    that takes the array as read from a file-like object, and returns the array
    we want.

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
    segments : list
        list of 2 element lists where lists are (offset, length), giving
        absolute memory offset in bytes and number of bytes to read
    read_shape : tuple
        shape with which to interpret memory as read from `segments`.
        Interpreting the memory read from `segments` with this shape, and a
        dtype, gives an intermediate array - call this ``R``
    post_slicers : tuple
        Any new slicing to be applied to the array ``R`` after reading via
        `segments` and reshaping via `read_shape`.  Slices are in terms of
        `read_shape`.  If empty, no new slicing to apply
    """
    if order not in "CF":
        raise ValueError("order should be one of 'CF'")
    sliceobj = expand_index(sliceobj, in_shape)
    # order fastest changing first (record reordering)
    if order == 'C':
        sliceobj = sliceobj[::-1]
        in_shape = in_shape[::-1]
    # Analyze sliceobj for new read_slicers and fixup post_slicers
    # read_slicers are the virtual slices; we don't slice with these, but use
    # the slice definitions to read the relevant memory from disk
    read_slicers, post_slicers = optimize_read_slicers(
        sliceobj, in_shape, itemsize, heuristic)
    # work out segments corresponding to read_slicers
    segments = slicers2segments(read_slicers, in_shape, offset, itemsize)
    # Make post_slicers empty if it is the slicing identity operation
    if all(s == slice(None) for s in post_slicers):
        post_slicers = []
    read_shape = guess_shape(read_slicers, in_shape)
    # If reordered, order shape, post_slicers
    if order == 'C':
        read_shape = read_shape[::-1]
        post_slicers = post_slicers[::-1]
    return list(segments), tuple(read_shape), tuple(post_slicers)


def optimize_read_slicers(sliceobj, in_shape, itemsize, heuristic):
    """ Calculates slices to read from disk, and apply after reading

    Parameters
    ----------
    sliceobj : object
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
    read_slicers : tuple
        `sliceobj` maybe rephrased to fill out dimensions that are better read
        from disk and later trimmed to their original size with `post_slicers`.
        `read_slicers` implies a block of memory to be read from disk. The
        actual disk positions come from `slicers2segments` run over
        `read_slicers`. Includes any ``newaxis`` dimensions in `sliceobj`
    post_slicers : tuple
        Any new slicing to be applied to the read array after reading.  The
        `post_slicers` discard any memory that we read to save time, but that
        we don't need for the slice.  Include any ``newaxis`` dimension added
        by `sliceobj`
    """
    read_slicers = []
    post_slicers = []
    real_no = 0
    stride = itemsize
    all_full = True
    for slicer in sliceobj:
        if slicer is None:
            read_slicers.append(None)
            post_slicers.append(slice(None))
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_last = real_no == len(in_shape)
        # make modified sliceobj (to_read, post_slice)
        read_slicer, post_slicer = optimize_read_slicer(
            slicer, dim_len, all_full, is_last, stride, heuristic)
        read_slicers.append(read_slicer)
        all_full = all_full and read_slicer == slice(None)
        if not isinstance(read_slicer, int):
            post_slicers.append(post_slicer)
        stride *= dim_len
    return tuple(read_slicers), tuple(post_slicers)


def optimize_read_slicer(slicer, dim_len, all_full, is_slowest, stride,
                         heuristic=threshold_heuristic):
    """ Return maybe modified slice and post-slice slicing for `slicer`

    Parameters
    ----------
    slicer : slice object or int
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
    to_read : slice object or int
        maybe modified slice based on `slicer` expressing what data should be
        read from an underlying file or buffer. `to_read` must always have
        positive ``step`` (because we don't want to go backwards in the buffer
        / file)
    post_slice : slice object
        slice to be applied after array has been read.  Applies any
        transformations in `slicer` that have not been applied in `to_read`. If
        axis will be dropped by `to_read` slicing, so no slicing would make
        sense, return string ``dropped``

    Notes
    -----
    This is the heart of the algorithm for making segments from slice objects.

    A contiguous slice is a slice with ``slice.step in (1, -1)``

    A full slice is a continuous slice returning all elements.

    The main question we have to ask is whether we should transform `to_read`,
    `post_slice` to prefer a full read and partial slice.  We only do this in
    the case of all_full==True.  In this case we might benefit from reading a
    continuous chunk of data even if the slice is not continuous, or reading
    all the data even if the slice is not full. Apply a heuristic `heuristic`
    to decide whether to do this, and adapt `to_read` and `post_slice` slice
    accordingly.

    Otherwise (apart from constraint to be positive) return `to_read` unaltered
    and `post_slice` as ``slice(None)``
    """
    # int or slice as input?
    try:  # if int - we drop a dim (no append)
        slicer = int(slicer)  # casts float to int as well
    except TypeError:  # slice
        # Deal with full cases first
        if slicer == slice(None):
            return slicer, slicer
        slicer = simplify_slice(slicer, dim_len)
        # actually equivalent to slice(None)
        if slicer == slice(None):
            return slice(None), slice(None)
        # full, but reversed
        if slicer == slice(None, None, -1):
            return slice(None), slice(None, None, -1)
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
            raise ValueError(f'Unexpected return {action} from heuristic')
        if is_int and action == 'contiguous':
            raise ValueError("int index cannot be contiguous")
        # If this is the slowest changing dimension, never upgrade None or
        # contiguous beyond contiguous (we've already covered the already-full
        # case)
        if is_slowest and action == 'full':
            action = None if is_int else 'contiguous'
        if action == 'full':
            return slice(None), slicer
        elif action == 'contiguous':  # Cannot be int
            # If this is already contiguous, default None behavior handles it
            step = slicer.step
            if step not in (None, -1, 1):
                if step < 0:
                    slicer = invert_slice(slicer, dim_len)
                return (slice(slicer.start, slicer.stop, None),
                        slice(None, None, step))
    # We only need to be positive
    if is_int:
        return slicer, 'dropped'
    if slicer.step and slicer.step < 0:
        return invert_slice(slicer, dim_len), slice(None, None, -1)
    return slicer, slice(None)


def calc_slicedefs_write(sliceobj, in_shape, itemsize, offset, order='C',
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
    sliceobj = expand_index(sliceobj, in_shape)
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
    sub_shape = guess_shape(write_slicers, in_shape)
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
        slicer = simplify_slice(slicer, dim_len)
        # full
        if slicer == slice(None):
            return slice(None), slice(None), slice(None)
        # full, but reversed
        if slicer == slice(None, None, -1):
            return slice(None, None, -1), slice(None), slice(None)
        # Not full, maybe continuous
        is_int = False
    else:  # int
        if slicer < 0:  # make negative offsets positive
            slicer = dim_len + slicer
        is_int = True
    if all_full:
        action = heuristic(slicer, dim_len, stride)
        # Check return values (we may be using a custom function)
        if action not in ('full', 'contiguous', None):
            raise ValueError(f'Unexpected return {action} from heuristic')
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
                    slicer = neg2pos(slicer, dim_len)
                return (slice(None, None, -1 if step < 0 else 1),
                        slice(slicer.start, slicer.stop, 1),
                        slice(None, None, slicer.step))
    # We only need to be positive
    if is_int:
        return None, slicer, slice(None)
    if slicer.step > 0:
        return slice(None), slicer, slice(None)
    return slice(None, None, -1), neg2pos(slicer, dim_len), slice(None)


def write_zeros(fileobj, count, block_size=8194):
    """ Write `count` zero bytes to `fileobj`

    Parameters
    ----------
    fileobj : file-like object
        with ``write`` method
    count : int
        number of bytes to write
    block_size : int, optional
        largest continuous block to write.
    """
    nblocks = int(count // block_size)
    rem = count % block_size
    blk = b'\x00' * block_size
    for bno in range(nblocks):
        fileobj.write(blk)
    fileobj.write(b'\x00' * rem)


def seek_tell(fileobj, offset, write0=False):
    """ Seek in `fileobj` or check we're in the right place already

    Parameters
    ----------
    fileobj : file-like
        object implementing ``seek`` and (if seek raises an IOError) ``tell``
    offset : int
        position in file to which to seek
    write0 : {False, True}, optional
        If True, and standard seek fails, try to write zeros to the file to
        reach `offset`.  This can be useful when writing bz2 files, that cannot
        do write seeks.
    """
    try:
        fileobj.seek(offset)
    except IOError as e:
        # This can be a negative seek in write mode for gz file object or any
        # seek in write mode for a bz2 file object
        pos = fileobj.tell()
        if pos == offset:
            return
        if not write0:
            raise IOError(str(e))
        if pos > offset:
            raise IOError("Can't write to seek backwards")
        fileobj.write(b'\x00' * (offset - pos))
        assert fileobj.tell() == offset

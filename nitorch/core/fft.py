"""Fourier transform module with backward compatibility

PyTorch introduced support for complex data types in version 1.6.
Before that, complex tensor were stored in real data types with an
additional dimension encoding the real and imaginary components.
Specifically, this convention was used in the fft/rfft/ifft functions.

In PyTorch 1.6 and 1.7, the old convention was still used in the fft-like
functions, which did not work with proper complex tensors. In PyTorch 1.7,
a new fft module was introduced. It implements functions that understand
complex tensors, but the old non-complex fft functions were kept as well
for backward compatibility. In PyTorch 1.8, the old functions were dropped
and additional helpers (fftshift, fftfreq) were implemented in the new
fft module.

This module mimics the PyTorch 1.8 fft module and calls the torch
functions when they are available. In PyTorch <= 1.5, the old convention
is used to represent complex tensors.

"""

import itertools
import torch
from . import py, utils

_torch_has_complex = utils.torch_version('>=', (1, 6))
_torch_has_fft_module = utils.torch_version('>=', (1, 7))
_torch_has_fftshift = utils.torch_version('>=', (1, 8))


if _torch_has_fft_module:
    import torch.fft as fft_mod


def fftshift(x, dim=None):
    """Move the first value to the center of the tensor.

    Notes
    -----
    .. If the dimension has an even shape, the center is the first
        position *after* the middle of the tensor: `c = s//2`
    .. This function triggers a copy of the data.
    .. If the dimension has an even shape, `fftshift` and `ifftshift`
       are equivalent.

    Parameters
    ----------
    x : tensor
        Input tensor
    dim : [sequence of] int, optional
        Dimensions to shift

    Returns
    -------
    x : tensor
        Shifted tensor

    """
    x = torch.as_tensor(x)
    if _torch_has_fftshift:
        return fft_mod.fftshift(x, dim)

    if dim is None:
        dim = list(range(x.dim()))
    dim = py.make_list(dim)
    if len(dim) > 1:
        x = x.clone()  # clone to get an additional buffer

    y = torch.empty_like(x)
    slicer = [slice(None)] * x.dim()
    for d in dim:
        # move front to back
        pre = list(slicer)
        pre[d] = slice(None, (x.shape[d]+1)//2)
        post = list(slicer)
        post[d] = slice(x.shape[d]//2, None)
        y[tuple(post)] = x[tuple(pre)]
        # move back to front
        pre = list(slicer)
        pre[d] = slice(None, x.shape[d]//2)
        post = list(slicer)
        post[d] = slice((x.shape[d]+1)//2, None)
        y[tuple(pre)] = x[tuple(post)]
        # exchange buffers
        x, y = y, x

    return x


def ifftshift(x, dim=None):
    """Move the center value to the front of the tensor.

    Notes
    -----
    .. If the dimension has an even shape, the center is the first
        position *after* the middle of the tensor: `c = s//2`
    .. This function triggers a copy of the data.
    .. If the dimension has an even shape, `fftshift` and `ifftshift`
       are equivalent.

    Parameters
    ----------
    x : tensor
        Input tensor
    dim : [sequence of] int, default=all
        Dimensions to shift

    Returns
    -------
    x : tensor
        Shifted tensor

    """
    x = torch.as_tensor(x)
    if _torch_has_fftshift:
        if isinstance(dim, range):
            dim = tuple(dim)
        return fft_mod.ifftshift(x, dim)

    if dim is None:
        dim = list(range(x.dim()))
    dim = py.make_list(dim)
    if len(dim) > 1:
        x = x.clone()  # clone to get an additional buffer

    y = torch.empty_like(x)
    slicer = [slice(None)] * x.dim()
    for d in dim:
        # move back to front
        pre = list(slicer)
        pre[d] = slice(None, (x.shape[d]+1)//2)
        post = list(slicer)
        post[d] = slice(x.shape[d]//2, None)
        y[tuple(pre)] = x[tuple(post)]
        # move front to back
        pre = list(slicer)
        pre[d] = slice(None, x.shape[d]//2)
        post = list(slicer)
        post[d] = slice((x.shape[d]+1)//2, None)
        y[tuple(post)] = x[tuple(pre)]
        # exchange buffers
        x, y = y, x

    return x


def fftfreq(n, d=1.0, *,
            dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """Computes the discrete Fourier Transform sample frequencies.

    Parameters
    ----------
    n : int
        Signal length
    d : float
        Voxel size

    Other Parameters
    ----------------
    dtype : torch.dtype, optional
    layout : torch.layout, optional
    device : torch.device, optional
    requires_grad : bool, default=False

    Returns
    -------
    freq : (n,) tensor
        [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)

    """
    dtype = dtype or torch.get_default_dtype()
    backend = dict(dtype=dtype, layout=layout,
                   device=device, requires_grad=requires_grad)

    if _torch_has_fftshift:
        return fft_mod.fftfreq(n, d, **backend)

    f = torch.empty(n, **backend)
    mid = (n - 1) // 2 + 1
    f[:mid] = torch.arange(mid, out=f[:mid], **backend)
    f[mid:] = torch.arange(-(n // 2), 0, out=f[mid:], **backend)
    f /= (d*n)
    return f


def rfftfreq(n, d=1.0, *,
            dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """Computes the discrete real Fourier Transform sample frequencies.

    Parameters
    ----------
    n : int
        Signal length
    d : float
        Voxel size

    Other Parameters
    ----------------
    dtype : torch.dtype, optional
    layout : torch.layout, optional
    device : torch.device, optional
    requires_grad : bool, default=False

    Returns
    -------
    freq : (n // 2 + 1,) tensor
        [0, 1, ..., n // 2 + 1]  / (d * n)

    """
    dtype = dtype or torch.get_default_dtype()
    backend = dict(dtype=dtype, layout=layout,
                   device=device, requires_grad=requires_grad)

    if _torch_has_fftshift:
        return fft_mod.rfftfreq(n, d, **backend)

    f = torch.arange(n // 2 + 1, **backend)
    f /= (d*n)
    return f


def fft(input, n=None, dim=-1, norm='backward', real=None):
    """One dimensional discrete Fourier transform.

    Parameters
    ----------
    input : tensor
        Input signal.
        If torch <= 1.5, the last dimension must be of length 2 and
        contain the real and imaginary parts of the signal, unless
        `real is True`.
    n : int, optional
        Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim : int, default=-1
        Dimension along which to take the Fourier transform.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    norm : {'forward', 'backward', 'ortho'}, default='backward'
        forward : normalize by 1/n (equivalent to ifft('backward'))
        backward : no normalization
        ortho :  normalize by 1/sqrt(n) (making the FFT orthonormal)
    real : bool, default=False
        Only used if torch <= 1.5.
        If True, the input signal has no imaginary component and the
        dimension encoding the real and imaginary parts does not exist.

    Returns
    -------
    output : tensor
        Fourier transform of the input signal. Complex tensor.
        If torch <= 1.5,  the last dimension is of length 2 and
        contain the real and imaginary parts of the signal.

    """
    if _torch_has_fft_module:
        return fft_mod.fft(input, n, dim, norm=norm)

    # Make real and move processed dimension to the right
    if _torch_has_complex:
        input = utils.movedim(input, dim, -1)
        if input.is_complex():
            input = torch.view_as_real(input)
        else:
            real = True
    elif real:
        input = utils.movedim(input, dim, -1)
    else:
        input = utils.movedim(input, dim if dim > 0 else dim - 1, -2)

    # Crop/pad
    dim1 = dim if dim > 0 else dim - (not real)
    if dim1 < 0:
        dim1 += input.dim()
    if n:
        if input.shape[dim1] > n:
            input = utils.slice_tensor(input, slice(n), dim1)
        elif input.shape[dim1] < n:
            pad = [0] * (dim1-1) + [n - input.shape[dim1]]
            input = utils.pad(input, pad, side='post')
    else:
        n = input.shape[dim1]

    # Do fft
    if real:
        output = torch.rfft(input, 1, normalized=(norm == 'ortho'),
                            onesided=False)
        if norm == 'forward':
            output /= float(n)
    else:
        if norm == 'forward':
            output = torch.ifft(input, 1)
        else:
            output = torch.fft(input, 1, normalized=(norm == 'ortho'))

    # Make complex and move back dimension to its original position
    if _torch_has_complex:
        output = torch.view_as_complex()
        output = utils.movedim(output, -1, dim)
    else:
        output = utils.movedim(output, -2, dim if dim > 0 else dim - 1)

    return output


def ifft(input, n=None, dim=-1, norm='backward', real=None):
    """One dimensional discrete inverse Fourier transform.

    Parameters
    ----------
    input : tensor
        Input signal.
        If torch <= 1.5, the last dimension must be of length 2 and
        contain the real and imaginary parts of the signal, unless
        `real is True`.
    n : int, optional
        Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim : int, default=-1
        Dimension along which to take the Fourier transform.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    norm : {'forward', 'backward', 'ortho'}, default='backward'
        forward : no normalization (equivalent to fft('backward'))
        backward : normalize by 1/n
        ortho :  normalize by 1/sqrt(n) (making the IFFT orthonormal)
    real : bool, default=False
        Only used if torch <= 1.5.
        If True, the input signal has no imaginary component and the
        dimension encoding the real and imaginary parts does not exist.
    symmetric : bool, default=False
        If True, assume the the input signal is hermitian-symmetric
        and return a real tensor.

    Returns
    -------
    output : tensor
        Inverse Fourier transform of the input signal. Complex tensor.
        If torch <= 1.5,  the last dimension is of length 2 and
        contain the real and imaginary parts of the signal.

    """
    if _torch_has_fft_module:
        return fft_mod.ifft(input, n, dim, norm=norm)

    # Make real and move processed dimension to the right
    if _torch_has_complex:
        input = utils.movedim(input, dim, -1)
        if input.is_complex():
            input = torch.view_as_real(input)
        else:
            real = True
    elif real:
        input = utils.movedim(input, dim, -1)
    else:
        input = utils.movedim(input, dim if dim > 0 else dim - 1, -2)

    # Crop/pad
    dim1 = dim if dim > 0 else dim - (not real)
    if dim1 < 0:
        dim1 += input.dim()
    if n:
        if input.shape[dim1] > n:
            input = utils.slice_tensor(input, slice(n), dim1)
        elif input.shape[dim1] < n:
            pad = [0] * (dim1-1) + [n - input.shape[dim1]]
            input = utils.pad(input, pad, side='post')
    else:
        n = input.shape[dim1]

    # Do ifft
    if real:
        output = torch.rfft(input, 1, normalized=(norm == 'ortho'),
                            onesided=False)
        if norm == 'backward':
            output /= float(n)
    else:
        if norm == 'forward':
            output = torch.fft(input, 1)
        else:
            output = torch.ifft(input, 1, normalized=(norm == 'ortho'))

    # Make complex and move back dimension to its original position
    if _torch_has_complex:
        output = torch.view_as_complex()
        output = utils.movedim(output, -1, dim)
    else:
        output = utils.movedim(output, -2, dim if dim > 0 else dim - 1)

    return output


def rfft(input, n=None, dim=-1, norm='backward'):
    """One dimensional real-to-complex inverse Fourier transform.

    The FFT of a real signal is Hermitian-symmetric, X[i] = conj(X[-i])
    so the output contains only the positive frequencies below the
    Nyquist frequency. To compute the full output, use fft()
    (or fft(real=True) in torch <= 1.5).

    Parameters
    ----------
    input : tensor
        Input real signal.
    n : int, optional
        Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim : int, default=-1
        Dimension along which to take the Fourier transform.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    norm : {'forward', 'backward', 'ortho'}, default='backward'
        forward : normalize by 1/n
        backward : no normalization
        ortho :  normalize by 1/sqrt(n) (making the FFT orthonormal)

    Returns
    -------
    output : tensor
        Fourier transform of the input signal. Complex tensor.
        Only positive frequencies are returned.
        If torch <= 1.5,  the last dimension is of length 2 and
        contain the real and imaginary parts of the signal.

    """
    if _torch_has_fft_module:
        return fft_mod.rfft(input, n, dim, norm=norm)

    # Move processed dimension to the right
    input = utils.movedim(input, dim, -1)

    # Crop/pad
    dim1 = dim
    if dim1 < 0:
        dim1 += input.dim()
    if n:
        if input.shape[dim1] > n:
            input = utils.slice_tensor(input, slice(n), dim1)
        elif input.shape[dim1] < n:
            pad = [0] * (dim1-1) + [n - input.shape[dim1]]
            input = utils.pad(input, pad, side='post')
    else:
        n = input.shape[dim1]

    # Do fft
    output = torch.rfft(input, 1, normalized=(norm == 'ortho'),
                        onesided=True)
    if norm == 'forward':
        output /= float(n)

    # Make complex and move back dimension to its original position
    if _torch_has_complex:
        output = torch.view_as_complex()
        output = utils.movedim(output, -1, dim)
    else:
        output = utils.movedim(output, -2, dim if dim > 0 else dim - 1)

    return output


def irfft(input, n=None, dim=-1, norm='backward', real=None):
    """One dimensional complex-to-real inverse Fourier transform.

    The input is interpreted as a one-sided Hermitian signal in the
    Fourier domain, as produced by rfft(). By the Hermitian property,
    the output will be real-valued.

    Notes
    -----
    .. The correct interpretation of the Hermitian input depends on the
       length of the original data, as given by `n`. This is because each
       input shape could correspond to either an odd or even length signal.
       By default, the signal is assumed to be even length and odd signals
       will not round-trip properly. So, it is recommended to always pass
       the signal length `n`.

    Parameters
    ----------
    input : tensor
        Input signal, containing only positive frequencies.
    n : int, optional
        Output signal length. If given, the input will either be
        zero-padded or trimmed to this length before computing the real
        IFFT. Defaults to even output: n=2*(input.size(dim) - 1).
    dim : int, default=-1
        Dimension along which to take the Fourier transform.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    norm : {'forward', 'backward', 'ortho'}, default='backward'
        forward : normalize by 1/n
        backward : no normalization
        ortho :  normalize by 1/sqrt(n) (making the FFT orthonormal)
    real : bool, default=False
        Only used if torch <= 1.5.
        If True, the input signal has no imaginary component and the
        dimension encoding the real and imaginary parts does not exist.

    Returns
    -------
    output : tensor
        Fourier transform of the input signal. Real tensor.

    """
    if _torch_has_fft_module:
        return fft_mod.irfft(input, n, dim, norm=norm)

    # Make real and move processed dimension to the right
    if _torch_has_complex:
        input = utils.movedim(input, dim, -1)
        if input.is_complex():
            input = torch.view_as_real(input)
        else:
            real = True
    elif real:
        input = utils.movedim(input, dim, -1)
    else:
        input = utils.movedim(input, dim if dim > 0 else dim - 1, -2)

    if real:
        # The output should be real *and* symmetric.
        # I can either allocate an empty imaginary component and
        # call irfft or build the full signal and call rfft + normalize
        zero = input.new_zeros([]).expand(input.shape)
        input = torch.stack([input, zero])

    # Do fft
    output = torch.irfft(input, 1, normalized=(norm == 'ortho'),
                         onesided=True, signal_sizes=[n])
    n = output.shape[-1]
    if norm == 'backward':
        output /= float(n)

    # Move back dimension to its original position
    output = utils.movedim(output, -1, dim)

    return output


def hfft(input, n=None, dim=-1, norm='backward', real=None):
    """One dimensional complex-to-real Fourier transform.

    Computes the one dimensional discrete Fourier transform of a
    Hermitian symmetric input signal.
    hfft('backward') is equivalent to irfft('forward').

    Notes
    -----
    .. The correct interpretation of the Hermitian input depends on the
       length of the original data, as given by `n`. This is because each
       input shape could correspond to either an odd or even length signal.
       By default, the signal is assumed to be even length and odd signals
       will not round-trip properly. So, it is recommended to always pass
       the signal length `n`.

    Parameters
    ----------
    input : tensor
        Input signal, containing only positive frequencies.
    n : int, optional
        Output signal length. If given, the input will either be
        zero-padded or trimmed to this length before computing the real
        IFFT. Defaults to even output: n=2*(input.size(dim) - 1).
    dim : int, default=-1
        Dimension along which to take the Fourier transform.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    norm : {'forward', 'backward', 'ortho'}, default='backward'
        forward : normalize by 1/n
        backward : no normalization (equivalent to irfft('forward'))
        ortho :  normalize by 1/sqrt(n) (making the FFT orthonormal)
    real : bool, default=False
        Only used if torch <= 1.5.
        If True, the input signal has no imaginary component and the
        dimension encoding the real and imaginary parts does not exist.

    Returns
    -------
    output : tensor
        Fourier transform of the input signal. Real tensor.

    """
    if _torch_has_fft_module:
        return fft_mod.hfft(input, n, dim, norm=norm)

    norm = ('forward' if norm == 'backward' else
            'backward' if norm == 'forward' else
            'ortho')
    return irfft(input, n, dim, norm=norm, real=real)


def ihfft(input, n=None, dim=-1, norm='backward'):
    """One dimensional real-to-complex inverse Fourier transform.

    The input must be a real-valued signal, interpreted in the Fourier
    domain. The IFFT of a real signal is Hermitian-symmetric,
    X[i] = conj(X[-i]). ihfft() represents this in the one-sided form
    where only the positive frequencies below the Nyquist frequency are
    included. To compute the full output, use ifft().

    ihfft('backward') is equivalent to rfft('forward').

    Parameters
    ----------
    input : tensor
        Input real signal.
    n : int, optional
        Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim : int, default=-1
        Dimension along which to take the Fourier transform.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    norm : {'forward', 'backward', 'ortho'}, default='backward'
        forward : normalize by 1/n
        backward : no normalization
        ortho :  normalize by 1/sqrt(n) (making the FFT orthonormal)

    Returns
    -------
    output : tensor
        Fourier transform of the input signal. Complex tensor.
        Only positive frequencies are returned.
        If torch <= 1.5,  the last dimension is of length 2 and
        contain the real and imaginary parts of the signal.

    """
    if _torch_has_fft_module:
        return fft_mod.ihfft(input, n, dim, norm=norm)

    norm = ('forward' if norm == 'backward' else
            'backward' if norm == 'forward' else
            'ortho')
    return rfft(input, n, dim, norm=norm)


def hpart(x, dim=-1, shift=False, real=None):
    """Extract only non-redundant information from Hermitian symmetric signal.

    Notes
    -----
    .. By default, the upper half of the signal is returned (i.e.,
       the first (zero) frequency is assumed to be in the first (zero) voxel).
    .. If shift is True, the lower half of the signal is returned  (i.e.,
       the first (zero) frequency is assumed to be in the central voxel).
       `hpart(x, shift=True)` is equivalent to `hpart(ifftshift(x))`.
    .. The Nyquist frequency (at n//2) is returned as well, although it is
       considered negative by convention. The output length is therefore
       n//2 + 1.

    Parameters
    ----------
    x : tensor
        Input signal.
    dim : int, default=-1
        Dimension to process.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    shift : bool, default=False
        If True, the (zero) frequency  is assumed to be in the center voxel.
        Else, it is assumed to be in the first (zero) voxel.
    real : bool, optional
        Only used if torch <= 1.5.
        If True, the input signal has no imaginary component and the
        dimension encoding the real and imaginary parts does not exist.

    Returns
    -------
    h : tensor
        If shift is False, this is a view in the input Tensor.
        Else, a copy is triggered.
    """
    x = torch.as_tensor(x)

    if not _torch_has_complex and not real and dim < 0:
        dim = dim - 1

    slicer = [slice(None)] * x.dim()
    if not shift:
        slicer[dim] = slice((x.shape[dim])//2+1)
        return x[tuple(slicer)]

    # the Nyquist frequency is in the first voxel, so we need to
    # combine two parts to create the half we want
    new_shape = list(x.shape)
    new_shape[dim] = x.shape[dim]//2 + 1
    new_x = x.new_empty(new_shape)

    # main part
    in_slicer = list(slicer)
    in_slicer[dim] = slice(x.shape[dim]//2, None)
    out_slicer = list(slicer)
    out_slicer[dim] = slice(-1)
    new_x[tuple(out_slicer)] = x[tuple(in_slicer)]

    # Nyquist frequency
    in_slicer = list(slicer)
    in_slicer[dim] = 0
    out_slicer = list(slicer)
    out_slicer[dim] = -1
    new_x[tuple(out_slicer)] = x[tuple(in_slicer)]

    return new_x


def ihpart(x, n=None, dim=-1, shift=False, real=None):
    """Reconstruct a full signal from a half signal by complex conjugacy.

    Notes
    -----
    .. By default, the input signal ends up in the upper half of the
       output signal
    .. If shift is True, he input signal ends up in the lower half of the
       output signal.
       `ihpart(x, shift=True)` is equivalent to `fftshift(ihpart(x))`.

    Parameters
    ----------
    x : tensor
        Input signal.
    n : [sequence of] int, default=2*(m - 1)
        The length of the output signal.
    dim : [sequence of] int, default=all
        Dimensions to process.
        If torch <= 1.5, the dimension encoding the real and imaginary
        parts are not taken into account in dimension indexing.
    shift : bool, default=False
        If True, the (zero) frequency  is assumed to be in the center voxel.
        Else, it is assumed to be in the first (zero) voxel.
    real : bool, default=False
        Only used if torch <= 1.5.
        If True, the input signal has no imaginary component and the
        dimension encoding the real and imaginary parts does not exist.

    Returns
    -------
    h : tensor
        View into the input tensor, with half the input shape along `dim`.

    """
    x = torch.as_tensor(x)

    y = None
    if not _torch_has_complex:
        if not real:
            x, y = x.unbind(-1)

    # allocate output
    n = n or 2*(x.shape[dim]-1)
    new_shape = list(x.shape)
    new_shape[dim] = n
    if y is not None:
        new_x0 = x.new_empty([*new_shape, 2])
        new_x = new_x0[..., 0]
        new_y = new_x0[..., 1]
    else:
        new_x = x.new_empty(new_shape)

    # positive side
    in_slicer = [slice(None)] * x.dim()
    out_slicer = list(in_slicer)
    if shift:
        in_slicer[dim] = slice(-1)
        out_slicer[dim] = slice(n//2, None)
    else:
        out_slicer[dim] = slice(x.shape[dim])
    new_x[out_slicer] = x[in_slicer]
    if y is not None:
        new_y[out_slicer] = y[in_slicer]

    # negative side
    x = torch.flip(x, [dim])
    if y is not None:
        y = torch.flip(y, [dim]).neg_()
    if _torch_has_complex and x.is_complex():
        x = x.conj_()
    in_slicer[dim] = slice((not shift and not n % 2), -1)
    if shift:
        out_slicer[dim] = slice(n//2)
    else:
        out_slicer[dim] = slice(x.shape[dim], None)
    print(out_slicer, in_slicer)
    new_x[out_slicer] = x[in_slicer]
    if y is not None:
        new_y[out_slicer] = y[in_slicer]

    if y is not None:
        return new_x0
    else:
        return new_x

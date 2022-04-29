"""Convolution layers."""

import math
import inspect
import random
from collections import OrderedDict
from typing import Sequence, Optional, Union, Callable, Type, TypeVar
import torch
import torch.nn as tnn
import torch.nn.functional as F
from nitorch.core.py import make_tuple
from nitorch.core import py, utils
from nitorch.spatial import BoundType, InterpolationType
from ..base import nitorchmodule, Module, Sequential, ModuleList
from ..activations import make_activation_from_name
from .norm import make_norm_from_name


ActivationLike = Union[str, Callable, Type]
NormalizationLike = Union[bool, str, Callable, Type]
_T = TypeVar('_T')
ScalarOrSequence = Union[_T, Sequence[_T]]
BoundLike = Union[str, BoundType]
InterpolationLike = Union[int, str, InterpolationType]


# padding options natively handled by pytorch
_native_padding_mode = ('zeros', 'reflect', 'replicate', 'circular')


_activation_doc = \
    """Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
            Alternatively, an activation name (from a set of common 
            functions) can be provided. See `nn.make_activation_from_name`
            for supported names and their default parameters."""

_norm_doc = \
    """Normalization layer.
            Can be one of {'batch', 'instance', 'layer', 'group'} or a
            `bool` or an `int` or a (non instantiated) class or a callable.
            
            If `False` or `None`, normalization is not used.
            
            If one of {'batch', 'instance', 'layer', 'group'} is used,
            it triggers the instantiation of the corresponding 
            normalization layer. `True` is equivalent to 'batch'.
            In the 'group' case, the number of groups in the convolution 
            is used. Otherwise, an integer can instead be provided, 
            which will be used as the number of groups.
            
            If a class (typically a Module) is provided, it is then 
            instantiated with parameters `(dim, nb_channels)`. Otherwise, 
            a callable (an already instantiated class or a more simple 
            function), can be provided. It is useful to accept both 
            these cases as they allow to either:
                * have normalization weights specific to this module
                * have normalization weights shared with other modules
                * have non-learnable normalization weights"""

_dropout_doc = \
    """If a `float` in (0, 1) is provided, it defines the dropout
            probability (1 sets all weights to zero, 0 does nothing).
            
            Alternatively, dropout can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function)."""


def _defer_property(prop: str, module: str,
                    setter: callable or bool = False,
                    getter: callable = None):
    """Return a 'property' objet that links to a submodule property

    prop (str) : property name
    module (str): module name
    setter (callable or bool, default=False) : define a setter
    getter (callable, optional) : function to apply to the returned value
    returns (property) : property object

    """
    getter = getter or (lambda x: x)
    if setter:
        if not callable(setter):
            setter = lambda x: x
        return property(lambda self: getter(getattr(getattr(self, module), prop)),
                        lambda self, val: setattr(getattr(self, module), prop, setter(val)))
    else:
        return property(lambda self: getter(getattr(getattr(self, module), prop)))


def _guess_output_shape(inshape, kernel_size, stride=1, dilation=1,
                        padding=0, output_padding=0, transposed=False):
    """Guess the output shape of a (pytorch) convolution"""
    inshape = make_tuple(inshape)
    dim = len(inshape)
    kernel_size = make_tuple(kernel_size, dim)
    stride = make_tuple(stride, dim)
    output_padding = make_tuple(output_padding, dim)
    dilation = make_tuple(dilation, dim)

    if padding == 'auto':
        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError('Padding "auto" only available with odd kernels')
        padding = [((k - 1) * d + 1) // 2 for k, d in zip(kernel_size, dilation)]
    padding = make_tuple(padding, dim)

    shape = []
    for L, S, Pi, D, K, Po in zip(inshape, stride, padding,
                                  dilation, kernel_size, output_padding):
        if transposed:
            shape += [(L - 1) * S - 2 * Pi + D * (K - 1) + Po + 1]
        else:
            shape += [math.floor((L + 2 * Pi - D * (K - 1) - 1) / S + 1)]
    return tuple(shape)


def _get_conv_class(dim, transposed=False):
    """Return the appropriate torch Conv class"""
    if transposed:
        if dim == 1:
            ConvKlass = nitorchmodule(tnn.ConvTranspose1d)
        elif dim == 2:
            ConvKlass = nitorchmodule(tnn.ConvTranspose2d)
        elif dim == 3:
            ConvKlass = nitorchmodule(tnn.ConvTranspose3d)
        else:
            raise NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
    else:
        if dim == 1:
            ConvKlass = nitorchmodule(tnn.Conv1d)
        elif dim == 2:
            ConvKlass = nitorchmodule(tnn.Conv2d)
        elif dim == 3:
            ConvKlass = nitorchmodule(tnn.Conv3d)
        else:
            raise NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
    return ConvKlass


def _get_dropout_class(dim):
    """Return the appropriate torch Dropout class"""
    if dim == 1:
        Klass = nitorchmodule(tnn.Dropout)
    elif dim == 2:
        Klass = nitorchmodule(tnn.Dropout2d)
    elif dim == 3:
        Klass = nitorchmodule(tnn.Dropout3d)
    else:
        raise NotImplementedError('ConvDropout is only implemented in 1, 2, or 3D.')
    return Klass


def _get_conv_fn(dim, transposed=False):
    """Return the appropriate torch conv function"""
    if transposed:
        if dim == 1:
            conv_fn = F.conv_transpose1d
        elif dim == 2:
            conv_fn = F.conv_transpose2d
        elif dim == 3:
            conv_fn = F.conv_transpose3d
        else:
            raise NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
    else:
        if dim == 1:
            conv_fn = F.conv1d
        elif dim == 2:
            conv_fn = F.conv2d
        elif dim == 3:
            conv_fn = F.conv3d
        else:
            raise NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
    return conv_fn


def _normalize_padding(padding):
    """Ensure that padding has format (left, right, top, bottom, ...)"""
    if all(isinstance(p, int) for p in padding):
        return padding
    else:
        npadding = []
        for p in padding:
            if isinstance(padding, int):
                npadding.append(p)
                npadding.append(p)
            else:
                npadding.extend(p)
        return npadding


class Conv(Module):
    """Simple convolution.

    This class is almost equivalent to torch's Conv/ConvTransposed,
    except that it has a single entry point for any number of spatial
    dimensions and optional transposed conv.

    We also deal with padding in a slightly different way in
    the **transposed** case:
      . `padding` is not applied to the input but used instead to crop
        the output so that -- all other arguments equal --
        ``conv(conv(x), transposed=True).shape == x.shape``
      . `output_padding` is not exposed, but an `output_shape` argument
        be provided instead at call time.
    """
    
    def __init__(self, 
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: ScalarOrSequence[int],
                 stride: ScalarOrSequence[int] = 1,
                 padding: Union[str, ScalarOrSequence[int]] = 0,
                 bound: BoundLike = 'zeros',
                 dilation: ScalarOrSequence[int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transposed: bool = False,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int
            Number of channels in the input image.
            
        out_channels : int
            Number of channels produced by the convolution.
            
        kernel_size : int or tuple[int]
            Size of the convolution kernel
            
        stride : int or sequence[int], default=1
            Stride of the convolution.
            
        padding : {'same', 'valid'} or int or sequence[int], optional
            Input padding (if not `transposed`) or output cropping
            (if `transposed`).
            Padding added to all three sides of the input.
            If ``'same'``, padding such that the output shape is the
            same as the input shape (up to strides) is used.
            ``'valid'`` is equivalent to ``0``.
            
        bound : bound_like, default='zeros'
            Padding mode. In transposed mode, only 'zeros' is handled.
            
        dilation : int or tuple[int], default=1
            Spacing between kernel elements.
            
        groups : int, default=1
            Number of groups in a grouped convolution.
            The number of input channels and of output channels 
            must be divisible by this number.
            
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            
        transposed : bool, default=False:
            Transposed convolution.

        """
        super().__init__()

        # --------------------------------------------------------------
        # Checks
        # --------------------------------------------------------------
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid', 'auto'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
        else:
            padding = py.ensure_list(padding, dim)

        # --------------------------------------------------------------
        # Store parameters
        # --------------------------------------------------------------
        self.transposed = transposed
        self.stride = py.ensure_list(stride, dim)
        self.bound = bound
        self.dilation = py.ensure_list(dilation, dim)
        self.groups = groups
        self.transposed = transposed
        self.padding = padding

        kernel_size = py.ensure_list(kernel_size, dim)

        # --------------------------------------------------------------
        # Allocate weights
        # --------------------------------------------------------------
        backend = dict(device=device, dtype=dtype)
        if transposed:
            self.weight = tnn.Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **backend))
        else:
            self.weight = tnn.Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **backend))
        if bias:
            self.bias = tnn.Parameter(torch.empty(out_channels, **backend))
        else:
            self.register_parameter('bias', None)

        # --------------------------------------------------------------
        # Initialize weights
        # --------------------------------------------------------------
        self.reset_parameters()

    @property
    def in_channels(self):
        return (self.weight.shape[0] if self.transposed else
                self.weight.shape[1] * self.groups)

    @property
    def out_channels(self):
        return (self.weight.shape[1] * self.groups if self.transposed else
                self.weight.shape[0])

    @property
    def dim(self):
        return self.weight.dim() - 2

    @property
    def kernel_size(self):
        return self.weight.shape[2:]

    def reset_parameters(self, method='kaiming', a=None, dist='uniform',
                         johnshift=False):
        """Initialize the values of the weights and bias
        
        Parameters
        ----------
        method : {'kaiming', 'xavier', None}, default='kaiming'
            Initialization method.
        a : float, default=sqrt(5)
            If method is 'kaiming', negative slope of the activation function 
            (0 for ReLU, some non-zero value for leaky ReLU)
            If method is 'xavier', gain.
            If method is None, FWHM of the distribution.
        dist : {'uniform', 'normal'}, default='uniform'
            Distribution to sample.
        johnshift : bool, default=True
            Each filter has one of its central input weights centered
            about one instead of zeros, making them close to an identity.
        """
        method = (method or '').lower()
        if method.startswith('k'):
            a = math.sqrt(5) if a is None else a
        else:
            a = 1. if a is None else a
            
        if method.startswith('k') and dist.startswith('u'):
            fn = tnn.init.kaiming_uniform_
        elif method.startswith('k') and dist.startswith('n'):
            fn = tnn.init.kaiming_normal_
        elif method.startswith('x') and dist.startswith('u'):
            fn = tnn.init.xavier_uniform_
        elif method.startswith('x') and dist.startswith('n'):
            fn = tnn.init.xavier_normal_
        elif not method and dist.startswith('u'):
            fn = lambda x, a: tnn.init.uniform_(x, a=-a/2, b=a/2)
        elif not method and dist.startswith('n'):
            fn = lambda x, a: tnn.init.normal_(x, std=a/2.355)
        else:
            raise ValueError(f'Unknown method {method} or dist {dist}.')

        fn(self.weight, a=a)

        if johnshift:
            # we make filters closer to an identity transform by
            # "opening" a path for each channel (i.e., initializing
            # a weight with a value close to one, instead of zero).
            # This trick was found by John Ashburner.
            with torch.no_grad():
                center = tuple(k//2 for k in self.weight.shape[2:])
                center = self.weight[(slice(None), slice(None), *center)]
                nfilters = min(center.shape[0], center.shape[1])
                cin = list(range(center.shape[0]))
                random.shuffle(cin)
                cin = cin[:nfilters]
                cout = list(range(center.shape[1]))
                random.shuffle(cout)
                cout = cout[:nfilters]
                center[cin, cout] += 1

        if self.bias is not None:
            fan_in, _ = tnn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            if dist.startswith('u'):
                tnn.init.uniform_(self.bias, -bound, bound)
            else:
                tnn.init.normal_(self.bias, std=2*bound/2.355)

    def forward(self, x, output_shape=None):
        """

        Parameters
        ----------
        x : (batch, in_channels, *spatial) tensor
        output_shape : sequence[int], optional
            This should only be used in `transposed` mode.

        Returns
        -------
        x : (batch, out_channels, *spatial_out) tensor

        """
        # --------------------------------------------------------------
        # Compute padding
        # . inpad is: padding on the input side if `transposed=False`
        #             cropping on the output side if `transposed=True`
        # . outpad is padding on the output side if `transposed=True`
        # --------------------------------------------------------------
        inpad, outpad = self._compute_padding(x.shape[2:], output_shape)

        # --------------------------------------------------------------
        # Perform our own padding
        #   . The underlying torch function only handle symmetric zero-padding.
        #     All other types of padding must be done by us.
        #   . Note that we never pad the input in the `transposed` case.
        #   . We know that we need to do the padding ourselves because
        #     in that case `_compute_padding` returns left and right
        #     padding values in alternate order.
        # --------------------------------------------------------------
        if len(inpad) == 2*(x.dim() - 2):
            x = utils.pad(x, inpad, mode=self.bound)
            inpad = 0

        # --------------------------------------------------------------
        # Call native convolution
        #   Output padding is done here (`transposed` only).
        #   The torch function only accepts output padding that serves
        #   to compensate for lines lost because of strides. It must
        #   therefore be less than the kernel size. Otherwise, an error
        #   is raised.
        # --------------------------------------------------------------
        kw = dict(output_padding=[max(0, p) for p in outpad]) if outpad else {}
        conv = _get_conv_fn(self.dim, self.transposed)
        x = conv(x, self.weight, self.bias, **kw, stride=self.stride,
                 padding=inpad, dilation=self.dilation, groups=self.groups)

        # --------------------------------------------------------------
        # Crop extra output
        #   This should only ever happens in 'same' mode, when the
        #   padding is asymmetric.
        # --------------------------------------------------------------
        if outpad and any(p < 0 for p in outpad):
            slicer = [slice(p) if p < 0 else slice(None) for p in outpad]
            x = x[(Ellipsis, *slicer)]

        return x

    def _compute_padding(self, input_shape, output_shape):
        # Note: the "padding" argument in pytorch's conv_transposed
        # function is equivalent to an "output cropping" argument: its
        # point is to "undo" what conv(padding) did in the forward pass.
        if not self.transposed:
            # -----------------------------------------------------------------
            # Compute padding on the input side when ``not transposed``.
            #   . 'same' ensures that the output shape (when stride == 1)
            #     is the same as the output shape. In the general case, it
            #     ensures that the output shape is input_shape//stride.
            #     If the amount of padding necessary has an odd shape,
            #     we pad the same amount on the left and right. If it
            #     has an even shape, we pad by 1 more voxel on the right than
            #     on the left.
            #   . By "kernel shape", we mean `(kernel_size - 1) * dilation + 1`.
            #   . 'auto' is an alias for 'same' that we keep for backward
            #     compatibility ('auto' was used in nitorch before pytorch
            #     supported 'same').
            #   . 'valid' is equivalent to `0`.
            #   . Note that we support more boundary conditions than pytorch.
            #     Although only zero-padding is efficient (other boundary
            #     conditions requires copying the input into a larger
            #     reallocated tensor.
            #   . If the padding is even, we are also forced to use an
            #     inefficient reallocation.
            # In the `transposed` case, we want to recover the original
            # input size in the output of the transposed convolution. This
            # "padding" will therefore effectively be used to *crop* the
            # output of the convolution.
            # -----------------------------------------------------------------
            if output_shape:
                raise ValueError('`output_shape` should only be used in '
                                 '`transposed` mode.`')
            if self.padding == 'valid':
                padding = [0] * self.dim
            elif self.padding in ('same', 'auto'):
                padding = []
                for L, K, D, S in zip(input_shape, self.kernel_size,
                                      self.dilation, self.stride):
                    K = (K - 1) * D + 1  # true kernel size
                    P = max(0, K - S - L % S)  # padding to get oL = floor(L/S)
                    P = P//2 if P % 2 == 0 else (P//2, P - P//2)
                    padding.append(P)
            else:
                padding = self.padding
            if (any(isinstance(x, (list, tuple)) for x in padding)
                    or self.bound not in ('zero', 'zeros')):
                # convert to (left, right, top, bottom, ...)
                padding = _normalize_padding(padding)
            return padding, None

        else:  # self.transposed
            # -----------------------------------------------------------------
            # Compute padding and cropping on the output side when ``transposed``.
            #   . We want ``conv_transposed(conv(x)).shape == x.shape``
            #   . When strides are used, multiple input shapes can yield the
            #     same output shape. The transposed component is therefore
            #     under-determined. To solve this, the `output_shape` argument
            #     can be used.
            #     When it is used, some padding can be performed on the
            #     output side.
            # -----------------------------------------------------------------
            if self.bound not in ('zero', 'zeros'):
                import warnings
                warnings.warn(RuntimeWarning("Bound not supported in mode "
                                             "transposed. Silently switching "
                                             "to 'zeros'"))
            if self.padding == 'valid':
                sumpadding = [0] * self.dim
            elif self.padding not in ('same', 'auto'):
                padding = self.padding
                sumpadding = [2*p for p in padding]
            elif not output_shape:
                assert self.padding in ('same', 'auto')
                # we don't know exactly how much padding as been done
                # on the input side. We have to make a guess
                sumpadding = []
                for K, S, D in zip(self.kernel_size, self.stride,
                                   self.dilation):
                    K = (K - 1) * D + 1
                    P = K - S
                    sumpadding.append(P)

            output_padding = [0] * self.dim
            if output_shape:
                if self.padding in ('same', 'auto'):
                    # we know the original input size so we can compute
                    # the exact padding that was applied
                    sumpadding = []
                    for L, K, S, D in zip(output_shape, self.kernel_size,
                                          self.stride, self.dilation):
                        K = (K - 1) * D + 1
                        P = (K - S - L % S)
                        sumpadding.append(P)
                else:
                    output_padding = []
                    for oL, iL, K, S, D, Pi in zip(output_shape, input_shape,
                                                   self.kernel_size, self.stride,
                                                   self.dilation, sumpadding):
                        K = (K - 1) * D + 1
                        Pe = oL - ((iL - 1) * S + K - Pi)
                        output_padding.append(Pe)

            padding = []
            for d, P in enumerate(sumpadding):
                padding.append(P//2)
                output_padding[d] -= P % 2

            return padding, output_padding

    def shape(self, x, output_shape=None):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : tuple or (batch, in_channel, *in_spatial) tensor
            Input tensor or its shape
        output_shape : sequence[int], optional

        Returns
        -------
        shape : tuple[int]
            Output shape

        """
        if torch.is_tensor(x):
            inshape = tuple(x.shape)
        else:
            inshape = x
        batch, _, *inshape = inshape
        dim = len(inshape)

        if output_shape:
            if not self.transposed:
                raise ValueError('`output_shape` should only be used in '
                                 '`transposed` mode.`')
            return (batch, self.out_channels, *output_shape)

        if self.padding in ('same', 'auto'):
            if self.transposed:
                output_shape = [sz*st for sz, st in zip(inshape, self.stride)]
            else:
                output_shape = [sz//st for sz, st in zip(inshape, self.stride)]
            return (batch, self.out_channels, *output_shape)
        elif self.padding == 'valid':
            sum_pad = [0] * dim
        else:
            sum_pad = [2*p for p in self.padding]

        # input padding
        if not self.transposed:
            inshape = [s + p for s, p in zip(inshape, sum_pad)]

        # compute shape in the absence of padding
        output_shape = _guess_output_shape(
            inshape, kernel_size=self.kernel_size,
            stride=self.stride, dilation=self.dilation,
            transposed=self.transposed)

        # output padding
        if self.transposed:
            output_shape = [s - p for s, p in zip(output_shape, sum_pad)]

        return (batch, self.out_channels, *output_shape)

    def extra_repr(self):
        s = [f'{self.in_channels}', f'{self.out_channels}']
        s += [f'kernel_size={list(self.kernel_size)}']
        if self.transposed:
            s += [f'transposed=True']
        if any(x > 1 for x in self.stride):
            s += [f'stride={list(self.stride)}']
        if any(x > 1 for x in self.dilation):
            s += [f'dilation={list(self.dilation)}']
        if self.groups > 1:
            s += [f'groups={self.groups}']
        s = ', '.join(s)
        return s


class Conv1d(Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class Conv2d(Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class Conv3d(Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class ConvTransposed1d(Conv):
    def __init__(self, *args, **kwargs):
        kwargs['transposed'] = True
        super().__init__(1, *args, **kwargs)


class ConvTransposed2d(Conv):
    def __init__(self, *args, **kwargs):
        kwargs['transposed'] = True
        super().__init__(2, *args, **kwargs)


class ConvTransposed3d(Conv):
    def __init__(self, *args, **kwargs):
        kwargs['transposed'] = True
        super().__init__(3, *args, **kwargs)


@nitorchmodule
class GroupedConv(ModuleList):
    """Simple imbalanced grouped convolution.
    
    Same as SimpleConv, but allows to have groups with non-equal number of channels.
    """

    # TODO:
    #    Maybe keep the `groups` option and simply let in_channels and 
    #    out_channels be lists *optionnally* if the user wants imbalanced
    #    groups
    
    def __init__(self,
                 dim: int,
                 in_channels: Sequence[int],
                 out_channels: Sequence[int],
                 *args,
                 groups: int = None,
                 **kwargs):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : sequence[int]
            Number of channels in the input image.
            
        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a scalar, must be divisible by the number of groups.
            
        kernel_size : int or sequence[int]
            Size of the convolution kernel.
            
        stride : int or sequence[int], default=1:
            Stride of the convolution.
            
        padding : int or sequence[int], default=0
            Input padding (if not `transposed`) or output cropping
            (if `transposed`).
            Padding added to all three sides of the input.
            If ``'same'``, padding such that the output shape is the
            same as the input shape (up to strides) is used.
            ``'valid'`` is equivalent to ``0``.
            
        bound : bound_like, default='zeros'
            Padding mode.
            
        dilation : int or sequence[int], default=1
            Spacing between kernel elements.
            
        groups : int, default=None
            Number of groups. Default is the maximum of the lengths 
            of ``in_channels`` and ``out_channels``.
            
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            
        transposed : bool, default=False:
            Transposed convolution.
        """
        in_channels = py.ensure_list(in_channels)
        out_channels = py.ensure_list(out_channels)
        nb_groups = groups or max(len(in_channels), len(out_channels))
        if len(in_channels) == 1:
            in_channels = [in_channels[0]//nb_groups] * nb_groups
        if len(out_channels) == 1:
            out_channels = [out_channels[0]//nb_groups] * nb_groups
        if len(in_channels) != nb_groups or len(out_channels) != nb_groups:
            raise ValueError(f'The number of input and output groups '
                             f'must be the same: {len(in_channels)} vs '
                             f'{len(out_channels)}')

        modules = [Conv(dim, i, o, *args, **kwargs)
                   for i, o in zip(in_channels, out_channels)]
        super().__init__(modules)
        self.in_channels = sum(in_channels)
        self.out_channels = sum(out_channels)

    def forward(self, input, output_shape=None):
        """Perform a grouped convolution with imbalanced channels.

        Parameters
        ----------
        input : (batch, sum(in_channels), *in_spatial) tensor
        output_shape : sequence[int], optional

        Returns
        -------
        output : (batch, sum(out_channels), *out_spatial) tensor

        """
        out_shape = (input.shape[0], self.out_channels, *input.shape[2:])
        output = input.new_empty(out_shape)
        input = input.split([c.in_channels for c in self], dim=1)
        out_channels = [c.out_channels for c in self]
        for d, (layer, inp) in enumerate(zip(self, input)):
            slicer = [slice(None)] * (self.dim + 2)
            slicer[1] = slice(sum(out_channels[:d]), sum(out_channels[:d+1]))
            output[slicer] = layer(inp, output_shape)
        return output

    def reset_parameters(self, *args, **kwargs):
        for layer in self:
            layer.reset_parameters(*args, **kwargs)
    
    @property
    def weight(self):
        return [layer.conv.weight for layer in self]

    @property
    def bias(self):
        return [layer.conv.bias for layer in self]

    @property
    def dim(self):
        for layer in self:
            return layer.dim
        
    @property
    def transposed(self):
        for layer in self:
            return layer.transposed
        
    @property
    def groups(self):
        return len(list(self.children()))

    @property
    def stride(self):
        for layer in self:
            return layer.stride

    @property
    def padding(self):
        for layer in self:
            return layer.padding

    @property
    def dilation(self):
        for layer in self:
            return layer.dilation

    @property
    def bound(self):
        for layer in self:
            return layer.bound

    def shape(self, x, output_shape=None):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : tuple or (batch, in_channel, *in_spatial) tensor
            Input tensor or its shape
        output_shape : sequence[int], optional

        Returns
        -------
        shape : tuple[int]
            Output shape

        """
        shape = x.shape
        for layer in self:
            shape = layer.shape(x, output_shape)
            break
        shape = list(shape)
        shape[1] = self.out_channels
        return tuple(shape)
    
    def __str__(self):
        in_channels = [l.in_channels for l in self]
        out_channels = [l.out_channels for l in self]
        in_channels = ', '.join(in_channels)
        out_channels = ', '.join(out_channels)
        s = [f'[{in_channels}]', f'[{out_channels}]']
        s += [f'kernel_size={self.kernel_size}']
        if self.transposed:
            s += [f'transposed=True']
        if any(x > 1 for x in self.stride):
            s += [f'stride={self.stride}']
        if any(x > 1 for x in self.dilation):
            s += [f'dilation={self.dilation}']
        if self.groups > 1:
            s += [f'groups={self.groups}']
        s = ', '.join(s)
        return f'GroupedConv({s})'
    
    __repr__ = __str__


class GroupedConv1d(GroupedConv):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class GroupedConv2d(GroupedConv):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class GroupedConv3d(GroupedConv):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class GroupedConvTransposed1d(GroupedConv):
    def __init__(self, *args, **kwargs):
        kwargs['transposed'] = True
        super().__init__(1, *args, **kwargs)


class GroupedConvTransposed2d(GroupedConv):
    def __init__(self, *args, **kwargs):
        kwargs['transposed'] = True
        super().__init__(2, *args, **kwargs)


class GroupedConvTransposed3d(GroupedConv):
    def __init__(self, *args, **kwargs):
        kwargs['transposed'] = True
        super().__init__(3, *args, **kwargs)


@nitorchmodule
class ConvBlock(Sequential):
    """Convolution layer (with batch norm and activation).

    Applies a convolution over an input signal.
    Optionally: apply an activation function to the output.

    """
    def __init__(self,
                 dim: int,
                 in_channels: ScalarOrSequence[int],
                 out_channels: ScalarOrSequence[int],
                 kernel_size: ScalarOrSequence[int],
                 stride: ScalarOrSequence[int] = 1,
                 padding: Union[str, ScalarOrSequence[int]] = 0,
                 bound: BoundLike = 'zeros',
                 dilation: ScalarOrSequence[int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 transposed: bool = False,
                 activation: Optional[ActivationLike] = None,
                 norm: Optional[NormalizationLike] = None,
                 dropout: float = 0,
                 order: str = 'ncda',
                 inplace: bool = False):
        """
        Parameters
        ----------
        dim : (1, 2, 3)
            Number of spatial dimensions.
            
        in_channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.
            
        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a sequence, grouped convolutions are used.
            
        kernel_size : int or sequence[int]
            Size of the convolution kernel.
            
        stride : int or sequence[int], default=1:
            Stride of the convolution.
            
        padding : int or sequence[int] or 'auto', default=0
            Zero-padding added to all three sides of the input.
            
        bound : ('zeros', 'reflect', 'replicate', 'circular'), default='zeros'
            Padding mode.
            
        dilation : int or sequence[int], default=1
            Spacing between kernel elements.
            
        groups : int, default=1
            Number of blocked connections from input channels to
            output channels. Using this parameter is an alternative to
            the use of 'sequence' input/output channels. In that case,
            the number of input and output channels in each group is
            found by dividing the ``input_channels`` and ``output_channels``
            with ``groups``.
            
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            
        transposed : bool, default=False:
            Transposed convolution.

        activation : str or type or callable, optional
            {activation}
                
        norm : bool or str or int or type or callable, optional
            {norm}

        dropout : float or type or callable, optional
            {dropout}

        order : permutation of 'ncda', default='ncda'
            Order in which to perform the normalization (n), convolution (c),
            dropout (d) and activation (a).

        inplace : bool, default=False
            Apply activation inplace if possible
            (i.e., not ``is_leaf and requires_grad``).

        """
        super().__init__()

        # store in-place
        self.inplace = inplace
        self.order = self._fix_order(order)

        # Build modules
        opt_conv = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bound=bound,
            dilation=dilation,
            transposed=transposed,
            bias=bias)
        conv = self._build_conv(dim, in_channels, out_channels, groups, **opt_conv)
        norm = self._build_norm(norm, conv, order)
        activation = self._build_activation(activation)
        dropout = self._build_dropout(dropout, dim)

        # Assign submodules in order so that they are nicely
        # ordered during pretty printing
        for o in order:
            if o == 'n':
                self.norm = norm
            elif o == 'c':
                self.conv = conv
            elif o == 'd':
                self.dropout = dropout
            elif o == 'a':
                self.activation = activation

        # Use appropriate weight initialization when possible
        self._init_weights(conv, activation)

    __init__.__doc__ = __init__.__doc__.format(
        activation=_activation_doc, norm=_norm_doc, dropout=_dropout_doc)

    @staticmethod
    def _fix_order(order):
        if 'n' not in order:
            order = order + 'n'
        if 'c' not in order:
            order = order + 'c'
        if 'd' not in order:
            order = order + 'd'
        if 'a' not in order:
            order = order + 'a'
        return order

    @staticmethod
    def _init_weights(conv, activation):
        if isinstance(activation, tnn.ReLU):
            conv.reset_parameters(a=0)
        elif isinstance(activation, tnn.LeakyReLU):
            conv.reset_parameters(a=activation.negative_slope)
        else:
            conv.reset_parameters()

    @staticmethod
    def _build_conv(dim, in_channels, out_channels, groups, **opt_conv):

        # Check if "manual" grouped conv are required
        in_channels = py.make_list(in_channels)
        out_channels = py.make_list(out_channels)
        if len(in_channels) > 1 and groups > 1:
            raise ValueError('Cannot use both `groups` and multiple '
                             'input channels, as both define grouped '
                             'convolutions.')
        if len(in_channels) == 1 and len(out_channels) == 1:
            ConvKlass = Conv
            in_channels = in_channels[0]
            out_channels = out_channels[0]
            opt_conv['groups'] = groups
        else:
            ConvKlass = GroupedConv
            opt_conv['groups'] = max(len(in_channels), len(out_channels))

        conv = ConvKlass(dim, in_channels, out_channels, **opt_conv)
        return conv

    @staticmethod
    def _build_activation(activation):
        #   an activation can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        #   it is useful to accept both these cases as they allow to either:
        #       * have a learnable activation specific to this module
        #       * have a learnable activation shared with other modules
        #       * have a non-learnable activation
        if not activation:
            return None
        if isinstance(activation, str):
            return make_activation_from_name(activation)
        activation = (activation() if inspect.isclass(activation)
                      else activation if callable(activation)
                      else None)
        return activation

    @staticmethod
    def _build_dropout(dropout, dim):
        dropout = (dropout() if inspect.isclass(dropout)
                   else dropout if callable(dropout)
                   else _get_dropout_class(dim)(p=float(dropout)) if dropout
                   else None)
        return dropout

    @staticmethod
    def _build_norm(norm, conv, order, groups=1):
        #   an normalization can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        if not norm:
            return None
        if isinstance(norm, bool) and norm:
            norm = 'batch'
        dim = conv.dim
        in_channels = (conv.in_channels if order.index('n') < order.index('c')
                       else conv.out_channels)
        if isinstance(norm, str):
            if norm.lower() == 'group':
                norm = groups
            return make_norm_from_name(norm, dim, in_channels)
        norm = (norm(dim, in_channels) if inspect.isclass(norm)
                else norm if callable(norm)
                else None)
        return norm

    weight = _defer_property('weight', 'conv')
    bias = _defer_property('bias', 'conv')
    dim = _defer_property('dim', 'conv')
    in_channels = _defer_property('in_channels', 'conv')
    out_channels = _defer_property('out_channels', 'conv')
    transposed = _defer_property('transposed', 'conv')
    stride = _defer_property('stride', 'conv', setter=True)
    padding = _defer_property('padding', 'conv', setter=True)
    dilation = _defer_property('dilation', 'conv', setter=True)
    bound = _defer_property('bound', 'conv', setter=True)
    groups = _defer_property('groups', 'conv', setter=True)
    p_dropout = _defer_property('p', 'dropout', setter=True)
            
    def forward(self, x, output_shape=None):
        """Forward pass.

        Parameters
        ----------
        x : (batch, in_channel, *in_spatial) tensor
            Input tensor
        output_shape : sequence[int], optional

        Returns
        -------
        x : (batch, out_channel, *out_spatial) tensor
            Convolved tensor

        Notes
        -----
        The output shape of an input tensor can be guessed using the
        method `shape`.

        """
        # Save inplace parameter
        activation_inplace, dropout_inplace = self._save_inplace()

        # Set inplace parameter
        if self.inplace and not (x.is_leaf and x.requires_grad):
            if self.activation and hasattr(self.activation, 'inplace'):
                self.activation.inplace = True
            if self.dropout:
                self.dropout.inplace = True
        else:
            # This function ensures that the input tensor is not
            # overwritten, but uses `inplace` operations for all operations
            # that are not applied directly to the input (that is, after
            # at least one new tensor has been allocated).
            self._set_inplace()

        # Apply layers (Norm + Convolution + Dropout + Activation)
        for layer in self:
            if isinstance(layer, (Conv, GroupedConv)):
                x = layer(x, output_shape)
            else:
                x = layer(x)

        # Reset inplace parameter
        self._reset_inplace(activation_inplace, dropout_inplace)

        return x

    def _save_inplace(self):
        activation_inplace = None
        if self.activation and hasattr(self.activation, 'inplace'):
            activation_inplace = self.activation.inplace
        dropout_inplace = None
        if self.dropout:
            dropout_inplace = self.dropout.inplace
        return activation_inplace, dropout_inplace

    def _reset_inplace(self, activation_inplace, dropout_inplace):
        if self.dropout:
            self.dropout.inplace = dropout_inplace
        if self.activation and hasattr(self.activation, 'inplace'):
            self.activation.inplace = activation_inplace

    def _set_inplace(self):
        if self.dropout:
            self.dropout.inplace = False
        if self.activation and hasattr(self.activation, 'inplace'):
            self.activation.inplace = False

        true_order = []
        for o in self.order:
            if o == 'b' and self.norm:
                true_order.append(o)
            elif o == 'a' and self.activation:
                true_order.append(o)
            elif o == 'd' and self.dropout:
                true_order.append(o)
            elif o == 'c':
                true_order.append(o)

        if true_order[0] != 'd' and self.dropout:
            self.dropout.inplace = True
        if true_order[0] != 'a' and self.activation \
                and hasattr(self.activation, 'inplace'):
            self.activation.inplace = True

    def shape(self, x, output_shape=None):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : tuple or (batch, in_channel, *in_spatial) tensor
            Input tensor or its shape
        output_shape : sequence[int], optional
            Instead of using 'output_padding', a target output shape
            can be provided (when using transposed convolutions).

        Returns
        -------
        shape : tuple[int]
            Output shape

        """
        return self.conv.shape(x, output_shape)


@nitorchmodule
class BottleneckConv(Sequential):
    """
    Squeeze and unsqueeze the number of channels around a (spatial)
    convolution using channel convolutions.
    """

    def __init__(self,
                 dim: int,
                 in_channels: ScalarOrSequence[int],
                 out_channels: ScalarOrSequence[int],
                 bottleneck: ScalarOrSequence[int],
                 kernel_size: ScalarOrSequence[int] = 3,
                 stride: ScalarOrSequence[int] = 1,
                 **kwargs):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.

        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a sequence, grouped convolutions are used.

        bottleneck : int or sequence[int]
            Number of channels in the bottleneck.
            If a sequence, grouped convolutions are used.

        kernel_size : int or sequence[int], default=3
            Size of the convolution kernel.

        stride : int or sequence[int], default=1
            Only used in the main convolution (not in the channel-wise ones).

        Other Parameters
        ----------------
        All parameters from `ConvBlock` are parameters of `BottleneckConv`.

        """
        super().__init__(OrderedDict(
            squeeze=ConvBlock(dim, in_channels, bottleneck, 1, **kwargs),
            conv=ConvBlock(dim, bottleneck, bottleneck, kernel_size,
                           stride=stride, **kwargs),
            unsqueeze=ConvBlock(dim, bottleneck, out_channels, 1, **kwargs),
        ))

    def forward(self, x, output_shape=None, **kwargs):
        x = self.squeeze(x, **kwargs)
        x = self.conv(x, output_shape=output_shape, **kwargs)
        x = self.unsqueeze(x, **kwargs)
        return x

    def shape(self, x, output_shape=None, **kwargs):
        x = self.squeeze.shape(x, **kwargs)
        x = self.conv.shape(x, output_shape=output_shape, **kwargs)
        x = self.unsqueeze.shape(x, **kwargs)
        return x


@nitorchmodule
class ConvGroup(Sequential):
    """A group of multiple convolutions with the same number of input
    and output channels. Usually used inside a ResBlock."""

    def __init__(self,
                 dim: int,
                 channels: ScalarOrSequence[int],
                 bottleneck: Optional[int] = None,
                 nb_conv: int = 2,
                 recurrent: bool = False,
                 kernel_size: ScalarOrSequence[int] = 3,
                 bound: BoundLike = 'zeros',
                 groups: int = 1,
                 bias : bool = True,
                 dilation: ScalarOrSequence[int] = 1,
                 activation: Optional[ActivationLike] = tnn.ReLU,
                 norm: Optional[NormalizationLike] = True,
                 dropout: float = 0,
                 order: str = 'nadc',
                 inplace: bool = False):
        """
        Parameters
        ----------
        dim : (1, 2, 3)
            Number of spatial dimensions.

        channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.

        bottleneck : int, optional
            If provided, a bottleneck architecture is used, where
            1d conv are used to project the input channels onto a
            lower-dimensional space, where the spatial convolutions
            are performed, before being mapped back to the original
            number of dimensions.

        nb_conv : int, default=2
            Number of convolutions

        recurrent : bool, default=False
            Use recurrent convolutions (weights are shared between blocks)

        kernel_size : int or sequence[int], default=3
            Size of the convolution kernel.

        bound : bound_like, default='zeros'
            Padding mode.

        groups : int, default=1
            Number of blocked connections from input channels to
            output channels. Using this parameter is an alternative to
            the use of 'sequence' input/output channels. In that case,
            the number of input and output channels in each group is
            found by dividing the ``input_channels`` and ``output_channels``
            with ``groups``.

        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.

        dilation : int, default=1

        activation : str or type or callable, default=ReLU
            {activation}

        norm : bool or str or int or type or callable, default=True
            {norm}

        dropout : float or type or callable, default=0
            {dropout}

        order : permutation of 'nadc', default='nadc'
            Order in which to perform the normalization (n), convolution (c),
            dropout (d) and activation (a).

        inplace : bool, default=False
            Apply activation inplace if possible
            (i.e., not ``is_leaf and requires_grad``).

        """
        super().__init__()

        conv_opt = dict(
            dim=dim,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding='same',
            bound=bound,
            groups=groups,
            bias=bias,
            dilation=dilation,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
            inplace=inplace,
        )
        if bottleneck:
            conv_opt['bottleneck'] = bottleneck
            Klass = BottleneckConv
        else:
            Klass = ConvBlock
        nb_conv_inde = 1 if recurrent else nb_conv
        convs = [Klass(**conv_opt) for _ in range(nb_conv_inde)]
        self.convs = super().__init__(*convs)
        self.recurrent = recurrent
        self.nb_conv = nb_conv

    __init__.__doc__ = __init__.__doc__.format(
        activation=_activation_doc, norm=_norm_doc, dropout=_dropout_doc)

    def shape(self, x, *args, **kwargs):
        if self.recurrent:
            for _ in range(self.nb_conv):
                for conv in self:
                    x = conv.shape(x, *args, **kwargs)
        else:
            for conv in self:
                x = conv.shape(x, *args, **kwargs)
        return x

    def forward(self, x, *args, **kwargs):
        """

        Parameters
        ----------
        x : (batch, channels, *spatial) tensor
            Input tensor

        Returns
        -------
        x : (batch, channels, *spatial) tensor
            Output tensor, with the same shape as the input tensor

        """

        if self.recurrent:
            for _ in range(self.nb_conv):
                for conv in self:
                    x = conv(x, *args, **kwargs)
        else:
            for conv in self:
                x = conv(x, *args, **kwargs)
        return x


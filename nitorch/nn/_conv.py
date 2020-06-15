# -*- coding: utf-8 -*-
"""Convolution layers."""

import torch
from torch import nn as tnn
from nitorch.utils import padlist, replist, getargs
from copy import copy
import math

# NOTE:
# My version of Conv allows parameters to be overridden at eval time.
# This probably clashes with these parameters being declared as __constants__
# in torch.nn._ConvND. I think this __constants__ mechanics is only used
# In TorchScript, so it's fine for now.


class Conv(tnn.Module):
    """Convolution layer (with activation).

    Applies a convolution over an input signal.
    Optionally: apply an activation function to the output.

    """
    def __init__(self, dim, *args, **kwargs):
        """
        Args:
            dim (int): Dimension of the convolving kernel (1|2|3)
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution.
                Default: 1
            padding (int or tuple, optional): Zero-padding added to all
                three sides of the input. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to
                the output. Default: ``True``
            transposed (bool, optional): Transposed convolution.
            `Default: ``True``
            activation (type or function, optional): Constructor of an
                activation function. Default: ``None``
            try_inplace (bool, optional): Apply activation inplace if
                possible (i.e., not (is_leaf and requires_grad).
                Default: True.

        """
        super().__init__()

        # Get additional arguments that are not present in torch's conv
        transposed, activation, try_inplace = getargs(
            [('transposed', 10, False),
             ('activation', 11, None),
             ('try_inplace', 12, True)],
            args, kwargs, consume=True)

        # Store dimension
        self.dim = dim
        self.try_inplace = try_inplace

        # Select Conv
        if transposed:
            if dim == 1:
                self.conv = tnn.ConvTranspose1d(*args, **kwargs)
            elif dim == 2:
                self.conv = tnn.ConvTranspose2d(*args, **kwargs)
            elif dim == 3:
                self.conv = tnn.ConvTranspose3d(*args, **kwargs)
            else:
                NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
        else:
            if dim == 1:
                self.conv = tnn.Conv1d(*args, **kwargs)
            elif dim == 2:
                self.conv = tnn.Conv2d(*args, **kwargs)
            elif dim == 3:
                self.conv = tnn.Conv3d(*args, **kwargs)
            else:
                NotImplementedError('Conv is only implemented in 1, 2, or 3D.')

        # Add activation
        self.activation = activation() if activation else None

    def forward(self, x, stride=None, padding=None, output_padding=None,
                dilation=None, padding_mode=None, activation=None):
        """Forward pass. Possibility to override constructed parameters.

        Args:
            x (torch.tensor): Input tensor
            stride (optional): Default: as constructed
            padding (optional): Default: as constructed
            output_padding (optional): Default: as constructed
            dilation (optional): Default: as constructed
            padding_mode (optional): Default: as constructed
            activation (optional): Default: as constructed

        Returns:
            x (torch.tensor): Convolved tensor

        """

        # Override constructed parameters
        conv = copy(self.conv)
        if stride is not None:
            conv.stride = padlist(stride, self.dim)
        if padding is not None:
            conv.padding = padlist(padding, self.dim)
            conv._padding_repeated_twice = replist(conv.padding, 2,
                                                   interleaved=True)
        if output_padding is not None:
            conv.output_padding = padlist(output_padding, self.dim)
        if dilation is not None:
            conv.dilation = padlist(dilation, self.dim)
        if padding_mode is not None:
            conv.padding_mode = padding_mode

        # Activation
        if activation is None:
            activation = copy(self.activation)
        else:
            activation = activation()
        if self.try_inplace \
                and hasattr(activation, 'inplace') \
                and not (x.is_leaf and x.requires_grad):
            activation.inplace = True

        # Convolution + Activation
        x = conv(x)
        if activation is not None:
            x = activation(x)
        return x


    def shape(self, x, stride=None, padding=None, output_padding=None,
              dilation=None, padding_mode=None, activation=None):
        """Compute output shape of the equivalent ``forward`` call.

        Args:
            x (torch.tensor): Input tensor
            stride (optional): Default: as constructed
            padding (optional): Default: as constructed
            output_padding (optional): Default: as constructed
            dilation (optional): Default: as constructed
            padding_mode (optional): Default: as constructed
            activation (optional): Default: as constructed

        Returns:
            x (tuple): Outptu shape

        """

        # Override constructed parameters
        conv = copy(self.conv)
        if stride is not None:
            conv.stride = padlist(stride, self.dim)
        if padding is not None:
            conv.padding = padlist(padding, self.dim)
            conv._padding_repeated_twice = replist(conv.padding, 2,
                                                   interleaved=True)
        if output_padding is not None:
            conv.output_padding = padlist(output_padding, self.dim)
        if dilation is not None:
            conv.dilation = padlist(dilation, self.dim)
        if padding_mode is not None:
            conv.padding_mode = padding_mode

        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation
        kernel_size = conv.kernel_size
        output_padding = conv.output_padding
        transposed = conv.transposed

        N = x.shape[0]
        C = self.conv.out_channels
        shape = [N, C]
        for i, inp in enumerate(x.shape[2:]):
            if transposed:
                shape.append(
                    (inp-1)*stride[i]-2*padding[i]
                    +dilation[i]*(kernel_size[i]-1)
                     +output_padding[i]+1
                )
            else:
                shape.append(math.floor(
                    (inp+2*padding[i]-dilation[i]*(kernel_size[i]-1)-1)
                    /stride[i] + 1
                ))
        return tuple(shape)


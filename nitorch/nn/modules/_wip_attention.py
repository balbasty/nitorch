from collections import OrderedDict
from nitorch.nn.activations.base import make_activation_from_name
import torch
from nitorch.nn.base import Module
from nitorch.core import py, utils, math, linalg
from nitorch import spatial
from nitorch.nn.modules.conv import ConvBlock


class ConvAttentionLayer(Module):
    """Convolutional dot product attention

    This module does not have learnable weights: it takes
    already split variables. For the module with learnable encoding
    weights, see `ConvAttention`.
    """

    def __init__(self,
                 kernel_size=3,
                 stride=1,
                 padding='auto',
                 padding_mode='zeros'):
        """

        Parameters
        ----------
        kernel_size : int, default=3
            Size of the neighbourhood in which the attention dot product
            is computed.
        stride : int, default=1
            Stride between output elements
        padding : sequence[int] or 'auto', default='auto'
            Amount of padding. 'auto' preserves the input dimensions.
        padding_mode : {'zeros', 'dft', 'dct1', 'dct2'}, default='zeros'
            Method used to invent padded values.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, q, k, v, **overload):
        """

        Parameters
        ----------
        q : (b, c, *spatial)
            Queries
        k : (b, c, *spatial)
            Keys
        v : (b, c, *spatial)
            Values

        Returns
        -------
        x : (b, c, *spatial)

        """
        kernel_size = overload.pop('kernel_size', self.kernel_size)
        stride = overload.pop('stride', self.kernel_size)
        padding = overload.pop('padding', self.padding)
        padding_mode = overload.pop('padding_mode', self.padding_mode)

        dim = q.dim() - 2
        if padding == 'auto':
            k = spatial.pad_same(dim, k, kernel_size, bound=padding_mode)
            v = spatial.pad_same(dim, v, kernel_size, bound=padding_mode)
        elif padding:
            padding = [0] * 2 + py.make_list(padding, dim)
            k = utils.pad(k, padding, side='both', mode=padding_mode)
            v = utils.pad(v, padding, side='both', mode=padding_mode)

        # compute weights by query/key dot product
        kernel_size = py.make_list(kernel_size, dim)
        k = utils.unfold(k, kernel_size, stride)
        k = k.reshape([*k.shape[:dim+2], -1])
        k = utils.movedim(k, 1, -1)
        q = utils.movedim(q[..., None], 1, -1)
        k = math.softmax(linalg.dot(k, q), dim=-1)
        k = k[:, None]  # add back channel dimension

        # compute new values by weight/value dot product
        v = utils.unfold(v, kernel_size, stride)
        v = v.reshape([*v.shape[:dim+2], -1])
        v = linalg.dot(k, v)

        return v


class ConvAttention(Module):

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding='auto',
                 padding_mode='zeros'):
        """

        Parameters
        ----------
        in_channels : int
        out_channels : int, default=in_channels
        kernel_size : int, default=3
            Size of the neighbourhood in which the attention dot product
            is computed.
        stride : int, default=1
            Stride between output elements
        padding : sequence[int] or 'auto', default='auto'
            Amount of padding. 'auto' preserves the input dimensions.
        padding_mode : {'zeros', 'dft', 'dct1', 'dct2'}, default='zeros'
            Method used to invent padded values.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dot = ConvAttentionLayer(kernel_size, stride, padding, padding_mode)
        self.linear = torch.nn.Linear(in_channels, 3*out_channels, bias=False)

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, in_channels, *spatial_in)

        Returns
        -------
        y : (batch, out_channels, *spatial_out)

        """

        x = utils.movedim(self.linear(utils.movedim(x, 1, -1)), -1, 1)
        q, k, v = torch.chunk(x, 3, dim=1)
        x = self.dot(q, k, v, **overload)
        return x


class MultiConvAttention(Module):

    def __init__(self,
                 nb_heads,
                 in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding='auto',
                 padding_mode='zeros'):
        """

        Parameters
        ----------
        nb_heads : int
        in_channels : int
        out_channels : int, default=in_channels
            Must be a multiple of `nb_heads`
        kernel_size : int, default=3
            Size of the neighbourhood in which the attention dot product
            is computed.
        stride : int, default=1
            Stride between output elements
        padding : sequence[int] or 'auto', default='auto'
            Amount of padding. 'auto' preserves the input dimensions.
        padding_mode : {'zeros', 'dft', 'dct1', 'dct2'}, default='zeros'
            Method used to invent padded values.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        if out_channels % nb_heads:
            raise ValueError('Output channels must be a multiple of the '
                             'number of heads')
        opt = dict(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   padding_mode=padding_mode)
        self.heads = torch.nn.ModuleList([ConvAttention(**opt)])
        self.linear = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, in_channels, *spatial_in)

        Returns
        -------
        y : (batch, out_channels, *spatial_out)

        """

        out = None
        for i, head in enumerate(self.heads):
            y = head(x, **overload)
            if out is None:
                out_shape = list(y.shape)
                out_shape[1] *= len(self.heads)
                out = y.new_empty(out_shape)
            out[:, i*y.shape[1]:(i+1)*y.shape[1]] = y
            del y
        out = utils.movedim(self.linear(utils.movedim(out, 1, -1)), -1, 1)
        return out


class AttentionGate(Module):
    """
    Attention gating module, as found in Attention U-Net model.

    References
    ----------
    ..[1] "Attention U-Net: Learning Where to Look for the Pancreas"
          Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, 
          Mattias Heinrich, Kazunari Misawa, Kensaku Mori, 
          Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, 
          Ben Glocker, Daniel Rueckert
          https://arxiv.org/abs/1804.03999
    """
    def __init__(self,
                 dim,
                 encoder_channels,
                 decoder_channels,
                 intermediate_channels=None,
                 norm=True,
                 dropout=None,
                 activation='ReLU'):
        """

        Parameters
        ----------

        dim : {1, 2, 3}
            Number of spatial dimensions.

        encoder_channels : int
            Number of input channels from encoder

        decoder_channels : int
            Number of input channels from decoder (typically will
            equal 2 * encoder_channels)

        intermediate_channels : int, default=None
            Number of channels to use in intermediate
            activations of attention gate. Typically use same
            value as encoder_channels.
            If None, will set intermediate_channels=decoder_channels

        """
        if not intermediate_channels:
            intermediate_channels = decoder_channels
        super().__init__(OrderedDict(gate=ConvBlock(dim, decoder_channels, intermediate_channels, kernel_size=1, norm=norm, activation=None, dropout=dropout),
                                     act=ConvBlock(dim, encoder_channels, intermediate_channels, kernel_size=1, norm=norm, activation=None, dropout=dropout),
                                     relu=make_activation_from_name(activation),
                                     psi=ConvBlock(dim, intermediate_channels, 1, kernel_size=1, norm=norm, activation='sigmoid', dropout=dropout)))

    def forward(self, x, x_cat=None):
        if not x_cat:
            x, x_cat = x
        g = self.gate(x)
        x = self.act(x_cat)
        x = self.relu(x + g)
        x = self.psi(x)
        return x * x_cat

import inspect
import torch.nn as tnn
from nitorch.core import py, utils
from ..base import Module, Sequential, ModuleList
from ..activations import make_activation_from_name
from .norm import make_norm_from_name
from .conv import _get_dropout_class
from .pool import Pool


def _defer_property(prop, module, setter=False):
    """Return a 'property' objet that links to a submodule property

    prop (str) : property name
    module (str): module name
    setter (bool, default=False) : define a setter
    returns (property) : property object

    """
    if setter:
        return property(lambda self: getattr(getattr(self, module), prop),
                        lambda self, val: setattr(getattr(self, module), prop, val))
    else:
        return property(lambda self: getattr(getattr(self, module), prop))


class Linear(Module):
    """Add options to torch.nn.Linear

    The fully connected layer is applied to the channels dimension
    by default (instead of between the last spatial dimension).
    """

    def __init__(self, in_channels, out_channels, bias=True, dim=1):
        """

        Parameters
        ----------
        in_channels : int
        out_channels : int
        bias : bool, default=True
        dim : int, default=1
        """
        super().__init__()
        self.linear = tnn.Linear(in_channels, out_channels, bias)
        self.dim = dim

    def forward(self, x):
        x = utils.fast_movedim(x, self.dim, -1)
        x = self.linear(x)
        x = utils.fast_movedim(x, -1, self.dim)
        return x

    in_channels = _defer_property('in_features', 'linear')
    out_channels = _defer_property('out_features', 'linear')
    bias = _defer_property('bias', 'linear')


class LinearBlock(Sequential):
    """
    Block of Linear-Norm-Activation.
    TODO: Add functionality to use custom order like in Conv
    """
    def __init__(self, in_channels, out_channels, norm=None, activation=None,
                 dim=1, linear_dim=1, bias=True, dropout=None):
        super().__init__()
        self.linear = Linear(in_channels, out_channels, bias, linear_dim)
        
        if isinstance(activation, str):
            activation = make_activation_from_name(activation)
        self.activation = activation
        
        if isinstance(norm, bool) and norm:
            norm = 'batch'
        if isinstance(norm, str):
            norm = make_norm_from_name(norm, dim, in_channels)
        self.norm = norm
        
        dropout = (dropout() if inspect.isclass(dropout)
                   else dropout if callable(dropout)
                   else _get_dropout_class(dim)(p=float(dropout)) if dropout
                   else None)
        self.dropout = dropout

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class MLP(Module):
    """
    Simple Multi-Layer Perceptron. Useful for projection head in contrastive learning.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=[2048], 
                 dim=3, linear_dim=1, bias=True, norm=None, prepool=False,
                 activation=None, dropout=None, final_activation=None, final_norm=None):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        self.prepool = prepool
        if prepool:
            dim=1
        self.first_layer = LinearBlock(in_channels, hidden_channels[0], norm=norm,
                                        activation=activation, dim=dim, linear_dim=linear_dim,
                                        bias=bias, dropout=dropout)
        if len(hidden_channels) > 1:
            self.hidden_layers = ModuleList([
                LinearBlock(hidden_channels[i], hidden_channels[i+1], norm=norm,
                            activation=activation, dim=dim, linear_dim=linear_dim,
                            bias=bias, dropout=dropout) for i in range(len(hidden_channels)-1)
            ])
        else:
            self.hidden_layers = None
        self.final_layer = LinearBlock(hidden_channels[-1], out_channels, norm=final_norm,
                                        activation=final_activation, dim=dim, linear_dim=linear_dim,
                                        bias=bias, dropout=dropout)

    def forward(self, x):
        if self.prepool:
            x = x.reshape(x.shape[0], x.shape[1], -1).mean(-1, keepdim=True)
        x = self.first_layer(x)
        if self.hidden_layers:
            for layer in self.hidden_layers:
                x = layer(x)
        x = self.final_layer(x)
        return x


def add_projection_head(model, model_out_channels=None,
                        proj_hidden_channels=2048, proj_channels=128, **kwargs):
    # if not model_out_channels:
    # TODO: figure out how to determine final channels in model
    #     model_out_channels = model.modules[-1]
    model.project = MLP(model_out_channels,
                        proj_channels,
                        proj_hidden_channels,
                        **kwargs)
    return model

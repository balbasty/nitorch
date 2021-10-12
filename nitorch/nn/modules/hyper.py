"""Utilities to build hyper-networks"""
from typing import Sequence
import inspect
import copy
import torch
import torch.nn as tnn
from nitorch.core import py
from ..base import Module
from .conv import ActivationLike
from ..activations import make_activation_from_name


class HyperNet(Module):
    """
    Generic hypernetwork.

    An hyper-network is a network whose weights are generated dynamically
    by a meta-network from a set of input features.

    Its forward pass is: HyperNet(x, feat) = SubNet(MetaNet(feat))(x)
    """

    # TODO: we maybe want to make it easier for people to build
    #   specializations of this class where not all weights are
    #   instantiated by the hyper network, but are trainable instead.
    #   We could define a filter function that selects which submodules
    #   of the main network have hyper-weights. Or have a `parameters`
    #   argument (like in optimizers) that let the user specify
    #   which parameters are dynamic.

    def __init__(self,
                 in_features: int,
                 network,
                 layers: Sequence[int] = (64, 64, 64, 64),
                 activation: ActivationLike = 'relu',
                 final_activation: ActivationLike = 'tanh',
                 ):
        """

        Parameters
        ----------
        in_features : int
            Number of input meta-features.
        network : Module
            Instantiated sub network, whose weights are dynamically generated.
            This network *will* be modified in place.
        layers : sequence[int], default=(64, 64, 64, 64)
            Number of output channels after each layer of the hyper-network.
        activation : activation_like, default='relu'
            Activation after each layer of the hyper-network.
        final_activation : activation_like, default='tanh'
            Final activation before the generated network weights.

        """
        super().__init__()

        # make hypernetwork
        nb_weights = sum(w.numel() for w in network.parameters())
        layers = [in_features, *layers]
        hyper = []
        for i in range(len(layers)-1):
            hyper.append(tnn.Linear(layers[i], layers[i+1]))
            a = self._make_activation(activation)
            if a:
                hyper.append(a)
        hyper.append(tnn.Linear(layers[-1], nb_weights))
        a = self._make_activation(final_activation)
        if a:
            hyper.append(a)
        self.hyper = tnn.Sequential(*hyper)

        # save main network
        self.network = network.requires_grad_(False)

    @property
    def in_features(self):
        return self.hyper[0].in_features

    @classmethod
    def detach_(cls, network):
        """Detach all weights of a generated network.

        Parameters
        ----------
        network : Module

        Returns
        -------
        network : Module

        """
        for param in network.parameters():
            param.detach_()
        return network

    @classmethod
    def _make_activation(cls, activation):
        if not activation:
            return None
        if isinstance(activation, str):
            return make_activation_from_name(activation)
        return (activation() if inspect.isclass(activation)
                else activation if callable(activation)
                else None)

    def _make_chunks(self, x):
        """Cut output of hypernetwork into weights with correct shape"""
        offset = 0
        all_shapes = [p.shape for p in self.network.parameters()]
        for shape in all_shapes:
            numel = py.prod(shape)
            w = x[offset:offset+numel].reshape(shape)
            offset += numel
            yield w

    def _set_weights(self, x, w, memo=None):
        """Sets the weights of the main module

        Parameters
        ----------
        x : Module
        w : tensor
        memo : set[Module]

        """
        # It's a bit tricky to mutate the network weights without breaking the
        # computational graph. The current implementation is probably less
        # efficient than something where we define lots of Meta Modules,
        # but I think that this one is easier to play with (we can just pass
        # it any "classic" network)
        param_names = [p[0] for p in x.named_parameters(recurse=False)]
        for name in param_names:
            w1 = next(w)
            old = getattr(x, name)
            delattr(x, name)
            new = tnn.Parameter(old.detach().clone(), requires_grad=False)
            setattr(x, name, new)
            getattr(x, name).copy_(w1)

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
        for _, module in x.named_modules(memo):
            self._set_weights(module, w, memo)

    def make_networks(self, feat, detach=False):
        """Generate networks from features

        Parameters
        ----------
        feat : (batch, in_features) tensor
            Input features
        detach : bool, default=False
            Detach all weights in the returned gradients.

        Returns
        -------
        networks : list[Module]

        """
        weights = self.hyper(feat)
        networks = []
        for batch_weights in weights:
            network = copy.deepcopy(self.network)
            self._set_weights(network, self._make_chunks(batch_weights))
            if detach:
                network = self.detach_(network)
            networks.append(network)
        return networks

    def forward(self, x, feat):
        """

        Parameters
        ----------
        x : (batch, ...)
            Input of the main network
        feat : (batch, in_features) tensor
            Input to the hyper-network

        Returns
        -------
        y : (batch, ...) tensor
            Output of the main network

        """
        # generate hyper weights
        weights = self.hyper(feat)

        if len(weights) == 1:
            self._set_weights(self.network, self._make_chunks(weights[0]))
            return self.network(x)

        # we have to loop over batches because network weights cannot have
        # a batch dimension
        output = []
        for batch_weights, batch_input in zip(weights, x):
            self._set_weights(self.network, self._make_chunks(batch_weights))
            output.append(self.network(batch_input[None]))
        output = torch.cat(output)
        return output

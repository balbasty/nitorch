"""Utilities to build hyper-networks"""
from typing import Sequence, Optional
import copy
import torch
import torch.nn as tnn
from nitorch.core import py
from ..base import Module
from .conv import ActivationLike, NormalizationLike
from .linear import LinearBlock
from nitorch.core.optionals import try_import
functorch = try_import('functorch')


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
                 network: Module,
                 nodes: Optional[Sequence[str or Module]] = None,
                 layers: Sequence[int] = (128,)*6,
                 activation: ActivationLike = 'relu',
                 final_activation: ActivationLike = None,
                 dropout: float = 0.,
                 norm: NormalizationLike = None
                 ):
        """
        Parameters
        ----------
        in_features : int
            Number of input meta-features.
        network : Module
            Instantiated sub network, whose weights are dynamically generated.
            This network *will* be modified in place.
        nodes : [sequence of] str or Module, optional
            Names or references of sub-modules, whose weights are
            hyper-generated. Names can include global patterns such as
            '*' or '**'.
            By default, all parameters of the subnetwork are hyper-generated.
        layers : sequence[int], default=(128,)*6
            Number of output channels after each layer of the hyper-network.
        activation : activation_like, default='relu'
            Activation after each layer of the hyper-network.
        final_activation : activation_like, default=None
            Final activation before the generated network weights.
        dropout : float or sequence[float], default=None
            dropout probability. if sequence, must match length of layers
        norm: bool or string, default=None
            Normalisation to use in Linear blocks
        """
        super().__init__()
        if nodes is not None and isinstance(nodes, (str, tnn.Module)):
            nodes = [nodes]
        self.nodes = set(nodes) if nodes is not None else None

        # convert parameters to buffers
        self.network = network
        self.preprocess_network_(self.network)

        # make hypernetwork
        nb_weights = sum(w.numel() for w in self._get_weights(network))
        layers = [in_features, *layers]
        hyper = []
        for i in range(len(layers)-1):
            hyper.append(LinearBlock(layers[i], layers[i+1],
                         activation=activation, norm=norm, dropout=dropout))
        hyper.append(LinearBlock(layers[-1], nb_weights,
                     activation=final_activation, norm=norm, dropout=dropout))
        self.hyper = tnn.Sequential(*hyper)

    @property
    def in_features(self):
        return self.hyper[0].in_features

    def _make_chunks(self, x):
        """Cut output of hypernetwork into weights with correct shape"""
        offset = 0
        all_shapes = [p.shape for p in self._get_weights(self.network)]
        for shape in all_shapes:
            numel = py.prod(shape)
            w = x[offset:offset+numel].reshape(shape)
            offset += numel
            yield w

    def _prefix_in_nodes(self, prefix, nodes):
        """Check if a module full-name is in the list of mutable nodes"""
        def _isequal(x, y):
            if x == y:
                return True
            elif y == '*':
                return True
            return False

        for node in nodes:
            if not isinstance(node, str):
                continue
            prefix = prefix.split('.')
            node = node.split('.')
            if '**' in node:
                node = (node[:node.index('**')]
                        + ['*'] * max(0, len(prefix)-len(nodes))
                        + node[node.index('**')+1:])
            if '**' in node:
                raise ValueError('There can be only one ** ellipsis in pattern')
            if len(node) != len(prefix):
                continue
            if all(_isequal(x, y) for x, y in zip(prefix, node)):
                return True
        return False

    def preprocess_network_(self, x, memo=None, nodes=None, prefix=''):
        """Convert all "generated" parameters into buffers
        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        nodes : set[str or Module]
            Set of nodes, whose parameters can be mutated.
        prefix : str
            Full name of the current module.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return
        if nodes is None:
            if self.nodes is None:
                nodes = None
            else:
                nodes = set(self.nodes)

        if nodes is None or x in nodes or self._prefix_in_nodes(prefix, nodes):
            if nodes is not None:
                nodes = nodes.union(set(x.children()))
            param_names = [p[0] for p in x.named_parameters(recurse=False)]
            for name in param_names:
                old = getattr(x, name)
                delattr(x, name)
                x.register_buffer(name, torch.Tensor(old.detach()))
                if not hasattr(x, 'generated_parameters'):
                    setattr(x, 'generated_parameters', [])
                x.generated_parameters.append(name)

        memo.add(x)
        for name, module in x.named_children():
            subprefix = prefix + ('.' if prefix else '') + name
            self.preprocess_network_(module, memo, nodes, subprefix)

    def postprocess_network_(self, x, memo=None):
        """Convert all "generated" buffers into parameters
        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, 'generated_parameters', []):
            old = getattr(x, name)
            delattr(x, name)
            setattr(x, name, torch.nn.Parameter(old.detach()))
        if hasattr(x, 'generated_parameters'):
            delattr(x, 'generated_parameters')

        memo.add(x)
        for name, module in x.named_children():
            self.postprocess_network_(module, memo)

    def _get_weights(self, x, memo=None):
        """Get all hyper-generated weights of the main module
        This method assumes that `preprocess_network_` has been called before.
        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        Yields
        ------
        Tensor
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, 'generated_parameters', []):
            yield getattr(x, name)

        memo.add(x)
        for name, module in x.named_children():
            for param in self._get_weights(module, memo):
                yield param

    def _set_weights(self, x, w, memo=None):
        """Sets the hyper-generated weights of the main module
        Parameters
        ----------
        x : Module
            Module to explore.
        w : iterator[tensor]
            Iterator over new weights.
        memo : set[Module]
            Sub-modules that have already been visited.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, 'generated_parameters', []):
            setattr(x, name, next(w))

        memo.add(x)
        for name, module in x.named_children():
            self._set_weights(module, w, memo)

    def detach_buffers_(self, x, memo=None):
        """Detach all buffers in a network"""
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name, buffer in x.named_buffers(recurse=False):
            setattr(x, name, getattr(x, name).detach())

        memo.add(x)
        for name, module in x.named_children():
            self.detach_buffers_(module, memo)


    def make_networks(self, feat):
        """Generate networks from features
        Parameters
        ----------
        feat : (batch, in_features) tensor
            Input features
        Returns
        -------
        networks : list[Module]
        """
        self.detach_buffers_(self.network)
        weights = self.hyper(feat)
        networks = []
        for batch_weights in weights:
            network = copy.deepcopy(self.network)
            self._set_weights(network, self._make_chunks(batch_weights))
            self.postprocess_network_(network)
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


class HyperNetV2(Module):
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
                 network: Module,
                 nodes: Optional[Sequence[str or Module]] = None,
                 layers: Sequence[int] = (128,)*6,
                 activation: ActivationLike = 'relu',
                 final_activation: ActivationLike = None,
                 dropout: float = 0.,
                 norm: NormalizationLike = None,
                 vmap: bool = False
                 ):
        """
        Parameters
        ----------
        in_features : int
            Number of input meta-features.
        network : Module
            Instantiated sub network, whose weights are dynamically generated.
            This network *will* be modified in place.
        nodes : [sequence of] str or Module, optional
            Names or references of sub-modules, whose weights are
            hyper-generated. Names can include global patterns such as
            '*' or '**'.
            By default, all parameters of the subnetwork are hyper-generated.
        layers : sequence[int], default=(128,)*6
            Number of output channels after each layer of the hyper-network.
        activation : activation_like, default='relu'
            Activation after each layer of the hyper-network.
        final_activation : activation_like, default=None
            Final activation before the generated network weights.
        dropout : float or sequence[float], default=None
            dropout probability. if sequence, must match length of layers
        norm: bool or string, default=None
            Normalisation to use in Linear blocks
        vmap: bool, default=False
            Use of torch.vmap to vectorize the multi-weight forward-pass.
            If True: forward pass is wrapped in torch.vmap() [NOTE: REQUIRES TORCH>=1.10]
            If False: Use standard for-loop method in forward-pass.
        """
        super().__init__()
        if nodes is not None and isinstance(nodes, (str, tnn.Module)):
            nodes = [nodes]
        self.nodes = set(nodes) if nodes is not None else None

        # convert parameters to buffers
        self.network = network
        self.preprocess_network_(self.network)

        # make hypernetwork
        nb_weights = sum(w.numel() for w in self._get_weights(network))
        layers = [in_features, *layers]
        hyper = []
        for i in range(len(layers)-1):
            hyper.append(LinearBlock(layers[i], layers[i+1],
                         activation=activation, norm=norm, dropout=dropout))
        hyper.append(LinearBlock(layers[-1], nb_weights,
                     activation=final_activation, norm=norm, dropout=dropout))
        self.hyper = tnn.Sequential(*hyper)

        if vmap and not functorch:
            self.vmap = False
            print("Selected 'vmap=True' flag but functorch not installed. To use vmap,\
                please follow instructions at https://github.com/pytorch/functorch#installing-functorch-preview-with-pytorch-110")
        else:
            self.vmap = vmap

    @property
    def in_features(self):
        return self.hyper[0].in_features

    def _make_chunks(self, x):
        """Cut output of hypernetwork into weights with correct shape"""
        offset = 0
        all_shapes = [p.shape for p in self._get_weights(self.network)]
        for shape in all_shapes:
            numel = py.prod(shape)
            w = x[offset:offset+numel].reshape(shape)
            offset += numel
            yield w

    def _prefix_in_nodes(self, prefix, nodes):
        """Check if a module full-name is in the list of mutable nodes"""
        def _isequal(x, y):
            if x == y:
                return True
            elif y == '*':
                return True
            return False

        for node in nodes:
            if not isinstance(node, str):
                continue
            prefix = prefix.split('.')
            node = node.split('.')
            if '**' in node:
                node = (node[:node.index('**')]
                        + ['*'] * max(0, len(prefix)-len(nodes))
                        + node[node.index('**')+1:])
            if '**' in node:
                raise ValueError('There can be only one ** ellipsis in pattern')
            if len(node) != len(prefix):
                continue
            if all(_isequal(x, y) for x, y in zip(prefix, node)):
                return True
        return False

    def preprocess_network_(self, x, memo=None, nodes=None, prefix=''):
        """Convert all "generated" parameters into buffers
        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        nodes : set[str or Module]
            Set of nodes, whose parameters can be mutated.
        prefix : str
            Full name of the current module.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return
        if nodes is None:
            if self.nodes is None:
                nodes = None
            else:
                nodes = set(self.nodes)

        if nodes is None or x in nodes or self._prefix_in_nodes(prefix, nodes):
            if nodes is not None:
                nodes = nodes.union(set(x.children()))
            param_names = [p[0] for p in x.named_parameters(recurse=False)]
            for name in param_names:
                old = getattr(x, name)
                delattr(x, name)
                x.register_buffer(name, torch.Tensor(old.detach()))
                if not hasattr(x, 'generated_parameters'):
                    setattr(x, 'generated_parameters', [])
                x.generated_parameters.append(name)

        memo.add(x)
        for name, module in x.named_children():
            subprefix = prefix + ('.' if prefix else '') + name
            self.preprocess_network_(module, memo, nodes, subprefix)

    def postprocess_network_(self, x, memo=None):
        """Convert all "generated" buffers into parameters
        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, 'generated_parameters', []):
            old = getattr(x, name)
            delattr(x, name)
            setattr(x, name, torch.nn.Parameter(old.detach()))
        if hasattr(x, 'generated_parameters'):
            delattr(x, 'generated_parameters')

        memo.add(x)
        for name, module in x.named_children():
            self.postprocess_network_(module, memo)

    def _get_weights(self, x, memo=None):
        """Get all hyper-generated weights of the main module
        This method assumes that `preprocess_network_` has been called before.
        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        Yields
        ------
        Tensor
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, 'generated_parameters', []):
            yield getattr(x, name)

        memo.add(x)
        for name, module in x.named_children():
            for param in self._get_weights(module, memo):
                yield param

    def _set_weights(self, x, w, memo=None):
        """Sets the hyper-generated weights of the main module
        Parameters
        ----------
        x : Module
            Module to explore.
        w : iterator[tensor]
            Iterator over new weights.
        memo : set[Module]
            Sub-modules that have already been visited.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, 'generated_parameters', []):
            setattr(x, name, next(w))

        memo.add(x)
        for name, module in x.named_children():
            self._set_weights(module, w, memo)

    def detach_buffers_(self, x, memo=None):
        """Detach all buffers in a network"""
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name, buffer in x.named_buffers(recurse=False):
            setattr(x, name, getattr(x, name).detach())

        memo.add(x)
        for name, module in x.named_children():
            self.detach_buffers_(module, memo)


    def make_networks(self, feat):
        """Generate networks from features
        Parameters
        ----------
        feat : (batch, in_features) tensor
            Input features
        Returns
        -------
        networks : list[Module]
        """
        self.detach_buffers_(self.network)
        weights = self.hyper(feat)
        networks = []
        for batch_weights in weights:
            network = copy.deepcopy(self.network)
            self._set_weights(network, self._make_chunks(batch_weights))
            self.postprocess_network_(network)
            networks.append(network)
        return networks

    
    def load_and_forward(self, weights, x):
        """
        Take as input weights and tensor for single-batch item.
        """
        self._set_weights(self.network, self._make_chunks(weights))
        return self.network(x)


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
            output = self.load_and_forward(weights[0], x)

        elif self.vmap:
            # input - feat [B, Weights]; x [B, C, Spatial]
            # in vmap want x -> [B, 1, C, Spatial]
            output = functorch.vmap(self.load_and_forward)(weights, x[:,None])[:,0]

        else:
            # we have to loop over batches because network weights cannot have
            # a batch dimension
            output = torch.cat([
                self.load_and_forward(wi, xi[None]) for (wi,xi) in zip(weights,x)
            ])

        return output

from .hyper import HyperNet
from .cnn import NeuriteUNet, NeuriteEncoder
from .reduction import reductions
from .segmentation import DiceHead
from .linear import LinearBlock
from ..activations import make_activation_from_name
from ..base import Module
from .. import check
from .seg_utils import board2
import torch
import inspect
tnn = torch.nn


class HyperU(Module):
    """Self-adaptive UNet"""

    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 embedding=None,
                 unet=None,
                 encoder=None,
                 hyper=None):
        """

        Parameters
        ----------
        dim : int
            Number of spatial dimensions
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        embedding : int
            Number of embedding/meta channels
        
        unet : dict
            nb_levels : int, default=5
            kernel_size : int, default=3
            nb_feat : int or sequence[int], default=16
            feat_mult : int, default=1
            pool_size : int, default=2
            padding' : int or 'same', default='same'
            dilation_rate_mult : int, default=1
            activation : str, default='elu'
            residual: bool, default=False
            final_activation : str, default='softmax'
            nb_conv_per_level : int, default=1
            dropout : float, default=0
            batch_norm : {'batch', 'instance', 'layer'} or None, default=None
        encoder : dict
            nb_levels : int, default=5
            kernel_size : int, default=3
            nb_feat : int or sequence[int], default=16
            feat_mult : int, default=1
            pool_size : int, default=2
            padding' : int or 'same', default='same'
            dilation_rate_mult : int, default=1
            activation : str, default='elu'
            residual: bool, default=False
            nb_conv_per_level : int, default=1
            dropout : float, default=0
            batch_norm : {'batch', 'instance', 'layer'} or None, default=None
            skip : bool, default=True
            reduction : str, default='mean'
            fc_layers : sequence[int], default=(64, 64, 64, 64)
            final_activation : str, default=None
        hyper : dict
            layers : sequence[int], default=(64, 64, 64, 64)
            activation : str, default='relu'
            final_activation : str, default='tanh'
        """
        super().__init__()

        unet = unet or dict()
        unet.setdefault('nb_levels', 5)
        unet.setdefault('kernel_size', 3)
        unet.setdefault('nb_feat', 16)
        unet.setdefault('feat_mult', 1)
        unet.setdefault('pool_size', 2)
        unet.setdefault('padding', 'same')
        unet.setdefault('dilation_rate_mult', 1)
        unet.setdefault('activation', 'elu')
        unet.setdefault('residual', False)
        unet.setdefault('final_activation', 'softmax')
        unet.setdefault('nb_conv_per_level', 1)
        unet.setdefault('dropout', 0)
        unet.setdefault('batch_norm', None)

        encoder = encoder or dict()
        encoder.setdefault('nb_levels', 5)
        encoder.setdefault('kernel_size', 3)
        encoder.setdefault('nb_feat', 16)
        encoder.setdefault('feat_mult', 1)
        encoder.setdefault('pool_size', 2)
        encoder.setdefault('padding', 'same')
        encoder.setdefault('dilation_rate_mult', 1)
        encoder.setdefault('activation', 'elu')
        encoder.setdefault('residual', False)
        encoder.setdefault('nb_conv_per_level', 1)
        encoder.setdefault('dropout', 0)
        encoder.setdefault('batch_norm', None)
        reduction = encoder.pop('reduction', 'mean')
        fc_layers = encoder.pop('fc_layers', (64, 64, 64, 64))
        final_activation = encoder.pop('final_activation', None)
        skip = encoder.pop('skip', True)

        # compute output number of channels
        nb_feat = encoder.get('nb_feat')
        nb_conv_per_level = encoder.get('nb_conv_per_level')
        if skip:
            if isinstance(nb_feat, (list, tuple)):
                nb_features = sum(nb_feat[-1::-nb_conv_per_level])
            else:
                nb_features = sum(nb_feat * 2 ** k 
                                  for k in range(encoder.get('nb_levels')))
        else:
            if skip:
                nb_features = nb_feat[-1]
            else:
                nb_features = nb_feat * 2 ** (encoder.get('nb_levels') - 1)

        hyper = hyper or dict()
        hyper.setdefault('layers', (64, 64, 64, 64))
        hyper.setdefault('activation', 'relu')
        hyper.setdefault('final_activation', 'relu')

        self.skip = skip
        self.encoder = NeuriteEncoder(dim, in_channels, **encoder)
        self.reduction = reductions[reduction]()
        fc_layers = [nb_features, *fc_layers]
        embedding = embedding or fc_layers[-1]
        fc_layers += [embedding]
        self.fc = self._make_fc(fc_layers, encoder['activation'], final_activation)
        unet = NeuriteUNet(dim, in_channels, out_channels, **unet)
        self.hyper = HyperNet(embedding, unet, **hyper)

    @classmethod
    def _make_activation(cls, activation):
        if not activation:
            return None
        if isinstance(activation, str):
            return make_activation_from_name(activation)
        return (activation() if inspect.isclass(activation)
                else activation if callable(activation)
                else None)
        
    @classmethod
    def _make_fc(cls, channels, activation, final_activation):
        layers = []
        for i in range(len(channels)-1):
            layers.append(LinearBlock(channels[i], channels[i+1],
                          activation=activation))
        layers.append(LinearBlock(channels[-2], channels[-1],
                     activation=final_activation))
        return tnn.Sequential(*layers)
            
    def forward(self, x, return_embedding=False):
        """

        Parameters
        ----------
        x : (B, in_channels, *spatial) tensor

        Returns
        -------
        y : (B, out_channels, *spatial) tensor

        """
        embedding = self.encoder(x, return_skip=self.skip)
        if self.skip:
            embedding = torch.cat([self.reduction(x).reshape([len(x), -1])
                                  for x in embedding], dim=-1)
        else:
            embedding = self.reduction(embedding).reshape([len(embedding), -1])
        embedding = self.fc(embedding)
        out = self.hyper(x, embedding)
        return (out, embedding) if return_embedding else out


class HyperSynthSeg(HyperU):

    def __init__(self,
                 dim=3,
                 in_channels=1,
                 out_channels=32,
                 embedding=64,
                 unet=None,
                 encoder=None,
                 hyper=None,
                 head=DiceHead):

        unet = unet or dict()
        unet.setdefault('nb_levels', 5)
        unet.setdefault('nb_feat', 8)
        unet.setdefault('feat_mult', 2)
        unet.setdefault('nb_conv_per_level', 2)
        unet.setdefault('batch_norm', 'batch')
        unet.setdefault('final_activation', None)

        encoder = encoder or dict()
        encoder.setdefault('nb_levels', 5)
        encoder.setdefault('nb_feat', 8)
        encoder.setdefault('feat_mult', 2)
        encoder.setdefault('nb_conv_per_level', 2)

        super().__init__(dim, in_channels, out_channels, embedding, unet, encoder, hyper)
        self.head = head if callable(head) else head()
        
        self.tags = ['score', 'posterior', 'embedding']

    def board(self, tb, **k):
        return board2(self, tb, **k, implicit=True)
    
    def forward(self, x, ref=None, meta=None, return_embedding=False,
                _loss=None, _metric=None):

        score, embedding = super().forward(x, return_embedding=True)
        prob = self.head.posterior(score)
        score = self.head.score(score)

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            self.compute(_loss, _metric,
                         score=[score, ref],
                         posterior=[prob, ref])
        if meta is not None:
            self.compute(_loss, _metric,
                         embedding=[embedding, meta])

        return (prob, embedding) if return_embedding else prob
    
    
class MetaSynthSeg(Module):

    def __init__(self,
                 dim=3,
                 in_channels=1,
                 out_channels=32,
                 in_features=3,
                 unet=None,
                 hyper=None,
                 head=DiceHead):
        super().__init__()

        unet = unet or dict()
        unet.setdefault('nb_levels', 5)
        unet.setdefault('nb_feat', 24)
        unet.setdefault('feat_mult', 2)
        unet.setdefault('nb_conv_per_level', 2)
        unet.setdefault('batch_norm', 'batch')
        unet.setdefault('final_activation', None)
        unet.setdefault('kernel_size', 3)
        unet.setdefault('pool_size', 2)
        unet.setdefault('padding', 'same')
        unet.setdefault('dilation_rate_mult', 1)
        unet.setdefault('activation', 'elu')
        unet.setdefault('residual', False)
        unet.setdefault('dropout', 0)

        hyper = hyper or dict()
        hyper.setdefault('layers', (64, 64, 64, 64))
        hyper.setdefault('activation', 'relu')
        hyper.setdefault('final_activation', None)

        unet = NeuriteUNet(dim, in_channels, out_channels, **unet)
        self.hyper = HyperNet(in_features, unet, **hyper)
        
        self.head = head if callable(head) else head()
        
        self.tags = ['score', 'posterior']

    def board(self, tb, **k):
        if 'inputs' in k:
            x, meta, *ref = k['inputs']
            k['inputs'] = (x, *ref)
        return board2(self, tb, **k, implicit=True)
    
    def forward(self, x, meta, ref=None, _loss=None, _metric=None):
        score = self.hyper(x, meta)
        prob = self.head.posterior(score)
        score = self.head.score(score)

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            self.compute(_loss, _metric,
                         score=[score, ref],
                         posterior=[prob, ref])

        return prob
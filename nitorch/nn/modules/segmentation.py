"""Dense (semantic) segmentation

ClassificationHead : base class for heads adapted to different losses
    LogisticHead
    FocalHead
    HingeHead
    DiceHead
ReweightingLayer : layer that reweights posterior probabilities
SegNet : a generic segmentation network (maps images to posterior probabilities)
"""

import torch.nn as tnn
import torch
from .. import check
from ..base import Module
from ..activations import SoftMax
from ..losses import cat as catloss
from .cnn import UNet2
from .conv import ConvBlock, Conv
from .seg_utils import board2


class ClassificationHead(Module):
    """Base class for classifiers.

    Classification head implement common subroutines for:
        - Computing implicit scores
        - Computing the maximum score labels
        - Computing (when possible) the true posterior
          (The softmax of the score is returned when the true posterior
           cannot be recovered. This is the case for e.g. the hinge loss)
    """

    def __init__(self, implicit='zero'):
        super().__init__()
        self.implicit = implicit

    def score(self, score, **overload):
        """Returns the score with all classes explicit

        Parameters:
        -----------
        score : (batch, classes[+1], *spatial) tensor
        implicit : {'sum', 'zero'} or False, optional

        Returns:
        --------
        score : (batch,, classes+1, *spatial) tensor

        """
        implicit = overload.get('implicit', self.implicit)
        if implicit == 'zero':
            new_shape = list(score.shape)
            new_shape[1] += 1
            new_score = score.new_zeros(new_shape)
            new_score[:, 1:] = score
            score, new_score = new_score, None
        elif implicit == 'sum':
            new_shape = list(score.shape)
            new_shape[1] += 1
            new_score = score.new_empty(new_shape)
            new_score[:, 1:] = score
            new_score[:, 0] = score.sum(1).neg_()
            score, new_score = new_score, None
        return score

    def classify(self, score, **overload):
        """Returns the index of the class with maximum score

        Parameters:
        -----------
        score : (batch, classes[+1], *spatial) tensor
        implicit : {'sum', 'zero'} or False, optional

        Returns:
        --------
        label : (batch, 1, *spatial) tensor[long]

        """
        score = self.score(score, **overload)
        return score.argmax(dim=1)


class LogisticHead(ClassificationHead):
    """Head for a logistic regression.

    The logistic regression optimizes the categorical cross-entropy
    and uses the softmax as a link function.

    The posterior is the softmaxed score.
    """

    def __init__(self, implicit='zero', weighted=False):
        super().__init__(implicit)
        self.loss = catloss.CategoricalLoss(weighted, logit=True, implicit=False)

    def posterior(self, score, **overload):
        """Returns posterior distribution (without background class)"""
        score = self.score(score, **overload)
        return SoftMax(implicit=(False, True))(score)


class FocalHead(ClassificationHead):
    """Head for a focal loss

    While the focal loss [1] is applied to the softmaxed score, additional
    manipulations must be performed to obtain the posterior [2].

    References
    ----------
    .. [1] "Focal Loss for Dense Object Detection"
           Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar
           ICCV (2017)
           https://arxiv.org/abs/1708.02002
    .. [2] "On Focal Loss for Class-Posterior Probability Estimation:
            A Theoretical Perspective"
           N Charoenphakdee, J Vongkulbhisal, N Chairatanakul, M Sugiyama
           CVPR (2021)
           https://arxiv.org/abs/2011.09172
    """

    def __init__(self, gamma=2, implicit='zero', weighted=False):
        """

        Parameters
        ----------
        gamma : float, default=2
            Focal parameter (0 == logistic regression)
        implicit : {'sum', 'zero'} or False, optional
            Whether an implicit class exist and how to compute it.
        weighted : bool or list[float], default=False
            Weight for each class. If True, use the inverse frequency.
        """
        super().__init__(implicit)
        self.loss = catloss.FocalLoss(gamma, weighted=weighted, logit=True, implicit=False)

    def posterior(self, score, **overload):
        """Returns posterior distribution (without background class)"""
        gamma = overload.get('gamma', self.loss.gamma)
        score = self.score(score, **overload)
        score = SoftMax(implicit=(False, False))(score)
        phi = (1 - score).pow(gamma) - gamma * (1 - score).pow(gamma - 1) * score * score.logit()
        score = score / phi
        score = score / score.sum(1, keepdim=True)
        return score[:, :-1]


class HingeHead(ClassificationHead):
    """Head for a hinge margin loss.

    """

    def __init__(self, implicit='sum', weighted=False):
        super().__init__(implicit)
        self.loss = catloss.HingeLoss(weighted, logit=True, implicit=False)

    def posterior(self, score, **overload):
        """Returns posterior distribution (without background class)"""
        score = self.score(score, **overload)
        return SoftMax(implicit=(False, True))(score)


class DiceHead(ClassificationHead):
    """Head for a Dice loss"""

    def __init__(self, exclude_background=False, implicit='zero', weighted=False):
        super().__init__(implicit)
        self.loss = catloss.DiceLoss(weighted=weighted,
                                     exclude_background=exclude_background,
                                     logit=True, implicit=False)

    def posterior(self, score, **overload):
        """Returns posterior distribution (without background class)"""
        score = self.score(score, **overload)
        return SoftMax(implicit=(False, True))(score)


class ReweightingLayer(Module):
    """Layer to reweight scores (1d conv k -> k)

    References
    ----------
    .. [1] "Posterior Re-calibration for Imbalanced Datasets"
           J Tian, YC Liu, N Glaser, YC Hsu
           NeurIPS (2020)
    """

    def __init__(self, nb_classes, implicit=False):
        """

        Parameters
        ----------
        nb_classes : int
            Number of input and output classes
        implicit : bool, default=False
            Whether there is an implicit class.
            If True, a bias term is added to the convolution.
        """
        super().__init__()
        self.conv = ConvBlock(1, nb_classes, nb_classes, bias=implicit)

    def forward(self, x):
        b, c, *shape = x.shape
        x = x.reshape([b, c, -1])
        x = self.conv(x)
        x = x.reshape([b, c, *shape])
        return x


class SegNet(Module):
    """Segmentation network.

    This is simply a UNet that ends with a softmax activation.

    Batch normalization is used by default.

    """
    def __init__(
            self,
            dim,
            output_classes=1,
            input_channels=1,
            head=LogisticHead,
            unet=None,
            *,
            encoder=None,
            decoder=None,
            conv_per_layer=1,
            kernel_size=3,
            pool=None,
            unpool=None,
            activation=tnn.LeakyReLU(0.2),
            norm='batch',
            to_feat=None,
            from_feat=None):
        """

        Parameters
        ----------
        dim : int
            Space dimension
        output_classes : int, default=1
            Number of classes, excluding background
        input_channels : int, default=1
            Number of input channels
        head : ClassificationHead, default=LogisticHead
            Either a type (which is then instantiated) or an instance.
        unet : Module, optional
            An instantiated network that maps from input feature to
            output feature space. Additional linear layers will be
            created to map from input channels to input features
            and from output features to output classes.
            If not provided, a default UNet is used.

        Other Parameters (only if `unet is None`)
        ----------------
        encoder : sequence[int], optional
            Number of features per encoding layer
        decoder : sequence[int], optional
            Number of features per decoding layer
        conv_per_layer : int, default=1
            Number of convolution per scale
        kernel_size : int or sequence[int], default=3
            Kernel size
        pool : str, default=None
        unpool : str, default=None
        activation : str or callable, default=LeakyReLU(0.2)
            Activation function in the UNet.
        norm : {'batch', 'instance', 'layer'} or int, default='batch'
            Normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).

        """
        super().__init__()

        self.dim = dim
        head = head if callable(head) else head()
        if unet is None:
            self.unet = UNet2(
                dim,
                in_channels=input_channels,
                out_channels=output_classes + (not head.implicit),
                encoder=encoder,
                decoder=decoder,
                conv_per_layer=conv_per_layer,
                kernel_size=kernel_size,
                pool=pool,
                unpool=unpool,
                activation=activation,
                norm=norm)
            self.to_feat = lambda x: x
            self.from_feat = lambda x: x
        else:
            if to_feat is True:
                self.to_feat = Conv(
                    dim,
                    in_channels=input_channels,
                    out_channels=unet.in_channels,
                    kernel_size=1)
            else:
                self.to_feat = lambda x: x
            self.unet = unet
            if from_feat is True:
                self.from_feat = Conv(
                    dim,
                    in_channels=unet.out_channels,
                    out_channels=output_classes + (not head.implicit),
                    kernel_size=1)
            else:
                self.from_feat = lambda x: x
        self.head = head

        # register loss tag
        self.tags = ['score', 'posterior']

    # defer properties
    # dim = property(lambda self: self.unet.dim)
    # encoder = property(lambda self: self.unet.encoder)
    # decoder = property(lambda self: self.unet.decoder)
    # kernel_size = property(lambda self: self.unet.kernel_size)
    # activation = property(lambda self: self.unet.activation)

    def board(self, tb, **k):
        return board2(self, tb, **k, implicit=True)
    
    def forward(self, image, ref=None, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        image : (batch, input_channels, *spatial) tensor
            Input image
        ref : (batch, output_classes[+1], *spatial) tensor, optional
            Ground truth segmentation, used by the loss function.
            Its data type should be integer if it contains hard labels,
            and floating point if it contains soft segmentations.
        _loss : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.
        _metric : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.

        Returns
        -------
        prob : (batch, output_classes[+1], *spatial)
            Tensor of class probabilities.
            If `implicit` is True, the background class is not returned.

        """
        image = torch.as_tensor(image)

        # sanity check
        check.dim(self.dim, image)

        # unet
        score = self.from_feat(self.unet(self.to_feat(image)))
        prob = self.head.posterior(score)
        score = self.head.score(score)

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim + 2))
            check.shape(prob, ref, dims=dims)
            self.compute(_loss, _metric,
                         score=[score, ref],
                         posterior=[prob, ref])

        return prob


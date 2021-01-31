import torch
import torch.nn as tnn
from ..modules._base import Module
from ..modules._cnn import (UNet, MRF)
from ...spatial import grid_pull
from .. import check


class SegMRFNet(Module):
    """Segmentation+MRF network.
    """
    def __init__(self, dim, output_classes=1, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), batch_norm_seg=True,
                 num_iter=10, w=1, num_extra=0, only_unet=False):
        super().__init__()

        self.only_unet = only_unet
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: board(
            dim, tb, inputs, outputs)

        self.unet = SegNet(dim,
                         input_channels=input_channels,
                         output_classes=output_classes,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=activation,
                         batch_norm=batch_norm_seg,
                         inc_final_activation=False)

        if not only_unet:
            self.mrf = MRFNet(dim,
                           num_classes=output_classes,
                           num_extra=num_extra,
                           activation=activation,
                           w=w,
                           num_iter=num_iter)

        # register loss tag
        self.tags = ['unet']
        if not only_unet:
            self.tags.append('mrf')

    # defer properties
    dim = property(lambda self: self.unet.dim)

    def forward(self, image, ref=None, *, _loss=None, _metric=None):
        """Forward pass.
        """
        image = torch.as_tensor(image)

        # sanity check
        check.dim(self.dim, image)

        # augmentation (only if reference is given, i.e., not at test-time)
        if ref is not None:
            for augmenter in self.augmenters:
                image, ref = augmenter(image, ref)

        # unet
        ll = self.unet(image)

        if not self.only_unet:
            # MRF
            p = self.mrf.iter_mrf(ll, ref)

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim+2))
            check.shape(ll, ref, dims=dims)
            if not self.only_unet:
                self.compute(_loss, _metric, mrf=[p, ref])
            else:
                self.compute(_loss, _metric, unet=[ll, ref])

        if self.only_unet:
           p = ll.softmax(dim=1)

        return p


class SegNet(Module):
    """Segmentation network.

    This is simply a UNet that ends with a softmax activation.

    Batch normalization is used by default.
    """

    def __init__(self, dim, output_classes=1, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), batch_norm=True,
                 implicit=True, inc_final_activation=True):
        """

        Parameters
        ----------
        dim : int
            Space dimension
        output_classes : int, default=1
            Number of classes, excluding background
        input_channels : int, default=1
            Number of input channels
        encoder : sequence[int], optional
            Number of features per encoding layer
        decoder : sequence[int], optional
            Number of features per decoding layer
        kernel_size : int or sequence[int], default=3
            Kernel size
        activation : str or callable, default=LeakyReLU(0.2)
            Activation function in the UNet.
        batch_norm : bool or callable, default=True
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
        implicit : bool, default=True
            Only return `output_classes` probabilities (the last one
            is implicit as probabilities must sum to 1).
            Else, return `output_classes + 1` probabilities.
        inc_final_activation : bool, default=True
           Append final activation function.
        """

        super().__init__()

        self.implicit = implicit
        self.output_classes = output_classes
        final_activation = None
        if inc_final_activation:
            if implicit and output_classes == 1:
                final_activation = tnn.Sigmoid
            else:
                final_activation = tnn.Softmax(dim=1)
        if implicit:
            output_classes += 1
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: board(
            dim, tb, inputs, outputs, implicit=implicit)

        self.unet = UNet(dim,
                         input_channels=input_channels,
                         output_channels=output_classes,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=activation,
                         final_activation=final_activation,
                         batch_norm=batch_norm)

        # register loss tag
        self.tags = ['segmentation']

    # defer properties
    dim = property(lambda self: self.unet.dim)
    encoder = property(lambda self: self.unet.encoder)
    decoder = property(lambda self: self.unet.decoder)
    kernel_size = property(lambda self: self.unet.kernel_size)
    activation = property(lambda self: self.unet.activation)

    def forward(self, image, ground_truth=None, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        image : (batch, input_channels, *spatial) tensor
            Input image
        ground_truth : (batch, output_classes[+1], *spatial) tensor, optional
            Ground truth segmentation, used by the loss function.
            Its data type should be integer if it contains hard labels,
            and floating point if it contains soft segmentations.
        _loss : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ground_truth`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.
        _metric : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ground_truth`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.

        Returns
        -------
        probability : (batch, output_classes[+1], *spatial)
            Tensor of class probabilities.
            If `implicit` is True, the background class is not returned.

        """

        image = torch.as_tensor(image)

        # sanity check
        check.dim(self.dim, image)

        # augmentation (only if reference is given, i.e., not at test-time)
        if ground_truth is not None:
            for augmenter in self.augmenters:
                image, ground_truth = augmenter(image, ground_truth)

        # unet
        prob = self.unet(image)
        if self.implicit and prob.shape[1] > self.output_classes:
            prob = prob[:, :-1, ...]

        # compute loss and metrics
        if ground_truth is not None:
            # sanity checks
            check.dim(self.dim, ground_truth)
            dims = [0] + list(range(2, self.dim+2))
            check.shape(prob, ground_truth, dims=dims)
            self.compute(_loss, _metric, segmentation=[prob, ground_truth])

        return prob


class MRFNet(Module):
    """MRF network.

    A network that aims at cleaning up categorical image data of num_classes
    classes. The first layer is a, so called, MRF layer. Sub-sequent layers are
    then 1x1 convolutions layers. The output of the network are softmaxed,
    categorical data. The idea is described in:

    Brudfors, Mikael, Yaël Balbastre, and John Ashburner.
    "Nonlinear markov random fields learned via backpropagation."
    IPMI. Springer, Cham, 2019.

    """

    def __init__(self, dim, num_classes, num_iter=10, num_extra=0, w=1,
                 activation=tnn.LeakyReLU(0.2)):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        num_classes : int
            Number of input classes.
        num_iter : int, default=5
            Number of mean-field iterations.
        num_extra : int, default=0
            Number of extra layers between MRF layer and final layer.
        w : float, default=1.0
            Weight between new and old prediction [0, 1].
        activation : str or type or callable, default='tnn.LeakyReLU(0.2)'
            Activation function.

        """
        super().__init__()

        self.num_classes = num_classes
        if num_iter < 1:
            raise ValueError('Parameter num_iter should be greater than 0 , got {:}'.format(num_iter))
        self.num_iter = num_iter
        if w < 0 or w > 1:
            raise ValueError('Parameter w should be between 0 and 1, got {:}'.format(w))
        self.w = w
        # As the final activation is applied at the end of the forward method
        # below, it is not applied in the final Conv layer
        final_activation = None
        # We use a first-order neigbourhood
        kernel_size = 3
        num_filters = self.get_num_filters(dim)
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: board(
            dim, tb, inputs, outputs)
        # Build MRF net
        self.mrf = MRF(dim,
                       num_classes=num_classes,
                       num_filters=num_filters,
                       num_extra=num_extra,
                       kernel_size=kernel_size,
                       activation=activation,
                       batch_norm=False,
                       final_activation=final_activation)
        # register loss tag
        self.tags = ['mrf']

    # defer properties
    dim = property(lambda self: self.mrf.dim)
    kernel_size = property(lambda self: self.mrf.kernel_size)
    activation = property(lambda self: self.mrf.activation)

    def forward(self, ll, ref=None, *, _loss=None, _metric=None):
        """Forward pass, with mean-field iterations.

        Parameters
        ----------
        ll : (batch, input_channels, *spatial) tensor
            Input image, the log-likelihood term.
        ref : (batch, output_classes[+1], *spatial) tensor, optional
            Reference segmentation, used by the loss function.
        _loss : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ground_truth`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.
        _metric : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ground_truth`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.

        Returns
        -------
        probability : (batch, output_classes[+1], *spatial)
            Tensor of class probabilities.

        """
        ll = torch.as_tensor(ll)

        # sanity check
        check.dim(self.dim, ll)

        # augmentation (only if reference is given, i.e., not at test-time)
        if ref is not None:
            for augmenter in self.augmenters:
                ll, ref = augmenter(ll, ref)

        # MRF
        p = self.iter_mrf(ll, ref)

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim+2))
            check.shape(p, ref, dims=dims)
            self.compute(_loss, _metric, mrf=[p, ref])

        return p

    def get_num_filters(self, dim):
        """Get number of MRF filters.
        """
        num_filters = self.num_classes**2
        if num_filters > 128:
            num_filters = 128
        return num_filters

    def iter_mrf(self, ll, ref):
        """Iterate over MRF.
        """
        p = torch.zeros_like(ll)
        for i in range(self.get_num_iter(ref)):
            op = p.clone()
            p = (ll + self.mrf(p)).softmax(dim=1)
            p = self.w*p + (1 - self.w)*op
        return p

    def get_num_iter(self, ref):
        """ Get number of VB iterations.
        """
        with torch.no_grad():
            if ref is not None:
                # Training
                num_iter = int(torch.LongTensor(1).random_(1, self.num_iter + 1))
            else:
                # Testing
                num_iter = self.num_iter
        return num_iter


def board(dim, tb, inputs, outputs, implicit=False):
    """TensorBoard visualisation of a segmentation model's inputs and outputs.

    Parameters
    ----------
    dim : int
        Space dimension
    tb : torch.utils.tensorboard.writer.SummaryWriter
        TensorBoard writer object.
    inputs : (tensor_like, tensor_like) tuple
        Input image (N, C, dim)  and reference segmentation (N, K, dim) .
    outputs : (N, K, dim) tensor_like
        Predicted segmentation.
    implicit : bool, default=False
        Only return `output_classes` probabilities (the last one
        is implicit as probabilities must sum to 1).
        Else, return `output_classes + 1` probabilities.

    """
    def get_slice(vol, plane, dim):
        if dim == 2:
            return vol.squeeze()
        if plane == 'z':
            z = round(0.5 * vol.shape[-1])
            slice = vol[..., z]
        elif plane == 'y':
            y = round(0.5 * vol.shape[-2])
            slice = vol[..., y, :]
        elif plane == 'x':
            x = round(0.5 * vol.shape[-3])
            slice = vol[..., x, :, :]

        return slice.squeeze()

    def input_view(slice_input):
        return slice_input

    def prediction_view(slice_prediction, implicit):
        if implicit:
            slice_prediction = torch.cat(
                (1 - slice_prediction.sum(dim=0, keepdim=True), slice_prediction), dim=0)
        K1 = float(slice_prediction.shape[0])
        slice_prediction = (slice_prediction.argmax(dim=0, keepdim=False))/(K1 - 1)
        return slice_prediction

    def target_view(slice_target):
        return slice_target.float()/slice_target.max().float()

    def to_grid(slice_input, slice_target, slice_prediction):
        return torch.cat((slice_input, slice_target, slice_prediction), dim=1)

    def get_slices(plane, inputs, outputs, dim, implicit):
        slice_input = input_view(get_slice(inputs[0][0, ...], plane, dim=dim))
        slice_target = target_view(get_slice(inputs[1][0, ...], plane, dim=dim))
        slice_prediction = prediction_view(get_slice(outputs[0, ...], plane, dim=dim),
                                           implicit=implicit)
        return slice_input, slice_target, slice_prediction

    def get_image(plane, inputs, outputs, dim, implicit):
        slice_input, slice_target, slice_prediction = \
            get_slices(plane, inputs, outputs, dim, implicit)
        if len(slice_input.shape) != len(slice_prediction.shape):
            K1 = float(slice_input.shape[0])
            slice_input = (slice_input.argmax(dim=0, keepdim=False)) / (K1 - 1)
            slice_target = (slice_target.argmax(dim=0, keepdim=False)) / (K1 - 1)
        return to_grid(slice_input, slice_target, slice_prediction)[None, ...]

    # Add to TensorBoard
    title = 'Image-Target-Prediction_'
    tb.add_image(title + 'z', get_image('z', inputs, outputs,
                                        dim, implicit))
    if dim == 3:
        tb.add_image(title + 'y', get_image('y', inputs, outputs,
                                            dim, implicit))
        tb.add_image(title + 'x', get_image('x', inputs, outputs,
                                            dim, implicit))
    tb.flush()

import torch
import torch.nn as tnn
from ..modules._base import Module
from ..modules._cnn import (UNet, MRF)
from ...spatial import grid_pull
from .. import check


class SegNet(Module):
    """Segmentation network.

    This is simply a UNet that ends with a softmax activation.

    Batch normalization is used by default.
    """

    def __init__(self, dim, output_classes=1, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), batch_norm=True,
                 implicit=True):
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
        """

        super().__init__()

        self.implicit = implicit
        self.output_classes = output_classes
        if implicit and output_classes == 1:
            final_activation = tnn.Sigmoid
        else:
            output_classes = output_classes + 1
            final_activation = tnn.Softmax(dim=1)
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
            check.shape(image, ground_truth, dims=dims)
            self.compute(_loss, _metric, segmentation=[prob, ground_truth])

        return prob


class MRFNet(Module):
    """MRF network.

    This network takes as input one-hot encoded, probabilistic segmentation
    data (one-hot encoded). It currently assumes an explicit representation of
    the number of segmentation classes. Its outputs same sized one-hot encoded,
    probabilistic segmentations.

    """

    def __init__(self, dim, num_classes, num_iter=20, num_filters=16, num_extra=0,
                 kernel_size=3, activation=tnn.LeakyReLU(0.2), batch_norm=False,
                 w=0.5):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        num_classes : int
            Number of input classes.
        num_iter : int, default=20
            Number of mean-field iterations.
        num_extra : int, default=0
            Number of extra layers between MRF layer and final layer.
        num_filters : int, default=16
            Number of conv filters in first, MRF layer.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        activation : str or type or callable, default='tnn.LeakyReLU(0.2)'
            Activation function.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        w : float, default=0.5
            Weight between new and old prediction [0, 1].

        """
        super().__init__()

        self.num_iter = num_iter
        if w < 0 or w > 1:
            raise ValueError('Parameter w should be between 0 and 1, got {w}'.format(w))
        self.w = w
        if num_classes == 1:
            final_activation = tnn.Sigmoid
        else:
            final_activation = tnn.Softmax(dim=1)
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: board(
            dim, tb, inputs, outputs)

        self.mrf = MRF(dim,
                       num_classes=num_classes,
                       num_filters=num_filters,
                       num_extra=num_extra,
                       kernel_size=kernel_size,
                       activation=activation,
                       batch_norm=batch_norm,
                       final_activation=final_activation)

        # register loss tag
        self.tags = ['mrf']

    # defer properties
    dim = property(lambda self: self.mrf.dim)
    kernel_size = property(lambda self: self.mrf.kernel_size)
    activation = property(lambda self: self.mrf.activation)

    def forward(self, seg, ref=None, *, _loss=None, _metric=None):
        """Forward pass, with mean-field iterations.
        """
        seg = torch.as_tensor(seg)

        # sanity check
        check.dim(self.dim, seg)

        # augmentation (only if reference is given, i.e., not at test-time)
        if ref is not None:
            for augmenter in self.augmenters:
                seg, ref = augmenter(seg, ref)

        # mrf
        with torch.no_grad():
            if self.train():
                # Training: variable number of iterations
                num_iter = int(torch.LongTensor(1).random_(1, self.num_iter))
            else:
                # Testing: fixed number of iterations
                num_iter = self.num_iter
        oseg = seg.clone()
        for i in range(num_iter):
            pred = self.mrf(seg)
            pred = self.w*pred + (1 - self.w)*oseg
            if i < num_iter - 1:
                oseg = pred.clone()

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim+2))
            check.shape(seg, ref, dims=dims)
            self.compute(_loss, _metric, mrf=[pred, ref])

        return pred


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

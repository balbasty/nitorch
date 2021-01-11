import torch
import torch.nn as tnn
from ._base import Module
from ._cnn import UNet
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


    def board(self, tb, inputs, outputs):
        """TensorBoard visualisation of model input image, reference segmentation
        and predicted segmentation.

        Parameters
        ----------
        tb : torch.utils.tensorboard.writer.SummaryWriter
            TensorBoard writer object.
        inputs : (tensor_like, tensor_like) tuple
            Input image (N, C, dim)  and reference segmentation (N, K, dim) .
        outputs : (N, K, dim) tensor_like
            Predicted segmentation.

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
            return to_grid(*get_slices(plane, inputs, outputs, dim, implicit))[None, ...]

        # Add to TensorBoard
        title = 'Image-Target-Prediction_'
        tb.add_image(title + 'z', get_image('z', inputs, outputs,
                                            self.dim, self.implicit))
        if self.dim == 3:
            tb.add_image(title + 'y', get_image('y', inputs, outputs,
                                                self.dim, self.implicit))
            tb.add_image(title + 'x', get_image('x', inputs, outputs,
                                                self.dim, self.implicit))
        tb.flush()
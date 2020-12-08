from ._base import Module
from ._cnn import UNet
import torch
import torch.nn as tnn


class SegNet(Module):
    """Segmentation network.

    This is simply a UNet that ends with a softmax activation.
    """

    def __init__(self, dim, output_classes=1, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), batch_norm=False,
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
        encoder : sequence[int]
            Number of features per encoding layer
        decoder : sequence[int]
            Number of features per decoding layer
        kernel_size : int or sequence[int], default=3
            Kernel size
        activation : str or callable, default=LeakyReLU(0.2)
            Activation function in the UNet.
        batch_norm : bool or callable, optional
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

    def forward(self, image, ground_truth=None, *, _loss, _metric):

        image = torch.as_tensor(image)

        # sanity check
        if image.dim() != self.dim + 2:
            raise ValueError('Expected image with shape '
                             '(batch, channels, *spatial) '
                             'with len(spatial) == {}, but found {}.'
                             .format(self.dim, image.shape))

        # unet
        prob = self.unet(image)
        if self.implicit and prob.shape[1] > self.output_classes:
            prob = prob[:, :-1, ...]

        # compute loss and metrics
        if ground_truth is not None:
            # sanity check
            if ground_truth.shape[0] != image.shape[0] or \
                    ground_truth.shape[2:] != image.shape[2:]:
                raise ValueError('Expected ground truth with shape '
                                 '(batch, classes|1, *spatial) '
                                 'but found {}.'
                                 .format(self.dim, ground_truth.shape))
            self.compute(_loss, _metric,
                         segmentation=[prob, ground_truth])

        return prob


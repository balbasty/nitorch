from ._base import Module
from ._cnn import UNet
import torch
import torch.nn as tnn


class SegNet(Module):
    """Segmentation network.

    This is simply a UNet that ends with a softmax activation.
    """

    def __init__(self, dim, output_classes, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2)):

        super().__init__()
        self.unet = UNet(dim,
                         input_channels=input_channels,
                         output_channels=output_classes,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=activation,
                         final_activation=tnn.Softmax(dim=1))

        # register loss tag
        self.tags = ['segmentation']

    # defer properties
    dim = property(lambda self: self.unet.dim)
    output_classes = property(lambda self: self.unet.output_channels)
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


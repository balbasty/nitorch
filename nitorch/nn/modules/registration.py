import torch
from torch import nn as tnn
from nitorch import core, spatial
from nitorch.core.py import make_list
from .cnn import UNet
from .base import Module
from .spatial import GridPull, GridResize, GridExp
from .. import check


class VoxelMorph(Module):
    """VoxelMorph warps a source/moving image to a fixed/target image.

    A VoxelMorph network is obtained by concatenating a UNet and a
    (diffeomorphic) spatial transformer. The loss is made of two terms:
    an image similarity loss and a velocity regularisation loss.

    The UNet used here is slightly different from the original one (we
    use a fully convolutional network -- based on strided convolutions --
    instead of maxpooling and upsampling).

    References
    ----------
    .. [1] "An Unsupervised Learning Model for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        CVPR 2018. eprint arXiv:1802.02604
    .. [2] "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231
    .. [3] "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MICCAI 2018. eprint arXiv:1805.04605
    .. [4] "Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
    """

    def __init__(self, dim, encoder=None, decoder=None, kernel_size=3,
                 interpolation='linear', grid_bound='dft', image_bound='dct2',
                 downsample_velocity=2, *, _input_channels=2):
        """

        Parameters
        ----------
        dim : int
            Dimensionalityy of the input (1|2|3)
        encoder : list[int], optional
            Number of channels after each encoding layer of the UNet.
        decoder : list[int], optional
            Number of channels after each decoding layer of the Unet.
        kernel_size : int or list[int], default=3
            Kernel size of the UNet.
        interpolation : int, default=1
            Interpolation order.
        grid_bound : bound_type, default='dft'
            Boundary conditions of the velocity field.
        image_bound : bound_type, default='dct2'
            Boundary conditions of the image.
        downsample_velocity : float, default=2
            Downsample the velocity field by a factor when exponentiating.
        """
        resize_factor = make_list(downsample_velocity, dim)
        resize_factor = [1/f for f in resize_factor]

        super().__init__()
        self.unet = UNet(dim,
                         input_channels=_input_channels,
                         output_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.resize = GridResize(interpolation=interpolation,
                                 bound=grid_bound,
                                 factor=resize_factor)
        self.velexp = GridExp(interpolation=interpolation,
                              bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'velocity', 'segmentation']

    def exp(self, velocity, displacement=False):
        """Generate a deformation grid from tangent parameters.

        Parameters
        ----------
        velocity : (batch, *spatial, nb_dim)
            Stationary velocity field
        displacement : bool, default=False
            Return a displacement field (voxel to shift) rather than
            a transformation field (voxel to voxel).

        Returns
        -------
        grid : (batch, *spatial, nb_dim)
            Deformation grid (transformation or displacement).

        """
        backend = dict(dtype=velocity.dtype, device=velocity.device)

        # generate grid
        shape = velocity.shape[1:-1]
        velocity_small = self.resize(velocity, type='displacement')
        grid = self.velexp(velocity_small)
        grid = self.resize(grid, shape=shape, type='grid')

        if displacement:
            grid = grid - spatial.identity_grid(grid.shape[1:-1], **backend)
        return grid

    def forward(self, source, target, source_seg=None, target_seg=None,
                *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image
        source_seg : tensor (batch, classes, *spatial), optional
            Source/moving segmentation
        target_seg : tensor (batch, classes, *spatial), optional
            Target/fixed segmentation

        Other Parameters
        ----------------
        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        deformed_source_seg : tensor (batch, classes, *spatial), optional
            Deformed source segmentation
        velocity : tensor (batch,, *spatial, len(spatial))
            Velocity field

        """
        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim+2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim+2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        velocity = self.unet(source_and_target)
        velocity = core.utils.channel2last(velocity)
        grid = self.exp(velocity)
        deformed_source = self.pull(source, grid)

        if source_seg is not None:
            if source_seg.shape[2:] != source.shape[2:]:
                grid = spatial.resize_grid(grid, shape=source_seg.shape[2:])
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = None

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     velocity=[velocity],
                     segmentation=[deformed_source_seg, target_seg])

        if source_seg is None:
            return deformed_source, velocity
        else:
            return deformed_source, deformed_source_seg, velocity

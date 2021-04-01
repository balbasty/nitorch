import torch
from torch import nn as tnn
from nitorch import core, spatial
from nitorch.core import py, utils
from .cnn import UNet2
from .base import Module
from .spatial import GridPull, GridResize, GridExp, GridShoot
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

    def __init__(self, dim, unet=None, pull=None, exp=None,
                 *, in_channels=2):
        """

        Parameters
        ----------
        dim : int
            Dimensionality of the input (1|2|3)
        unet : dict
            Dictionary of U-Net parameters with fields:
                encoder : sequence[int], default=[16, 32, 32, 32]
                decoder : sequence[int], default=[32, 32, 32, 32, 32, 16, 16]
                conv_per_layer : int, default=1
                kernel_size : int, default=3
                activation : str or callable, default=LeakyReLU(0.2)
                pool : {'max', 'conv', 'down', None}, default='max'
                unpool : {'conv', 'up', None}, default='up'
        pull : dict
            Dictionary of Transformer parameters with fields:
                interpolation : {0..7}, default=1
                bound : str, default='dct2'
                extrapolate : bool, default=False
        exp : dict
            Dictionary of Exponentiation parameters with fields:
                interpolation : {0..7}, default=1
                bound : str, default='dft'
                steps : int, default=8
                shoot : bool, default=False
                downsample : float, default=2
            If shoot is True, these fields are also present:
                absolute : float, default=0.0001
                membrane : float, default=0.001
                bending : float, default=0.2
                lame : (float, float), default=(0.05, 0.2)
        """
        # default parameters
        unet = dict(unet or {})
        unet.setdefault('encoder', [16, 32, 32, 32])
        unet.setdefault('decoder', [16, 32, 32, 32])
        unet.setdefault('kernel_size', 3)
        unet.setdefault('pool', 'max')
        unet.setdefault('unpool', 'up')
        unet.setdefault('activation', tnn.LeakyReLU(0.2))
        pull = dict(pull or {})
        pull.setdefault('interpolation', 1)
        pull.setdefault('bound', 'dct2')
        pull.setdefault('extrapolate', False)
        exp = dict(exp or {})
        exp.setdefault('interpolation', 1)
        exp.setdefault('bound', 'dft')
        exp.setdefault('steps', 8)
        exp.setdefault('shoot', False)
        exp.setdefault('downsample', 2)
        exp.setdefault('absolute', 0.0001)
        exp.setdefault('membrane', 0.001)
        exp.setdefault('bending', 0.2)
        exp.setdefault('lame', (0.05, 0.2))
        do_shoot = exp.pop('shoot')
        downsample_vel = utils.make_vector(exp.pop('downsample'), dim).tolist()
        vel_inter = exp['interpolation']
        vel_bound = exp['bound']
        if do_shoot:
            exp.pop('interpolation')
            exp.pop('bound')
            exp.pop('voxel_size', downsample_vel)
            vol = py.prod(downsample_vel)
            exp['absolute'] *= vol
            exp['membrane'] *= vol
            exp['bending'] *= vol
            exp['lame'] = [l * vol for l in py.make_list(exp['lame'])]
        else:
            exp.pop('absolute')
            exp.pop('membrane')
            exp.pop('bending')
            exp.pop('lame')

        # prepare layers
        super().__init__()
        self.unet = UNet2(dim, in_channels, dim, **unet,)
        self.resize = GridResize(interpolation=vel_inter, bound=vel_bound,
                                 factor=[1/f for f in downsample_vel])
        self.velexp = GridShoot(**exp) if do_shoot else GridExp(**exp)
        self.pull = GridPull(**pull)
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
        # generate grid
        shape = velocity.shape[1:-1]
        velocity_small = self.resize(velocity, type='displacement')
        grid = self.velexp(velocity_small, displacement=displacement)
        grid = self.resize(grid, shape=shape,
                           type='disp' if displacement else 'grid')
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

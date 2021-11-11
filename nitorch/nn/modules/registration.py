"""Registration networks

These are mostly re-implementations of VoxelMorph and its derivatives.

VoxelMorph : pair-wise registration network
AtlasMorph : (unconditional) atlas building

References
----------
.. [1] "An Unsupervised Learning Model for Deformable Medical Image Registration"
    Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
    CVPR 2018. eprint arXiv:1802.02604
.. [2] "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
    Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
    IEEE TMI 2019. eprint arXiv:1809.05231
.. [3] "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration"
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    MICCAI 2018. eprint arXiv:1805.04605
.. [4] "Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces"
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    MedIA 2019. eprint arXiv:1903.03545
.. [5] "Learning Conditional Deformable Templates with Convolutional Networks"
    A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
    NeurIPS 2019. eprint arXiv:1908.02738
"""

import math as pymath
import torch
from torch import nn as tnn
from nitorch import core, spatial
from nitorch.core import py, utils
from .cnn import UNet2, Decoder, StackedConv
from nitorch.nn.base import Module
from .spatial import GridPull, GridResize, GridExp, GridShoot
from .linear import Linear
from ..activations import SoftMax
from .. import check


class VoxelMorph(Module):
    """VoxelMorph warps a source/moving image to a fixed/target image.

    A VoxelMorph network is obtained by concatenating a UNet and a
    (diffeomorphic) spatial transformer. The loss is made of two terms:
    an image similarity loss and a velocity regularisation loss.

    The original U-Net structure used by VoxelMorph is described in [2].
    It works at 5 different resolutions, with the number of features at 
    each encoding scale being [16, 32, 32, 32, 32]. The first number 
    corresponds to feature extraction at the initial resolution, and 
    the last number is the number of output features at the coarsest 
    resolution (the bottleneck). In the decoder, the number of features
    at each scale is [32, 32, 32, 32], each of these feature map is 
    concatenated with the output from the decoder at the same scale.
    Finally, two convolutions with 16 output features are applied 
    (without change of scale) followed by a final convolution with 
    3 output features (the three components of the displacement or
    velocity field). Note that all encoding and decoding convolutions
    have kernel size 3 and stride 2 -- therefore no max-pooling or
    linear upsampling is used. All convolutions are followed by a 
    leaky ReLU activation with slope 0.2. The default parameters of
    our implementation follow this architecture.
    
    Note that a slighlty different architecture was proposed in [1], 
    where two convolutions were applied at the second-to-last scale.
    This module does not implement this architecture. However, 
    our U-Net is highly parameterised, and alternative pooling and 
    upsampling methods, activation functions and number of convolutions
    per scale can be used.
    
    A scaling and squaring layer is used to integrate the output 
    velocity field and generate a diffeomorphic transformation, 
    as in [3, 4]. If the number of integration steps is set at 0, 
    a small deformation model (without integration) is used. 
    Alternatively, a novel geodesic shooting layer can be used 
    by setting `shoot=True` in the exponentiation structure.

    References
    ----------
    .. [1] "An Unsupervised Learning Model for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        CVPR 2018. eprint arXiv:1802.02604
    .. [2] "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        IEEE TMI 2019. eprint arXiv:1809.05231
    .. [3] "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MICCAI 2018. eprint arXiv:1805.04605
    .. [4] "Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA 2019. eprint arXiv:1903.03545
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
                encoder : sequence[int], default=[16, 32, 32, 32, 32]
                decoder : sequence[int], default=[32, 32, 32, 32, 16, 16]
                conv_per_layer : int, default=1
                kernel_size : int, default=3
                activation : str or callable, default=LeakyReLU(0.2)
                pool : {'max', 'conv', 'down', None}, default=None
                    'max'  -> 2x2x2 max-pooling
                    'conv' -> 2x2x2 strided convolution (no bias, no activation)
                    'down' -> downsampling
                     None  -> use strided convolutions in the encoder
                unpool : {'conv', 'up', None}, default=None
                    'conv' -> 2x2x2 strided convolution (no bias, no activation)
                    'up'   -> linear upsampling
                     None  -> use strided convolutions in the decoder
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
        unet.setdefault('encoder', [16, 32, 32, 32, 32])
        unet.setdefault('decoder', [32, 32, 32, 32, 16, 16])
        unet.setdefault('kernel_size', 3)
        unet.setdefault('pool', None)
        unet.setdefault('unpool', None)
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
        exp.setdefault('factor', 1)
        do_shoot = exp.pop('shoot')
        downsample_vel = exp.pop('downsample')
        vel_inter = exp['interpolation']
        vel_bound = exp['bound']
        if do_shoot:
            exp.pop('interpolation')
            exp.pop('bound')
            exp.pop('voxel_size', downsample_vel)
            exp['factor'] *= py.prod(downsample_vel)
        else:
            exp.pop('absolute')
            exp.pop('membrane')
            exp.pop('bending')
            exp.pop('lame')
            exp.pop('factor')
        exp['displacement'] = True
        unet['skip_decoder_level'] = int(pymath.floor(pymath.log(downsample_vel) / pymath.log(2)))

        # prepare layers
        super().__init__()
        self.unet = UNet2(dim, in_channels, dim, **unet,)
        self.velexp = GridShoot(**exp) if do_shoot else GridExp(**exp)
        self.resize = GridResize(interpolation=vel_inter, bound=vel_bound,
                                 factor=downsample_vel, anchor='f',
                                 type='displacement')
        self.pull = GridPull(**pull)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'velocity', 'segmentation']

    def exp(self, velocity, shape=None, displacement=False):
        """Generate a deformation grid from tangent parameters.

        Parameters
        ----------
        velocity : (batch, *spatial, nb_dim)
            Stationary velocity field
        shape : sequence[int], optional
        displacement : bool, default=False
            Return a displacement field (voxel to shift) rather than
            a transformation field (voxel to voxel).

        Returns
        -------
        grid : (batch, *spatial, nb_dim)
            Deformation grid (transformation or displacement).

        """
        # generate grid
        grid = self.velexp(velocity)
        grid = self.resize(grid, output_shape=shape)
        if not displacement:
            grid = spatial.add_identity_grid_(grid)
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
        shape = source.shape[2:]
        source_and_target = torch.cat((source, target), dim=1)
        velocity = self.unet(source_and_target)
        velocity = core.utils.channel2last(velocity)
        grid = self.exp(velocity, shape=shape)

        deformed_source = self.pull(source, grid)

        if source_seg is not None:
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

    def board(self, tb, **k):
        """Tensorboard visualization function"""
        implicit = getattr(self, 'implicit', False)
        return registration_board(self, tb, **k, implicit=implicit)


class AtlasMorph(Module):
    """AtlasMorph learns an atlas *and* learns to warp it to data.

    Instead of being fixed, as in VoxelMorph, the moving image is here
    learnable. In a classic LDDMM framework, an unbiased atlas can be
    obtained by ensuring that all velocity fields sum to zero.
    In a stochastic optimization framework, the model does not see
    all images at the same time. Instead, a running mean of velocity
    fields is computed and its deviation from zero is penalized.

    References
    ----------
    .. [1] "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MICCAI 2018. eprint arXiv:1805.04605
    .. [2] "Learning Conditional Deformable Templates with Convolutional Networks"
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        NeurIPS 2019. eprint arXiv:1908.02738
    """

    # TODO:
    #   - GMM loss
    #   - Graph template (like samseg)?

    def __init__(self, dim, unet=None, pull=None, exp=None, template=None):
        """

        Parameters
        ----------
        dim : int
            Dimensionality of the input (1|2|3)
        unet : dict
            Dictionary of U-Net parameters with fields:
                encoder : sequence[int], default=[16, 32, 32, 32, 32]
                decoder : sequence[int], default=[32, 32, 32, 32, 16, 16]
                conv_per_layer : int, default=1
                kernel_size : int, default=3
                activation : str or callable, default=LeakyReLU(0.2)
                pool : {'max', 'conv', 'down', None}, default=None
                    'max'  -> 2x2x2 max-pooling
                    'conv' -> 2x2x2 strided convolution (no bias, no activation)
                    'down' -> downsampling
                     None  -> use strided convolutions in the encoder
                unpool : {'conv', 'up', None}, default=None
                    'conv' -> 2x2x2 strided convolution (no bias, no activation)
                    'up'   -> linear upsampling
                     None  -> use strided convolutions in the decoder
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
        template : dict
            Dictionary of Template parameters with fields:
                shape : tuple[int], default=(192,) * dim
                mom : float or int, default=100
                    If in (0, 1), momentum of the running mean.
                    The mean is updated according to:
                        `new_mean = (1-mom) * old_mean + mom * new_sample`
                    If 0, use cumulative average:
                        `new_n = old_n + 1`
                        `mom = 1/new_n`
                    If > 1, cap the weight of a new sample in the average:
                        `new_n = min(cap, old_n + 1)`
                        `mom = 1/new_n`
                cat : bool or int, default=False
                    Build a categorical template.
                implicit : bool, default=True
                    Whether the template has an implicit background class.

        """
        # default parameters
        unet = dict(unet or {})
        unet.setdefault('encoder', [16, 32, 32, 32, 32])
        unet.setdefault('decoder', [32, 32, 32, 32, 16, 16])
        unet.setdefault('kernel_size', 3)
        unet.setdefault('pool', None)
        unet.setdefault('unpool', None)
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
        exp.setdefault('factor', 1)
        do_shoot = exp.pop('shoot')
        downsample_vel = utils.make_vector(exp.pop('downsample'), dim).tolist()
        vel_inter = exp['interpolation']
        vel_bound = exp['bound']
        if do_shoot:
            exp.pop('interpolation')
            exp.pop('bound')
            exp.pop('voxel_size', downsample_vel)
            exp['factor'] *= py.prod(downsample_vel)
            if do_shoot == 'approx':
                exp['approx'] = True
        else:
            exp.pop('absolute')
            exp.pop('membrane')
            exp.pop('bending')
            exp.pop('lame')
            exp.pop('factor')
        exp['displacement'] = True
        template = dict(template or {})
        template.setdefault('shape', (192,)*dim)
        template.setdefault('mom', 100)
        template.setdefault('cat', False)
        template.setdefault('implicit', True)

        self.cat = template['cat']
        self.implicit = template['implicit']

        # prepare layers
        super().__init__()
        template_channels = (self.cat + (not self.implicit)) if self.cat else 1
        self.template = tnn.Parameter(torch.zeros([template_channels, *template['shape']]))
        self.unet = UNet2(dim, template_channels + 1, dim, **unet)
        self.resize = GridResize(interpolation=vel_inter, bound=vel_bound,
                                 factor=[1 / f for f in downsample_vel],
                                 type='displacement')
        self.velexp = GridShoot(**exp) if do_shoot else GridExp(**exp)
        self.pull = GridPull(**pull)
        self.dim = dim
        self.mom = template['mom']

        # register losses/metrics
        self.tags = ['match', 'velocity',  'template', 'mean']

    def init_template(self, images, one_hot_map=None):
        """Initialize template with the average of a series of images

        Parameters
        ----------
        images : sequence of (b, 1, *shape) tensor
        one_hot_map : sequence[int], default=identity
            Mapping from hard label to soft class.

        """
        with torch.no_grad():
            if self.cat:
                self._init_template_cat(images, one_hot_map)
            else:
                self._init_template_mse(images)

    def _init_template_cat(self, images, one_hot_map):
        self.template.data.zero_()
        n = 0
        for i, image in enumerate(images):
            image = image.to(self.template.device)
            # check shape
            shape = image.shape[2:]
            if self.template.shape[1:] != shape:
                if i == 0:
                    shape = [self.cat + (not self.implicit), *shape]
                    self.template = tnn.Parameter(torch.zeros(shape))
                else:
                    raise ValueError('All images must have the same shape')
            # mean probabilities
            if image.dtype.is_floating_point:
                image = image.to(self.template.dtype)
                self.template.data[:self.cat] += image[:, :self.cat].sum(0)
            else:
                if not one_hot_map:
                    one_hot_map = list(range(1, self.cat+1))
                for soft, label in enumerate(one_hot_map):
                    label = py.make_list(label)
                    if len(label) == 1:
                        self.template.data[soft] += (image == label).sum(0)[0]
                    else:
                        self.template.data[soft] += utils.isin(image, label).sum(0)[0]
            n += image.shape[0]

        k = self.cat
        self.template.data /= n
        self.template.data[:k] += 1e-5
        self.template.data[:k] /= 1+1e-5
        norm = self.template.data[:k].sum(0).neg_().add_(1)
        if not self.implicit:
            self.template.data[-1] = norm
            self.template.data.clamp_min_(1e-5).log_()
        else:
            self.template.data.clamp_min_(1e-5).log_()
            self.template.data -= norm.clamp_min_(1e-5).log_()

    def _init_template_mse(self, images):
        self.template.data.zero_()
        n = 0
        for i, image in enumerate(images):
            image = image.to(device=self.template.device,
                             dtype=self.template.dtype)
            if self.template.shape[1:] != image.shape[2:]:
                if i == 0:
                    shape = image.shape[2:]
                    self.template = tnn.Parameter(torch.zeros([1, *shape]))
                else:
                    raise ValueError('All images must have the same shape')
            n += image.shape[0]
            self.template.data += image.sum(0)[0]
        self.template.data /= n

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
        velocity_small = self.resize(velocity)
        grid = self.velexp(velocity_small)
        grid = self.resize(grid, output_shape=shape, factor=None)
        if not displacement:
            grid = spatial.add_identity_grid_(grid)
        return grid

    def update_mean(self, velocity):
        """Update running mean with a new sample."""
        batch = velocity.shape[0]
        if not hasattr(self, 'mean'):
            self.register_buffer('mean', velocity.mean(0))
            self.tracked = batch
        else:
            self.tracked += batch
            if not self.mom:
                mom = 1/self.tracked
                velocity = velocity.sum(0)
            elif self.mom > 1:
                mom = 1/min(self.mom, self.tracked)
                velocity = velocity.sum(0)
            else:
                mom = self.mom
                velocity = velocity.mean(0)
            # we need to detach the previous mean so that gradients do
            # not propagate through the previous iterations (we only 
            # want to penalize the latest velocity)
            self.mean = self.mean.detach()
            self.mean *= (1 - mom)
            self.mean += mom * velocity

    def forward(self, target,
                *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        target : tensor (batch, 1|classes, *spatial)
            Target/fixed image

        Other Parameters
        ----------------
        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_template : tensor (batch, 1|classes, *spatial)
            Deformed template
        velocity : tensor (batch, *spatial, len(spatial))
            Velocity field

        """
        # sanity checks
        check.dim(self.dim, self.template[None], target)
        check.shape(target, self.template[None], dims=range(2, self.dim + 2))

        # chain operations
        batch = target.shape[0]
        template = self.template
        if self.cat:
            template = SoftMax(implicit=self.implicit, dim=0)(template)
        template = template.expand([batch, *self.template.shape])
        source_and_target = torch.cat((template, target), dim=1)
        del template
        velocity = self.unet(source_and_target)
        del source_and_target
        velocity = core.utils.channel2last(velocity)
        grid = self.exp(velocity)
        deformed_template = self.pull(self.template, grid)
        if self.cat:
            if not self.pull.extrapolate:
                msk = (deformed_template == 0).all(dim=1, keepdim=True)
            deformed_template = SoftMax(implicit=self.implicit)(deformed_template)
            if not self.pull.extrapolate:
                # we can't just let out-of-bound values to be just zero, as
                # it makes the probability be equi-probable. Instead, We want 
                # the background class to have probability one.
                deformed_template[:self.cat][msk] = 1e-5
                if not self.implicit:
                    deformed_template[self.cat:][msk] = 1 - (self.cat*1e-5)

        # running mean
        if self.training:
            self.update_mean(velocity)

        # compute loss and metrics
        losses = dict(velocity=[velocity])
        if self.training:
            losses['mean'] = [self.mean]
            losses['template'] = [self.template]
        losses['match'] = [deformed_template, target]
        self.compute(_loss, _metric, **losses)

        return deformed_template, velocity

    def board(self, tb, **k):
        """Tensorboard visualization function"""
        implicit = getattr(self, 'implicit', False)
        if k.get('inputs', None):
            batch = k['inputs'][0].shape[0]
            template = self.template.expand([batch, *self.template.shape])
            if self.cat:
                template = SoftMax(implicit=self.implicit)(template)
            k['inputs'] = (template, *k['inputs'])
        return registration_board(self, tb, **k, implicit=implicit)


class _MetaToImage(Module):
    """Generate an image from a vector of metadata"""
    # WIP: this probably does not work

    def __init__(self, shape, in_channels, out_channels, nb_levels=0,
                 decoder=(32, 32, 32, 32), kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), unpool=None):
        """

        Parameters
        ----------
        shape : sequence[int]
            Output spatial shape
        in_channels : int
            Number of input channels (= meta variables)
        out_channels : int
            Number of output channels
        nb_levels : int, default=0
            Number of levels in the decoder.
            If 0: directly generate the image using a dense layer.
        decoder : sequence[int], default=(32, 32, 32, 32)
            Number of features after each layers.
            If len(decoder) is larger than the number of levels, additional
            stride-1 convolutions are applied.
        kernel_size : [sequence of] int, default=3
        activation : str or callable, default=LeakyReLU(0.2)
        unpool : {'conv', 'up', None}, default=None
                'conv' -> 2x2x2 strided convolution (no bias, no activation)
                'up'   -> linear upsampling
                 None  -> use strided convolutions in the decoder
        """
        super().__init__()
        shape = py.make_list(shape)
        dim = len(shape)
        small_shape = [s // 2**nb_levels for s in shape]
        in_feat, *decoder = decoder
        self.dense = Linear(in_channels, py.prod(small_shape)*in_feat)
        self.reshape = lambda x: x.reshape([-1, in_feat, *small_shape])
        decoder, stack = decoder[:nb_levels], decoder[nb_levels:]
        if decoder:
            self.decoder = Decoder(dim, in_feat, decoder,
                                   kernel_size=kernel_size,
                                   activation=activation,
                                   unpool=unpool)
            in_feat = decoder[-1]
        else:
            self.decoder = lambda x: x
        if stack:
            self.stack = StackedConv(dim, in_feat, stack,
                                     kernel_size=kernel_size,
                                     activation=activation)
            in_feat = stack[-1]
        else:
            self.stack = lambda x: x
        self.final = StackedConv(dim, in_feat, out_channels,
                                 kernel_size=kernel_size,
                                 activation=None)

    def forward(self, x):
        """

        Parameters
        ----------
        x : (batch, in_channels)
            Meta-vector

        Returns
        -------
        image : (batch, out_channels, *shape)
            Generated spatial tensor

        """
        x = self.dense(x)
        x = self.reshape(x)
        x = self.decoder(x)
        x = self.stack(x)
        x = self.final(x)
        return x


class _ConditionalAtlasMorph(AtlasMorph):
    """A conditional atlas is an atlas that depends on a set of
    (phenotypic) parameters. They can in theory be anything, but would
    typically be something like age, gender, or some biological state.

    In the original paper, the template was directly predicted from the
    phenotypic parameters (of a subject) and deformed to match the MR image
    of that same subject. A major inconvenient of this approach is that
    spatial correspondence between phenotypes is (mostly) lost. Here, we
    instead learn a common template (as in AtlasMorph) and a conditional
    "shape and appearance" morphing of this template. That is, the template
    for a given subject is the template for all subjects, plus an (additive)
    appearance field, with the whole thing being conditionnaly deformed.
    A _residual_ warp is then estimated to match the subject's MRI.
    We further make use of the approximation
            exp(v1) o exp(v2) \approx exp(v1 + v2),
    which means that we integrate the sum of the conditional and
    subject-specific velocities, and use the resulting transformation to
    warp the template.

    Furthermore, we could:
        - compose the two exponentiated fields
        - use a multiplicative appearance change, rather than an additive one
        - predict an additional rigid transform (on the subject side)
    But these options are not currently implemented.

    References
    ----------
    .. [1] "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MICCAI 2018. eprint arXiv:1805.04605
    .. [2] "Learning Conditional Deformable Templates with Convolutional Networks"
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        NeurIPS 2019. eprint arXiv:1908.02738
    """
    # WIP: This probably does not work
    # TODO:
    #   Deform the template using the average velocity across subjects every
    #   few epochs (and reset the mean velocity variable). I feel that
    #   this would give a nicer unbiased template than the running mean
    #   used by Adrian.

    def __init__(self, dim, dnet=None, unet=None, pull=None, exp=None, template=None):
        """

        Parameters
        ----------
        dim : int
            Dimensionality of the input (1|2|3)
        dnet : dict
            Dictionary of metadata decoding parameters with fields
                in_channels : int, default=1
                nb_levels : int, default=0
                decoder : sequence[int], default=[32, 32, 32]
                kernel_size : int, default=3
                activation : str or callable, default=LeakyReLU(0.2)
                unpool : {'conv', 'up', None}, default=None
                    'conv' -> 2x2x2 strided convolution (no bias, no activation)
                    'up'   -> linear upsampling
                     None  -> use strided convolutions in the decoder
        unet : dict
            Dictionary of U-Net parameters with fields:
                encoder : sequence[int], default=[16, 32, 32, 32, 32]
                decoder : sequence[int], default=[32, 32, 32, 32, 16, 16]
                conv_per_layer : int, default=1
                kernel_size : int, default=3
                activation : str or callable, default=LeakyReLU(0.2)
                pool : {'max', 'conv', 'down', None}, default=None
                    'max'  -> 2x2x2 max-pooling
                    'conv' -> 2x2x2 strided convolution (no bias, no activation)
                    'down' -> downsampling
                     None  -> use strided convolutions in the encoder
                unpool : {'conv', 'up', None}, default=None
                    'conv' -> 2x2x2 strided convolution (no bias, no activation)
                    'up'   -> linear upsampling
                     None  -> use strided convolutions in the decoder
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
        template : dict
            Dictionary of Template parameters with fields:
                shape : tuple[int], default=(192,) * dim
                mom : float or int, default=100
                    If in (0, 1), momentum of the running mean.
                    The mean is updated according to:
                        `new_mean = (1-mom) * old_mean + mom * new_sample`
                    If 0, use cumulative average:
                        `new_n = old_n + 1`
                        `mom = 1/new_n`
                    If > 1, cap the weight of a new sample in the average:
                        `new_n = min(cap, old_n + 1)`
                        `mom = 1/new_n`
                cat : int, default=False
                    If 0:
                        Build an intensity template
                    If > 0:
                        Build a categorical template with this number of classes.
                implicit : bool, default=True
                    Whether the template has an implicit background class.

        """
        # default parameters
        dnet = dict(dnet or {})
        dnet.setdefault('in_channels', 1)
        dnet.setdefault('nb_levels', 0)
        dnet.setdefault('decoder', [32, 32, 32, 32])
        dnet.setdefault('kernel_size', 3)
        dnet.setdefault('unpool', None)
        dnet.setdefault('activation', tnn.LeakyReLU(0.2))
        super().__init__(dim, unet, pull, exp, template)

        shape = template.get('shape', [192]*dim)
        out_channels = template.get('cat', [192]*dim) or 1
        out_channels = out_channels + dim
        self.decoder = _MetaToImage(shape, out_channels=out_channels, **dnet)

    def softmax(self, template):
        if not self.pull.extrapolate:
            msk = (template == 0).all(dim=1, keepdim=True)
        template = SoftMax(implicit=self.implicit)(template)
        if not self.pull.extrapolate:
            # we can't just let out-of-bound values to be just zero, as
            # it makes the probability be equi-probable. Instead, We want
            # the background class to have probability one.
            template[-self.cat:][msk] = 1e-5
            if not self.implicit:
                template[self.cat:][msk] = 1 - (self.cat*1e-5)
        return template

    def forward(self, target, meta, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        target : (batch, in_channels, *spatial) tensor
            Target/fixed image
        meta : (batch, in_feat) tensor
            Metadata

        Other Parameters
        ----------------
        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_template : tensor (batch, 1|classes, *spatial)
            Deformed template
        velocity : tensor (batch, *spatial, len(spatial))
            Velocity field

        """
        # sanity checks
        check.dim(self.dim, self.template[None], target)
        check.shape(target, self.template[None], dims=range(2, self.dim + 2))

        # generate template
        template_action = self.decoder(meta)
        velocity0 = template_action[:, :self.dim]
        template0 = template_action[:, self.dim:]
        template0 += self.template
        velocity0 = core.utils.channel2last(velocity0)
        grid0 = self.exp(velocity0)
        template = self.pull(template0, grid0)
        if self.cat:
            template = self.softmax(template)

        # chain operations
        source_and_target = torch.cat((template, target), dim=1)
        del template0
        velocity = self.unet(source_and_target)
        del source_and_target
        velocity = core.utils.channel2last(velocity)
        velocity += velocity0
        del velocity0
        grid = self.exp(velocity)
        template0 = self.pull(template0, grid)
        if self.cat:
            deformed_template = self.softmax(template0)

        # running mean
        if self.training:
            self.update_mean(velocity)

        # compute loss and metrics
        losses = dict(velocity=[velocity])
        if self.training:
            losses['mean'] = [self.mean]
            losses['template'] = [template0]
        losses['match'] = [deformed_template, target]
        self.compute(_loss, _metric, **losses)

        return template0, velocity


def registration_board(
        self, tb,
        inputs=None, outputs=None, epoch=None, minibatch=None, mode=None,
        implicit=False, do_eval=True, do_train=True, **kwargs):
    """Plug-and-play tensorboard method for registration networks

    Parameters
    ----------
    self : Module
    tb : SummaryWriter
    inputs : tuple of tensor
        (source, target, [source_seg, target_seg])
    outputs : tuple of tensor
        (deformed_source, [deformed_source_seg], velocity)
    epoch : int
        Index of current epoch
    minibatch : int
        Index of current minibatch
    mode : {'train', 'eval'}
    implicit : bool, default=False
        Does the deformed segmentation have an implicit class?
    do_eval : bool, default=True
    do_train : bool, default=True
    kwargs : dict

    """
    if torch.is_grad_enabled():
        # run without gradients
        with torch.no_grad():
            return registration_board(self, tb, inputs, outputs, epoch,
                                      minibatch, mode, implicit, do_eval,
                                      do_train, **kwargs)

    if ((not do_eval and mode == 'eval') or
        (not do_train and mode == 'train') or
        inputs is None):
        return

    from nitorch.plot import get_orthogonal_slices, get_slice
    from nitorch.plot.colormaps import prob_to_rgb, intensity_to_rgb, disp_to_rgb
    import matplotlib.pyplot as plt

    def get_slice_seg(x):
        """Get slice + convert to probabilities (one-hot) if needed."""
        if x.dtype in (torch.float, torch.double):
            x = get_slice(x)
        else:
            x = get_slice(x[0])
            x = torch.stack([x == i for i in range(1, x.max().item() + 1)])
            x = x.float()
        return x

    def get_orthogonal_slices_seg(x):
        """Get slices + convert to probabilities (one-hot) if needed."""
        if x.dtype in (torch.float, torch.double):
            xs = get_orthogonal_slices(x)
        else:
            xs = get_orthogonal_slices(x[0])
            xs = [torch.stack([x == i for i in range(1, x.max().item() + 1)])
                  for x in xs]
            xs = [x.float() for x in xs]
        return xs

    def prepare(x):
        is_seg = x.shape[1] > 1
        if x.dim() - 2 == 2:  # 2d
            if is_seg:
                nk = x.shape[1] + implicit
                x = get_slice_seg(x[0])
                x = prob_to_rgb(x, implicit=x.shape[0] < nk)
            else:
                x = get_slice(x[0, 0])
                x = intensity_to_rgb(x)
            x = x.clip(0, 1)
        else:  # 3d
            if is_seg:
                nk = x.shape[1] + implicit
                x = get_orthogonal_slices_seg(x[0])
                x = [prob_to_rgb(y, implicit=y.shape[0] < nk) for y in x]
            else:
                x = get_orthogonal_slices(x[0, 0])
                x = [intensity_to_rgb(y) for y in x]
            x = [y.clip(0, 1) for y in x]
        return x

    *outputs, vel = outputs
    images = [*inputs, *outputs]
    is2d = inputs[0].dim() - 2 == 2

    fig = plt.figure()
    if is2d:  # 2D
        ncol = len(images)
        nrow = 1
        for i, image in enumerate(images):
            image = prepare(image)
            plt.subplot(nrow, ncol, i+1)
            plt.imshow(image.detach().cpu())
            plt.axis('off')
    else:  # 3D
        ncol = len(images)
        nrow = 3
        for i, image in enumerate(images):
            image = prepare(image)
            for j in range(3):
                plt.subplot(nrow, ncol, i + j*ncol + 1)
                plt.imshow(image[j].detach().cpu())
                plt.axis('off')
    plt.tight_layout()

    if not hasattr(self, 'tbstep'):
        self.tbstep = dict()
    self.tbstep.setdefault(mode, 0)
    self.tbstep[mode] += 1
    tb.add_figure(f'warps/{mode}', fig, global_step=self.tbstep[mode])

    fig = plt.figure()
    if is2d:
        vel = get_slice(utils.movedim(vel[0], -1, 0))
        vel = disp_to_rgb(vel, amplitude='saturation')
        plt.imshow(vel.detach().cpu())
        plt.axis('off')
    else:
        vel = get_orthogonal_slices(utils.movedim(vel[0], -1, 0))
#         vel = [disp_to_rgb(v, amplitude='saturation') for v in vel]
#         plt.subplot(1, 3, 1)
#         plt.imshow(vel[0].detach().cpu())
#         plt.axis('off')
#         plt.subplot(1, 3, 2)
#         plt.imshow(vel[1].detach().cpu())
#         plt.axis('off')
#         plt.subplot(1, 3, 3)
#         plt.imshow(vel[2].detach().cpu())
#         plt.axis('off')
        for i in range(3):
            for j in range(3):
                plt.subplot(3, 3, 1+j+i*3)
                plt.imshow(vel[j][i].detach().cpu())
                plt.colorbar()
                plt.axis('off')
    plt.tight_layout()

    tb.add_figure(f'vel/{mode}', fig, global_step=self.tbstep[mode])

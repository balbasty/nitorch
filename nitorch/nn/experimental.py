"""Experimental models -> may not work well (or work at all)."""

import torch
import torch.nn as tnn
from nitorch.nn.modules._base import Module
from nitorch.nn.modules._cnn import UNet
from nitorch.nn.modules._spatial import GridPull, GridPush, GridExp
from .. import spatial
from ..core.utils import broadcast_to
from ..core.pyutils import make_list


class VoxelMorphSymmetric(Module):
    """VoxelMorph network with a symmetric loss.

    Contrary to what's done in voxelmorph, I predict a midpoint image
    and warp it to both native spaces.

    NOTE:
        It doesn't seem to work very well for pairs of images. There's
        just two much of each in the midpoint, and the deformation
        just tries to squeeze values it doesn't want out.
    """

    def __init__(self, dim, encoder=None, decoder=None, kernel_size=3,
                 interpolation='linear', grid_bound='dft', image_bound='dct2'):
        super().__init__()
        self.unet = UNet(dim,
                         input_channels=2,
                         output_channels=dim+1,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.exp = GridExp(fwd=True, inv=True,
                           interpolation=interpolation,
                           bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim

    def forward(self, source, target):
        # checks
        if len(source.shape) != self.dim+2:
            raise ValueError('Expected `source` to have shape (B, C, *spatial) '
                             'with len(spatial) == {} but found {}.'
                             .format(self.dim, source.shape))
        if len(target.shape) != self.dim+2:
            raise ValueError('Expected `target` to have shape (B, C, *spatial) '
                             'with len(spatial) == {} but found {}.'
                             .format(self.dim, target.shape))
        if not (target.shape[0] == source.shape[0] or
                target.shape[0] == 1 or source.shape[0] == 1):
            raise ValueError('Batch dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[0], target.shape[0]))
        if target.shape[2:] != source.shape[2:]:
            raise ValueError('Spatial dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[2:], target.shape[2:]))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        velocity_and_template = self.unet(source_and_target)
        template = velocity_and_template[:, -1:, ...]
        velocity = velocity_and_template[:, :-1, ...]
        velocity = spatial.channel2grid(velocity)
        grid, igrid = self.exp(velocity)
        deformed_to_source = self.pull(template, grid)
        deformed_to_target = self.pull(template, igrid)

        return deformed_to_source, deformed_to_target, velocity, template


class VoxelMorphPlus(Module):
    """A VoxelMorph network augmented with a morphing field.
    """

    def __init__(self, dim, encoder=None, decoder=None, kernel_size=3,
                 interpolation='linear', grid_bound='dft', image_bound='dct2'):
        super().__init__()
        self.unet = UNet(dim,
                         input_channels=2,
                         output_channels=dim+1,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.exp = GridExp(interpolation=interpolation,
                           bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim

    def forward(self, source, target):
        # checks
        if len(source.shape) != self.dim+2:
            raise ValueError('Expected `source` to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(self.dim, source.shape))
        if len(target.shape) != self.dim+2:
            raise ValueError('Expected `target` to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(self.dim, target.shape))
        if not (target.shape[0] == source.shape[0] or
                target.shape[0] == 1 or source.shape[0] == 1):
            raise ValueError('Batch dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[0], target.shape[0]))
        if target.shape[2:] != source.shape[2:]:
            raise ValueError('Spatial dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[2:], target.shape[2:]))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        velocity_and_morph = self.unet(source_and_target)
        morph = velocity_and_morph[:, -1:, ...]
        velocity = velocity_and_morph[:, :-1, ...]
        velocity = spatial.channel2grid(velocity)
        grid = self.exp(velocity)
        deformed_source = self.pull(source+morph, grid)

        return deformed_source, velocity, morph


class DiffeoMovie(Module):
    """Compute the deformation at intermediate time steps.

    The output tensor has time steps in the channel dimension, which
    can be used as frames in an animation.
    """

    def __init__(self, nb_frames=100, interpolation='linear',
                 grid_bound='dft', image_bound='dct2'):

        super().__init__()
        self.nb_frames = nb_frames
        self.exp = GridExp(interpolation=interpolation,
                           bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)

    def forward(self, image, velocity):

        if image.shape[1] != 1:
            raise ValueError('DiffeoMovie only accepts single channel '
                             'images (for now).')
        scale = torch.linspace(0, 1, self.nb_frames)
        frames = []
        for s in scale:
            grid = self.exp(velocity * s)
            frames.append(self.pull(image, grid))
        frames = torch.cat(frames, dim=1)

        return frames


class IterativeVoxelMorph(Module):
    """VoxelMorph warps a source/moving image to a fixed/target image.

    This network aims to bridge the gap between UNets and iterative
    optimization methods. The network is 'residual' (it predicts the
    difference between the previous velocity estimate and the optimal
    one), 'recurrent' (there is a core voxelmorph network that is
    applied iteratively) and 'adjoint' (since a spatial transformer T is
    applied after the UNet, its adjoint T' is applied before)
    """

    def __init__(self, dim, max_iter=None, encoder=None, decoder=None, kernel_size=3,
                 interpolation='linear', grid_bound='dft', image_bound='dct2'):
        """

        Parameters
        ----------
        dim : int
            Dimensionalityy of the input (1|2|3)
        max_iter : int or [int, int], default=[1, 20]
            Number of iterations of the recurrent network.
            If list, max_iter is randomly sampled in the corresponding
            range at each minibatch.
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
        """
        super().__init__()
        self.push = GridPush(interpolation=interpolation,
                             bound=image_bound)
        self.unet = UNet(dim,
                         input_channels=3,
                         output_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.exp = GridExp(interpolation=interpolation,
                           bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim
        if max_iter is None:
            max_iter = [1, 20]
        if isinstance(max_iter, range):
            inf = max_iter.start if max_iter.start else 1
            sup = max_iter.stop if max_iter.stop else max(1, inf)
            max_iter = [inf, sup]
        max_iter = make_list(max_iter, 2)
        if max_iter[0] == max_iter[1]:
            max_iter = max_iter[0]
        self.max_iter = max_iter
        self.image_losses = []
        self.velocity_losses = []
        self.image_metrics = {}
        self.velocity_metrics = {}

    def add_image_loss(self, *loss_fn):
        """Add one or more image loss functions.

        The image loss should measure similarity between the deformed
        source and target images.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_losses += list(loss_fn)

    def set_image_loss(self, *loss_fn):
        """Set one or more image loss functions.

        The image loss should measure similarity between the deformed
        source and target images.

        This function discards all previously held image losses.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_losses = list(loss_fn)

    def add_velocity_loss(self, *loss_fn):
        """Add one or more velocity loss functions.

        The velocity loss should penalize features of the velocity field.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_losses += list(loss_fn)

    def set_velocity_loss(self, *loss_fn):
        """Set one or more image loss functions.

        The velocity loss should penalize features of the velocity field.

        This function discards all previously held velocity losses.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_losses = list(loss_fn)

    def compute_loss(self, deformed_source, target, velocity):
        """Compute all losses."""
        loss = []
        for loss_fn in self.image_losses:
            if isinstance(loss_fn, (list, tuple)):
                _loss_fn, weight = loss_fn
                loss_fn = lambda *a, **k: weight*_loss_fn(*a, **k)
            loss.append(loss_fn(deformed_source, target))
        for loss_fn in self.velocity_losses:
            if isinstance(loss_fn, (list, tuple)):
                _loss_fn, weight = loss_fn
                loss_fn = lambda *a, **k: weight*_loss_fn(*a, **k)
            loss.append(loss_fn(velocity))
        return loss

    def add_image_metric(self, **metric_fn):
        """Add one or more image metric functions.

        The image metric should measure similarity between the deformed
        source and target images.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_metrics.update(dict(metric_fn))

    def set_image_metric(self, **metric_fn):
        """Set one or more image metric functions.

        The image metric should measure similarity between the deformed
        source and target images.

        This function discards all previously held image metrics.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_metrics = dict(metric_fn)

    def add_velocity_metric(self, **metric_fn):
        """Add one or more velocity metric functions.

        The velocity metric should penalize features of the velocity field.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_metrics.update(dict(metric_fn))

    def set_velocity_metric(self, **metric_fn):
        """Set one or more image metric functions.

        The velocity metric should penalize features of the velocity field.

        This function discards all previously held velocity metrics.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_metrics = dict(metric_fn)

    def compute_metric(self, deformed_source, target, velocity):
        """Compute all metrics."""
        metric = {}
        for key, metric_fn in self.image_metrics.items():
            key = '{}/{}/{}'.format(self.__class__.__name__, 'image', key)
            metric[key] = metric_fn(deformed_source, target)
        for key, metric_fn in self.velocity_metrics.items():
            key = '{}/{}/{}'.format(self.__class__.__name__, 'velocity', key)
            metric[key] = metric_fn(velocity)
        return metric

    def forward(self, source, target, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : list, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        velocity : tensor (batch,, *spatial, len(spatial))
            Velocity field

        """
        lm = {'_loss': _loss, '_metric': _metric}

        # checks
        if len(source.shape) != self.dim+2:
            raise ValueError('Expected `source` to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(self.dim, source.shape))
        if len(target.shape) != self.dim+2:
            raise ValueError('Expected `target` to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(self.dim, target.shape))
        if not (target.shape[0] == source.shape[0] or
                target.shape[0] == 1 or source.shape[0] == 1):
            raise ValueError('Batch dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[0], target.shape[0]))
        if target.shape[2:] != source.shape[2:]:
            raise ValueError('Spatial dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[2:], target.shape[2:]))

        # parameters
        batch = source.shape[0]
        channels = source.shape[1]
        shape = source.shape[2:]
        dtype = source.dtype
        device = source.device
        max_iter = self.max_iter
        if isinstance(max_iter, list):
            # random number of iterations
            max_iter = torch.randint(max_iter[0], max_iter[1], [],
                                     dtype=torch.long,
                                     device=torch.device('cpu')).item()

        # initialize
        ones = torch.ones((1, 1, *shape), dtype=dtype, device=device)
        velocity = 0
        grid = spatial.identity_grid(shape, dtype=dtype, device=device)
        grid = grid[None, ...]

        # chain operations
        for n_iter in range(max_iter):
            p_target = self.push(target, grid)
            pp_source = self.push(self.pull(source, grid), grid)
            pp_ones = self.push(self.pull(ones, grid), grid)
            pp_ones = broadcast_to(pp_ones, [batch, 1, *shape])
            source_and_target = torch.cat((pp_source, p_target, pp_ones), dim=1)
            increment = self.unet(source_and_target, **lm)
            increment = spatial.channel2grid(increment)
            velocity = velocity + increment
            grid = self.exp(velocity)
            deformed_source = self.pull(source, grid)

        # compute loss and metrics
        if _loss is not None:
            assert isinstance(_loss, list)
            _loss += self.compute_loss(deformed_source, target, velocity)
        if _metric is not None:
            assert isinstance(_metric, dict)
            metrics = self.compute_metric(deformed_source, target, velocity)
            self.update_metrics(_metric, metrics)

        return deformed_source, velocity

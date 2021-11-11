import torch
import torch.nn as tnn
from nitorch.nn.base import Module
from ..modules.cnn import UNet
from ..modules.spatial import GridPull, GridPush, GridExp
from ..modules.registration import VoxelMorph
from nitorch import spatial
from nitorch.core.utils import expand
from nitorch.core.py import make_list


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


class IterativeVoxelMorph(VoxelMorph):
    """VoxelMorph warps a source/moving image to a fixed/target image.

    This network aims to bridge the gap between UNets and iterative
    optimization methods. The network is 'residual' (it predicts the
    difference between the previous velocity estimate and the optimal
    one), 'recurrent' (there is a core voxelmorph network that is
    applied iteratively) and 'adjoint' (since a spatial transformer T is
    applied after the UNet, its adjoint T' is applied before)
    """

    def __init__(self, dim, max_iter=None, residual=True, feed=None,
                 *args, **kwargs):
        """

        Parameters
        ----------
        dim : int
            Dimensionality of the input (1|2|3)
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
        if feed is None:
            # possible feeds
            # target, push(target), source, push(pull(source)),
            # push(ones), push(pull(ones)), velocity
            feed = ['push(target)', 'source', 'velocity']
        input_channels = sum(dim if f == 'velocity' else 1 for f in feed)
        super().__init__(dim, *args, **kwargs, _input_channels=input_channels)
        self.feed = feed
        self.residual = residual
        self.push = GridPush(interpolation=self.pull.interpolation,
                             bound=self.pull.bound)
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

    def forward(self, source, target, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : dict, optional
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
        velocity = torch.zeros((), dtype=dtype, device=device)
        grid = spatial.identity_grid(shape, dtype=dtype, device=device)
        grid = grid[None, ...]

        # chain operations
        for n_iter in range(max_iter):
            # broadcast velocity just in case
            velocity = expand(velocity, [batch, *shape, self.dim])

            # concatenate inputs to the UNet
            input_unet = []
            for f in self.feed:
                if f == 'source':
                    input_unet += [source]
                elif f == 'pull(source)':
                    input_unet += [self.pull(source, grid)]
                elif f == 'push(pull(source))':
                    input_unet += [self.push(self.pull(source, grid), grid)]
                elif f == 'target':
                    input_unet += [target]
                elif f == 'push(target)':
                    input_unet += [self.push(target, grid)]
                elif f == 'push(pull(ones))':
                    c = self.push(self.pull(ones, grid), grid)
                    c = expand(c, [batch, 1, *shape])
                    input_unet += [c]
                elif f == 'push(ones)':
                    c = self.push(ones, grid)
                    c = expand(c, [batch, 1, *shape])
                    input_unet += [c]
                elif f == 'velocity':
                    input_unet += [spatial.grid2channel(velocity)]
                else:
                    raise ValueError('Unknown feed tensor {}'.format(f))
            input_unet = torch.cat(input_unet, dim=1)

            # increment velocity
            increment = self.unet(input_unet, **lm)
            increment = spatial.channel2grid(increment)
            if self.residual:
                velocity = velocity + increment
            else:
                velocity = increment
            velocity_small = self.resize(velocity)
            grid = self.exp(velocity_small)
            grid = self.resize(grid, shape=target.shape[2:])
            deformed_source = self.pull(source, grid)

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     velocity=[velocity])

        return deformed_source, velocity

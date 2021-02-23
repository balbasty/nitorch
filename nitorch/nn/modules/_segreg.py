from ._base import Module
from ._cnn import UNet
from ._spatial import GridResize, GridExp, GridPull
from nitorch.nn.activations import SoftMax
from nitorch.nn import check
from nitorch.core import py, utils
from nitorch import spatial
import torch


class SegMorphUNet(Module):
    """
    Joint segmentation and registration using a dual-branch UNet.

    The UNet outputs both a velocity field and two native-space segmentations.
    One loss function acts on the native-space segmentations and tries to
    make them as accurate as possible (in the supervised case) and another
    loss function acts on the warped moving segmentation and tries to
    make it match the fixed-space segmentation (in the supervised or
    semi-supervised case).
    """

    def __init__(self, dim, output_classes=1, encoder=None, decoder=None,
                 kernel_size=3, activation=torch.nn.LeakyReLU(0.2),
                 interpolation='linear', grid_bound='dft', image_bound='dct2',
                 downsample_velocity=2, batch_norm=True, implicit=True):
        """

        Parameters
        ----------
        dim : int
            Number of spatial dimensions
        output_classes : int
            Number of classes in the segmentation (excluding background)
        encoder : sequence[int]
            Number of output channels in each encoding layer
        decoder : sequence[int]
            Number of output channels in each decoding layer
        kernel_size : int, default=3
        activation : callable, default=LeakyReLU(0.2)
        interpolation : str, default='linear'
        grid_bound : str, default='dft'
        image_bound : str, default='dct2'
        downsample_velocity : int, default=2
        batch_norm : bool, default=True
        implicit : bool or (bool, bool), default=True
            If the first element is True, the UNet only outputs `output_classes`
            channels (i.e., it does not model the background). The missing
            channels is assumed all zero when performing the softmax.
            If the second element is True, the network only outputs
            `output_classes` channels (i.e., the background is implicit).
            Using `implicit=True` extends the behaviour of a Sigmoid
            activation to the multi-class case.
        """
        super().__init__()

        resize_factor = py.make_list(downsample_velocity, dim)
        resize_factor = [1/f for f in resize_factor]

        self.implicit = py.make_list(implicit, 2)
        self.output_classes = output_classes
        if not self.implicit[0]:
            output_classes = output_classes + 1
        self.softmax = SoftMax(implicit=implicit)

        self.unet = UNet(dim,
                         input_channels=2,
                         output_channels=output_classes*2 + dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=activation,
                         final_activation=None,
                         batch_norm=batch_norm)

        self.resize = GridResize(interpolation=interpolation,
                                 bound=grid_bound,
                                 factor=resize_factor)
        self.velexp = GridExp(interpolation=interpolation,
                              bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'velocity', 'segmentation', 'source', 'target']

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
        target_seg_pred : tensor (batch, classes, *spatial), optional
            Predicted target segmentation
        source_seg_pred : tensor (batch, classes, *spatial), optional
            Predicted source segmentation
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        velocity : tensor (batch,, *spatial, len(spatial))
            Velocity field

        """
        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim+2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim+2))

        # unet
        source_and_target = torch.cat((source, target), dim=1)
        if (~torch.isfinite(source_and_target)).any():
            raise RuntimeError('UNet input NaNs')
        velocity_and_seg = self.unet(source_and_target)
        if (~torch.isfinite(velocity_and_seg)).any():
            raise RuntimeError('UNet output NaNs')
        del source_and_target
        velocity = velocity_and_seg[:, :self.dim]
        output_classes = self.output_classes + (not self.implicit[0])
        target_seg_pred = velocity_and_seg[:, self.dim:self.dim + output_classes]
        source_seg_pred = velocity_and_seg[:, self.dim + output_classes:]
        del velocity_and_seg

        # sigmoid
        target_seg_pred = self.softmax(target_seg_pred)
        if (~torch.isfinite(target_seg_pred)).any():
            raise RuntimeError('Softmax (target) output NaNs')
        source_seg_pred = self.softmax(source_seg_pred)
        if (~torch.isfinite(source_seg_pred)).any():
            raise RuntimeError('Softmax (source) output NaNs')

        # deformation
        velocity = utils.channel2last(velocity)
        grid = self.exp(velocity)
        if (~torch.isfinite(grid)).any():
            raise RuntimeError('Exp (vel) output NaNs')
        deformed_source = self.pull(source, grid)
        deformed_source_seg_pred = self.pull(source_seg_pred, grid)

        # compute loss and metrics
        tensors = dict(
             image=[deformed_source, target],
             velocity=[velocity],
             segmentation=[deformed_source_seg_pred, target_seg_pred])
        if source_seg is not None:
            tensors['source'] = [source_seg_pred, source_seg]
        if target_seg is not None:
            tensors['target'] = [target_seg_pred, target_seg]
        self.compute(_loss, _metric, **tensors)
        
        return source_seg_pred, target_seg_pred, deformed_source, velocity

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

    def board(self, tb, inputs, outputs, epoch=None, minibatch=None, **k):
        """TensorBoard visualisation of a segmentation model's inputs and outputs.

        Parameters
        ----------
        tb : torch.utils.tensorboard.writer.SummaryWriter
            TensorBoard writer object.
        inputs : (tensor, tensor, [tensor], [tensor])
            Input source and target images (N, C, *spatial) their
            ground truth segmentations (N, K|1, *spatial).
        outputs : (tensor, tensor, tensor, tensor)
            Predicted source and target segmentations (N, K[-1], *spatial),
            deformed source image (N, C, *spatial) and velocity field
            (N, *spatial, dim).
        """
        if minibatch != 0:
            return

        dim = self.dim
        implicit = self.implicit[1]

        def get_slice(plane, vol):
            if plane == 'z':
                z = round(0.5 * vol.shape[-1])
                slice = vol[..., z]
            elif plane == 'y':
                y = round(0.5 * vol.shape[-2])
                slice = vol[..., y, :]
            elif plane == 'x':
                x = round(0.5 * vol.shape[-3])
                slice = vol[..., x, :, :]
            else:
                assert False
            slice = slice.squeeze()
            return slice

        def get_image(plane, vol):
            vol = get_slice(plane, vol)
            return vol

        def get_label(plane, vol):
            vol = get_slice(plane, vol)
            if vol.dim() == 2:
                vol = vol.float()
                vol /= vol.max()
            else:
                nb_classes = vol.shape[0]
                if implicit:
                    background = vol.sum(dim=0, keepdim=True).neg().add(1)
                    vol = torch.cat((vol, background), dim=0)
                vol = vol.argmax(dim=0)
                vol += 1
                vol[vol == nb_classes+1] = 0
                vol = vol.float()
                vol /= float(nb_classes)
            return vol

        # unpack
        source, target, *seg = inputs
        source_seg = target_seg = None
        if seg:
            source_seg, *seg = seg
        if seg:
            target_seg, *seg = seg
        source_pred, target_pred, source_warped, velocity = outputs

        planes = ['z'] + (['y', 'x'] if dim == 3 else [])
        title = 'Image-Target-Prediction_'
        for plane in planes:
            slices = []
            slices += [get_image(plane, source)]
            if source_seg is not None:
                slices += [get_label(plane, source_seg)]
            slices += [get_label(plane, source_pred)]
            slices += [get_image(plane, target)]
            if target_seg is not None:
                slices += [get_label(plane, target_seg)]
            slices += [get_label(plane, target_pred)]
            slices += [get_image(plane, source_warped)]
            slices = torch.cat(slices, dim=1)
            tb.add_image(title + plane, slices[None], global_step=epoch)
        tb.flush()

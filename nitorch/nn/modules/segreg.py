from .base import Module
from .cnn import UNet, UUNet
from .spatial import GridResize, GridExp, GridPull
from nitorch.nn.activations import SoftMax
from nitorch.nn import check
from nitorch.core import py, utils
from nitorch import spatial
import torch


class BaseMorph(Module):
    """Secondary base class for model that implement a morph component"""

    def __init__(self, dim, interpolation='linear', grid_bound='dft',
                 image_bound='dct2', downsample_velocity=2, ):
        super().__init__()
        
        resize_factor = py.make_list(downsample_velocity, dim)
        resize_factor = [1/f for f in resize_factor]

        self.resize = GridResize(interpolation=interpolation,
                                 bound=grid_bound,
                                 factor=resize_factor)
        self.velexp = GridExp(interpolation=interpolation,
                              bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim

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

    def board(self, tb, inputs=None, outputs=None, epoch=None, minibatch=None,
              mode=None, **k):
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
        if minibatch is None:
            return
        if mode == 'train' and minibatch != 0:
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
            slice = slice.squeeze().detach().cpu()
            return slice

        def get_image(plane, vol, batch=0):
            vol = get_slice(plane, vol[batch])
            return vol

        def get_velocity(plane, vol, batch=0):
            vol = vol.square().sum(-1).sqrt()
            vol = get_slice(plane, vol[batch])
            return vol

        def get_label(plane, vol, batch=0):
            vol = get_slice(plane, vol[batch])
            if vol.dim() == 2:
                vol = vol.float()
                vol /= vol.max()
            else:
                if implicit:
                    background = 1 - vol.sum(dim=0, keepdim=True)
                    vol = torch.cat((vol, background), dim=0)
                nb_classes = vol.shape[0]
                vol = vol.argmax(dim=0)
                vol += 1
                vol[vol == nb_classes] = 0
                vol = vol.float()
                vol /= float(nb_classes - 1)
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
        title = f'{mode}/Image-Target-Prediction'

        def plot_slices():
            import matplotlib.pyplot as plt
            nplanes = len(planes)
            nbatch = len(source)
            nrow = nplanes * nbatch
            ncol = 6 + int(source_seg is not None) + int(target_seg is not None)
            fig = plt.figure(figsize=(2 * ncol, 2 * nrow),
                             tight_layout={'w_pad': 0, 'h_pad': 0, 'pad': 0})
            idx = 0
            prm = dict(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
            for b in range(nbatch):
                for p, plane in enumerate(planes):
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_image(plane, source, b))
                    if source_seg is not None:
                        idx += 1
                        ax = fig.add_subplot(nrow, ncol, idx, **prm)
                        ax.imshow(get_label(plane, source_seg, b))
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_label(plane, source_pred, b))
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_image(plane, target, b))
                    if target_seg is not None:
                        idx += 1
                        ax = fig.add_subplot(nrow, ncol, idx, **prm)
                        ax.imshow(get_label(plane, target_seg, b))
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_label(plane, target_pred, b))
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_image(plane, source_warped, b))
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_velocity(plane, velocity, b))
            fig.subplots_adjust(wspace=0, hspace=0)
            return fig

        if not hasattr(self, 'tbstep'):
            self.tbstep = dict()
        self.tbstep.setdefault(mode, 0)
        self.tbstep[mode] += 1
        tb.add_figure(title, plot_slices(), global_step=self.tbstep[mode])

        
def _is_bn(layer):
    return isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)


class SegMorphWNet(BaseMorph):
    """
    Joint segmentation and registration using a WNet.

    A first UNet is applied to both input images and outputs a native space
    segmentation for each (with a loss that tries to make them as accurate
    as possible).
    The second UNet is applied to all input images and predicted segmentations,
    with a loss function acts on the warped moving segmentation and tries to
    make it match the fixed-space segmentation.
    """

    def __init__(self, dim, output_classes=1, encoder=None, decoder=None,
                 kernel_size=3, activation=torch.nn.LeakyReLU(0.2),
                 interpolation='linear', grid_bound='dft', image_bound='dct2',
                 downsample_velocity=2, batch_norm=True, implicit=True,
                 unet_inputs='image+seg'):
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
        unet_inputs : {'seg', 'image', 'image+seg'}, default='image+seg'
        """
        super().__init__(
            dim,
            interpolation,
            grid_bound,
            image_bound,
            downsample_velocity,
        )

        self.unet_inputs = unet_inputs
        self.implicit = py.make_list(implicit, 2)
        self.output_classes = output_classes
        self.softmax = SoftMax(implicit=implicit)

        output_channels = output_classes + int(not self.implicit[0])
        self.segnet = UNet(dim,
                           input_channels=1,
                           output_channels=output_channels,
                           encoder=encoder,
                           decoder=decoder,
                           kernel_size=kernel_size,
                           activation=[activation, ..., self.softmax],
                           batch_norm=batch_norm)

        input_channels = int('image' in unet_inputs) \
                        + int('seg' in unet_inputs) \
                        * (output_classes + int(not self.implicit[1]))
        self.unet = UNet(dim,
                         input_channels=input_channels * 2,
                         output_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=[activation, ..., None],
                         batch_norm=batch_norm)
        
#         self.bn_source = self.get_bn()
#         self.bn_target = self.get_bn()
        
        # register losses/metrics
        self.tags = ['image', 'velocity', 'segmentation', 'source', 'target']

    def set_bn(self, values, backtrack=False):
        bnlayers = [child for child in self.segnet.modules() if _is_bn(child)]
        for layer, mean, var, mom in zip(bnlayers, values['mean'], values['var'], values['momentum']):
            layer.running_mean = mean.to(device=layer.running_mean.device)
            layer.running_var = var.to(device=layer.running_var.device)
            layer.momentum = mom
            if backtrack:
                layer.num_batches_tracked -= 1
    
    def get_bn(self):
        values = dict(mean=[], var=[], momentum=[])
        bnlayers = [child for child in self.segnet.modules() if _is_bn(child)]
        for layer in bnlayers:
            values['mean'].append(layer.running_mean)
            values['var'].append(layer.running_var)
            values['momentum'].append(layer.momentum)
        return values
        
        
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

        # segnet
        source_seg_pred = self.segnet(source)
        target_seg_pred = self.segnet(target)
        
        # unet
        inputs = []
        if 'seg' in self.unet_inputs:
            inputs += [source_seg_pred, target_seg_pred]
        if 'image' in self.unet_inputs:
            inputs += [source, target]
        inputs = torch.cat(inputs, dim=1)

        # deformation
        velocity = self.unet(inputs)
        velocity = utils.channel2last(velocity)
        grid = self.exp(velocity)
        deformed_source = self.pull(source, grid)
        
        if source_seg is not None:
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = self.pull(source_seg_pred, grid)
        if target_seg is None:
            target_seg_for_deformed = target_seg_pred
        else:
            target_seg_for_deformed = target_seg
            
        # compute loss and metrics
        tensors = dict(
             image=[deformed_source, target],
             velocity=[velocity],
             segmentation=[deformed_source_seg, target_seg_for_deformed])
        if source_seg is not None:
            tensors['source'] = [source_seg_pred, source_seg]
        if target_seg is not None:
            tensors['target'] = [target_seg_pred, target_seg]
        self.compute(_loss, _metric, **tensors)
        
        return source_seg_pred, target_seg_pred, deformed_source, velocity


class SegMorphUNet(BaseMorph):
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
                 downsample_velocity=2, batch_norm=True, implicit=True,
                 groups=None, stitch=1):
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
        super().__init__(
            dim,
            interpolation,
            grid_bound,
            image_bound,
            downsample_velocity,
        )

        self.implicit = py.make_list(implicit, 2)
        self.output_classes = output_classes
        if not self.implicit[0]:
            output_classes = output_classes + 1
        self.softmax = SoftMax(implicit=implicit)

        groups = py.make_list(groups)
        stitch = py.make_list(stitch)
        if (groups[-1] or stitch[-1]) == 2:
            output_channels = [output_classes * 2, dim]
        else:
            output_channels = output_classes * 2 + dim
        self.unet = UNet(dim,
                         input_channels=2,
                         output_channels=output_channels,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=[activation, ..., None],
                         batch_norm=batch_norm,
                         groups=groups,
                         stitch=stitch)

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
        check.shape(target, source, dims=range(2, self.dim + 2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim + 2))

        # unet
        source_and_target = torch.cat((source, target), dim=1)
        velocity_and_seg = self.unet(source_and_target)
        del source_and_target
        velocity = velocity_and_seg[:, :self.dim]
        output_classes = self.output_classes + (not self.implicit[0])
        target_seg_pred = velocity_and_seg[:,
                          self.dim:self.dim + output_classes]
        source_seg_pred = velocity_and_seg[:, self.dim + output_classes:]
        del velocity_and_seg

        # sigmoid
        target_seg_pred = self.softmax(target_seg_pred)
        source_seg_pred = self.softmax(source_seg_pred)

        # deformation
        velocity = utils.channel2last(velocity)
        grid = self.exp(velocity)
        deformed_source = self.pull(source, grid)

        if source_seg is not None:
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = self.pull(source_seg_pred, grid)
        if target_seg is None:
            target_seg_for_deformed = target_seg_pred
        else:
            target_seg_for_deformed = target_seg

        # compute loss and metrics
        tensors = dict(
            image=[deformed_source, target],
            velocity=[velocity],
            segmentation=[deformed_source_seg, target_seg_for_deformed])
        if source_seg is not None:
            tensors['source'] = [source_seg_pred, source_seg]
        if target_seg is not None:
            tensors['target'] = [target_seg_pred, target_seg]
        self.compute(_loss, _metric, **tensors)

        return source_seg_pred, target_seg_pred, deformed_source, velocity


class SegMorphRUNet(BaseMorph):
    """
    Joint segmentation and registration using a recursive dual-branch UNet.

    The UNet outputs both a velocity field and two native-space segmentations.
    One loss function acts on the native-space segmentations and tries to
    make them as accurate as possible (in the supervised case) and another
    loss function acts on the warped moving segmentation and tries to
    make it match the fixed-space segmentation (in the supervised or
    semi-supervised case).

    The UNet takes as inputs both intensity images as well as the current
    prediction (segments and velocities). Initially, the velocities and
    log-segments are set to zero. The UNet is then called recursively with
    the output at an iteration serving as inputs to the next iteration.

    In residual mode the velocity and log-segments are incremented with
    the output of the UNet (i.e., the UNet predicts the residual between
    the current estimate and the next best guess).
    """

    def __init__(
            self,
            dim,
            output_classes=1,
            encoder=None,
            decoder=None,
            kernel_size=3,
            activation=torch.nn.LeakyReLU(0.2),
            interpolation='linear',
            grid_bound='dft',
            image_bound='dct2',
            downsample_velocity=2,
            batch_norm=True,
            implicit=True,
            residual=True,
            nb_iter=5,
            random='train'):
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
        super().__init__(
            dim,
            interpolation,
            grid_bound,
            image_bound,
            downsample_velocity,
        )

        self.nb_iter = nb_iter
        self.random = random or ''
        self.implicit = py.make_list(implicit, 2)
        self.output_classes = output_classes
        if not self.implicit[0]:
            output_classes = output_classes + 1
        self.softmax = SoftMax(implicit=implicit)

        output_channels = output_classes * 2 + dim
        self.unet = UUNet(dim,
                          input_channels=2 + output_channels,
                          output_channels=output_channels,
                          encoder=encoder,
                          decoder=decoder,
                          kernel_size=kernel_size,
                          activation=[activation, None],
                          batch_norm=batch_norm,
                          nb_iter=self.nb_iter,
                          residual=self.residual)

        # register losses/metrics
        self.tags = ['image', 'velocity', 'segmentation', 'source', 'target']


    def get_nb_iter(self):
        train_random = 'train' in self.random
        test_random = ('eval' in self.random) or ('test' in self.random)
        if ((self.training and train_random) or 
                (not self.training and test_random)):
            with torch.no_grad():
                return torch.randint(low=1, high=self.nb_iter*2, size=[]).item()
        else:
            return self.nb_iter

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
        nb_iter = self.get_nb_iter()
        output_classes = self.output_classes + (not self.implicit[0])

        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim + 2))

        backend = utils.backend(source)
        batch = source.shape[0]
        spshape = source.shape[2:]

        # unet
        source_and_target = torch.cat((source, target), dim=1)
        velocity_and_seg = self.unet(source_and_target, nb_iter=nb_iter)
        del source_and_target
        velocity = velocity_and_seg[:, :self.dim]
        output_classes = self.output_classes + (not self.implicit[0])
        target_seg_pred = velocity_and_seg[:,
                          self.dim:self.dim + output_classes]
        source_seg_pred = velocity_and_seg[:, self.dim + output_classes:]
        del velocity_and_seg

        # deformation
        velocity = utils.channel2last(velocity)
        grid = self.exp(velocity)
        deformed_source = self.pull(source, grid)

        if source_seg is not None:
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = self.pull(source_seg_pred, grid)
            deformed_source_seg = self.softmax(deformed_source_seg)
        # sigmoid
        target_seg_pred = self.softmax(target_seg_pred)
        source_seg_pred = self.softmax(source_seg_pred)
        if target_seg is None:
            target_seg_for_deformed = target_seg_pred
        else:
            target_seg_for_deformed = target_seg

        # compute loss and metrics
        tensors = dict(
            image=[deformed_source, target],
            velocity=[velocity],
            segmentation=[deformed_source_seg, target_seg_for_deformed])
        if source_seg is not None:
            tensors['source'] = [source_seg_pred, source_seg]
        if target_seg is not None:
            tensors['target'] = [target_seg_pred, target_seg]
        self.compute(_loss, _metric, **tensors)

        return source_seg_pred, target_seg_pred, deformed_source, velocity


class SegMorphRWNet(BaseMorph):
    """
    Joint segmentation and registration using a recurrent WNet.

    A first UNet is applied to both input images and outputs a native space
    segmentation for each (with a loss that tries to make them as accurate
    as possible).
    The second UNet is applied to all input images and predicted segmentations,
    with a loss function acts on the warped moving segmentation and tries to
    make it match the fixed-space segmentation.
    """

    def __init__(self, dim, output_classes=1, encoder=None, decoder=None,
                 kernel_size=3, activation=torch.nn.LeakyReLU(0.2),
                 interpolation='linear', grid_bound='dft', image_bound='dct2',
                 downsample_velocity=2, batch_norm=True, implicit=True,
                 unet_inputs='image+seg', nb_iter=5, random='train'):
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
        unet_inputs : {'seg', 'image', 'image+seg'}, default='image+seg'
        """
        super().__init__(
            dim,
            interpolation,
            grid_bound,
            image_bound,
            downsample_velocity,
        )

        self.nb_iter = nb_iter
        self.random = random or ''
        self.unet_inputs = unet_inputs
        self.implicit = py.make_list(implicit, 2)
        self.output_classes = output_classes
        self.softmax = SoftMax(implicit=implicit)

        output_channels = output_classes + int(not self.implicit[0])
        self.segnet = UNet(dim,
                           input_channels=1 + output_channels,
                           output_channels=output_channels,
                           encoder=encoder,
                           decoder=decoder,
                           kernel_size=kernel_size,
                           activation=[activation, ..., None],
                           batch_norm=batch_norm)

        input_channels = int('image' in unet_inputs) \
                         + int('seg' in unet_inputs) \
                         * (output_classes + int(not self.implicit[1]))
        self.unet = UNet(dim,
                         input_channels=input_channels * 2,
                         output_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=[activation, ..., None],
                         batch_norm=batch_norm)

        # register losses/metrics
        self.tags = ['image', 'velocity', 'segmentation', 'source', 'target',
                     'images', 'velocities', 'segmentations', 'sources', 'targets']

    def get_nb_iter(self):
        train_random = 'train' in self.random
        test_random = ('eval' in self.random) or ('test' in self.random)
        if ((self.training and train_random) or 
                (not self.training and test_random)):
            with torch.no_grad():
                return torch.randint(low=1, high=self.nb_iter*2, size=[]).item()
        else:
            return self.nb_iter

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
        nb_iter = self.get_nb_iter()

        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim + 2))

        backend = utils.backend(source)
        batch = source.shape[0]
        spshape = source.shape[2:]

        output_classes = self.output_classes
        target_seg_pred_deformed = torch.full([batch, output_classes, *spshape],
                                              1/(output_classes+1), **backend)
        source_seg_pred_deformed = torch.full([batch, output_classes, *spshape],
                                              1/(output_classes+1), **backend)

        if _loss is not None:
            types = ['velocities', 'segmentations', 'sources', 'targets']
            for type in types:
                for tag in self.losses[type].keys():
                    if tag:
                        _loss[f'{type}/{tag}'] = 0
        for n_iter in range(nb_iter):

            if target_seg_pred_deformed.shape[1] > output_classes:
                target_seg_pred_deformed = target_seg_pred_deformed[:, :-1]
                source_seg_pred_deformed = source_seg_pred_deformed[:, :-1]

            # segnet
            source_seg_pred = self.segnet(torch.cat([source, target_seg_pred_deformed], dim=1))
            target_seg_pred = self.segnet(torch.cat([target, source_seg_pred_deformed], dim=1))
            del target_seg_pred_deformed, source_seg_pred_deformed

            # unet
            inputs = []
            if 'seg' in self.unet_inputs:
                inputs += [source_seg_pred, target_seg_pred]
            if 'image' in self.unet_inputs:
                inputs += [source, target]
            inputs = torch.cat(inputs, dim=1)

            # deformation
            velocity = self.unet(inputs)
            del inputs
            velocity = utils.channel2last(velocity)

            if _loss is not None:
                for tag, fn in self.losses.get('velocities', {}).items():
                    if not tag:
                        continue
                    fn, w = py.make_tuple(fn, 2, default=1)
                    _loss[f'velocities/{tag}'] += fn(velocity) * w
                        
            igrid = self.exp(-velocity)
            target_seg_pred_deformed = self.pull(target_seg_pred, igrid)
            target_seg_pred_deformed = self.softmax(target_seg_pred_deformed)
            del igrid
            grid = self.exp(velocity)
            source_seg_pred_deformed = self.pull(source_seg_pred, grid)
            source_seg_pred_deformed = self.softmax(source_seg_pred_deformed)

            if _loss is not None:

                deformed_source = self.pull(source, grid)
                if source_seg is not None:
                    deformed_source_seg = self.pull(source_seg, grid)
                else:
                    deformed_source_seg = self.pull(source_seg_pred, grid)
                    source_seg_pred = self.softmax(source_seg_pred)
                if target_seg is None:
                    target_seg_for_deformed = self.softmax(target_seg_pred)
                else:
                    target_seg_for_deformed = target_seg

                for tag, fn in self.losses.get('images', {}).items():
                    if not tag:
                        continue
                    fn, w = py.make_tuple(fn, 2, default=1)
                    _loss[f'images/{tag}'] += fn(deformed_source, target) * w
                for tag, fn in self.losses.get('segmentations', {}).items():
                    if not tag:
                        continue
                    fn, w = py.make_tuple(fn, 2, default=1)
                    _loss[f'segmentations/{tag}'] += fn(deformed_source_seg, target_seg_for_deformed) * w
                for tag, fn in self.losses.get('sources', {}).items():
                    if not tag:
                        continue
                    fn, w = py.make_tuple(fn, 2, default=1)
                    _loss[f'sources/{tag}'] += fn(self.softmax(source_seg_pred), source_seg) * w
                for tag, fn in self.losses.get('targets', {}).items():
                    if not tag:
                        continue
                    fn, w = py.make_tuple(fn, 2, default=1)
                    _loss[f'targets/{tag}'] += fn(self.softmax(target_seg_pred), target_seg) * w

        del target_seg_pred_deformed, source_seg_pred_deformed

        if _loss is not None:
            types = ['velocities', 'segmentations', 'sources', 'targets']
            for type in types:
                for tag in self.losses[type].keys():
                    if tag:
                        _loss[f'{type}/{tag}'] = _loss[f'{type}/{tag}'] / nb_iter

        deformed_source = self.pull(source, grid)
        if source_seg is not None:
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = self.pull(source_seg_pred, grid)
            source_seg_pred = self.softmax(source_seg_pred)
        if target_seg is None:
            target_seg_for_deformed = self.softmax(target_seg_pred)
        else:
            target_seg_for_deformed = target_seg

        # compute loss and metrics
        tensors = dict(
            image=[deformed_source, target],
            velocity=[velocity],
            segmentation=[deformed_source_seg, target_seg_for_deformed])
        if source_seg is not None:
            tensors['source'] = [self.softmax(source_seg_pred), source_seg]
        if target_seg is not None:
            tensors['target'] = [self.softmax(target_seg_pred), target_seg]
        self.compute(_loss, _metric, **tensors)

        return source_seg_pred, target_seg_pred, deformed_source, velocity

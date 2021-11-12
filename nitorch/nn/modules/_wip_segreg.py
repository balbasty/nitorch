"""Work In Progress: joint segmentation and registration"""

from nitorch.nn.base import Module
from .cnn import UNet, UUNet, WNet, UNet2, SEWNet
from .spatial import GridResize, GridExp, GridPull
from nitorch.nn.activations import SoftMax
from nitorch.nn import check
from nitorch.core import py, utils, math
from nitorch import spatial
import torch


class BaseMorph(Module):
    """Secondary base class for model that implement a morph component"""

    def __init__(self, dim, interpolation='linear', grid_bound='dft',
                 image_bound='dct2', downsample_velocity=2, anagrad=False):
        super().__init__()
        
        resize_factor = py.make_list(downsample_velocity, dim)
        resize_factor = [1/f for f in resize_factor]

        self.resize = GridResize(interpolation=interpolation,
                                 bound=grid_bound,
                                 factor=resize_factor,
                                 type='displacement')
        self.velexp = GridExp(interpolation=interpolation,
                              bound=grid_bound,
                              displacement=True,
                              anagrad=anagrad)
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
        # generate grid
        shape = velocity.shape[1:-1]
        velocity_small = self.resize(velocity)
        grid = self.velexp(velocity_small)
        grid = self.resize(grid, output_shape=shape)
        if not displacement:
            grid = spatial.add_identity_grid_(grid)

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
            vol = utils.movedim(vol, -1, 1)
            vol = get_slice(plane, vol[batch])
            vol = utils.movedim(vol, 0, -1)
            bound = vol.reshape(-1, self.dim).abs().max(dim=0).values
            vol = vol + bound
            vol = vol / (2*bound)
            vol = vol.clip_(0, 1)
            return vol

        def get_label(plane, vol, batch=0, logit=False):
            vol = get_slice(plane, vol[batch])
            if vol.dim() == 2:
                if logit:
                    vol = math.softmax(vol[None], dim=0, implicit=True)[0]
                else:
                    vol = vol.float()
                    vol /= vol.max()
            else:
                if logit:
                    vol = math.softmax(vol, dim=0, implicit=(implicit, False))
                elif implicit:
                    background = 1 - vol.sum(dim=0, keepdim=True)
                    vol = torch.cat((background, vol), dim=0)
                nb_classes = vol.shape[0]
                vol = vol.argmax(dim=0).float()
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
                    if b == 0 and p == 0:
                        ax.set_title('Source')
                    if source_seg is not None:
                        idx += 1
                        ax = fig.add_subplot(nrow, ncol, idx, **prm)
                        ax.imshow(get_label(plane, source_seg, b))
                        if b == 0 and p == 0:
                            ax.set_title('Source [seg]')
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_label(plane, source_pred, b, logit=True))
                    if b == 0 and p == 0:
                        ax.set_title('Source [pred]')
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_image(plane, target, b))
                    if b == 0 and p == 0:
                        ax.set_title('Target')
                    if target_seg is not None:
                        idx += 1
                        ax = fig.add_subplot(nrow, ncol, idx, **prm)
                        ax.imshow(get_label(plane, target_seg, b))
                        if b == 0 and p == 0:
                            ax.set_title('Target [seg]')
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_label(plane, target_pred, b, logit=True))
                    if b == 0 and p == 0:
                        ax.set_title('Target [pred]')
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_image(plane, source_warped, b))
                    if b == 0 and p == 0:
                        ax.set_title('Warped')
                    idx += 1
                    ax = fig.add_subplot(nrow, ncol, idx, **prm)
                    ax.imshow(get_velocity(plane, velocity, b))
                    if b == 0 and p == 0:
                        ax.set_title('Velocity')
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
                 unet_inputs='image+seg', variant=1):
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

        Klass = UNet2 if variant == 2 else UNet
        
        out_channels = output_classes + int(not self.implicit[0])
        self.segnet = Klass(dim,
                           in_channels=1,
                           out_channels=out_channels,
                           encoder=encoder,
                           decoder=decoder,
                           kernel_size=kernel_size,
                           activation=[activation, ..., self.softmax],
                           batch_norm=batch_norm)

        in_channels = int('image' in unet_inputs) \
                        + int('seg' in unet_inputs) \
                        * (output_classes + int(not self.implicit[1]))
        self.unet = Klass(dim,
                         in_channels=in_channels * 2,
                         out_channels=dim,
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
                 downsample_velocity=2, norm='batch', implicit=True,
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
        norm : bool, default='batch'
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

        self.implicit = py.ensure_list(implicit, 2)
        self.output_classes = output_classes
        if not self.implicit[0]:
            output_classes = output_classes + 1
        self.softmax = SoftMax(implicit=implicit)

        groups = py.ensure_list(groups)
        stitch = py.ensure_list(stitch)
        if (groups[-1] or stitch[-1]) == 2:
            out_channels = [output_classes * 2, dim]
        else:
            out_channels = output_classes * 2 + dim
        self.unet = UNet(dim,
                         in_channels=2,
                         out_channels=out_channels,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=[activation, ..., None],
                         norm=norm,
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
            conv_per_layer=1,
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

        out_channels = output_classes * 2 + dim
        self.unet = UUNet(dim,
                          in_channels=2,
                          out_channels=out_channels,
                          encoder=encoder,
                          decoder=decoder,
                          kernel_size=kernel_size,
                          conv_per_layer=conv_per_layer,
                          activation=[activation, None],
                          norm=batch_norm,
                          nb_iter=nb_iter,
                          residual=residual)

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

        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim + 2))

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

        out_channels = output_classes + int(not self.implicit[0])
        self.segnet = UNet(dim,
                           in_channels=1 + out_channels,
                           out_channels=out_channels,
                           encoder=encoder,
                           decoder=decoder,
                           kernel_size=kernel_size,
                           activation=[activation, ..., None],
                           norm=batch_norm)

        in_channels = int('image' in unet_inputs) \
                         + int('seg' in unet_inputs) \
                         * (output_classes + int(not self.implicit[1]))
        self.unet = UNet(dim,
                         in_channels=in_channels * 2,
                         out_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=[activation, ..., None],
                         norm=batch_norm)

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


class SegMorphWNet2(BaseMorph):
    """
    Joint segmentation and registration using a dual branch WNet.

    Image A + Image B -> Unet -> Unet -> Seg A + Seg B + Velocity

    The WNet outputs both a velocity field and two native-space segmentations.
    One loss function acts on the native-space segmentations and tries to
    make them as accurate as possible (in the supervised case) and another
    loss function acts on the warped moving segmentation and tries to
    make it match the fixed-space segmentation (in the supervised or
    semi-supervised case).
    """

    def __init__(self, dim, output_classes=1, encoder=None, decoder=None,
                 encoder2=None, decoder2=None, skip=True,
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

        out_channels = output_classes * 2 + dim
        self.wnet = WNet(dim,
                         in_channels=2,
                         out_channels=out_channels,
                         encoder=encoder,
                         decoder=decoder,
                         encoder2=encoder2,
                         decoder2=decoder2,
                         skip=skip,
                         kernel_size=kernel_size,
                         activation=[activation, None],
                         norm=batch_norm)

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

        # wnet
        source_and_target = torch.cat((source, target), dim=1)
        velocity_and_seg = self.wnet(source_and_target)
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


class SegMorphWNet3(BaseMorph):
    """
    Joint segmentation and registration using a WNet.

    Image A + Image B -> Unet -|-> Unet -> Velocity
                               |-> Seg A + Seg B

    A first UNet outputs both segmentations (with a segmentation loss).
    The last *features* (before the final conv) are fed to a second UNet
    along with the input images, which outputs a velocity (with a
    registration loss).
    Optionally, skip connections exist between the two UNets.
    """

    def __init__(self, dim, output_classes=1, encoder=None, decoder=None,
                 encoder2=None, decoder2=None, skip=True,
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

        self.wnet = WNet(dim,
                         in_channels=2,
                         mid_channels=output_classes * 2,
                         out_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         encoder2=encoder2,
                         decoder2=decoder2,
                         skip=skip,
                         kernel_size=kernel_size,
                         activation=[activation, None],
                         norm=batch_norm)

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

        # wnet
        source_and_target = torch.cat((source, target), dim=1)
        velocity, seg = self.wnet(source_and_target)
        del source_and_target
        output_classes = self.output_classes + (not self.implicit[0])
        target_seg_pred = seg[:, :output_classes]
        source_seg_pred = seg[:, output_classes:]
        del seg

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


class SegMorphWNet4(BaseMorph):
    """
    Joint segmentation and registration using a WNet.

    Image A -> UNet -> Feat A [+ Image A] -> |
                                             + -> UNet2 -> Velocity
    Image B ->   "  -> Feat B [+ Image B] -> |
    """

    def __init__(self, dim, output_classes=1, encoder=None, decoder=None,
                 kernel_size=3, activation=torch.nn.LeakyReLU(0.2),
                 interpolation='linear', grid_bound='dft', image_bound='dct2',
                 downsample_velocity=2, batch_norm=True, implicit=True,
                 unet_inputs='image+feat'):
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

        out_channels = output_classes + int(not self.implicit[0])
        self.segnet = UNet2(dim,
                            in_channels=1,
                            out_channels=out_channels,
                            encoder=encoder,
                            decoder=decoder,
                            kernel_size=kernel_size,
                            activation=activation,
                            norm=batch_norm)

        nb_feat = self.segnet.final.in_channels
        in_channels = int('image' in unet_inputs) \
                      + int('seg' in unet_inputs) \
                      * (output_classes + int(not self.implicit[1])) \
                      + int('feat' in unet_inputs) * nb_feat
        self.unet = UNet2(dim,
                          in_channels=in_channels * 2,
                          out_channels=dim,
                          encoder=encoder,
                          decoder=decoder,
                          kernel_size=kernel_size,
                          activation=activation,
                          norm=batch_norm)

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

        # segnet
        source_feat = self.segnet(source)
        target_feat = self.segnet(target)
        source_seg_pred = self.softmax(source_feat)
        target_seg_pred = self.softmax(target_feat)

        # unet
        inputs = []
        if 'feat' in self.unet_inputs:
            inputs += [source_feat, target_feat]
        if 'seg' in self.unet_inputs:
            inputs += [source_seg_pred, target_seg_pred]
        if 'image' in self.unet_inputs:
            inputs += [source, target]
        del source_feat, target_feat
        inputs = torch.cat(inputs, dim=1)

        # deformation
        velocity = self.unet(inputs)
        del inputs
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


class SewMorph(BaseMorph):
    """
    Joint segmentation and registration using a SewNet.

    Image A -> UNet -> Feat A [+ Image A] -> |
                                             + -> UNet2 -> Velocity
    Image B ->   "  -> Feat B [+ Image B] -> |
    """

    def __init__(self,
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
                 norm='batch',
                 implicit=True,
                 skip=False,
                 anagrad=False):
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
        norm : {'batch', 'instance', 'layer'}, default='batch'
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
            dim=dim,
            interpolation=interpolation,
            grid_bound=grid_bound,
            image_bound=image_bound,
            downsample_velocity=downsample_velocity,
            anagrad=anagrad,
        )

        self.implicit = py.ensure_list(implicit, 2)
        self.output_classes = output_classes
        self.softmax = SoftMax(implicit=implicit)

        mid_channels = output_classes + int(not self.implicit[0])
        self.sewnet = SEWNet(dim,
                             nb_twins=2,
                             in_channels=1,
                             feat_channels=mid_channels,
                             out_channels=dim,
                             encoder=encoder,
                             decoder=decoder,
                             kernel_size=kernel_size,
                             activation=activation,
                             norm=norm,
                             skip=skip)

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
        source_seg : tensor (batch, 1|classes, *spatial), optional
            Source/moving segmentation
        target_seg : tensor (batch, 1|classes, *spatial), optional
            Target/fixed segmentation

        Other Parameters
        ----------------
        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        target_logits : tensor (batch, classes[-1], *spatial), optional
            Predicted target logits
        source_logits : tensor (batch, classes[-1], *spatial), optional
            Predicted source logits
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

        # apply net
        # -> returns velocity and pre-softmax predicted segmentations
        vel, (src_seg_pred, tgt_seg_pred) = self.sewnet(source, target, return_feat=True)

        # exponentiate deformation and deform source image
        vel = utils.channel2last(vel)
        grid = self.exp(vel)
        wsrc = self.pull(source, grid)

        # deform source segmentation
        if source_seg is not None:
            wsrc_seg = source_seg
            if not wsrc_seg.dtype.is_floating_point:
                wsrc_seg = wsrc_seg.squeeze(1)
                wsrc_seg = utils.one_hot(wsrc_seg,
                                         dim=1,
                                         implicit=self.implicit[1],
                                         dtype=wsrc.dtype)
            wsrc_seg = wsrc_seg.clamp_(1e-5, 1-1e-5)
            wsrc_seg = math.logit(wsrc_seg, dim=1, implicit=self.implicit[1])
        else:
            wsrc_seg = src_seg_pred
        wsrc_seg = self.pull(wsrc_seg, grid)
        tgt_seg_w = tgt_seg_pred if target_seg is None else target_seg

        # compute loss and metrics
        tensors = dict(
            image=[wsrc, target],
            velocity=[vel],
            segmentation=[wsrc_seg, tgt_seg_w])
        if source_seg is not None:
            tensors['source'] = [src_seg_pred, source_seg]
        if target_seg is not None:
            tensors['target'] = [tgt_seg_pred, target_seg]
        self.compute(_loss, _metric, **tensors)

        return src_seg_pred, tgt_seg_pred, wsrc, vel


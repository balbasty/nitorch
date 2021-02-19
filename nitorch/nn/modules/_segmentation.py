import torch
import torch.nn as tnn
from ..modules._base import Module
from ..modules._cnn import (UNet, MRF)
from ..modules._spatial import (GridPull, GridPushCount)
from ...spatial import (affine_grid, identity_grid, voxel_size)
from ...core.constants import eps
from ...core.utils import logsumexp
from .. import check


class MeanSpaceNet(Module):
    """Segmentation network that trains in a shared mean space, but computes
    losses in native space.

    OBS: For optimal performance, both memory- and prediction-wise, it is important
    that the input data is aligned in the common space.

    """
    def __init__(self, dim, common_space, output_classes=1, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), batch_norm=True,
                 implicit=True, coord_conv=False):
        """

        Parameters
        ----------
        dim : int
            Space dimension
        common_space : [2,] sequence
            First element is the mean space affine matrix ([dim + 1, dim + 1] tensor),
            second element is the mean space dimensions ([3,] sequence)
        output_classes : int, default=1
            Number of classes, excluding background
        input_channels : int, default=1
            Number of input channels
        encoder : sequence[int], optional
            Number of features per encoding layer
        decoder : sequence[int], optional
            Number of features per decoding layer
        kernel_size : int or sequence[int], default=3
            Kernel size
        activation : str or callable, default=LeakyReLU(0.2)
            Activation function in the UNet.
        batch_norm : bool or callable, default=True
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
        implicit : bool, default=True
            Only return `output_classes` probabilities (the last one
            is implicit as probabilities must sum to 1).
            Else, return `output_classes + 1` probabilities.
        coord_conv : bool, default=False
            Use mean space coordinate grid as input to the UNet.

        """
        super().__init__()

        self.implicit = implicit
        self.output_classes = output_classes
        self.mean_mat = common_space[0].unsqueeze(0)
        self.mean_dim = common_space[1]
        self.coord_conv = coord_conv
        if implicit and output_classes == 1:
            final_activation = tnn.Sigmoid
        else:
            final_activation = tnn.Softmax(dim=1)
        if implicit:
            output_classes += 1
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: self.board_custom(
            dim, tb, inputs, outputs, implicit=implicit)
        # Push/pull settings
        interpolation = 'linear'
        bound = 'dct2'  # symmetric
        extrapolate = False
        # Add push operators
        self.push = GridPushCount(interpolation=interpolation,
                                  bound=bound, extrapolate=extrapolate)
        # Add channel to take into account count image
        input_channels += 1
        if coord_conv:
            input_channels += dim
        # Add UNet
        self.unet = UNet(dim,
                         input_channels=input_channels,
                         output_channels=output_classes,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=activation,
                         final_activation=None,  # so that pull is performed in log-space
                         batch_norm=batch_norm)
        # Add pull operators
        self.pull = GridPull(interpolation=interpolation,
                             bound=bound, extrapolate=extrapolate)
        # register loss tag
        self.tags = ['native']

    # defer properties
    dim = property(lambda self: self.unet.dim)
    encoder = property(lambda self: self.unet.encoder)
    decoder = property(lambda self: self.unet.decoder)
    kernel_size = property(lambda self: self.unet.kernel_size)
    activation = property(lambda self: self.unet.activation)

    def forward(self, image, mat_native, ref=None, *, _loss=None, _metric=None):
        """Forward pass.

        OBS: Supports only batch size one, because images can be of
        different dimensions.

        Parameters
        ----------
        image : (1, input_channels, *spatial) tensor
            Input image
        mat_native : (1, dim + 1, dim + 1) tensor
            Input image's affine matrix.
        ref : (1, output_classes[+1], *spatial) tensor, optional
            Ground truth segmentation, used by the loss function.
            Its data type should be integer if it contains hard labels,
            and floating point if it contains soft segmentations.
        _loss : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.
        _metric : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.

        Returns
        -------
        pred : (1, output_classes[+1], *spatial)
            Tensor of class probabilities.
            If `implicit` is True, the background class is not returned.

        """
        image = torch.as_tensor(image)

        if image.shape[0] != 1:
            raise NotImplementedError('Only batch size 1 supported, got {:}'.format(image.shape[0]))

        # sanity check
        check.dim(self.dim, image)
        check.shape(self.mean_mat, mat_native)

        # augment (taking voxel size into account)
        vx = voxel_size(mat_native).squeeze().tolist()
        image, ref = augment(image, ref, self.augmenters, vx)

        # Compute grid
        dim_native = image.shape[2:]
        grid = self.compute_grid(mat_native, dim_native)

        # Push image into common space
        image_pushed, count = self.push(image, grid, shape=self.mean_dim)
        image_pushed = torch.cat((image_pushed, count), dim=1)  # add count image as channel

        if self.coord_conv:
            # Append mean space coordinate grid to UNet input
            id = identity_grid(self.mean_dim, dtype=image_pushed.dtype, device=image_pushed.device)
            id = id.unsqueeze(0)
            if self.dim == 3:
                id = id.permute((0, 4, 1, 2, 3))
                id = id.repeat(image_pushed.shape[0], 1, 1, 1, 1)
            else:
                id = id.permute((0, 3, 1, 2))
                id = id.repeat(image_pushed.shape[0], 1, 1, 1)
            image_pushed = torch.cat((image_pushed, id), dim=1)

        # Apply UNet
        pred = self.unet(image_pushed)

        # Remove last class, if implicit
        if self.implicit and pred.shape[1] > self.output_classes:
            pred = pred[:, :-1, ...]

        # Pull prediction into native space (whilst in log space)
        pred = self.pull(pred, grid)

        # compute loss and metrics (in native, log space)
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim + 2))
            check.shape(pred, ref, dims=dims)
            self.compute(_loss, _metric, native=[pred, ref])

        # softmax
        pred = pred.softmax(dim=1)

        return pred

    def compute_grid(self, mat_native, dim_native):
        """Computes resampling grid for pulling/pushing from/to common space.

        Parameters
        ----------
        mat_native : (1, dim + 1, dim + 1) tensor
            Native image affine matrix.
        dim_native : [3, ] sequence
            Native image dimensions.

        Returns
        ----------
        grid : (batch, *spatial, dim) tensor
            Resampling grid.

        """
        self.mean_mat = self.mean_mat.type(mat_native.dtype).to(mat_native.device)
        mat = mat_native.solve(self.mean_mat)[0]
        grid = affine_grid(mat, dim_native)

        return grid

    def board_custom(self, dim, tb, inputs, outputs, implicit=False):
        """Removes affine matrix from inputs before calling board.
        """
        board(dim, tb, (inputs[0], inputs[2]), outputs, implicit=implicit)


class SegMRFNet(Module):
    """Combines a SegNet with a MRFNet.

    The idea is that a simple, explicit, spatial prior on the categorical data
    should improve generalisation to out-of-distribution data, and allow for less
    parameters.

    """
    def __init__(self, dim, output_classes=1, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), batch_norm_seg=True,
                 num_iter=10, w=1.0, num_extra=0, only_unet=False):
        """

        Parameters
        ----------
        dim : int
            Space dimension
        output_classes : int, default=1
            Number of classes, excluding background
        input_channels : int, default=1
            Number of input channels
        encoder : sequence[int], optional
            Number of features per encoding layer
        decoder : sequence[int], optional
            Number of features per decoding layer
        kernel_size : int or sequence[int], default=3
            Kernel size
        activation : str or callable, default=LeakyReLU(0.2)
            Activation function in the UNet.
        batch_norm_seg : bool or callable, default=True
            Batch normalization layer in UNet?
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
        num_iter : int, default=10
            Number of mean-field iterations.
        w : float, default=1.0
            Weight between new and old prediction [0, 1].
        num_extra : int, default=0
            Number of extra layers between MRF layer and final layer.
        only_unet : bool, default=False
            This allows for fitting just the UNet, without the MRF part.
            This is good for testing and comparing the two methods.

        """
        super().__init__()

        self.only_unet = only_unet
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: board(
            dim, tb, inputs, outputs)

        self.unet = SegNet(dim,
                           input_channels=input_channels,
                           output_classes=output_classes,
                           encoder=encoder,
                           decoder=decoder,
                           kernel_size=kernel_size,
                           activation=activation,
                           batch_norm=batch_norm_seg,
                           implicit=False,
                           skip_final_activation=True)

        if not only_unet:
            self.mrf = MRFNet(dim,
                              num_classes=output_classes,
                              num_extra=num_extra,
                              activation=activation,
                              w=w,
                              num_iter=num_iter)

        # register loss tag
        self.tags = ['unetmrf']

    # defer properties
    dim = property(lambda self: self.unet.dim)

    def forward(self, image, ref=None, *, _loss=None, _metric=None):
        """Forward pass, with mean-field iterations.

        Parameters
        ----------
        image : (batch, input_channels, *spatial) tensor
            Input image.
        ref : (batch, output_classes[+1], *spatial) tensor, optional
            Reference segmentation, used by the loss function.
        _loss : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.
        _metric : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.

        Returns
        -------
        p : (batch, output_classes[+1], *spatial)
            Tensor of class probabilities.

        """
        is_train = True if ref is not None else False

        image = torch.as_tensor(image)

        # sanity check
        check.dim(self.dim, image)

        if ref is not None:
            # augment
            image, ref = augment(self.augmenters, image, ref=ref)

        # unet
        p = self.unet(image)

        if self.only_unet:
            p = p.softmax(dim=1)
        else:
            p = p - logsumexp(p, dim=1, keepdim=True)
            p = self.mrf.apply(p, is_train)
            # hook to zero gradients of central filter weights
            self.mrf.parameters().__next__() \
                .register_hook(self.mrf.zero_grad_centre)

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim + 2))
            check.shape(p, ref, dims=dims)
            self.compute(_loss, _metric, unetmrf=[p, ref])

        return p


class SegNet(Module):
    """Segmentation network.

    This is simply a UNet that ends with a softmax activation.

    Batch normalization is used by default.

    """
    def __init__(self, dim, output_classes=1, input_channels=1,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.LeakyReLU(0.2), batch_norm=True,
                 implicit=True, skip_final_activation=False):
        """

        Parameters
        ----------
        dim : int
            Space dimension
        output_classes : int, default=1
            Number of classes, excluding background
        input_channels : int, default=1
            Number of input channels
        encoder : sequence[int], optional
            Number of features per encoding layer
        decoder : sequence[int], optional
            Number of features per decoding layer
        kernel_size : int or sequence[int], default=3
            Kernel size
        activation : str or callable, default=LeakyReLU(0.2)
            Activation function in the UNet.
        batch_norm : bool or callable, default=True
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
        implicit : bool, default=True
            Only return `output_classes` probabilities (the last one
            is implicit as probabilities must sum to 1).
            Else, return `output_classes + 1` probabilities.
        skip_final_activation : bool, default=False
           Append final activation function.

        """
        super().__init__()

        self.implicit = implicit
        self.output_classes = output_classes
        final_activation = None
        if not skip_final_activation:
            if implicit and output_classes == 1:
                final_activation = tnn.Sigmoid
            else:
                final_activation = tnn.Softmax(dim=1)
        if implicit:
            output_classes += 1
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: board(
            dim, tb, inputs, outputs, implicit=implicit)

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

    def forward(self, image, ref=None, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        image : (batch, input_channels, *spatial) tensor
            Input image
        ref : (batch, output_classes[+1], *spatial) tensor, optional
            Ground truth segmentation, used by the loss function.
            Its data type should be integer if it contains hard labels,
            and floating point if it contains soft segmentations.
        _loss : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.
        _metric : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.

        Returns
        -------
        prob : (batch, output_classes[+1], *spatial)
            Tensor of class probabilities.
            If `implicit` is True, the background class is not returned.

        """
        image = torch.as_tensor(image)

        # sanity check
        check.dim(self.dim, image)

        if ref is not None:
            # augment
            image, ref = augment(self.augmenters, image, ref=ref)

        # unet
        prob = self.unet(image)
        if self.implicit and prob.shape[1] > self.output_classes:
            prob = prob[:, :-1, ...]

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim + 2))
            check.shape(prob, ref, dims=dims)
            self.compute(_loss, _metric, segmentation=[prob, ref])

        return prob


class MRFNet(Module):
    """MRF network.

    A network that aims at cleaning up categorical image data of num_classes
    classes. The first layer is a, so called, MRF layer. Sub-sequent layers are
    then 1x1 convolutions layers. The output of the network are softmaxed,
    categorical data. The idea is described in:

    Brudfors, Mikael, Yaël Balbastre, and John Ashburner.
    "Nonlinear markov random fields learned via backpropagation."
    IPMI. Springer, Cham, 2019.

    """
    def __init__(self, dim, num_classes, num_iter=10, num_extra=0, w=1.0,
                 activation=tnn.LeakyReLU(0.2)):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        num_classes : int
            Number of input classes.
        num_iter : int, default=10
            Number of mean-field iterations.
        num_extra : int, default=0
            Number of extra layers between MRF layer and final layer.
        w : float, default=1.0
            Weight between new and old prediction [0, 1].
        activation : str or type or callable, default='tnn.LeakyReLU(0.2)'
            Activation function.

        """
        super().__init__()

        self.num_classes = num_classes
        if num_iter < 1:
            raise ValueError(
                'Parameter num_iter should be greater than 0 , got {:}'.format(
                    num_iter))
        self.num_iter = num_iter
        if w < 0 or w > 1:
            raise ValueError(
                'Parameter w should be between 0 and 1, got {:}'.format(w))
        self.w = w
        # As the final activation is applied at the end of the forward method
        # below, it is not applied in the final Conv layer
        final_activation = None
        # We use a first-order neigbourhood
        kernel_size = 3
        num_filters = self.get_num_filters()
        # Add tensorboard callback
        self.board = lambda tb, inputs, outputs: board(
            dim, tb, inputs, outputs)
        # Build MRF net
        self.mrf = MRF(dim,
                       num_classes=num_classes,
                       num_filters=num_filters,
                       num_extra=num_extra,
                       kernel_size=kernel_size,
                       activation=activation,
                       batch_norm=False,
                       bias=True,
                       final_activation=final_activation)
        # register loss tag
        self.tags = ['mrf']

    # defer properties
    dim = property(lambda self: self.mrf.dim)
    kernel_size = property(lambda self: self.mrf.kernel_size)
    activation = property(lambda self: self.mrf.activation)

    def forward(self, resp, ref=None, *, _loss=None, _metric=None):
        """Forward pass, with mean-field iterations.

        Parameters
        ----------
        resp : (batch, input_channels, *spatial) tensor
            Input responsibilities.
        ref : (batch, output_classes[+1], *spatial) tensor, optional
            Reference segmentation, used by the loss function.
        _loss : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.
        _metric : dict, optional
            Dictionary of losses that will be modified in place.
            If provided along with `ref`, all registered loss
            functions will be applied and stored under the key
            '<tag>/<name>' in the dictionary.

        Returns
        -------
        p : (batch, output_classes[+1], *spatial)
            Tensor of class probabilities.

        """
        is_train = True if ref is not None else False

        resp = torch.as_tensor(resp)

        # sanity check
        check.dim(self.dim, resp)

        if ref is not None:
            # augment
            resp, ref = augment(self.augmenters, resp, ref=ref)

        # MRF
        p = self.apply(resp, is_train)

        # hook to zero gradients of central filter weights
        self.mrf.parameters().__next__() \
            .register_hook(self.mrf.zero_grad_centre)

        # compute loss and metrics
        if is_train:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim + 2))
            check.shape(p, ref, dims=dims)
            self.compute(_loss, _metric, mrf=[p, ref])

        return p

    def zero_grad_centre(self, grad):
        """Hook to zero gradients of central filter weights.

        Parameters
        ----------
        grad : tensor
            Input gradients.

        Returns
        -------
        grad_clone : tensor
            Output gradients.

        """
        k = grad.shape[2:]
        grad_clone = grad.clone()
        if self.dim == 3:
            grad_clone[:, :, k[0] // 2, k[1] // 2, k[2] // 2] = 0
        elif self.dim == 2:
            grad_clone[:, :, k[0] // 2, k[1] // 2] = 0
        else:
            grad_clone[:, :, k[0] // 2] = 0

        return grad_clone

    def get_num_filters(self):
        """Returns number of MRF filters. We simply use
        num_classes**2.

        Returns
        ----------
        num_filters : int
            Number of Conv filters in MRF layer.

        """
        num_filters = self.num_classes ** 2
        if num_filters > 64:
            num_filters = 64

        return num_filters

    def get_num_iter(self, is_train):
        """Returns number of VB iterations.

        Parameters
        ----------
        is_train : bool
            Is model in training or testing mode?

        Returns
        ----------
        num_iter : int
            Number of VB iterations.

        """
        if is_train:
            rng = 0.5
            mn = rng*self.num_iter
            mx = (1 + rng) * self.num_iter
            num_iter = torch.FloatTensor(1).uniform_(mn, mx + 1).int()
        else:
            num_iter = self.num_iter

        return num_iter

    def apply(self, ll, is_train):
        """Apply MRFNet.

        For training, one forward pass through the MRFNet is performed
        with a softmax at the end. For testing, the forward pass is iterated over
        as part of a VB update of the responsibilities.

        Parameters
        ----------
        ll : (batch, input_channels, *spatial) tensor
            Log-likelihood term.
        is_train : bool
            Is model training?

        Returns
        ----------
        p : (batch, input_channels, *spatial) tensor
            VB optimal tissue posterior, under the given MRF assumption.

        """
        K = ll.shape[1]
        p = torch.ones_like(ll)/K
        for i in range(self.get_num_iter(is_train)):
            op = p.clone()
            p = (ll + self.mrf(p)).softmax(dim=1)
            p = self.w * p + (1 - self.w) * op

        return p


def augment(augmenters, image, ref=None, vx=None):
    """Applies various augmentation techniques.

    Parameters
    ----------
    image : (batch, input_channels, *spatial) tensor
        Input image.
    ref : (batch, output_classes[+1], *spatial) tensor, optional
        Reference segmentation.
    augmenters : list
        List of augmentation functions (defined in nn.seg_augmentation).
    vx : [ndim, ] sequence, optional
        Image voxel size (in mm), defaults to 1 mm isotropic.

    Returns
    ----------
    image : (batch, input_channels, *spatial) tensor
        Augmented input image.
    ref : (batch, output_classes[+1], *spatial) tensor, optional
        Augmented reference segmentation.

    """
    for augmenter in augmenters:
        image, ref = augmenter(image, ref, vx)
                
    return image, ref


def board(dim, tb, inputs, outputs, implicit=False):
    """TensorBoard visualisation of a segmentation model's inputs and outputs.

    Parameters
    ----------
    dim : int
        Space dimension
    tb : torch.utils.tensorboard.writer.SummaryWriter
        TensorBoard writer object.
    inputs : (tensor_like, tensor_like) tuple
        Input image (N, C, dim)  and reference segmentation (N, K, dim) .
    outputs : (N, K, dim) tensor_like
        Predicted segmentation.
    implicit : bool, default=False
        Only return `output_classes` probabilities (the last one
        is implicit as probabilities must sum to 1).
        Else, return `output_classes + 1` probabilities.

    """
    def get_slice(vol, plane, dim):
        if dim == 2:
            return vol.squeeze()
        if plane == 'z':
            z = round(0.5 * vol.shape[-1])
            slice = vol[..., z]
        elif plane == 'y':
            y = round(0.5 * vol.shape[-2])
            slice = vol[..., y, :]
        elif plane == 'x':
            x = round(0.5 * vol.shape[-3])
            slice = vol[..., x, :, :]

        return slice.squeeze()

    def input_view(slice_input):
        return slice_input

    def prediction_view(slice_prediction, implicit):
        if implicit:
            slice_prediction = \
                torch.cat((1 - slice_prediction.sum(dim=0, keepdim=True), slice_prediction), dim=0)
        K1 = float(slice_prediction.shape[0])
        slice_prediction = \
            (slice_prediction.argmax(dim=0, keepdim=False)) / (K1 - 1)
        return slice_prediction

    def target_view(slice_target):
        if len(slice_target.shape) == 3 and slice_target.shape[0] > 1:
            K1 = float(slice_target.shape[0])
            slice_target = (slice_target.argmax(dim=0, keepdim=False))/(K1 - 1)
        else:
            slice_target =  slice_target.float() / slice_target.max().float()
        return slice_target

    def to_grid(slice_input, slice_target, slice_prediction):
        return torch.cat((slice_input, slice_target, slice_prediction), dim=1)

    def get_slices(plane, inputs, outputs, dim, implicit):
        slice_input = input_view(get_slice(inputs[0][0, ...], plane, dim=dim))
        slice_target = target_view(get_slice(inputs[1][0, ...], plane, dim=dim))
        slice_prediction = prediction_view(
            get_slice(outputs[0, ...], plane, dim=dim), implicit=implicit)
        return slice_input.detach().cpu(), \
               slice_target.detach().cpu(), \
               slice_prediction.detach().cpu()

    def get_image(plane, inputs, outputs, dim, implicit):
        slice_input, slice_target, slice_prediction = \
            get_slices(plane, inputs, outputs, dim, implicit)
        if len(slice_input.shape) != len(slice_prediction.shape):
            K1 = float(slice_input.shape[0])
            slice_input = (slice_input.argmax(dim=0, keepdim=False)) / (K1 - 1)
            slice_target = (slice_target.argmax(dim=0, keepdim=False)) / (K1 - 1)
        return to_grid(slice_input, slice_target, slice_prediction)[None, ...]

    # from nitorch.plot import show_slices
    # show_slices(get_image('z', inputs, outputs, dim, implicit).squeeze())

    # Add to TensorBoard
    title = 'Image-Target-Prediction_'
    tb.add_image(title + 'z', get_image('z', inputs, outputs, dim, implicit))
    if dim == 3:
        tb.add_image(title + 'y', get_image('y', inputs, outputs, dim, implicit))
        tb.add_image(title + 'x', get_image('x', inputs, outputs, dim, implicit))
    tb.flush()
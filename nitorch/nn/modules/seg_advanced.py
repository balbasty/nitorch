"""Advanced segmentation networks (by Mikael Brudfors)

MeanSpaceNet : splat/pull images to/from a common space
MRFNet : A recurrent head that refines segmentation by optimising a Markov Random Field
SegMRFNet : A MRFNet plugged at the end of a SegNet

References
----------
..[1] "An MRF-UNet Product of Experts for Image Segmentation"
      Mikael Brudfors, Yaël Balbastre, John Ashburner, Geraint Rees,
      Parashkev Nachev, Sébastien Ourselin, M. Jorge Cardoso
      MIDL (2021)
      https://arxiv.org/abs/2104.05495
..[2] "A Splat Layer for Improved CNN Segmentation at Multiple Resolutions"
      Mikael Brudfors, Yaël Balbastre, John Ashburner, Geraint Rees,
      Parashkev Nachev, Sébastien Ourselin, M. Jorge Cardoso
      In prep. (2021)
"""

import torch
import torch.nn as tnn
from nitorch import spatial
from nitorch.core import utils, math
from nitorch.core.linalg import lmdiv
from .. import check
from nitorch.nn.base import Module
from .cnn import UNet, MRF
from .spatial import GridPull, GridPushCount
from ..generators import RandomDiffeo
from .seg_utils import augment_params, augment, warp_image, warp_label, board
from .segmentation import SegNet


class MeanSpaceNet(Module):
    """Segmentation network that trains in a shared mean space, but computes
    losses in native space.

    OBS: For optimal performance, both memory- and prediction-wise, it is important
    that the input data is aligned in the common space.

    """
    def __init__(self,
                 dim,
                 common_space,
                 output_classes=1,
                 input_channels=1,
                 encoder=None,
                 decoder=None,
                 kernel_size=3,
                 activation=tnn.LeakyReLU(0.2),
                 batch_norm=True,
                 implicit=True,
                 coord_conv=False,
                 bg_class=0,
                 augmentation=None):
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
        bg_class : int, default=0
            Index of background class in reference segmentation.
        augmentation : str or sequence[str], default=None
            Apply various augmentation techniques, available methods are:
            * 'warp' : Nonlinear warp of input image and target label
            * 'noise' : Additive gaussian noise to image
            * 'inu' : Multiplicative intensity non-uniformity (INU) to image

        """
        super().__init__()

        self.bg_class = bg_class
        if not isinstance(augmentation, (list, tuple)):  augmentation = [augmentation]
        augmentation = ['warp-img-lab' if a == 'warp' else a for a in augmentation]
        self.augmentation = augmentation
        self.implicit = implicit
        self.output_classes = output_classes
        self.mean_mat = common_space[0]
        self.vx = spatial.voxel_size(common_space[0])
        self.mean_dim = tuple(common_space[1])
        self.coord_conv = coord_conv
        if implicit and output_classes == 1:
            final_activation = tnn.Sigmoid
        else:
            final_activation = tnn.Softmax(dim=1)
        if implicit:
            output_classes += 1
        # Add tensorboard callback
        self.board = lambda tb, *args, **kwargs: self.board_custom(tb, *args, **kwargs, implicit=implicit, dim=dim)
        # Push/pull settings
        interpolation = 'linear'
        bound = 'dct2'  # symmetric
        extrapolate = False
        # Add push operators
        self.push = GridPushCount(interpolation=interpolation,
                                  bound=bound, extrapolate=extrapolate)
        # Add channel to take into account count image
        input_channels += input_channels
        if coord_conv:
            input_channels += dim
        # Add UNet
        #   no final activation so that pull is performed in log-space
        self.unet = UNet(dim,
                         in_channels=input_channels,
                         out_channels=output_classes,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=[activation, ..., None],
                         norm=batch_norm)
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

    def forward(self, image, mat_native, ix_ref, ref=None, *, _loss=None, _metric=None):
        """MeanSpaceNet forward pass.

        OBS: Supports only batch size one, because images can be of
        different dimensions.

        Parameters
        ----------
        image : [tensor, ...]
            List of n_channels input images.
        mat_native : (n_channels, dim + 1, dim + 1)
            Input images' affine matrices.
        ix_ref : int
            Index of which image the reference maps to,
        ref : [tensor, ...]
            List of ground truth segmentation, used by the loss function.
            None is used for images with no segmentation.
            The ground truth segmentation's data type should be integer if
            it contains hard labels, and floating point if it contains soft segmentations.
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
        # parameters
        n_channels = len(image)  # number of channels
        self.mean_mat = self.mean_mat.type(mat_native.dtype)

        # augment (takes voxel size into account)
        if ref is not None:
            if 'warp' in self.augmentation:
                # warping deformation defined in mean-space
                image, ref = \
                    self.warping_augmentation(image, ref, mat_native, ix_ref)
            # intensity-based
            for n in range(n_channels):
                vx = spatial.voxel_size(mat_native[n, ...])
                for aug_method in self.augmentation:
                    if aug_method == 'warp':  continue
                    image[n], ref[n] = augment(aug_method, image[n], ref[n], vx=vx)

        # Create common space input
        inputs = torch.zeros(((1, 2*n_channels,) +  self.mean_dim),
            device=image[0].device, dtype=image[0].dtype)
        # for loop index, so that pull image's grid is retained for later use
        nix = [x for x in range(n_channels) if x != ix_ref]
        nix.append(ix_ref)
        for n in nix:
            # Compute grid
            grid = self.compute_grid(mat_native[n, ...], image[n].shape[2:])[None]
            # Push image into common space
            inputs[:, n, ...], inputs[:, n_channels + n, ...] = \
                self.push(image[n], grid, shape=self.mean_dim) # add count image as channel

        if self.coord_conv:
            # Append mean space coordinate grid to UNet input
            id = spatial.identity_grid(self.mean_dim, dtype=inputs.dtype, device=inputs.device)
            id = id.unsqueeze(0)
            if self.dim == 3:
                id = id.permute((0, 4, 1, 2, 3))
                id = id.repeat(inputs.shape[0], 1, 1, 1, 1)
            else:
                id = id.permute((0, 3, 1, 2))
                id = id.repeat(inputs.shape[0], 1, 1, 1)
            inputs = torch.cat((inputs, id), dim=1)

        # Apply UNet
        pred = self.unet(inputs)

        # Remove last class, if implicit
        if self.implicit and pred.shape[1] > self.output_classes:
            pred = pred[:, :-1, ...]

        # Pull prediction into native space (whilst in log space)
        pred = self.pull(pred, grid)

        with torch.no_grad():
            # deal with background voxels
            bg = (pred.sum(dim=1, keepdim=False) == 0).type(pred.dtype)
            bg[bg == 1] = bg[bg == 1] + pred.max()
        pred[:, self.bg_class, ...] = pred[:, self.bg_class, ...] + bg

        # softmax
        pred = pred.softmax(dim=1)

        # i = 0
        # for i in range(2*n_channels):
        #     debug_view(inputs, ix_channel=i, fig_num=i)
        # debug_view(ref[ix_ref], one_hot=True, fig_num=i + 1)
        # debug_view(pred, one_hot=True, fig_num=i + 2)

        # compute loss and metrics (in native space)
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref[ix_ref])
            dims = [0] + list(range(2, self.dim + 2))
            check.shape(pred, ref[ix_ref], dims=dims)
            self.compute(_loss, _metric, native=[pred, ref[ix_ref]])

        return pred

    def warping_augmentation(self, image, ref, mats, ix_ref):
        """Applies a random nonlinear deformation to images and reference.
        The deformation is defined in the mean-space, and then composed to each
        image's native space by: inv_mat_image(mat_aff(diffeo(mat_mean))), here,
        the random affine matrix (mat_aff) is identity.

        Parameters
        ----------
        image : [tensor, ...]
            List of n_channels input images.
        ref : [tensor, ...]
            List of ground truth segmentation, used by the loss function.
            None is used for images with no segmentation.
        mats : (n_channels, dim + 1, dim + 1)
            Input images' affine matrices.
        ix_ref : int
            Index of which image the reference maps to,

        Returns
        ----------
        image : [tensor, ...]
            List of n_channels input images, with augmentation applied.
        ref : [tensor, ...]
            List of ground truth segmentation, used by the loss function,
            with augmentation applied.

        """
        # augmentation parameters
        amplitude = augment_params['warp']['amplitude']
        fwhm = (augment_params['warp']['fwhm'],) * self.dim
        fwhm = [f / v for f, v in zip(fwhm, self.vx)]  # modulate FWHM with voxel size
        # instantiate augmenter
        aug = RandomDiffeo(amplitude=amplitude, fwhm=fwhm, bound='zero',
                           device=image[0].device, dtype=image[0].dtype)
        # get random grid in mean space
        grid0 = aug(batch=1, shape=self.mean_dim, dim=self.dim)[0, ...]
        # loop over images
        for n in range(len(image)):
            # do composition to native space
            dm = image[n].shape[2:]
            grid = self.compose(mats[n, ...], grid0, self.mean_mat, shape_out=dm)[None]
            # apply deformation
            image[n] = warp_image(image[n], grid)
            if n == ix_ref:
                ref[n] = warp_label(ref[n], grid)

        return image, ref

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
        mat = lmdiv(self.mean_mat, mat_native)
        grid = spatial.affine_grid(mat, dim_native)

        return grid

    def board_custom(self, tb, inputs, outputs, *args, **kwargs):
        """Removes affine matrix from inputs before calling board.
        """
        ix_ref = inputs[2]
        image = inputs[0][ix_ref]
        label = outputs[ix_ref]
        board(tb, (image, label), *args, **kwargs)

    def compose(self, orient_in, deformation, orient_mean, affine=None, orient_out=None, shape_out=None):
        """Composes a deformation defined in a mean space to an image space.

        Parameters
        ----------
        orient_in : (4, 4) tensor
            Orientation of the input image
        deformation : (*shape_mean, 3) tensor
            Random deformation
        orient_mean : (4, 4) tensor
            Orientation of the mean space (where the deformation is)
        affine : (4, 4) tensor, default=identity
            Random affine
        orient_out : (4, 4) tensor, default=orient_in
            Orientation of the output image
        shape_out : sequence[int], default=shape_mean
            Shape of the output image

        Returns
        -------
        grid : (*shape_out, 3)
            Voxel-to-voxel transform

        """
        if orient_out is None:
            orient_out = orient_in
        if shape_out is None:
            shape_out = deformation.shape[:-1]
        if affine is None:
            affine = torch.eye(4, 4,
                device=orient_in.device, dtype=orient_in.dtype)
        shape_mean = deformation.shape[:-1]

        orient_in, affine, deformation, orient_mean, orient_out \
            = utils.to_max_backend(orient_in, affine, deformation, orient_mean, orient_out)
        backend = utils.backend(deformation)
        eye = torch.eye(4, **backend)

        # Compose deformation on the right
        right_affine = spatial.affine_lmdiv(orient_mean, orient_out)
        if not (shape_mean == shape_out and right_affine.all_close(eye)):
            # the mean space and native space are not the same
            # we must compose the diffeo with a dense affine transform
            # we write the diffeo as an identity plus a displacement
            # (id + disp)(aff) = aff + disp(aff)
            # -------
            # to displacement
            deformation = deformation - spatial.identity_grid(deformation.shape[:-1], **backend)
            trf = spatial.affine_grid(right_affine, shape_out)
            deformation = spatial.grid_pull(utils.movedim(deformation, -1, 0)[None], trf[None],
                                    bound='dft', extrapolate=True)
            deformation = utils.movedim(deformation[0], 0, -1)
            trf = trf + deformation  # add displacement

        # Compose deformation on the left
        #   the output of the diffeo(right) are mean_space voxels
        #   we must compose on the left with `in\(aff(mean))`
        # -------
        left_affine = spatial.affine_matmul(spatial.affine_inv(orient_in), affine)
        left_affine = spatial.affine_matmul(left_affine, orient_mean)
        trf = spatial.affine_matvec(left_affine, trf)

        return trf


class SegMRFNet(Module):
    """Combines a SegNet with a MRFNet.

    The idea is that a simple, explicit, spatial prior on the categorical data
    should improve generalisation to out-of-distribution data, and allow for less
    parameters. The idea is described in:

    Brudfors, M., Balbastre, Y., Ashburner, J., Rees, G., Nachev, P., Ourselin, S., & Cardoso, M. J.
    "An MRF-UNet Product of Experts for Image Segmentation."
    MIDL 2021.

    """
    def __init__(
            self,
            dim,
            output_classes=1,
            input_channels=1,
            encoder=None,
            decoder=None,
            kernel_size=3,
            activation=tnn.LeakyReLU(0.2),
            batch_norm_seg=True,
            num_iter=10,
            w=1.0,
            num_extra=0,
            only_unet=False,
            augmentation=None):
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
        augmentation : str or sequence[str], default=None
            Apply various augmentation techniques, available methods are:
            * 'warp' : Nonlinear warp of input image and target label
            * 'noise' : Additive gaussian noise to image
            * 'inu' : Multiplicative intensity non-uniformity (INU) to image

        """
        super().__init__()

        self.only_unet = only_unet
        if not isinstance(augmentation, (list, tuple)):  augmentation = [augmentation]
        augmentation = ['warp-img-lab' if a == 'warp' else a for a in augmentation]
        self.augmentation = augmentation
        # Add tensorboard callback
        self.board = lambda tb, *args, **kwargs: board(tb, *args, **kwargs, dim=dim)

        self.unet = SegNet(dim,
                           input_channels=input_channels,
                           output_classes=output_classes,
                           encoder=encoder,
                           decoder=decoder,
                           kernel_size=kernel_size,
                           activation=[activation, ..., None],
                           norm=batch_norm_seg,
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
            for aug_method in self.augmentation:
                image, ref = augment(aug_method, image, ref)

        # unet
        p = self.unet(image)

        if self.only_unet:
            p = p.softmax(dim=1)
        else:
            p = p - math.logsumexp(p, dim=1, keepdim=True)
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



class MRFNet(Module):
    """MRF network.

    A network that aims at cleaning up categorical image data of num_classes
    classes. The first layer is a, so called, MRF layer. Sub-sequent layers are
    then 1x1 convolutions layers. The output of the network are softmaxed,
    categorical data. The idea is described in:

    Brudfors, Mikael, Yaël Balbastre, and John Ashburner.
    "Nonlinear markov random fields learned via backpropagation."
    IPMI 2019.

    """
    def __init__(self,
                 dim,
                 num_classes,
                 num_iter=10,
                 num_extra=0,
                 w=1.0,
                 activation=tnn.LeakyReLU(0.2),
                 augmentation=False):
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
        augmentation : bool, default=False
            Nonlinear warp of input label and target label.

        """
        super().__init__()

        self.num_classes = num_classes
        self.augmentation = augmentation
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
        self.board = lambda tb, *args, **kwargs: board(tb, *args, **kwargs, dim=dim)
        # Build MRF net
        self.mrf = MRF(dim,
                       num_classes=num_classes,
                       num_filters=num_filters,
                       num_extra=num_extra,
                       kernel_size=kernel_size,
                       activation=[activation, ..., final_activation],
                       batch_norm=False,
                       bias=True)
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

        if ref is not None and self.augmentation:
            resp, ref = augment('warp-lab-lab', resp, ref)

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
        kernel_size = grad.shape[2:]
        center = tuple(k//2 for k in kernel_size)
        center = (slice(None),) * 2 + center
        grad_clone = grad.clone()
        grad_clone[center] = 0
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

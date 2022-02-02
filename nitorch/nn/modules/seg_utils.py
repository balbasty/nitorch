from nitorch import spatial
from ..generators import (RandomBiasFieldTransform, RandomDiffeo)
import torch
from ...core.constants import eps


# Default parameters for various augmentation techniques
augment_params = {'inu': {'amplitude': 0.25, 'fwhm': 15.0},
                  'warp': {'amplitude': 2.0, 'fwhm': 15.0},
                  'noise': {'std_prct': 0.025}}


def board2(self, tb, inputs=None, outputs=None, epoch=None, minibatch=None, mode=None,
           implicit=False, do_eval=True, do_train=True, **kwargs):
    if not do_eval and mode == 'eval':
        return
    if not do_train and mode == 'train':
        return
    if inputs is None:
        return
    from nitorch.plot import get_orthogonal_slices, get_slice
    from nitorch.plot.colormaps import prob_to_rgb, intensity_to_rgb
    import matplotlib.pyplot as plt

    image, ref = inputs
    pred = outputs
    fig = plt.figure()

    if image.dim() - 2 == 2:
        image = image[0, 0]
        image = intensity_to_rgb(image)
        nk = pred.shape[1] + implicit
        pred = pred[0]
        pred = prob_to_rgb(pred, implicit=implicit)
        if ref.dtype in (torch.float, torch.double):
            ref = ref[0]
        else:
            ref = ref[0, 0]
            ref = torch.stack \
                ([ref == i for i in range(1, ref.max().item( ) +1)]).float()
        ref = prob_to_rgb(ref, implicit=ref.shape[0] < nk)
        plt.subplot(1, 3, 1)
        plt.imshow(image.detach().cpu())
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(pred.detach().cpu())
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(ref.detach().cpu())
        plt.axis('off')
    else:
        images = get_orthogonal_slices(image[0, 0])
        images = [intensity_to_rgb(image) for image in images]
        nk = pred.shape[1] + implicit
        preds = get_orthogonal_slices(pred[0])
        preds = [prob_to_rgb(pred, implicit=implicit) for pred in preds]
        if ref.dtype in (torch.float, torch.double):
            refs = get_orthogonal_slices(ref[0])
        else:
            mx = ref.max().item()
            refs = get_orthogonal_slices(ref[0, 0])
            refs = [torch.stack
                ([ref == i for i in range(1, mx + 1)]).float() for ref in refs]
        refs = [prob_to_rgb(ref, implicit=ref.shape[0] < nk) for ref in refs]
        plt.subplot(3, 3, 1)
        plt.imshow(images[0].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 4)
        plt.imshow(images[1].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 7)
        plt.imshow(images[2].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 2)
        plt.imshow(preds[0].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 5)
        plt.imshow(preds[1].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 8)
        plt.imshow(preds[2].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 3)
        plt.imshow(refs[0].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 6)
        plt.imshow(refs[1].detach().cpu())
        plt.axis('off')
        plt.subplot(3, 3, 9)
        plt.imshow(refs[2].detach().cpu())
        plt.axis('off')


    if not hasattr(self, 'tbstep'):
        self.tbstep = dict()
    self.tbstep.setdefault(mode, 0)
    self.tbstep[mode] += 1
    tb.add_figure(f'prediction/{mode}', fig, global_step=self.tbstep[mode])



def board(tb, inputs=None, outputs=None, epoch=None, minibatch=None,
          mode=None, loss=None, losses=None, metrics=None, implicit=False, dim=3):
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
            slice_target = (slice_target.argmax(dim=0, keepdim=False) ) / \
                        (K1 - 1)
        else:
            slice_target =  slice_target.float() / slice_target.max().float()
        return slice_target

    def to_grid(slice_input, slice_target, slice_prediction):
        return torch.cat((slice_input, slice_target, slice_prediction), dim=1)

    def get_slices(plane, inputs, outputs, dim, implicit):
        slice_input = input_view(get_slice(inputs[0][0, ...], plane, dim=dim))
        slice_target = target_view \
            (get_slice(inputs[1][0, ...], plane, dim=dim))
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
            slice_target = (slice_target.argmax(dim=0, keepdim=False)) / \
                        (K1 - 1)
        return to_grid(slice_input, slice_target, slice_prediction)[None, ...]

    if inputs is None or outputs is None:
        return
    # Add to TensorBoard
    title = 'Image-Target-Prediction_'
    tb.add_image(title + 'z', get_image('z', inputs, outputs, dim, implicit))
    if dim == 3:
        tb.add_image(title + 'y',
                     get_image('y', inputs, outputs, dim, implicit))
        tb.add_image(title + 'x',
                     get_image('x', inputs, outputs, dim, implicit))
    tb.flush()


def debug_view(dat, ix_batch=0, ix_channel=0, one_hot=False, fig_num=1):
    """A simple viewer for inspecting network inputs/outputs.
    """
    from nitorch.plot import show_slices
    if one_hot:
        show_slices(dat[ix_batch, ...].argmax(dim=0, keepdim=False),
                    fig_num=fig_num)
    else:
        show_slices(dat[ix_batch, ix_channel, ...], fig_num=fig_num)


def augment(method, image, label=None, vx=None):
    """Augmentation methods for segmentation network, with parameters that
    should, hopefully, work well by default.

    OBS: Grount truth input only required when doing warping augmentation.

    Parameters
    -------
    method : str
        Augmentation method:
        'warp-img-img' : Nonlinear warp of input image and target image
        'warp-img-lab' : Nonlinear warp of input image and target label
        'warp-lab-img' : Nonlinear warp of input label and target image
        'warp-lab-lab' : Nonlinear warp of input label and target label
        'noise-gauss' : Additive gaussian noise to image
        'inu' : Multiplicative intensity non-uniformity (INU) to image
    image : (batch, input_channels, *spatial) tensor
        Input image
    label : (batch, output_classes[+1], *spatial) tensor, optional
        Ground truth segmentation, used by the loss function.
        Its data type should be integer if it contains hard labels,
        and floating point if it contains soft segmentations.
    vx : [ndim, ] sequence, optional
        Image voxel size (in mm), defaults to 1 mm isotropic.

    Returns
    -------
    image : (batch, input_channels, *spatial) tensor
        Augmented input image.
    label : (batch, output_classes[+1], *spatial) tensor, optional
        Augmented ground truth segmentation.

    """
    if method is None:
        return image, label
    # sanity check
    valid_methods = ['warp-img-img', 'warp-img-lab', 'warp-lab-img',
                     'warp-lab-lab', 'noise-gauss', 'inu']
    if method not in valid_methods:
        raise ValueError(
            'Undefined method {:}, need to be one of {:}'.format(method,
                                                                 valid_methods))
    nbatch = image.shape[0]
    nchan = image.shape[1]
    dim = tuple(image.shape[2:])
    ndim = len(dim)
    nvox = int(torch.as_tensor(image.shape[2:]).prod())
    # voxel size
    if vx is None:
        vx = (1.0,) * ndim
    vx = torch.as_tensor(vx, device=image.device, dtype=image.dtype)
    vx = vx.clamp_min(1.0)
    # Augmentation method
    if 'warp' in method:
        # Nonlinear warp
        # Parameters
        amplitude = augment_params['warp']['amplitude']
        fwhm = (augment_params['warp']['fwhm'],) * ndim
        fwhm = [f / v for f, v in
                zip(fwhm, vx)]  # modulate FWHM with voxel size
        # Instantiate augmenter
        aug = RandomDiffeo(amplitude_exp=amplitude, fwhm_exp=fwhm, bound='zero',
                           device=image.device, dtype=image.dtype)
        # Get random grid
        grid = aug(batch=nbatch, shape=dim)
        # Warp
        if method == 'warp-img-img':
            image = warp_image(image, grid)
            if label is not None:
                label = warp_image(label, grid)
        elif method == 'warp-img-lab':
            image = warp_image(image, grid)
            if label is not None:
                label = warp_label(label, grid)
        elif method == 'warp-lab-img':
            image = warp_label(image, grid)
            if label is not None:
                label = warp_image(label, grid)
        elif method == 'warp-lab-lab':
            image = warp_label(image, grid)
            if label is not None:
                label = warp_label(label, grid)
        else:
            raise ValueError('')
    elif method == 'noise-gauss':
        # Additive gaussian noise to image
        # Parameter
        std_prct = augment_params['noise'][
            'std_prct']  # percentage of max intensity of batch and channel
        # Get max intensity in for each batch and channel
        mx = image.reshape((nbatch, nchan, nvox)).max(dim=-1, keepdim=True)[0]
        # Add 'lost' dimensions
        for d in range(ndim - 1):
            mx = mx.unsqueeze(-1)
        # Add noise to image
        image += std_prct * mx * torch.randn_like(image)
    elif method == 'inu':
        # Multiplicative intensity non-uniformity (INU) to image
        # Parameters
        amplitude = augment_params['inu']['amplitude']
        fwhm = (augment_params['inu']['fwhm'],) * ndim
        fwhm = [f / v for f, v in
                zip(fwhm, vx)]  # modulate FWHM with voxel size
        # Instantiate augmenter
        aug = RandomBiasFieldTransform(amplitude=amplitude, fwhm=fwhm)
        # Augment image
        image = aug(image)

    return image, label


def warp_image(image, grid):
    """Warp image according to grid.
    """
    image = spatial.grid_pull(image, grid,
                              bound='dct2', extrapolate=True, interpolation=1)

    return image


def warp_label(label, grid):
    """Warp label image according to grid.
    """
    ndim = len(label.shape[2:])
    dtype_seg = label.dtype
    if dtype_seg not in (torch.half, torch.float, torch.double):
        # hard labels to one-hot labels
        n_batch = label.shape[0]
        u_labels = label.unique()
        n_labels = len(u_labels)
        label_w = torch.zeros((n_batch, n_labels,) + tuple(label.shape[2:]),
                              device=label.device, dtype=torch.float32)
        for i, l in enumerate(u_labels):
            label_w[..., i, ...] = label == l
    else:
        label_w = label
    # warp
    label_w = spatial.grid_pull(label_w, grid,
                                bound='dct2', extrapolate=True,
                                interpolation=1)
    if dtype_seg not in (torch.half, torch.float, torch.double):
        # one-hot labels to hard labels
        label_w = label_w.argmax(dim=1, keepdim=True).type(dtype_seg)
    else:
        # normalise one-hot labels
        label_w = label_w / (label_w.sum(dim=1, keepdim=True) + eps())

    return label_w

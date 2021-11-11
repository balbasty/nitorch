import torch
from nitorch.core import utils, py
from nitorch import io
import math
import random


class Dataset:
    """Base class for datasets"""

    def __init__(self, dim=None, qmin=0, qmax=0.95, shape=None, shape_min=None,
                 shape_max=None, shape_mult=None, device=None, dtype=None):
        self.qmin = qmin
        self.qmax = qmax
        self.shape = shape
        self.shape_min = shape_min
        self.shape_max = shape_max
        self.shape_mult = shape_mult
        self.dim = dim
        self.dtype = dtype
        self.device = device

    def rescale(self, image):
        """Affine rescaling of an image by mapping quantiles to (0, 1)

        Parameters
        ----------
        image : tensor
            Input image
        qmin : (0..1), default=0
            Lower quantile
        qmax : (0..1), default=0.95
            Upper quantile

        Returns
        -------
        image : tensor
            Rescale image

        """
        if self.qmin == 0 and self.qmax == 1:
            return image
        qmin, qmax = utils.quantile(image, [self.qmin, self.qmax])
        image -= qmin
        image /= (qmax - qmin)
        return image

    def to_shape(self, image, bound='zero'):
        """Crop/Pad a volume to match a target shape

        Parameters
        ----------
        image : (channel, *spatial) tensor
            Input image
        bound : str, default='zero'
            Method to fill out-of-bounds.

        Returns
        -------
        image : (channel, *shape) tensor
            Cropped/padded image

        """
        oshape = image.shape[1:]
        if self.shape:
            oshape = (*image.shape[:-len(self.shape)], *self.shape)
            return utils.ensure_shape(image, oshape, mode=bound, side='both')
        if self.shape_min:
            shape_min = py.ensure_list(self.shape_min, len(oshape))
            oshape = [max(s, mn) for s, mn in zip(oshape, shape_min)]
        if self.shape_max:
            shape_max = py.ensure_list(self.shape_max, len(oshape))
            oshape = [min(s, mx) for s, mx in zip(oshape, shape_max)]
        if self.shape_mult:
            shape_mult = py.ensure_list(self.shape_mult, len(oshape))
            oshape = [(s//m)*m for s, m in zip(oshape, shape_mult)]
        oshape = (*image.shape[:-len(oshape)], *oshape)
        return utils.ensure_shape(image, oshape, mode=bound, side='both')

    def load(self, fname, dtype=None, device=None):
        """Load a volume from disk

        Parameters
        ----------
        fname : str
        dtype : torch.dtype, optional

        Returns
        -------
        dat : (channels, *spatial) tensor

        """
        dtype = dtype or self.dtype
        device = device or self.device
        if not dtype or dtype.is_floating_point:
            dat = io.loadf(fname, dtype=dtype, device=device)
            dat = self.rescale(dat)
        else:
            dat = io.load(fname, dtype=dtype, device=device)
        dat = dat.squeeze()
        dim = self.dim or dat.dim()
        dat = utils.unsqueeze(dat, -1, max(0, dim - dat.dim()))
        dat = dat.reshape([*dat.shape[:dim], -1])
        dat = utils.movedim(dat, -1, 0)
        dat = self.to_shape(dat)
        return dat

    def loadvol(self, fnames, dtype=None, device=None):
        """Load a volume from disk

        Parameters
        ----------
        fnames : str or sequence[str]
        dtype : torch.dtype, optional

        Returns
        -------
        dat : (channels, *spatial) tensor

        """
        fnames = py.make_list(fnames)
        channels = []
        for fname in fnames:  # loop across channels
            dat = self.load(fname, dtype=dtype, device=device)
            channels.append(dat)
        if len(channels) > 1:
            channels = torch.cat(channels, 0)
        else:
            channels = channels[0]
        return channels

    def loadseg(self, fnames, segtype='label', lookup=None,
                dtype=None, device=None):
        """Load a volume from disk

        Parameters
        ----------
        fnames : str or sequence[str]
        segtype : [tuple] {'label', 'implicit', 'explicit'}, default='label'
        lookup : list of [list of] int, optional
        dtype : torch.dtype, optional

        Returns
        -------
        dat : (channels | 1, *spatial) tensor

        """
        insegtype, outsegtype = py.make_list(segtype, 2)
        sdtype = dtype if insegtype != 'label' else torch.int
        fnames = py.make_list(fnames)
        channels = []
        for fname in fnames:  # loop across channels
            dat = self.load(fname, dtype=sdtype, device=device)
            channels.append(dat)
        if len(channels) > 1:
            channels = torch.cat(channels, 0)
        else:
            channels = channels[0]
        if insegtype == 'label' and lookup:
            channels = utils.merge_labels(channels, lookup)
        if insegtype == 'label' and outsegtype != 'label':
            channels = utils.one_hot(channels, dim=0,
                                     implicit=outsegtype == 'implicit',
                                     implicit_index=0, dtype=dtype)
            if outsegtype == 'implicit':
                channels /= len(channels) + 1
            else:
                channels /= len(channels)
        return channels

    def __len__(self):
        return int(math.ceil(self.nb_dat / self.batch_size))

    def __iter__(self):
        return self.iter()


class UnsupervisedDataset(Dataset):
    """A simple dataset"""

    def __init__(self, filenames, batch_size=1, **kwargs):
        """

        Parameters
        ----------
        filenames : sequence of [sequence of] str
            List of filenames.
            If nested, the inner loop corresponds to channels.
        batch_size : int, default=1
            Batch size
        qmin : (0..1), default=0
            Lower quantile
        qmax : (0..1), default=0.95
            Upper quantile
        dim : int, optional
            Number of spatial dimensions.
        shape : sequence[int], optional
            Crop/pad images to this shape.
        device : torch.device, optional
            Load the data to a specific device
        dtype : torch.dtype, optional
            Load the data in a specific data type.
        """
        super().__init__(**kwargs)
        self.filenames = list(filenames)
        self.nb_dat = len(self.filenames)
        self.batch_size = batch_size

    def iter(self, batch_size=None, device=None, dtype=None):
        """Dataset iterator

        Load, concatenate and yield images from the dataset.

        Parameters
        ----------
        batch_size : int, default=self.batch_size
        device : torch.device, default=self.device
        dtype : torch.dtype, default=self.dtype

        Returns
        -------
        dat : (batch_size, channels, *spatial) tensor

        """
        batch_size = batch_size or self.batch_size
        dtype = dtype or self.dtype or torch.get_default_dtype()
        filenames = list(self.filenames)

        while filenames:
            dats = []
            for _ in range(batch_size):
                if not filenames:
                    break
                channels = self.loadvol(filenames.pop(0),
                                        dtype=dtype,
                                        device=device)
                dats.append(channels)
            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            yield dats


class DatasetWithSeg(Dataset):
    """A dataset with ground truth segmentations"""

    def __init__(self, filenames, segnames, segtype='label', lookup=None,
                 batch_size=1, **kwargs):
        """

        Parameters
        ----------
        filenames : sequence of [sequence of] str
            List of image filenames.
            If nested, the inner loop corresponds to channels.
        segnames : sequence of [sequence of] str
            List of segmentation filenames.
            If nested, the inner loop corresponds to classes.
        segtype : [tuple] {'implicit', 'explicit', 'label'}, default='label'
            Type of segmentation.
                'label' : each class has a unique identifier
                'explicit' : each class has its own channel, including the background
                'implicit' : each class has its own channel, except the background
            If a tuple, the first element corresponds to the on-disk
            layout while the second element corresponds to the required
            in-memory layout.
        lookup : list of [list of] int, optional
            Mapping from soft index to label value.
            Only used if `segtype` is ('label', 'implicit' or 'explicit')
        batch_size : int, default=1
            Batch size
        qmin : (0..1), default=0
            Lower quantile
        qmax : (0..1), default=0.95
            Upper quantile
        dim : int, optional
            Number of spatial dimensions.
        device : torch.device, optional
            Load the data to a specific device
        dtype : torch.dtype, optional
            Load the data in a specific data type.
        """
        super().__init__(**kwargs)
        self.filenames = list(filenames)
        self.segnames = list(segnames)
        self.nb_dat = len(self.filenames)
        self.segtype = segtype
        self.lookup = lookup
        self.batch_size = batch_size

    def iter(self, batch_size=None, device=None, dtype=None):
        """Dataset iterator

        Load, concatenate and yield images from the dataset.

        Parameters
        ----------
        batch_size : int, default=self.batch_size
        device : torch.device, default=self.device
        dtype : torch.dtype, default=self.dtype

        Returns
        -------
        dat : (batch_size, channels, *spatial) tensor
        seg : (batch_size, channels | 1, *spatial) tensor

        """
        batch_size = batch_size or self.batch_size
        dtype = dtype or self.dtype or torch.get_default_dtype()
        filenames = list(self.filenames)
        segnames = list(self.segnames)

        while filenames:
            dats = []
            segs = []
            for _ in range(batch_size):
                if not filenames or not segnames:
                    break
                # images
                channels = self.loadvol(filenames.pop(0),
                                        dtype=dtype,
                                        device=device)
                dats.append(channels)

                # segmentations
                channels = self.loadseg(segnames.pop(0),
                                        segtype=self.segtype,
                                        lookup=self.lookup,
                                        dtype=dtype,
                                        device=device)
                segs.append(channels)

            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            segs = torch.stack(segs) if len(segs) > 1 else segs[0][None]
            yield dats, segs


class PairedDataset(Dataset):
    """A dataset made of pairs of images"""

    def __init__(self, filenames, refnames=None, pairs=None,
                 batch_size=1, **kwargs):
        """

        Parameters
        ----------
        filenames : sequence of [sequence of] str
            First list of filenames.
            If nested, the inner loop corresponds to channels.
        refnames : sequence of [sequence of] str, optional
            Second list of filenames.
            If nested, the inner loop corresponds to channels.
            Exactly one of `pairs` or `refnames` must be provided.
        pairs : int or sequence of (int, int), optional
            List of pairs.
            In an int, maximum number of pairs to build.
            Exactly one of `pairs` or `refnames` must be provided.
        batch_size : int, default=1
            Batch size
        qmin : (0..1), default=0
            Lower quantile
        qmax : (0..1), default=0.95
            Upper quantile
        dim : int, optional
            Number of spatial dimensions.
        shape : sequence[int], optional
            Crop/pad to ths shape
        device : torch.device, optional
            Load the data to a specific device
        dtype : torch.dtype, optional
            Load the data in a specific data type.
        """
        super().__init__(**kwargs)
        filenames = list(filenames)
        if refnames:
            self.filenames = filenames
            self.refnames = list(refnames)
        else:
            n = len(filenames)
            pairs = pairs or (n*(n-1)//2)
            if isinstance(pairs, int):
                nmax = pairs
                pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                state = random.getstate()
                random.seed(1234)
                random.shuffle(pairs)
                random.setstate(state)
                pairs = pairs[:nmax]
            self.pairs = pairs
            self.filenames = [filenames[i] for i, _ in pairs]
            self.refnames = [filenames[j] for _, j in pairs]
        self.nb_dat = len(self.filenames)
        self.batch_size = batch_size

    def iter(self, batch_size=None, device=None, dtype=None):
        """Dataset iterator

        Load, concatenate and yield images from the dataset.

        Parameters
        ----------
        batch_size : int, default=self.batch_size
        device : torch.device, default=self.device
        dtype : torch.dtype, default=self.dtype

        Returns
        -------
        dat : (batch_size, channels, *spatial) tensor
        ref : (batch_size, channels, *spatial) tensor

        """
        batch_size = batch_size or self.batch_size
        device = device or self.device
        dtype = dtype or self.dtype or torch.get_default_dtype()
        filenames = list(self.filenames)
        refnames = list(self.refnames)

        while filenames and refnames:
            if not filenames or not refnames:
                break
            dats = []
            refs = []
            for _ in range(batch_size):
                channels = self.loadvol(filenames.pop(0),
                                        dtype=dtype,
                                        device=device)
                dats.append(channels)

                channels = self.loadvol(refnames.pop(0),
                                        dtype=dtype,
                                        device=device)
                refs.append(channels)
            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            refs = torch.stack(refs) if len(refs) > 1 else refs[0][None]
            yield dats, refs


class PairedDatasetWithSeg(Dataset):
    """A dataset made of pairs of images and segmentations"""

    def __init__(self, filenames, segnames, refnames=None, segrefnames=None,
                 pairs=None, batch_size=1, segtype='label', segreftype=None,
                 lookup=None, **kwargs):
        """

        Parameters
        ----------
        filenames : sequence of [sequence of] str
            First list of filenames.
            If nested, the inner loop corresponds to channels.
        segnames : sequence of [sequence of] str
            List of segmentation filenames.
            If nested, the inner loop corresponds to classes.
        refnames : sequence of [sequence of] str, optional
            Second list of filenames.
            If nested, the inner loop corresponds to channels.
            Exactly one of `pairs` or `refnames` must be provided.
        segrefnames : sequence of [sequence of] str
            List of segmentation filenames.
            If nested, the inner loop corresponds to classes.
        pairs : int or sequence of (int, int), optional
            List of pairs.
            In an int, maximum number of pairs to build.
            Exactly one of `pairs` or `refnames` must be provided.
        batch_size : int, default=1
            Batch size
        segtype : [tuple] {'implicit', 'explicit', 'label'}, default='label'
            Type of segmentation.
                'label' : each class has a unique identifier
                'explicit' : each class has its own channel, including the background
                'implicit' : each class has its own channel, except the background
            If a tuple, the first element corresponds to the on-disk
            layout while the second element corresponds to the required
            in-memory layout.
        segreftype : [tuple] {'implicit', 'explicit', 'label'}, default=`segtype`
        lookup : list of [list of] int, optional
            Mapping from soft index to label value.
            Only used if `segtype` is ('label', 'implicit' or 'explicit')
        qmin : (0..1), default=0
            Lower quantile
        qmax : (0..1), default=0.95
            Upper quantile
        dim : int, optional
            Number of spatial dimensions.
        device : torch.device, optional
            Load the data to a specific device
        dtype : torch.dtype, optional
            Load the data in a specific data type.
        """
        super().__init__(**kwargs)
        filenames = list(filenames)
        segnames = list(segnames)
        if refnames:
            self.filenames = filenames
            self.segnames = segnames
            self.refnames = list(refnames)
            self.segrefnames = list(segrefnames)
        else:
            n = len(filenames)
            pairs = pairs or (n*(n-1)//2)
            if isinstance(pairs, int):
                nmax = pairs
                pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                state = random.getstate()
                random.seed(1234)
                random.shuffle(pairs)
                random.setstate(state)
                pairs = pairs[:nmax]
            self.pairs = pairs
            self.filenames = [filenames[i] for i, _ in pairs]
            self.refnames = [filenames[j] for _, j in pairs]
            self.segnames = [segnames[i] for i, _ in pairs]
            self.segrefnames = [segnames[j] for _, j in pairs]
        self.nb_dat = len(self.filenames)
        self.batch_size = batch_size
        self.segtype = segtype
        self.segreftype = segreftype or segtype
        self.lookup = lookup

    def iter(self, batch_size=None, device=None, dtype=None):
        """Dataset iterator

        Load, concatenate and yield images from the dataset.

        Parameters
        ----------
        batch_size : int, default=self.batch_size
        device : torch.device, default=self.device
        dtype : torch.dtype, default=self.dtype

        Returns
        -------
        dat : (batch_size, channels, *spatial) tensor
        ref : (batch_size, channels, *spatial) tensor
        seg : (batch_size, channels | 1, *spatial) tensor
        segref : (batch_size, channels | 1, *spatial) tensor

        """
        batch_size = batch_size or self.batch_size
        filenames = list(self.filenames)
        segnames = list(self.segnames)
        refnames = list(self.refnames)
        segrefnames = list(self.segrefnames)

        while filenames and refnames:
            if not filenames or not refnames:
                break
            dats = []
            segs = []
            refs = []
            segrefs = []
            for _ in range(batch_size):
                channels = self.loadvol(filenames.pop(0),
                                        dtype=dtype, device=device)
                dats.append(channels)

                channels = self.loadseg(segnames.pop(0),
                                        segtype=self.segtype,
                                        lookup=self.lookup,
                                        dtype=dtype)
                segs.append(channels)

                channels = self.loadvol(refnames.pop(0),
                                        dtype=dtype, device=device)
                refs.append(channels)

                channels = self.loadseg(segrefnames.pop(0),
                                        segtype=self.segreftype,
                                        lookup=self.lookup,
                                        dtype=dtype)
                segrefs.append(channels)
            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            refs = torch.stack(refs) if len(refs) > 1 else refs[0][None]
            segs = torch.stack(segs) if len(segs) > 1 else segs[0][None]
            segrefs = torch.stack(segrefs) if len(segrefs) > 1 else segrefs[0][None]
            yield dats, refs, segs, segrefs


class SynthDataset(Dataset):
    """A dataset of segmentations to generate synthetic data, as per SynthSeg methods.
       Also optionally pass paired images for segmentation, e.g. for training SynthSR.
    """

    def __init__(self, segnames, imgnames=None,
                 segtype='label', lookup=None,
                 batch_size=1,
                 channel=1,
                 vel_amplitude=3,
                 vel_fwhm=20,
                 translation=20,
                 rotation=15,
                 zoom=0.15,
                 shear=0.012,
                 gmm_fwhm=10,
                 bias_amplitude=0.5,
                 bias_fwhm=50,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 noise=48,
                 gfactor_amplitude=0.5,
                 gfactor_fwhm=64,
                 gmm_cat='labels',
                 bag=0.5,
                 droppable_labels=None,
                 predicted_labels=None,
                 **kwargs):
        """
        Parameters
        ----------
        imgnames : sequence of [sequence of] str
            List of image filenames.
            If nested, the inner loop corresponds to channels.
        segnames : sequence of [sequence of] str
            List of segmentation filenames.
            If nested, the inner loop corresponds to classes.
        segtype : [tuple] {'implicit', 'explicit', 'label'}, default='label'
            Type of segmentation.
                'label' : each class has a unique identifier
                'explicit' : each class has its own channel, including the background
                'implicit' : each class has its own channel, except the background
            If a tuple, the first element corresponds to the on-disk
            layout while the second element corresponds to the required
            in-memory layout.
        lookup : list of [list of] int, optional
            Mapping from soft index to label value.
            Only used if `segtype` is ('label', 'implicit' or 'explicit')
        batch_size : int, default=1
            Batch size
        dim : int, optional
            Number of spatial dimensions.
        device : torch.device, optional
            Load the data to a specific device
        dtype : torch.dtype, optional
            Load the data in a specific data type.
        """
        super().__init__(**kwargs)
        self.segnames = list(segnames)
        if imgnames:
            self.imgnames = list(imgnames)
        else:
            self.imgnames = None
        self.nb_dat = len(self.segnames)
        self.segtype = segtype
        self.lookup = lookup
        self.batch_size = batch_size
        from nitorch.nn.generators import SynthMRI
        self.brain_generator = SynthMRI(
            channel=channel,
            vel_amplitude=vel_amplitude,
            vel_fwhm=vel_fwhm,
            translation=translation,
            rotation=rotation,
            zoom=zoom,
            shear=shear,
            gmm_fwhm=gmm_fwhm,
            bias_amplitude=bias_amplitude,
            bias_fwhm=bias_fwhm,
            gamma=gamma,
            motion_fwhm=motion_fwhm,
            resolution=resolution,
            noise=noise,
            gfactor_amplitude=gfactor_amplitude,
            gfactor_fwhm=gfactor_fwhm,
            gmm_cat=gmm_cat,
            bag=bag,
            droppable_labels=droppable_labels,
            predicted_labels=predicted_labels
        )

    def iter(self, batch_size=None, device=None, dtype=None):
        """Dataset iterator
        Load, concatenate and yield images from the dataset.
        Parameters
        ----------
        batch_size : int, default=self.batch_size
        device : torch.device, default=self.device
        dtype : torch.dtype, default=self.dtype
        Returns
        -------
        dat : (batch_size, channels, *spatial) tensor
        seg : (batch_size, channels | 1, *spatial) tensor
        """
        batch_size = batch_size or self.batch_size
        dtype = dtype or self.dtype or torch.get_default_dtype()
        segnames = list(self.segnames)
        if self.imgnames is not None:
            imgnames = list(self.imgnames)
        else:
            imgnames = None
        while segnames:
            segs = []
            imgs = []
            for _ in range(batch_size):
                if not segnames:
                    break

                # segmentations
                channels = self.loadseg(segnames.pop(0),
                                        segtype=self.segtype,
                                        lookup=self.lookup,
                                        dtype=dtype,
                                        device=device)
                segs.append(channels)

                if self.imgnames is not None:
                    # images
                    channels = self.loadvol(imgnames.pop(0),
                                            dtype=dtype,
                                            device=device)
                    channels = 255 * self.rescale(channels)
                    imgs.append(channels)

            segs = torch.stack(segs) if len(segs) > 1 else segs[0][None]
            if len(imgs) > 0:
                imgs = torch.stack(imgs) if len(imgs) > 1 else imgs[0][None]
                dats, segs, imgs = self.brain_generator(segs, img=imgs)
                yield dats, segs, imgs

            else:
                dats, segs = self.brain_generator(segs)
                yield dats, segs

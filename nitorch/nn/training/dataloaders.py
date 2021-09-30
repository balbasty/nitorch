import torch
from nitorch.core import utils, py
from nitorch import io
import math
import random


class Dataset:
    """Base class for datasets"""

    @staticmethod
    def rescale(image, qmin=0, qmax=0.95):
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
        if qmin == 0 and qmax == 1:
            return image
        qmin, qmax = utils.quantile(image, [qmin, qmax])
        image -= qmin
        image /= (qmax - qmin)
        return image

    @staticmethod
    def load(fname, dim=None, qmin=0, qmax=1, device=None, dtype=None):
        """Load a volume from disk

        Parameters
        ----------
        fname : str
        dim : int, optional
        qmin : (0..1), default=0
        qmax : (0..1), default=1
        device : torch.device, optional
        dtype : torch.dtype, optional

        Returns
        -------
        dat : (channels, *spatial) tensor

        """
        if dtype.is_floating_point:
            dat = io.loadf(fname, dtype=dtype, device=device)
            dat = Dataset.rescale(dat, qmin, qmax)
        else:
            dat = io.load(fname, dtype=dtype, device=device)
        dat = dat.squeeze()
        dim = dim or dat.dim()
        dat = utils.unsqueeze(dat, -1, max(0, dim - dat.dim()))
        dat = dat.reshape([*dat.shape[:dim], -1])
        dat = utils.movedim(dat, -1, 0)
        return dat

    @staticmethod
    def loadvol(fnames, dim=None, qmin=0, qmax=1, device=None, dtype=None):
        """Load a volume from disk

        Parameters
        ----------
        fnames : str or sequence[str]
        dim : int, optional
        qmin : (0..1), default=0
        qmax : (0..1), default=1
        device : torch.device, optional
        dtype : torch.dtype, optional

        Returns
        -------
        dat : (channels, *spatial) tensor

        """
        fnames = py.make_list(fnames)
        channels = []
        for fname in fnames:  # loop across channels
            dat = Dataset.load(fname, dim, qmin, qmax, device, dtype)
            channels.append(dat)
        if len(channels) > 1:
            channels = torch.cat(channels, 0)
        else:
            channels = channels[0]
        return channels

    @staticmethod
    def loadseg(fnames, dim=None, segtype='label', lookup=None,
                device=None, dtype=None):
        """Load a volume from disk

        Parameters
        ----------
        fnames : str or sequence[str]
        dim : int, optional
        segtype : [tuple] {'label', 'implicit', 'explicit'}, default='label'
        lookup : list of [list of] int, optional
        device : torch.device, optional
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
            dat = Dataset.load(fname, dim, device=device, dtype=sdtype)
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

    def __init__(self, filenames, batch_size=1, qmin=0, qmax=0.95,
                 dim=None, device=None, dtype=None):
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
        device : torch.device, optional
            Load the data to a specific device
        dtype : torch.dtype, optional
            Load the data in a specific data type.
        """
        self.filenames = list(filenames)
        self.nb_dat = len(self.filenames)
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.qmin = qmin
        self.qmax = qmax
        self.dim = dim

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
        device = device or self.device
        dtype = dtype or self.dtype or torch.get_default_dtype()
        filenames = list(self.filenames)

        while filenames:
            dats = []
            for _ in range(batch_size):
                if not filenames:
                    break
                channels = self.loadvol(filenames.pop(0),
                                        dim=self.dim,
                                        qmin=self.qmin,
                                        qmax=self.qmax,
                                        device=device,
                                        dtype=dtype)
                dats.append(channels)
            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            yield dats


class DatasetWithSeg(Dataset):
    """A dataset with ground truth segmentations"""

    def __init__(self, filenames, segnames, segtype='label', lookup=None,
                 batch_size=1, qmin=0, qmax=0.95,
                 dim=None, device=None, dtype=None):
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
        self.filenames = list(filenames)
        self.segnames = list(segnames)
        self.nb_dat = len(self.filenames)
        self.segtype = segtype
        self.lookup = lookup
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.qmin = qmin
        self.qmax = qmax
        self.dim = dim

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
        device = device or self.device
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
                                        dim=self.dim,
                                        qmin=self.qmin,
                                        qmax=self.qmax,
                                        device=device,
                                        dtype=dtype)
                dats.append(channels)

                # segmentations
                channels = self.loadseg(segnames.pop(0),
                                        dim=self.dim,
                                        segtype=self.segtype,
                                        lookup=self.lookup,
                                        device=device,
                                        dtype=dtype)
                segs.append(channels)

            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            segs = torch.stack(segs) if len(segs) > 1 else segs[0][None]
            yield dats, segs


class PairedDataset(Dataset):
    """A dataset made of pairs of images"""

    def __init__(self, filenames, refnames=None, pairs=None,
                 batch_size=1, qmin=0, qmax=0.95,
                 dim=None, device=None, dtype=None):
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
        device : torch.device, optional
            Load the data to a specific device
        dtype : torch.dtype, optional
            Load the data in a specific data type.
        """
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
        self.device = device
        self.dtype = dtype
        self.qmin = qmin
        self.qmax = qmax
        self.dim = dim

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
                                        dim=self.dim,
                                        qmin=self.qmin,
                                        qmax=self.qmax,
                                        device=device,
                                        dtype=dtype)
                dats.append(channels)

                channels = self.loadvol(refnames.pop(0),
                                        dim=self.dim,
                                        qmin=self.qmin,
                                        qmax=self.qmax,
                                        device=device,
                                        dtype=dtype)
                refs.append(channels)
            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            refs = torch.stack(refs) if len(refs) > 1 else refs[0][None]
            yield dats, refs


class PairedDatasetWithSeg(Dataset):
    """A dataset made of pairs of images and segmentations"""

    def __init__(self, filenames, segnames, refnames=None, segrefnames=None,
                 pairs=None, batch_size=1, segtype='label', segreftype=None,
                 lookup=None, qmin=0, qmax=0.95, dim=None, device=None,
                 dtype=None):
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
        self.device = device
        self.dtype = dtype
        self.qmin = qmin
        self.qmax = qmax
        self.segtype = segtype
        self.segreftype = segreftype or segtype
        self.lookup = lookup
        self.dim = dim

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
        device = device or self.device
        dtype = dtype or self.dtype or torch.get_default_dtype()
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
                                        dim=self.dim,
                                        qmin=self.qmin,
                                        qmax=self.qmax,
                                        device=device,
                                        dtype=dtype)
                dats.append(channels)

                channels = self.loadseg(segnames.pop(0),
                                        dim=self.dim,
                                        segtype=self.segtype,
                                        lookup=self.lookup,
                                        device=device,
                                        dtype=dtype)
                segs.append(channels)

                channels = self.loadvol(refnames.pop(0),
                                        dim=self.dim,
                                        qmin=self.qmin,
                                        qmax=self.qmax,
                                        device=device,
                                        dtype=dtype)
                refs.append(channels)

                channels = self.loadseg(segrefnames.pop(0),
                                        dim=self.dim,
                                        segtype=self.segreftype,
                                        lookup=self.lookup,
                                        device=device,
                                        dtype=dtype)
                segrefs.append(channels)
            dats = torch.stack(dats) if len(dats) > 1 else dats[0][None]
            refs = torch.stack(refs) if len(refs) > 1 else refs[0][None]
            segs = torch.stack(segs) if len(segs) > 1 else segs[0][None]
            segrefs = torch.stack(segrefs) if len(segrefs) > 1 else segrefs[0][None]
            yield dats, refs, segs, segrefs

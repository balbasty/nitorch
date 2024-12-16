import torch
import numpy as np
from types import GeneratorType as generator
from nibabel.streamlines.trk import TrkFile, get_affine_trackvis_to_rasmm
from nibabel.streamlines.tractogram import Tractogram
from nitorch.core import dtypes
from nitorch.spatial import affine_matvec
from nitorch.io.mapping import AccessType
from .mapping import MappedStreamlines
from .readers import reader_classes
from .writers import writer_classes


class TrkStreamlines(MappedStreamlines):
    """Streamlines stored in a TRK file"""

    readable: AccessType = AccessType.Full
    writable: AccessType = AccessType.Full

    @classmethod
    def possible_extensions(cls):
        return ('.trk',)

    def __init__(self, file_like=None, mode='r', keep_open=False):
        """

        Parameters
        ----------
        file_like : str of file object
            File to map
        mode : {'r', 'r+'}, default='r'
            Read in read-only ('r') or read-and-write ('r+') mode.
            Modifying the file in-place is only possible in 'r+' mode.
        keep_open : bool, default=False
            Does nothing.
        """
        self.filename = None
        self.mode = mode
        if file_like is None:
            self._struct = TrkFile(Tractogram())
        elif isinstance(file_like, TrkStreamlines):
            self._struct = file_like._struct
        elif isinstance(file_like, TrkFile):
            self._struct = file_like
        elif isinstance(file_like, Tractogram):
            self._struct = TrkFile(file_like)
        else:
            self.filename = file_like
            if 'r' in mode:
                self._struct = TrkFile.load(file_like, lazy_load=True)
            else:
                self._struct = TrkFile(Tractogram())

    @property
    def _loaded(self):
        if not self._struct:
            # not loaded at all
            return False
        if isinstance(self._struct.streamlines, generator):
            # lazilty loaded
            return False
        return True

    def __len__(self):
        return self._struct.header["nb_streamlines"]

    def shape(self):
        return torch.Size([len(self)])

    @property
    def dtype(self):
        return np.dtype('float64')

    @classmethod
    def _cast_generator(cls, generator, dtype=None, device=None, numpy=False):
        if not numpy:
            if dtype is not None:
                dtype = dtypes.dtype(dtype)
                if dtype.torch is None:
                    raise TypeError(
                        f'Data type {dtype} does not exist in PyTorch.'
                    )
            dtype = dtypes.dtype(dtype or np.float64).torch_upcast
        else:
            dtype = dtypes.dtype(dtype or np.float64).numpy_upcast

        for elem in generator:
            yield (
                np.asarray(elem, dtype=dtype) if numpy else
                torch.as_tensor(elem, dtype=dtype, device=device)
            )

    @classmethod
    def _apply_affine_generator(cls, generator, affine):
        for elem in generator:
            elem = affine_matvec(affine.to(elem), elem)
            yield elem

    def fdata(self, dtype=None, device=None, numpy=False):
        yield from self._cast_generator(
            self._struct.tractogram.streamlines,
            dtype=dtype, device=device, numpy=numpy
        )

    def data(self, dtype=None, device=None, numpy=False):
        affine = self.affine.inverse()
        if numpy:
            np_dtype = dtypes.dtype(dtype).numpy_upcast
        for streamline in self.fdata(dtype=dtype, device=device):
            streamline = affine_matvec(affine.to(streamline), streamline)
            if numpy:
                streamline = np.asarray(streamline.cpu(), dtype=np_dtype)
            yield streamline

    def scalars(self, dtype=None, device=None, numpy=False, keys=None):
        all_scalars = self._struct.tractogram.data_per_point
        all_keys = list(all_scalars.keys())
        keys = keys or all_keys
        return_dict = True
        if isinstance(keys, str):
            return_dict = False
            keys = [keys]
        scalars = {
            key: self._cast_generator(
                all_scalars[key], dtype=dtype, device=device, numpy=numpy
            )
            for key in keys
        }
        if not return_dict:
            scalars = next(iter(scalars.values()))
        return scalars

    def properties(self, dtype=None, device=None, numpy=False, keys=None):
        all_props = self._struct.tractogram.data_per_streamline
        all_keys = list(all_props.keys())
        keys = keys or all_keys
        return_dict = True
        if isinstance(keys, str):
            return_dict = False
            keys = [keys]
        props = {
            key: self._cast_generator(
                all_props[key], dtype=dtype, device=device, numpy=numpy
            )
            for key in keys
        }
        if not return_dict:
            props = next(iter(props.values()))
        return props

    @property
    def affine(self):
        """
        Vertex to world transformation matrix.
        """
        return torch.as_tensor(
            get_affine_trackvis_to_rasmm(self._struct.header)
        )

    def metadata(self, keys=None):
        raise NotImplementedError

    def set_fdata(self, streamlines):
        raise NotImplementedError

    def set_data(self, affine):
        raise NotImplementedError

    def set_metadata(self, **meta):
        raise NotImplementedError

    def save(self, file_like=None, **meta):
        # FIXME: meta
        self._struct.save(file_like)

    @classmethod
    def save_new(cls, streamlines, file_like, like=None, **meta):
        # FIXME: meta
        header = None
        if like:
            like = cls(like)
            header = like._struct.header

        obj = streamlines

        affine = meta.get("affine", None)
        if affine is None and like:
            affine = like.affine

        if not isinstance(obj, TrkStreamlines):
            if not isinstance(obj, (Tractogram, TrkFile)):
                obj = Tractogram(obj, affine_to_rasmm=affine)
            if not isinstance(TrkStreamlines, TrkFile):
                obj = TrkFile(obj, header)
            obj = TrkStreamlines(obj)

        if like:
            obj._struct = TrkFile(obj._struct.tractogram, header)

        if "scalars" in meta:
            obj._struct.tractogram.data_per_point = meta.pop("scalars")
        elif like:
            try:
                obj._struct.tractogram.data_per_point \
                    = like._struct.tractogram.data_per_point
            except Exception:
                pass

        if "properties" in meta:
            obj._struct.tractogram.data_per_streamline = meta.pop("properties")
        elif like:
            try:
                obj._struct.tractogram.data_per_streamline \
                    = like._struct.tractogram.data_per_streamline
            except Exception:
                pass

        for key, value in meta.items():
            if key in header.keys():
                obj._struct.header[key] = value

        obj._struct.save(file_like)

    @classmethod
    def savef_new(cls, streamlines, file_like, like=None, **meta):
        if like:
            like = cls(like)
        affine = meta.get("affine", None)
        if affine is None and like:
            affine = like.affine
        if not isinstance(streamlines, (TrkStreamlines, TrkFile, Tractogram)):
            if affine is not None:
                affine = affine.inverse()
                streamlines = cls._apply_affine_generator(streamlines, affine)
        cls.save_new(streamlines, file_like, like, **meta)


reader_classes.append(TrkStreamlines)
writer_classes.append(TrkStreamlines)

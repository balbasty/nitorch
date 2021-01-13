# python
import warnings
import copy
from uuid import uuid4, UUID
import torch
# nitorch
import nitorch as ni
from nitorch import io
from nitorch.spatial import affine_default
from nitorch.core.utils import same_storage


# TODO:
#   - set proper affine after call to __getitem__
#   - read/write 4d volumes
#   - test, test, test


def set_same_scanner_position(*images):
    """Set the same unique scanner position to all images.

    Parameters
    ----------
    images : sequence[MRI]

    Raises
    ------
    ValueError
        If two or more different positions are already present in the
        input set.

    """
    uids = [img.scanner_position for img in images
            if img.scanner_position is not None]
    uids = set(uids)
    if len(uids) > 1:
        raise ValueError('Several images are already locked to different '
                         'positions. Call `detach_position` on some of them '
                         'first.')
    uid = uids.pop() if uids else uuid4()
    for img in images:
        img.scanner_position = uid


def is_same_scanner_position(*images):
    """Check that all images have the same unique scanner position.

    Parameters
    ----------
    images : sequence[MRI]

    Returns
    -------
    True if all elements of the input set have the same unique scanner
    position (which is not `None`).

    """
    uids = [img.scanner_position for img in images
            if img.scanner_position is not None]
    uids = set(uids)
    return len(uids) == 1


class BaseND:
    """Represents a ND volume.

    Properties
    ----------
    volume : io.MappedArray or tensor       Mapped volume file
    affine : tensor                         Orientation matrix
    scanner_position : UUID                 Allows to link scanner positions

    Methods
    -------
    detach_position()                       Detach scanner position (-> None)
    detach_position_()                      > Same but in-place
    fdata(cache=True)                       Load data to memory
    discard()                               Delete cached data

    """
    _volume: io.MappedArray or torch.Tensor = None  # Mapped volume file
    affine: torch.Tensor = None                     # Orientation matrix
    scanner_position: UUID = None                   # Link positions in scanner
    _fdata: torch.Tensor = None                     # Cached scaled data
    _dtype: torch.dtype = None                      # default dtype
    _device: torch.device = None                    # default device
    dtype = property(lambda self: self._dtype  or torch.get_default_dtype())
    device = property(lambda self: self._device  or torch.device('cpu'))

    @property
    def volume(self):
        # just in case volume is modified
        self._fdata = None
        return self._volume

    @volume.setter
    def volume(self, val):
        self._volume = val
        self._fdata = None

    @property
    def shape(self):
        return self.volume.shape

    def __new__(cls, input=None, *args, **kwargs):
        if isinstance(input, str):
            return cls.from_fname(input, *args, **kwargs)
        if isinstance(input, io.MappedArray):
            return cls.from_mapped(input, *args, **kwargs)
        if torch.is_tensor(input):
            return cls.from_tensor(input, *args, **kwargs)
        if isinstance(input, BaseND):
            return cls.from_instance(input, *args, **kwargs)
        return object.__new__(cls).set_attributes(**kwargs)

    def __str__(self):
        if self.volume is None:
            return '{}(<unset>)'.format(type(self).__name__)
        return '{}(shape={})'.format(type(self).__name__, self.volume.shape)

    __repr__ = __str__

    def copy(self):
        """Copy an instance"""
        return copy.copy(self)

    def attributes(self):
        """Return the name of all attributes"""
        return ['affine', 'scanner_position']

    def set_attributes(self, **attributes):
        """Set specified attributes."""
        for key, val in attributes.items():
            setattr(self, key, val)
        return self

    def reset_attributes(self):
        """Reset attributes to their default values (when they have one)."""
        if isinstance(self.volume, io.MappedArray):
            self.affine = self.volume.affine
            if isinstance(self.volume.affine, (tuple, list)):
                self.affine = self.affine[0]
        elif torch.is_tensor(self.volume):
            self.affine = affine_default(self.volume.shape[:3])
        return self

    @classmethod
    def from_attributes(cls, **attributes):
        """Build an MRI object from attributes only. `volume` is not set."""
        return cls(**attributes)

    @classmethod
    def from_fname(cls, fname, permission='r', keep_open=False, **attributes):
        """Build an MRI object from a file name.

        We accept paths of the form 'path/to/file.nii,1,2', which
        mean that only the subvolume `[:, :, :, 1, 2]` should be read.
        The first three (spatial) dimensions are always read.
        """
        fname = str(fname)
        fname, *index = fname.split(',')
        mapped = io.map(fname, permission=permission, keep_open=keep_open)
        if index:
            index = tuple(int(i) for i in index)
            index = (slice(None),) * 3 + index
            mapped = mapped[index]
        return cls.from_mapped(mapped, **attributes)

    @classmethod
    def from_mapped(cls, mapped, **attributes):
        """Build an MRI object from a mapped array"""
        if not isinstance(mapped, io.MappedArray):
            raise TypeError('Expected a MappedArray but got a {}.'
                            .format(type(mapped)))
        new = cls()
        new.volume = mapped
        new.reset_attributes()
        new.set_attributes(**attributes)
        new.atleast_3d_()
        return new

    @classmethod
    def from_tensor(cls, tensor, **attributes):
        """Build an MRI object from a mapped array"""
        tensor = torch.as_tensor(tensor)
        new = cls()
        new.volume = tensor
        new.reset_attributes()
        new.set_attributes(**attributes)
        new.atleast_3d_()
        return new

    @classmethod
    def from_instance(cls, instance, **attributes):
        """Build an MRI object from an other instance"""
        new = cls()
        new.volume = instance.volume
        new.affine = instance.affine
        new.scanner_position = instance.scanner_position
        old_attributes = [getattr(instance, key) 
                          for key in new.attributes()
                          if key in instance.attributes()]
        old_attributes.update(attributes)
        new.set_attributes(**old_attributes)
        new.atleast_3d_()
        return new

    def atleast_nd_(self, dim):
        """Make sure there are at least N dimensions."""
        dim_in = len(self.volume.shape)
        if dim_in < dim:
            warnings.warn('Volume is {}D. Making it {}D by appending '
                          'singleton dimensions.'.format(dim_in, dim),
                          RuntimeWarning)
            index = (Ellipsis,) + (None,) * (dim - dim_in)
            self.volume = self.volume[index]
        return self

    def atleast_nd(self, dim):
        """Make sure there are at least N dimensions (inplace)."""
        return self.copy().atleast_nd_(dim)

    def ensure_nd_(self, dim):
        """Make sure there are exactly N dimensions (inplace)."""
        while len(self.volume.shape) > dim:
            if self.volume.shape[-1] != 1:
                raise ValueError('Cannot squeeze extra dimension {}.'
                                 .format(len(self.volume.shape)))
            self.volume = self.volume.squeeze(-1)
        return self.atleast_nd_(dim)

    def ensure_nd(self, dim):
        """Make sure there are exactly N dimensions."""
        return self.copy().ensure_nd_(dim)

    def atleast_3d_(self):
        """Make sure there are at least 3 dimensions (inplace)."""
        return self.atleast_nd_(3)

    def atleast_3d(self):
        """Make sure there are at least 3 dimensions."""
        return self.copy().atleast_3d_()

    def ensure_3d_(self):
        """Make sure there are exactly 3 dimensions (inplace)."""
        return self.ensure_nd_(3)

    def ensure_3d(self):
        """Make sure there are exactly 3 dimensions."""
        return self.copy().ensure_3d_()

    def detach_position_(self):
        """Set `scanner_position = None` in this object."""
        self.scanner_position = None
        return self

    def detach_position(self):
        """Return a copy of the object with `scanner_position = None`."""
        return self.copy().detach_position_()

    def __getitem__(self, item):
        """Slice the array"""
        new = self.copy()
        new.volume = self.volume[item]
        if self._fdata is self.volume:
            new._fdata = new.volume
        elif self._fdata is not None:
            new._fdata = self._fdata[item]
        return new

    def __setitem__(self, item, value):
        """Write array slice"""
        self.volume[item] = value
        if self._fdata is not None and self._fdata is not self.volume:
            self._fdata[item] = value
        return self

    def fdata(self, dtype=None, device=None, rand=True, cache=True, **kwargs):
        """Get scaled floating-point data.

        Parameters
        ----------
        dtype : torch.dtype, default='`torch.get_default_dtype()`
        device : torch.device, default='cpu'
        rand : bool, default=True
            Add random noise if raw data is integer
        cache : bool, default=True
            Cache the data in memory so that it does not need to be
            loaded again next time

        Returns
        -------
        dat : torch.tensor[dtype]

        """
        dtype = dtype or self._dtype or torch.get_default_dtype()
        device = device or self._device
        backend = dict(dtype=dtype, device=device)

        if not cache or self._fdata is None:
            if isinstance(self.volume, io.MappedArray):
                _fdata = self.volume.fdata(rand=rand, **backend)
            else:
                _fdata = self.volume.to(**backend)
            if cache:
                self._fdata = _fdata
        else:
            _fdata = self._fdata
        return _fdata.to(**backend)

    def savef(self, fname, *args, **kwargs):
        """Save to disk"""
        io.savef(self.volume, fname, *args, **kwargs)

    def discard(self):
        """Delete cached data."""
        self._fdata = None

    def to(self, dtype=None, device=None):
        """Move data to a different dtype/device

        Parameters
        ----------
        dtype : torch.dtype
        device : torch.device

        Returns
        -------
        self

        """
        self._dtype = dtype or self._dtype
        self._device = device or self._device
        backend = dict(dtype=self._dtype, device=self._device)
        if self._fdata is not None:
            self._fdata = self._fdata.to(**backend)
        if self.affine is not None:
            self.affine = self.affine.to(**backend)
        return self


class Volume3D(BaseND):
    @classmethod
    def from_mapped(cls, mapped, **attributes):
        return super().from_mapped(mapped, **attributes).ensure_3d_()

    @classmethod
    def from_tensor(cls, tensor, **attributes):
        return super().from_tensor(tensor, **attributes).ensure_3d_()


class GradientEcho(BaseND):
    """Represents a single volume acquired with a Gradient Echo sequence.

    Properties
    ----------
    volume : io.MappedArray or tensor       Mapped volume file
    affine : tensor                         Orientation matrix
    scanner_position : UUID                 Link scanner positions
    te : float                              Echo time (in sec)
    tr : float                              Repetition time (in sec)
    ti : float                              Inversion time (in sec)
    fa : float                              Flip angle (in deg)
    mt : float or bool                      Off-resonance pulse (in hz, or bool)

    """
    te: float = None                # Echo time (in sec)
    tr: float = None                # Repetition time (in sec)
    ti: float = None                # Inversion time (in sec)
    fa: float = None                # Flip angle (in deg)
    mt: float or bool = None        # Off-resonance pulse (in hz, or bool)

    def attributes(self):
        """Return the name of all attributes"""
        return super().attributes() + ['te', 'tr', 'ti', 'fa', 'mt']

    @classmethod
    def from_mapped(cls, mapped, **attributes):
        missing = [key for key in ['te', 'tr', 'ti', 'fa', 'mt']
                  if key not in attributes ]
        meta = mapped.metadata(missing)
        if not isinstance(meta, dict):
            meta = meta[0]
        meta = {key: val for key, val in meta.items() if val is not None}
        if 'te' in meta:
            if meta['te_unit'] in ('ms', 'msec'):
                meta['te'] /= 1e3
        if 'tr' in meta:
            if meta['tr_unit'] in ('ms', 'msec'):
                meta['tr'] /= 1e3
        if 'ti' in meta:
            if meta['ti_unit'] in ('ms', 'msec'):
                meta['ti'] /= 1e3
        if 'fa' in meta:
            if meta['fa_unit'] in ('rad',):
                meta['fa'] *= 180. / ni.core.constants.pi
                
        meta.update(attributes)
        attributes = meta
        return super().from_mapped(mapped, **attributes)

    @classmethod
    def from_instance(cls, instance, **attributes):
        new = super().from_instance(instance)
        new.te = getattr(instance, 'te', None)
        new.tr = getattr(instance, 'tr', None)
        new.ti = getattr(instance, 'ti', None)
        new.fa = getattr(instance, 'fa', None)
        new.mt = getattr(instance, 'mt', None)
        new.set_attributes(**attributes)
        return new


class GradientEchoSingle(GradientEcho, Volume3D):
    pass


class GradientEchoMulti(GradientEcho):
    """Multi-Echo Gradient Echo series.

    Properties
    ----------
    volume : io.MappedArray or tensor       Mapped volume file
    affine : tensor                         Orientation matrix
    scanner_position : UUID                 Link scanner positions
    te : list[float]                        Echo times (in sec)
    tr : float                              Repetition time (in sec)
    ti : float                              Inversion time (in sec)
    fa : float                              Flip angle (in deg)
    mt : float or bool                      Off-resonance pulse (in hz, or bool)
    """
    te: list = None

    @classmethod
    def from_instance(cls, instance, **attributes):
        """Build a multi-echo gradient-echo volume from an
        instance of GradientEchoMulti or GradientEcho."""
        if isinstance(instance, GradientEchoMulti):
            new = instance.copy()
            new.set_attributes(**attributes)
            return new
        if isinstance(instance, GradientEcho):
            new = super().from_instance(instance)
            new.volume = new.volume[None, ...]
            new.te = [new.te]
            new.set_attributes(**attributes)
            return new
        return super().from_instance(instance, **attributes)

    @classmethod
    def from_instances(cls, echoes, **attributes):
        """Build a multi-echo gradient-echo volume from mutiple
        instances of GradientEchoSingle."""
        volume = io.stack([echo.volume for echo in echoes], dim=0)
        if 'tr' not in attributes:
            trs = set([echo.tr for echo in echoes if echo.tr is not None])
            tr = trs.pop() if trs else None
            if len(trs) > 0:
                warnings.warn("TR not consistent across echoes. Using {}."
                              .format(tr))
            attributes['tr'] = tr
        if 'ti' not in attributes:
            tis = set([echo.ti for echo in echoes if echo.ti is not None])
            ti = tis.pop() if tis else None
            if len(tis) > 0:
                warnings.warn("TI not consistent across echoes. Using {}."
                              .format(ti))
            attributes['ti'] = ti
        if 'fa' not in attributes:
            fas = set([echo.fa for echo in echoes if echo.fa is not None])
            fa = fas.pop() if fas else None
            if len(fas) > 0:
                warnings.warn("FA not consistent across echoes. Using {}."
                              .format(fa))
            attributes['fa'] = fa
        if 'mt' not in attributes:
            mts = set([echo.mt for echo in echoes if echo.mt is not None])
            mt = mts.pop() if mts else None
            if len(mts) > 0:
                warnings.warn("MT not consistent across echoes. Using {}."
                              .format(mt))
            attributes['mt'] = mt
        if 'te' not in attributes:
            attributes['te'] = [echo.te for echo in echoes]
        return cls.from_mapped(volume, **attributes)

    @classmethod
    def from_fnames(cls, fnames, **attributes):
        echoes = [GradientEchoSingle.from_fname(fname) for fname in fnames]
        return cls.from_instances(echoes, **attributes)

    def echo(self, index):
        volume = self.volume[index, ...]
        te = self.te[index]
        attributes = {key: getattr(self, key) for key in self.attributes()
                      if key != 'te'}
        attributes['te'] = te
        return GradientEcho.from_mapped(volume, **attributes)

    def __getitem__(self, item):
        item = ni.core.pyutils.make_tuple(item)
        if (not item) or item[0] in (slice(None), Ellipsis):
            return super().__getitem__(item)
        elif item[0] is None:
            raise IndexError('Cannot left-pad dimensions of a '
                             '<GradientEchoMulti>')
        else:
            subecho, *item = item
            if isinstance(subecho, slice):
                subecho = self.echo(subecho)
                if item:
                    subecho = subecho[(slice(None), *item)]
                return subecho
            else:
                subecho = self.echo(subecho)
                if item:
                    subecho = subecho[item]
                return subecho

    def __len__(self):
        return len(self.volume)

    def __iter__(self):
        for e in range(len(self)):
            yield self.echo(e)


class PrecomputedFieldMap(Volume3D):
    """A field map, eventually associated to a structural image in the
    same space."""

    def attributes(self):
        """Return the name of all attributes"""
        return super().attributes() + ['magnitude', 'unit']

    _magnitude: Volume3D or type(None) = None
    unit: str = '%'

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, val):
        if val is not None:
            self._magnitude = Volume3D(val)
        else:
            self._magnitude = None

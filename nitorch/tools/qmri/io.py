# python
import warnings
import copy
from uuid import uuid4, UUID
import torch
# nitorch
import nitorch as ni
from nitorch import io
from nitorch.core import py, utils
from nitorch.core.dtypes import dtype as dtype_info
from nitorch.spatial import affine_default, voxel_size as get_voxel_size


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
    fdata(cache=False)                      Load data to memory
    discard()                               Delete cached data

    """
    _volume: io.MappedArray or torch.Tensor = None  # Mapped volume file
    _mask: io.MappedArray or torch.Tensor = None    # Mapped mask file
    affine: torch.Tensor = None                     # Orientation matrix
    scanner_position: UUID = None                   # Link positions in scanner
    _fdata: torch.Tensor = None                     # Cached scaled data
    _dtype: torch.dtype = None                      # default dtype
    _device: torch.device = None                    # default device
    spatial_dim: int = None                         # Number of spatial dimensions

    @property
    def voxel_size(self):
        if self.affine is None:
            return None
        return get_voxel_size(self.affine)

    @property
    def spatial_shape(self):
        return self.shape[-self.spatial_dim:] if self.spatial_dim else None

    @property
    def dtype(self):
        return self._dtype or torch.get_default_dtype()
    
    @dtype.setter
    def dtype(self, value):
        return self.to(dtype=value)
    
    @property
    def device(self):
        return self._device or torch.device('cpu')
    
    @device.setter
    def device(self, value):
        return self.to(device=value)
    
    @property
    def volume(self):
        # just in case volume is modified
        self._fdata = None
        return self._volume

    @volume.setter
    def volume(self, val):
        self._volume = val
        self._fdata = None
        if isinstance(val, io.MappedArray):
            self.affine = val.affine
        if isinstance(self.affine, (list, tuple)):
            self.affine = self.affine[0]

    @property
    def mask(self):
        # just in case mask is modified
        self._fdata = None
        return self._mask

    @mask.setter
    def mask(self, val):
        self._mask = val
        self._fdata = None

    @property
    def shape(self):
        return self.volume.shape

    def __init__(self, input=None, *args, **kwargs):
        if isinstance(input, str):
            self._init_from_fname(input, *args, **kwargs)
        elif isinstance(input, io.MappedArray):
            self._init_from_mapped(input, *args, **kwargs)
        elif torch.is_tensor(input):
            self._init_from_tensor(input, *args, **kwargs)
        elif isinstance(input, BaseND):
            self._init_from_instance(input, *args, **kwargs)
        else:
            self.set_attributes(**kwargs)

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

    def _init_from_fname(new, fname, permission='r', keep_open=False, **attributes):
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
        return new._init_from_mapped(mapped, **attributes)

    @classmethod
    def from_fname(cls, fname, permission='r', keep_open=False, **attributes):
        """Build an MRI object from a file name.

        We accept paths of the form 'path/to/file.nii,1,2', which
        mean that only the subvolume `[:, :, :, 1, 2]` should be read.
        The first three (spatial) dimensions are always read.
        """
        new = cls.__new__(cls)
        new._init_from_fname(fname, permission, keep_open, **attributes)
        return new

    def _init_from_mapped(new, mapped, **attributes):
        """Build an MRI object from a mapped array"""
        if not isinstance(mapped, io.MappedArray):
            raise TypeError('Expected a MappedArray but got a {}.'
                            .format(type(mapped)))
        new.volume = mapped
        new.reset_attributes()
        new.set_attributes(**attributes)
        new.atleast_3d_()

    @classmethod
    def from_mapped(cls, mapped, **attributes):
        """Build an MRI object from a mapped array"""
        new = cls.__new__(cls)
        new._init_from_mapped(mapped, **attributes)
        return new

    def _init_from_tensor(new, tensor, **attributes):
        """Build an MRI object from a mapped array"""
        tensor = torch.as_tensor(tensor)
        new.volume = tensor
        new.reset_attributes()
        new.set_attributes(**attributes)
        new.atleast_3d_()
        new.device = tensor.device

    @classmethod
    def from_tensor(cls, tensor, **attributes):
        """Build an MRI object from a mapped array"""
        new = cls.__new__(cls)
        new._init_from_tensor(tensor, **attributes)
        return new

    def _init_from_instance(new, instance, **attributes):
        """Build an MRI object from an other instance"""
        # new = BaseND.__new__(cls)
        new.volume = instance.volume
        new.affine = instance.affine
        new.scanner_position = instance.scanner_position
        old_attributes = {key: getattr(instance, key)
                          for key in new.attributes()
                          if key in instance.attributes()}
        old_attributes.update(attributes)
        new.set_attributes(**old_attributes)
        new.atleast_3d_()

    @classmethod
    def from_instance(cls, instance, **attributes):
        """Build an MRI object from an other instance"""
        new = cls.__new__(cls)
        new._init_from_instance(instance, **attributes)
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

    def fdata(self, dtype=None, device=None, rand=False, missing=None,
              cache=False, copy=False, **kwargs):
        """Get scaled floating-point data.

        Note that if a mask is registered in the object, all voxels
        outside of this mask will be set to NaNs.

        Parameters
        ----------
        dtype : torch.dtype, default='`torch.get_default_dtype()`
        device : torch.device, default='cpu'
        rand : bool, default=True
            Add random noise if raw data is integer
        missing : scalar or sequence, default=0
            Values that should be considered missing data.
            All of these values will be transformed to NaNs.
        cache : bool, default=False
            Cache the data in memory so that it does not need to be
            loaded again next time
        copy : bool, default=False
            Ensure that a copy of the original data is performed.

        Returns
        -------
        dat : torch.tensor[dtype]

        """
        dtype = dtype or self.dtype
        device = device or self.device
        backend = dict(dtype=dtype, device=device)
        if missing is not None:
            missing = py.ensure_list(missing)

        do_copy = copy or rand or (missing is not None)

        if not cache or self._fdata is None:
            if isinstance(self.volume, io.MappedArray):
                _fdata = self.volume.fdata(**backend)
            else:
                _fdata = self.volume.to(**backend, copy=do_copy)
            do_copy = False
            mask = torch.isfinite(_fdata).bitwise_not_()
            if missing:
                mask.bitwise_or_(utils.isin(_fdata, missing))
            if self._mask is not None:
                if isinstance(self._mask, io.MappedArray):
                    _mask = self._mask.data(dtype=torch.bool, device=mask.device)
                else:
                    _mask = self._mask.to(dtype=torch.bool, device=mask.device, copy=True)
                _mask.bitwise_not_()
                mask.bitwise_or_(_mask)
            disk_dtype = self.volume.dtype
            if isinstance(disk_dtype, (list, tuple)):
                disk_dtype = disk_dtype[0]
            if rand and not dtype_info(disk_dtype).is_floating_point:
                slope = getattr(self.volume, 'slope', None) or 1
                _fdata.add_(torch.rand_like(_fdata).mul_(slope))
            _fdata[mask] = float('nan')
            if cache:
                self._fdata = _fdata
                do_copy = copy
        else:
            _fdata = self._fdata
        return _fdata.to(**backend, copy=do_copy)

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
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('Expected a torch.dtype but got '
                            f'{type(dtype)}')
        self._dtype = dtype or self._dtype
        device = device or self._device
        self._device = torch.device(device) if device else None
        backend = dict(dtype=self.dtype, device=self.device)
        if self._fdata is not None:
            self._fdata = self._fdata.to(**backend)
        if self.affine is not None:
            self.affine = self.affine.to(**backend)
        return self


class Volume3D(BaseND):
    spatial_dim = 3

    def _init_from_mapped(new, mapped, **attributes):
        super()._init_from_mapped(mapped, **attributes)
        new.ensure_3d_()

    def from_tensor(new, tensor, **attributes):
        super().from_tensor(tensor, **attributes)
        new.ensure_3d_()


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
    readout : {0, 1, 2}                     Readout dimension
    blip : {1, -1}                          Readout direction (up or down)
    noise : float                           Variance of the noise
    dof : float                             Degrees of freedom of the noise

    """
    spatial_dim = 3
    te: float = None                # Echo time (in sec)
    tr: float = None                # Repetition time (in sec)
    ti: float = None                # Inversion time (in sec)
    fa: float = None                # Flip angle (in deg)
    mt: float or bool = None        # Off-resonance pulse (in hz, or bool)
    readout: int = None             # Readout dimension
    blip: int = None                # Readout direction (up or down)
    noise: float = None             # Variance of the noise
    dof: float = None               # DOF of coils

    def attributes(self):
        """Return the name of all attributes"""
        return super().attributes() + ['te', 'tr', 'ti', 'fa', 'mt',
                                       'readout', 'blip', 'noise', 'dof']

    def _init_from_mapped(new, mapped, **attributes):
        missing = [key for key in ['te', 'tr', 'ti', 'fa', 'mt']
                   if key not in attributes ]
        meta = mapped.metadata(missing) if missing else {}
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
        super()._init_from_mapped(mapped, **attributes)

    def _init_from_instance(new, instance, **attributes):
        super()._init_from_instance(instance)
        new.te = getattr(instance, 'te', None)
        new.tr = getattr(instance, 'tr', None)
        new.ti = getattr(instance, 'ti', None)
        new.fa = getattr(instance, 'fa', None)
        new.mt = getattr(instance, 'mt', None)
        new.readout = getattr(instance, 'readout', None)
        new.blip = getattr(instance, 'blip', None)
        new.noise = getattr(instance, 'noise', None)
        new.dof = getattr(instance, 'dof', None)
        new.set_attributes(**attributes)

    def fdata(self, *args, **kwargs):
        kwargs.setdefault('missing', 0)
        return super().fdata(*args, **kwargs)


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
    readout : {0, 1, 2}                     Readout dimension
    blip : list[{1, -1}]                    Readout direction (up or down)
    """
    te: list = None

    def _init_from_instance(new, echoes, **attributes):
        """Build a multi-echo gradient-echo volume from mutiple
        instances of GradientEchoSingle."""
        if isinstance(echoes, GradientEchoMulti):
            copy_attributes = {k: getattr(echoes, k) for k in
                               echoes.attributes()}
            copy_attributes.update(attributes)
            new.set_attributes(**copy_attributes)
            super()._init_from_instance(echoes, **attributes)
            return
        if isinstance(echoes, GradientEcho):
            super()._init_from_instance(echoes)
            new.volume = new.volume[None, ...]
            new.te = [new.te]
            new.blip = [getattr(new, 'blip', None)]
            new.set_attributes(**attributes)
            return
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
        if 'readout' not in attributes:
            rds = set([echo.readout for echo in echoes if echo.readout is not None])
            rd = rds.pop() if rds else None
            if len(rds) > 0:
                warnings.warn(f"readout direection not consistent across "
                              f"echoes. Using {rd}.")
            attributes['readout'] = rd
        if 'blip' not in attributes:
            attributes['blip'] = [echo.blip for echo in echoes]
        if 'noise' not in attributes:
            noises = set([echo.noise for echo in echoes if echo.noise is not None])
            noise = noises.pop() if noises else None
            if len(noises) > 0:
                warnings.warn("noise not consistent across echoes. Using {}."
                              .format(noise))
            attributes['noise'] = noise
        if 'dof' not in attributes:
            dofs = set([echo.dof for echo in echoes if echo.dof is not None])
            dof = dofs.pop() if dofs else None
            if len(dofs) > 0:
                warnings.warn("Degrees of freedom not consistent across echoes. Using {}."
                              .format(dof))
            attributes['dof'] = dof
        super()._init_from_mapped(volume, **attributes)

    def _init_from_fname(new, fnames, permission='r', keep_open=False,  **attributes):
        fnames = py.make_list(fnames)
        fs = []
        for fname in fnames:
            f = io.map(fname, permission=permission, keep_open=keep_open)
            while f.dim < 4:
                f = f.unsqueeze(-1)
            fs += [f]
        fs = io.cat(fs, -1).permute([-1, 0, 1, 2])
        new._init_from_mapped(fs, **attributes)

    def _init_from_mapped(new, fs, **attributes):
        echoes = [GradientEchoSingle.from_mapped(f) for f in fs]
        new._init_from_instance(echoes, **attributes)

    def echo(self, index):
        volume = self.volume[index, ...]
        te = self.te[index]
        blip = self.blip[index] if self.blip else None
        attributes = {key: getattr(self, key) for key in self.attributes()
                      if key not in ('te', 'blip')}
        attributes['te'] = te
        attributes['blip'] = blip
        if isinstance(volume, io.MappedArray):
            e = GradientEcho.from_mapped(volume, **attributes)
        else:
            e = GradientEcho.from_tensor(volume, **attributes)
        if self._mask is not None:
            if self._mask.dim() == self.spatial_dim + 1:
                e.mask = self._mask[index]
            else:
                e.mask = self._mask
        return e

    def __getitem__(self, item):
        item = ni.core.py.make_tuple(item)
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

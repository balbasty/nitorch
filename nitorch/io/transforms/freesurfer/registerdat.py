from warnings import warn
import os
import torch
from .fsutils import read_values, write_values
from nitorch.core.struct import Structure
from nitorch.io.transforms.mapping import MappedAffine
from nitorch.io.mapping import AccessType
from nitorch.io.loadsave import map as map_affine
from nitorch.io.utils.volutils import cast
from nitorch.io.utils.opener import Opener
from nitorch.core.optionals import numpy as np
from ..readers import reader_classes
from ..writers import writer_classes


class RegisterDatStruct(Structure):
    """Structure encoding a register.dat file

    This representation mimics the representation on dist and is not
    exposed to the user.
    """

    subject: str = None         # FS subject name
    resolution: float = None    # In-plane resolution
    thickness: float = None     # Slice thickness
    intensity: float = None     # Mean image intensity
    affine: np.ndarray = None   # 4x4 affine matrix tkRAS-to-tkRAS

    @classmethod
    def from_filename(cls, fname):
        """Build from path to LTA file"""
        with Opener(fname) as f:
            return cls.from_lines(f)

    @classmethod
    def from_lines(cls, f):
        """Build from an iterator over lines"""
        struct = RegisterDatStruct()
        section = 'subject'
        affine = []
        for line in f:
            if section == 'subject':
                struct.subject = read_values(line, str)
                section = 'resolution'
                continue
            if section == 'resolution':
                struct.resolution = read_values(line, float)
                section = 'thickness'
                continue
            if section == 'thickness':
                struct.thickness = read_values(line, float)
                section = 'intensity'
                continue
            if section == 'intensity':
                struct.intensity = read_values(line, float)
                section = 'affine'
                continue
            if section == 'affine':
                if line.startswith('round'):
                    section = 'finished'
                    break
                row = read_values(line, [float] * 4)
                affine.append(list(row))
                continue
            warn(f'Don`t know what to do with line: {line}\n'
                 f'Skipping it.', RuntimeWarning)
        if section != 'finished':
            warn(f'Footer never found.')
        if affine is not None:
            struct.affine = np.asarray(affine).reshape([4, 4])
        return struct

    def to_lines(self):
        """Iterator over lines"""
        yield write_values(self.subject)
        yield write_values(self.resolution or 0.)
        yield write_values(self.thickness or 0.)
        yield write_values(self.intensity or 0.)
        for row in self.affine:
            yield write_values(row)
        yield write_values('round')

    def to_file_like(self, f):
        """Write to file-like object"""
        for line in self.to_lines():
            f.write(line + os.linesep)
        return

    def to_filename(self, fname):
        """Write to file"""
        with Opener(fname, 'w') as f:
            return self.to_file_like(f)

    def to(self, thing):
        """Write to something"""
        if isinstance(thing, str):
            return self.to_filename(thing)
        else:
            return self.to_file_like(thing)


class RegisterDat(MappedAffine):
    """MGH format for (some) affine transformations."""

    readable: AccessType = AccessType.Full
    writable: AccessType = AccessType.Full

    @classmethod
    def possible_extensions(cls):
        return ('.dat',)

    def __init__(self, file_like, mode='r', keep_open=False):
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
        if isinstance(file_like, RegisterDatStruct):
            self._struct = file_like
        elif isinstance(file_like, str):
            self.filename = file_like
            self._struct = RegisterDatStruct.from_filename(file_like)
        else:
            self.file_like = file_like
            self._struct = RegisterDatStruct.from_lines(file_like)
        self.mode = mode

    @property
    def shape(self):
        if self._struct.affine is not None:
            return tuple(self._struct.affine.shape)
        else:
            return tuple()

    def fdata(self, dtype=None, device=None, numpy=False):
        if self._struct.affine is None:
            return None
        if dtype is None:
            dtype = torch.get_default_dtype()
        affine = cast(self._struct.affine, dtype)
        if numpy:
            return np.asarray(affine)
        else:
            return torch.as_tensor(affine, dtype=dtype, device=device)

    def data(self, dtype=None, device=None, numpy=False):
        return self.fdata(dtype, device, numpy)

    def type(self):
        return 'ras', 'ras'

    def metadata(self, keys=None):
        """Read additional metadata

        Parameters
        ----------
        keys : sequence of str, optional
            Keys should be in:
            'subject' : str
                FreeSurfer subject name
            'resolution' : float
                In-plane resolution
            'thickness' : float
                Slice thickness
            'intensity' : float
                Mean image intensity

        Returns
        -------
        dict

        """
        known_keys = ('subject', 'resolution', 'thickness', 'intensity')
        keys = keys or known_keys
        meta = dict()
        for key in keys:
            if key in known_keys:
                meta[key] = getattr(self._struct, key)
            else:
                meta[key] = None
        return meta

    def set_fdata(self, affine):
        affine = np.asarray(affine)
        if affine.shape != (4, 4):
            raise ValueError('Expected a 4x4 matrix')
        self._struct.affine = affine
        return self

    def set_data(self, affine):
        return self.set_fdata(affine)

    def set_metadata(self, **meta):
        """Set additional metadata

        Parameters
        ----------
        subject : str, optional
            FreeSurfer subject name
        resolution : float, optional
            In-plane resolution
        thickness : float, optional
            Slice thickness
        intensity : float, optional
            Mean image intensity

        Returns
        -------
        self

        """
        known_keys = dict(subject=str, resolution=float,
                          thickness=float, intensity=float)
        for key, value in meta.items():
            if key in known_keys:
                conv = known_keys[key]
                setattr(self._struct, key, conv(value))
        return self

    def save(self, file_like=None, *args, **meta):
        if '+' not in self.mode:
            raise RuntimeError('Cannot write into read-only volume. '
                               'Re-map in mode "r+" to allow in-place '
                               'writing.')
        self.set_metadata(**meta)
        file_like = file_like or self.filename or self.file_like
        self._struct.to(file_like)
        return self

    @classmethod
    def save_new(cls, affine, file_like, like=None, *args, **meta):

        if isinstance(affine, MappedAffine):
            if like is None:
                like = affine
            affine = affine.data(numpy=True)
        if torch.is_tensor(affine):
            affine = affine.detach().cpu()
        affine = np.asanyarray(affine)
        if like is not None:
            like = map_affine(like)

        if affine.shape != (4, 4):
            raise ValueError('Expected a 4x4 matrix')

        struct = RegisterDatStruct()
        struct.affine = affine
        known_keys = dict(subject=str, resolution=float,
                          thickness=float, intensity=float)
        metadata = like.metadata(known_keys.keys())
        metadata.update(meta)
        for key, value in metadata.items():
            if key in known_keys:
                conv = known_keys[key]
                value = conv(value) if value is not None else value
                setattr(struct, key, value)

        struct.to(file_like)


reader_classes.append(RegisterDat)
writer_classes.append(RegisterDat)

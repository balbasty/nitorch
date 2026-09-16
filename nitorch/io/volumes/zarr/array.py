"""Implementation of MappedArray based on zarr.

Reads both nifti-zarr stores (an embedded NIfTI header alongside a chunked
Zarr array) and plain OME-Zarr stores (0.4/0.5, no embedded header -- header
metadata is derived from the store's own OME-Zarr metadata instead, mirroring
the reference `nifti-zarr-py` implementation; see
`specs/003-nifti-zarr-support/research.md`).
"""

import numpy as np
import torch

from nitorch.core import py
from nitorch.io.mapping import AccessType
from nitorch.io.volumes.mapping import MappedArray
from nitorch.io.volumes.readers import reader_classes
from nitorch.io.volumes.babel.metadata import header_to_metadata
from nitorch.io.metadata import keys as metadata_keys
from nitorch.io.utils.indexing import split_operation

from . import metadata as niizarr_metadata
from . import ome_header as niizarr_ome


def _get_ome_metadata(attrs):
    """Return OME-Zarr `multiscales` metadata (`ome[0]` convention), or None."""
    multiscales = attrs.get('multiscales')
    if multiscales:
        return multiscales
    # OME-Zarr 0.5 nests version-scoped metadata under an "ome" key
    ome = attrs.get('ome') or {}
    if ome.get('multiscales'):
        return ome['multiscales']
    return None


def _get_level_paths(root, ome, is_group):
    """Return the array path of each pyramid level, finest first."""
    if ome:
        return [dataset['path'] for dataset in ome[0]['datasets']]
    if is_group:
        # a plain (non-OME) group of numbered levels, e.g. "0", "1", ...
        keys = sorted((k for k in root.array_keys() if k.isdigit()), key=int)
        return keys
    return []


class NiftiZarrArray(MappedArray):
    """MappedArray that relies on zarr to read nifti-zarr / OME-Zarr stores.

    Behaves like any other `MappedArray` backend (`.data()`/`.fdata()` are
    eager, `.affine`/`.voxel_size`/`.dtype`/`.shape` reflect the bound
    resolution level) and is reachable through the same format-agnostic
    `nitorch.io.map()`/`load()` entry points as `BabelArray`/`TiffArray`.

    Two capabilities are additive to the base `MappedArray` contract:

    * `.as_dask()` -- a lazily evaluated `dask.array.Array` view of this
      level's data (FR-004), for reading a sub-region of a store too large
      to fit in memory without loading the whole array.
    * `.level(i)` / `.nb_levels` -- for a multiscale store, fetch a
      specific native resolution level by index (FR-007); the default
      (no-argument) construction is always bound to the finest level.
    """

    FailedReadError = RuntimeError

    def __init__(self, file_like, mode='r', keep_open=True,
                 _level_index=0, _shared=None):
        """

        Parameters
        ----------
        file_like : str or zarr.Array or zarr.Group
            Path to a nifti-zarr/OME-Zarr store, or an already-open zarr
            object.
        mode : {'r', 'r+'}, default='r'
            File permission (only 'r' is currently supported -- writing is
            out of scope for this feature).
        keep_open : bool, default=True
            Kept for interface compatibility with other `MappedArray`
            backends; zarr stores are not held open between calls.

        """
        import zarr

        if mode not in ('r', 'r+'):
            raise ValueError(f"Mode expected in ('r', 'r+'). Got {mode}.")
        self.mode = mode
        self.keep_open = keep_open

        if _shared is not None:
            # constructing a sibling level (see `.level()`) -- reuse
            # everything already resolved by the original instance
            self.filename = _shared['filename']
            self._root = _shared['root']
            self._ome = _shared['ome']
            self._level_paths = _shared['level_paths']
            self._header = _shared['header']
        else:
            if isinstance(file_like, (zarr.Array, zarr.Group)):
                self.filename = None
                root = file_like
            else:
                self.filename = str(file_like)
                try:
                    root = zarr.open(store=self.filename, mode='r')
                except Exception as e:
                    raise self.FailedReadError(
                        f"{self.filename!r} could not be opened as a Zarr "
                        f"store: {e}"
                    ) from e

            self._root = root
            is_group = isinstance(root, zarr.Group)
            attrs = dict(getattr(root, 'attrs', {}) or {})
            self._ome = _get_ome_metadata(attrs)
            self._level_paths = _get_level_paths(root, self._ome, is_group)

            header = niizarr_metadata.read_embedded_header(root)
            if header is None:
                if self._ome is None:
                    raise self.FailedReadError(
                        f"{self.filename!r} is neither a nifti-zarr store "
                        f"(no embedded NIfTI header found) nor a "
                        f"recognizable OME-Zarr store (no multiscale "
                        f"metadata found). Supply a valid nifti-zarr or "
                        f"OME-Zarr path."
                    )
                header = niizarr_ome.derive_header(
                    self._level_array(0), self._ome)
            self._header = header

        self._level_index = _level_index
        self._array = self._level_array(_level_index)

        # The raw Zarr array is stored in whatever axis order the store
        # itself declares (OME-Zarr axes, e.g. z/y/x -- or, lacking OME
        # metadata, the nifti-zarr spec's own reversed-NIfTI-order
        # convention), which is generally *not* the same order as the
        # x/y/z/t/c order `self._shape`/`self._affine` (derived from the
        # header) describe. Set `self.permutation` so `raw_data()` reorders
        # the raw array to match -- mirroring `nifti-zarr-py`'s own
        # `_zarr2nii.py` axis-permutation logic exactly (see
        # specs/003-nifti-zarr-support/research.md).
        names = niizarr_ome.axis_order(len(self._array.shape), self._ome)
        perm = niizarr_ome.ome_permutation(names)
        if len(perm) == len(self._array.shape) == len(self._shape):
            self.permutation = perm

        super().__init__()

    def _level_array(self, index):
        """Return the raw zarr array for pyramid level `index`."""
        import zarr
        if isinstance(self._root, zarr.Array):
            return self._root
        if not self._level_paths:
            raise self.FailedReadError(
                f"{self.filename!r} has no readable array at level {index}."
            )
        return self._root[self._level_paths[index]]

    # ------------------------------------------------------------------
    #    FORMAT RECOGNITION
    # ------------------------------------------------------------------

    @classmethod
    def possible_extensions(cls):
        return ('.zarr',)

    @classmethod
    def sniff(cls, file_like):
        try:
            cls(file_like)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    #    ATTRIBUTES
    # ------------------------------------------------------------------

    _spatial = property(
        lambda self: tuple([True] * 3 + [False] * max(0, self._dim - 3)))
    dtype = property(lambda self: self._array.dtype)

    @property
    def _shape(self):
        # The header describes the *finest* level (it is shared, unchanged,
        # across all levels of a store -- see `.level()`). For any other
        # level, its own array is smaller/larger than the header's declared
        # shape, so re-derive the (correctly axis-ordered) shape from this
        # level's own raw array shape instead of trusting the header here.
        if self._level_index == 0 or self._ome is None:
            return tuple(int(d) for d in self._header.get_data_shape())
        return tuple(int(d) for d in niizarr_ome.ome_shape(self._array.shape, self._ome))

    @property
    def _affine(self):
        return torch.as_tensor(self._header.get_best_affine(), dtype=torch.double)

    @property
    def slope(self):
        slope, _ = self._header.get_slope_inter()
        return float(slope) if slope is not None else 1.0

    @property
    def inter(self):
        _, inter = self._header.get_slope_inter()
        return float(inter) if inter is not None else 0.0

    @property
    def readable(self):
        return AccessType.TruePartial

    @property
    def writable(self):
        return AccessType.No

    def metadata(self, keys=None):
        keys = keys or metadata_keys
        return header_to_metadata(self._header, keys)

    # ------------------------------------------------------------------
    #    MULTISCALE / LEVEL-FETCH (FR-007)
    # ------------------------------------------------------------------

    @property
    def nb_levels(self):
        """Number of native resolution levels this store provides."""
        return max(1, len(self._level_paths))

    def level(self, index):
        """Return a `NiftiZarrArray` bound to native resolution `index`.

        Shares this store's already-parsed header and OME metadata --
        the header is not re-read or re-derived per level.

        Parameters
        ----------
        index : int

        Returns
        -------
        NiftiZarrArray

        """
        if not (0 <= index < self.nb_levels):
            raise IndexError(
                f"Level {index} out of range: this store has "
                f"{self.nb_levels} native level(s)."
            )
        shared = dict(filename=self.filename, root=self._root, ome=self._ome,
                      level_paths=self._level_paths, header=self._header)
        return type(self)(self.filename, mode=self.mode,
                           keep_open=self.keep_open, _level_index=index,
                           _shared=shared)

    # ------------------------------------------------------------------
    #    LAZY ARRAY ACCESS (FR-004)
    # ------------------------------------------------------------------

    def as_dask(self):
        """Return this level's data as a lazily evaluated dask array, in
        the same (x, y, z, ...) axis order as `.shape`/`.fdata()` (not
        necessarily the store's own raw on-disk axis order).

        Returns
        -------
        dask.array.Array

        """
        return self._as_dask_raw().transpose(self.permutation)

    def _as_dask_raw(self):
        """This level's data as a dask array, in the store's own raw
        on-disk axis order (i.e. before `self.permutation` is applied)."""
        import dask.array as da
        return da.from_array(self._array)

    # ------------------------------------------------------------------
    #    LOW-LEVEL IMPLEMENTATION
    # ------------------------------------------------------------------

    def raw_data(self):
        """Load this level's data, honoring the current (symbolic) slicer.

        Returns
        -------
        dat : np.ndarray[self.dtype]

        """
        if py.prod(self.shape) == 0:
            return np.zeros(self.shape, dtype=self.dtype)
        slicer, perm, newdim = split_operation(self.permutation, self.slicer, 'r')
        dat = np.asarray(self._as_dask_raw()[slicer].compute())
        dat = dat.transpose(perm)[newdim]
        return dat


reader_classes.append(NiftiZarrArray)

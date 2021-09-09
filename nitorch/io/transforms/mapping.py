from nitorch.core.optionals import numpy as np
from ..mapping import MappedFile


class MappedAffine(MappedFile):
    """An affine matrix stored on disk"""

    @classmethod
    def possible_extensions(cls):
        """List all possible extensions"""
        return tuple()

    def __str__(self):
        try:
            affine = self.fdata()
        except NotImplementedError:
            affine = None
        if affine is None:
            return '{}(None)'.format(type(self).__name__)
        affine = str(np.asarray(affine))
        klass = type(self).__name__
        affine = affine.split('\n')
        pad = len(klass) + 1
        affine = ('\n' + ' ' * pad).join(affine)
        return '{}({})'.format(type(self).__name__, affine)

    __repr__ = __str__

    def fdata(self, dtype=None, device=None, numpy=False):
        """Load the affine matrix from file.

        If the file encodes a transformation, this function tries to
        return a world-to-world matrix, in mm.

        If the file encodes an orientation matrix, this function tries
        to return a voxel-to-world, in mm.
        In neuroimaging, world is usually a RAS space:
            - left is -x       ->  right is +x
            - posterior is -y  ->  anterior is +y
            - inferior is -z   ->  superior is +z

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        numpy : bool, default=False

        Returns
        -------
        affine : (..., 4, 4) tensor or array
            Affine matrix

        """
        raise NotImplementedError

    def data(self, dtype=None, device=None, numpy=False):
        """Load the affine matrix from file.

        The "raw" matrix is loaded even if it is not a world-to-world
        transformation matrix or voxel-to-world orientation matrix.
        When known, its type is returned by the `type()` function.

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        numpy : bool, default=False

        Returns
        -------
        affine : (..., 4, 4) tensor or array
            Affine matrix
        """

    def type(self):
        """Type of affine matrix returned by `data()`

        Returns
        -------
        tuple : (source_space, destination_space)
            Spaces are one of {'voxel', 'physical', 'world', 'mni'} or
            permutations of ('rl', 'ap', 'si'). Some format specific spaces that
            have no generic equivalent can also be returned in some (rare)
            cases.
            - 'world' spaces have their coordinates (usually) expressed in mm.
            - 'physical' means 'voxel' scaled by voxel size.
              It is a world space aligned with the voxel lattice, where
              the origin is in the center of the field-of-view.
              There is no general anatomical meaning associated with its
              dimensions.
            - RAS-type spaces are world coordinates specific to neuroimaging,
              where the x/y/z axes have anatomical meaning (right/left,
              anterior/posterior, superior/inferior).
              By default, 'world' is assumed equivalent to 'ras'.
            - 'mni' is a specific 'ras' space that is attached to the MNI
              atlas.

        """

    def metadata(self, keys=None):
        """Read additional metadata from the transform

        Parameters
        ----------
        keys : sequence of str, optional
            List of metadata keys to read.
            If not provided, all (format-specific) known keys are read/

        Returns
        -------
        dict

        """
        raise NotImplementedError

    def set_fdata(self, affine):
        """Set the affine matrix

        This function only modifies the in-memory representation of the
        transform. The file is not modified. To overwrite the file,
        call `save` afterward.

        Parameters
        ----------
        affine : (..., 4, 4) tensor or array

        Returns
        -------
        self

        """
        raise NotImplementedError

    def set_data(self, affine):
        """Set the (raw) affine matrix

        This function only modifies the in-memory representation of the
        transform. The file is not modified. To overwrite the file,
        call `save` afterward.

        Parameters
        ----------
        affine : (..., 4, 4) tensor or array

        Returns
        -------
        self

        """
        raise NotImplementedError

    def set_metadata(self, **meta):
        """Set additional metadata in the transform.

        This function only modifies the in-memory representation of the
        transform. The file is not modified. To overwrite the file,
        use `save`.

        Parameters
        ----------
        **meta : dict
            Only keys that make sense to the format will effectively
            be set.

        Returns
        -------
        self

        """
        raise NotImplementedError

    def save(self, file_like=None, **meta):
        """Save the current transform to disk.

        Parameters
        ----------
        file_like : str or file object, default=self.file_like
            Target file to write the transform.
        **meta : dict
            Additional metadata to set before saving.

        Returns
        -------
        self

        """
        raise NotImplementedError

    @classmethod
    def save_new(cls, affine, file_like, like=None, **meta):
        """Save a new affine to disk in the `cls` format

        Parameters
        ----------
        affine : MappedAffine or (..., 4, 4) tensor or array
            Affine matrix to write
        file_like : str or file object
            Target file
        like : MappedAffine or str or file object, optional
            Template affine. Its metadata fields will be copied unless
            they are overwritten by `meta`.
        **meta : dict
            Additional metadata to set before writing.

        """
        raise NotImplementedError

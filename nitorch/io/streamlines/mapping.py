import numpy as np
from ..mapping import MappedFile


class MappedStreamlines(MappedFile):
    """Streamlines stored on disk"""

    @classmethod
    def possible_extensions(cls):
        """List all possible extensions"""
        return tuple()

    def __str__(self):
        return '{}()'.format(type(self))

    __repr__ = __str__

    @property
    def affine(self):
        """
        Vertex to world transformation matrix.
        """
        raise NotImplementedError

    @property
    def dtype(self):
        return np.dtype("float64")

    def fdata(self, dtype=None, device=None, numpy=False):
        """Load the streamlines from file.

        This function tries to return vertices in RAS space.

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        numpy : bool, default=False

        Returns
        -------
        streamlines : iterator[(N, 3) tensor or array]
            Streamlines

        """
        raise NotImplementedError

    def data(self, dtype=None, device=None, numpy=False):
        """Load the streamlines from file.

        The "raw" streamlines are loaded even if they are not in RAS space.

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        numpy : bool, default=False

        Returns
        -------
        streamlines : iterator[(N, 3) tensor or array]
            Streamlines
        """
        raise NotImplementedError

    def scalars(self, dtype=None, device=None, numpy=False, keys=None):
        """
        Load the scalars associated with each vertex.

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        numpy : bool, default=False
        keys : [list of] str, optional
            Keys to load

        Returns
        -------
        scalars : [dict of] iterator[(N, K) tensor or array]
            Scalars.
            If `keys` is a string, return the list of scalars directly.
            Else, return a dictionary mapping keys to scalars.
        """
        raise NotImplementedError

    def properties(self, dtype=None, device=None, numpy=False, keys=None):
        """
        Load the properties associated with each streamline.

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        numpy : bool, default=False
        keys : [list of] str, optional
            Keys to load

        Returns
        -------
        scalars : [dict of] iterator[(N, K) tensor or array]
            Scalars.
            If `keys` is a string, return the list of properties directly.
            Else, return a dictionary mapping keys to properties.
        """
        raise NotImplementedError

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

    def set_fdata(self, streamlines):
        """Set the streamlines data

        This function only modifies the in-memory representation of the
        streamlines. The file is not modified. To overwrite the file,
        call `save` afterward.

        Parameters
        ----------
        streamlines : list[(N, 3) tensor or array]

        Returns
        -------
        self

        """
        raise NotImplementedError

    def set_data(self, affine):
        """Set the (raw) streamlines data

        This function only modifies the in-memory representation of the
        streamlines. The file is not modified. To overwrite the file,
        call `save` afterward.

        Parameters
        ----------
        streamlines : list[(N, 3) tensor or array]

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
        """Save the current streamlines to disk.

        Parameters
        ----------
        file_like : str or file object, default=self.file_like
            Target file to write the streamlines.
        **meta : dict
            Additional metadata to set before saving.

        Returns
        -------
        self

        """
        raise NotImplementedError

    savef = save

    @classmethod
    def save_new(cls, streamlines, file_like, like=None, **meta):
        """Save a new affine to disk in the `cls` format

        Parameters
        ----------
        streamlines : MappedStreamlines or list[(N, 3) tensor or array]
            Streamlines to write
        file_like : str or file object
            Target file
        like : MappedStreamlines or str or file object, optional
            Template streamlines. Its metadata fields will be copied unless
            they are overwritten by `meta`.
        **meta : dict
            Additional metadata to set before writing.

        """
        raise NotImplementedError

    @classmethod
    def savef_new(cls, streamlines, file_like, like=None, **meta):
        """Save a new streamlines to disk in the `cls` format

        Parameters
        ----------
        streamlines : MappedStreamlines or list[(N, 3) tensor or array]
            Streamlines to write
        file_like : str or file object
            Target file
        like : MappedStreamlines or str or file object, optional
            Template streamlines. Its metadata fields will be copied unless
            they are overwritten by `meta`.
        **meta : dict
            Additional metadata to set before writing.

        """
        raise NotImplementedError

""""Conversions between generic and nibabel-specific metadata"""
import torch
from warnings import warn
from ..spatial import voxel_size
from ..core.optionals import numpy as np
from ..core.optionals import try_import
MGHHeader = try_import('nibabel.freesurfer.mghformat', 'MGHHeader')
Nifti1Header, Spm2AnalyzeHeader, Spm99AnalyzeHeader, AnalyzeHeader = try_import(
    'nibabel', ['Nifti1Header', 'Spm2AnalyzeHeader','Spm99AnalyzeHeader', 'AnalyzeHeader'])


def metadata_to_header(header, metadata, shape=None, dtype=None):
    """Register metadata into a nibabel Header object

    Parameters
    ----------
    header : Header or type
        Original header _or_ Header class.
        If it is a class, default values are used to populate the
        missing fields.
    metadata : dict
        Dictionary of metadata

    Returns
    -------
    header : Header

    """
    is_empty = type(header) is type
    if is_empty:
        header = header()

    # --- generic fields ---

    if metadata.get('voxel_size', None) is not None:
        vx0 = header.get_zooms()
        vx = metadata['voxel_size']
        vx = [vx[i] if i < len(vx) else vx0[i] for i in range(len(vx0))]
        header.set_zooms(vx)

    if metadata.get('affine', None) is not None:
        affine = metadata['affine']
        if torch.is_tensor(affine):
            affine = affine.detach()
        affine = np.asanyarray(affine)
        if isinstance(header, MGHHeader):
            if shape is None:
                warn('Cannot set the affine of a MGH file without '
                     'knowing the data shape', RuntimeWarning)
            elif affine.shape not in ((3, 4), (4, 4)):
                raise ValueError('Expected a (3, 4) or (4, 4) affine matrix. '
                                 'Got {}'.format(affine.shape))
            else:
                affine = np.asanyarray(affine)
                vx = voxel_size(affine)
                Mdc = affine[:3, :3] / vx
                c_ras = affine.dot(np.hstack((shape / 2.0, [1])))[:3]

                # Assign after we've had a chance to raise exceptions
                header['delta'] = vx
                header['Mdc'] = Mdc.T
                header['Pxyz_c'] = c_ras
        elif isinstance(header, Nifti1Header):
            header.set_sform(affine)
        elif isinstance(header, AnalyzeHeader):
            header.set_zooms(voxel_size(affine))
            if isinstance(header, Spm99AnalyzeHeader):
                header.set_origin_from_affine(affine)
        else:
            warn('Format {} does not accept orientation matrices. '
                 'It will be discarded.'.format(type(header).__name__),
                 RuntimeWarning)

    if (metadata.get('slope', None) is not None or
        metadata.get('inter', None) is not None):
        slope = metadata.get('slope', 1.)
        inter = metadata.get('inter', None)
        if isinstance(header, Spm99AnalyzeHeader):
            header.set_slope_inter(slope, inter)
        else:
            if slope not in (1, None) or inter not in (0, None):
                warn('Format {} does not accept intensity transforms. '
                     'It will be discarded.'.format(type(header).__name__),
                     RuntimeWarning)

    if (metadata.get('time_step', None) is not None or
        metadata.get('tr', None) is not None):
        time_step = metadata.get('time_step', None) or metadata['tr']
        if isinstance(header, Nifti1Header):
            time_step = time_step / header.get_n_slices()
            header.set_slice_duration(time_step)
        elif isinstance(header, MGHHeader):
            zooms = header.get_zooms()[:-1]
            zooms = (*zooms, time_step)
            header.set_zooms(zooms)
        else:
            warn('Format {} does not accept time steps. '
                 'It will be discarded.'.format(type(header).__name__),
                 RuntimeWarning)

    # TODO: time offset / intent for nifti format
    # TODO: te/ti/fa for MGH format
    #       maybe also nifti from description field?

    if dtype is not None or metadata.get('dtype', None) is not None:
        dtype = dtype or metadata.get('dtype', None)
        header.set_data_dtype(dtype)

    return header


def header_to_metadata(header, metadata):
    """Read metadata from nibabel header

    Parameters
    ----------
    header : Header
        Original header.
    metadata : dict or sequence
        If sequence of keys, build dictionary and populate/
        If dict, populate dictionary.

    Returns
    -------
    metadata : dict
        Dictionary of metadata

    """

    if not isinstance(metadata, dict):
        metadata = {key: None for key in metadata}

    if 'voxel_size' in metadata:
        metadata['voxel_size'] = header.get_zooms()[:3]

    if 'affine' in metadata:
        metadata['affine'] = torch.as_tensor(header.get_best_affine())

    if 'slope' in metadata:
        metadata['slope'], _ = header.get_slope_inter()

    if 'inter' in metadata:
        _, metadata['inter'] = header.get_slope_inter()

    if 'time_step' in metadata or 'tr' in metadata:
        if isinstance(header, Nifti1Header):
            try:
                time_step = header.get_slice_duration() * header.get_n_slices()
            except:
                time_step = None
            if 'time_step' in metadata:
                metadata['time_step'] = time_step
            if 'tr' in metadata:
                metadata['tr'] = time_step
        elif isinstance(header, MGHHeader):
            time_step = header.get_zooms()[-1]
            if 'time_step' in metadata:
                metadata['time_step'] = time_step
            if 'tr' in metadata:
                metadata['tr'] = time_step
        else:
            if 'time_step' in metadata:
                metadata['time_step'] = None
            if 'tr' in metadata:
                metadata['tr'] = None
            warn('Format {} does not store time steps. ', RuntimeWarning)

    # TODO: time offset / intent for nifti format
    # TODO: te/ti/fa for MGH format

    if 'dtype' in metadata:
        metadata['dtype'] = header.get_data_dtype()

    if 'format' in metadata:
        metadata['format'] = header.__class__.__name__.split('Header')[0]

    return metadata


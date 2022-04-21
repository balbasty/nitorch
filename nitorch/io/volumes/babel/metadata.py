""""Conversions between generic and nibabel-specific metadata"""
import torch
from warnings import warn
import numpy as np
import re
import ast
from nibabel.freesurfer.mghformat import MGHHeader
from nibabel import (Nifti1Header, Spm99AnalyzeHeader, AnalyzeHeader)
from nibabel.spatialimages import HeaderDataError
from nitorch.spatial import voxel_size
from nitorch.core import dtypes, constants
from nitorch.core.utils import make_vector


def set_affine(header, affine, shape=None):
    if torch.is_tensor(affine):
        affine = affine.detach().cpu()
    affine = np.asanyarray(affine)
    vx = np.asanyarray(voxel_size(affine))
    vx0 = header.get_zooms()
    vx = [vx[i] if i < len(vx) else vx0[i] for i in range(len(vx0))]
    header.set_zooms(vx)
    if isinstance(header, MGHHeader):
        if shape is None:
            warn('Cannot set the affine of a MGH file without '
                 'knowing the data shape', RuntimeWarning)
        elif affine.shape not in ((3, 4), (4, 4)):
            raise ValueError('Expected a (3, 4) or (4, 4) affine matrix. '
                             'Got {}'.format(affine.shape))
        else:
            Mdc = affine[:3, :3] / vx[:3]
            shape = np.asarray(shape[:3])
            c_ras = affine.dot(np.hstack((shape / 2.0, [1])))[:3]

            # Assign after we've had a chance to raise exceptions
            header['delta'] = vx[:3]
            header['Mdc'] = Mdc.T
            header['Pxyz_c'] = c_ras
    elif isinstance(header, Nifti1Header):
        header.set_sform(affine)
        header.set_qform(affine)
    elif isinstance(header, Spm99AnalyzeHeader):
        header.set_origin_from_affine(affine)
    else:
        warn('Format {} does not accept orientation matrices. '
             'It will be discarded.'.format(type(header).__name__),
             RuntimeWarning)
    return header
    

def set_voxel_size(header, vx, shape=None):
    vx0 = header.get_zooms()
    nb_dim = max(len(vx0), len(vx))
    vx = [vx[i] if i < len(vx) else vx0[i] for i in range(nb_dim)]
    header.set_zooms(vx)
    aff = torch.as_tensor(header.get_best_affine())
    vx = torch.as_tensor(vx, dtype=aff.dtype, device=aff.device)
    vx0 = voxel_size(aff)
    aff[:-1, :] *= vx[:3, None] / vx0[:3, None]
    header = set_affine(header, aff, shape)
    return header
    

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

    if metadata.get('voxel_size_unit', None) is not None:
        val = metadata['voxel_size_unit']
        if hasattr(header, 'set_xyzt_units'):
            header.set_xyzt_units(val, None)
        else:
            format_name = type(header).__name__.split('Header')[0]
            warn('Format {} does not accept voxel size units. '
                 'It will be discarded.'.format(format_name),
                 RuntimeWarning)

    if metadata.get('voxel_size', None) is not None:
        val = metadata['voxel_size']
        if isinstance(val, str):
            val = ast.literal_eval(val)
        header = set_voxel_size(header, val, shape)

    if metadata.get('affine', None) is not None:
        val = metadata['affine']
        if isinstance(val, str):
            val = ast.literal_eval(val)
        header = set_affine(header, val, shape)

    if (metadata.get('slope', None) is not None or
        metadata.get('inter', None) is not None):
        slope = metadata.get('slope', 1.)
        inter = metadata.get('inter', None)
        if isinstance(slope, str):
            slope = float(ast.literal_eval(slope))
        if isinstance(inter, str):
            slope = float(ast.literal_eval(inter))
        if isinstance(header, (Spm99AnalyzeHeader, Nifti1Header)):
            header.set_slope_inter(slope, inter)
        else:
            if slope not in (1, None) or inter not in (0, None):
                format_name = type(header).__name__.split('Header')[0]
                warn('Format {} does not accept intensity transforms. '
                     'It will be discarded.'.format(format_name),
                     RuntimeWarning)

    if (metadata.get('time_step', None) is not None or
        metadata.get('tr', None) is not None):
        time_step = metadata.get('time_step', None) or metadata['tr']
        unit = None
        if isinstance(time_step, str):
            if time_step.endswith('sec'):
                time_step = time_step[:-3]
                unit = 'sec'
            elif time_step.endswith('ms'):
                time_step = time_step[:-2]
                unit = 'ms'
            elif time_step.endswith('s'):
                time_step = time_step[:-1]
                unit = 'sec'
            time_step = float(ast.literal_eval(time_step))
        unit = unit or metadata.get('te_unit', 'sec')
        if isinstance(header, (MGHHeader, Nifti1Header)):
            if unit == 'sec':
                time_step = time_step * 1e3  # TODO: unit for niftis?
            zooms = header.get_zooms()[:3]
            zooms = (*zooms, time_step)
            try:
                # only possible if 4-th dimension is explicit
                header.set_zooms(zooms)
            except HeaderDataError:
                if isinstance(header, MGHHeader):
                    # set tr manually
                    header['tr'] = time_step

        else:
            warn('Format {} does not accept time steps. '
                 'It will be discarded.'.format(type(header).__name__),
                 RuntimeWarning)

    # TODO: time offset / intent for nifti format

    # TODO: te/ti/fa for MGH format
    #       maybe also nifti from description field?
    if metadata.get('te', None) is not None:
        val = metadata['te']
        unit = None
        if isinstance(val, str):
            if val.endswith('sec'):
                val = val[:-3]
                unit = 'sec'
            elif val.endswith('ms'):
                val = val[:-2]
                unit = 'ms'
            elif val.endswith('s'):
                val = val[:-1]
                unit = 'sec'
            val = float(ast.literal_eval(val))
        unit = unit or metadata.get('te_unit', 'sec')
        if isinstance(header, MGHHeader):
            if unit == 'sec':
                val = val * 1e3
            header['te'] = val

    if metadata.get('ti', None) is not None:
        val = metadata['ti']
        unit = None
        if isinstance(val, str):
            if val.endswith('sec'):
                val = val[:-3]
                unit = 'sec'
            elif val.endswith('ms'):
                val = val[:-2]
                unit = 'ms'
            elif val.endswith('s'):
                val = val[:-1]
                unit = 'sec'
            val = float(ast.literal_eval(val))
        unit = unit or metadata.get('ti_unit', 'sec')
        if isinstance(header, MGHHeader):
            if unit == 'sec':
                val = val * 1e3
            header['ti'] = val

    if metadata.get('fa', None) is not None:
        val = metadata['fa']
        unit = None
        if isinstance(val, str):
            if val.endswith('deg'):
                val = val[:-3]
                unit = 'deg'
            elif val.endswith('rad'):
                val = val[:-3]
                unit = 'rad'
            val = float(ast.literal_eval(val))
        unit = unit or metadata.get('fa_unit', 'deg')
        if isinstance(header, MGHHeader):
            if unit == 'deg':
                val = val * constants.pi / 180.
            header['flip_angle'] = val

    if dtype is not None or metadata.get('dtype', None) is not None:
        dtype = dtype or metadata.get('dtype', None)
        dtype = dtypes.dtype(dtype).numpy
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
        if hasattr(header, 'get_xyzt_units'):
            metadata['voxel_size_unit'], _ = header.get_xyzt_units()
        else:
            metadata['voxel_size_unit'] = 'mm'

    elif 'voxel_size_unit' in metadata:
        if hasattr(header, 'get_xyzt_units'):
            metadata['voxel_size_unit'], _ = header.get_xyzt_units()
        else:
            metadata['voxel_size_unit'] = 'mm'

    if 'affine' in metadata:
        metadata['affine'] = torch.as_tensor(header.get_best_affine())

    if 'slope' in metadata:
        if hasattr(header, 'get_slope_inter'):
            metadata['slope'], _ = header.get_slope_inter()
        else:
            metadata['slope'] = None

    if 'inter' in metadata:
        if hasattr(header, 'get_slope_inter'):
            _, metadata['inter'] = header.get_slope_inter()
        else:
            metadata['inter'] = None

    if 'time_step' in metadata or 'tr' in metadata:
        zooms = header.get_zooms()
        time_step = zooms[3] if len(zooms) > 3 else None
        time_step = time_step or None
        if isinstance(header, MGHHeader):
            time_step = float(header['tr']) or None
            time_unit = 'ms'
        elif hasattr(header, 'get_xyzt_units'):
            _, time_unit = header.get_xyzt_units()
        else:
            time_unit = 'ms' if isinstance(header, MGHHeader) else 'sec'
        if time_step is None and isinstance(header, Nifti1Header):
            descrip = header['descrip'].tobytes().decode().rstrip('\x00')
            parse = re.search('tr=(?P<tr>[\d.]+)\s*(?P<unit>([m]?s)?)',
                              descrip, re.IGNORECASE)
            if parse:
                time_step = float(parse.group('tr'))
                time_unit = parse.group('unit') or time_unit

        if 'time_step' in metadata:
            metadata['time_step'] = time_step
            if time_step is not None:
                metadata['time_step_unit'] = time_unit
        if 'tr' in metadata:
            metadata['tr'] = time_step
            if time_step is not None:
                metadata['tr_unit'] = time_unit

    if 'time_offset' in metadata:
        if isinstance(header, Nifti1Header):
            metadata['time_offset'] = float(header['toffset'])
        else:
            metadata['time_offset'] = None

    if 'te' in metadata:
        if isinstance(header, MGHHeader):
            te = float(header['te']) or None
            te_unit = 'ms'
        else:
            te = None
            te_unit = 'sec'
        if te is None and isinstance(header, Nifti1Header):
            descrip = header['descrip'].tobytes().decode().rstrip('\x00')
            parse = re.search('te=(?P<te>[\d.]+)\s*(?P<unit>([m]?s)?)',
                              descrip, re.IGNORECASE)
            if parse:
                te = float(parse.group('te'))
                te_unit = parse.group('unit') or te_unit

        metadata['te'] = te
        if te:
            metadata['te_unit'] = te_unit

    if 'ti' in metadata:
        if isinstance(header, MGHHeader):
            ti = float(header['ti']) or None
            ti_unit = 'ms'
        else:
            ti = None
            ti_unit = 'sec'
        if ti is None and isinstance(header, Nifti1Header):
            descrip = header['descrip'].tobytes().decode().rstrip('\x00')
            parse = re.search('ti=(?P<ti>[\d.]+)\s*(?P<unit>([m]?s)?)',
                              descrip, re.IGNORECASE)
            if parse:
                ti = float(parse.group('ti'))
                ti_unit = parse.group('unit') or ti_unit

        metadata['ti'] = ti
        if ti:
            metadata['ti_unit'] = ti_unit

    if 'fa' in metadata:
        if isinstance(header, MGHHeader):
            fa = float(header['flip_angle']) or None
            fa_unit = 'rad'
        else:
            fa = None
            fa_unit = 'deg'
        if fa is None and isinstance(header, Nifti1Header):
            descrip = header['descrip'].tobytes().decode().rstrip('\x00')
            parse = re.search('fa=(?P<fa>[\d.]+)\s*(?P<unit>(deg|rad)?)',
                              descrip, re.IGNORECASE)
            if parse:
                fa = float(parse.group('fa'))
                fa_unit = parse.group('unit') or fa_unit

        metadata['fa'] = fa
        if fa:
            metadata['fa_unit'] = fa_unit

    if 'dtype' in metadata:
        metadata['dtype'] = header.get_data_dtype()

    if 'format' in metadata:
        metadata['format'] = header.__class__.__name__.split('Header')[0]

    return metadata


import re
from warnings import warn
import torch
from nitorch.core import utils
from nitorch.spatial import affine_matmul, affine_inv, layout_matrix
from nitorch.spatial import voxel_size as get_voxel_size
from nitorch.io.transforms.conversions import Orientation, XYZC, HomogeneousAffineMatrix


# Regex patterns for different value types
patterns = {
    int: r'(\d+)',
    float: r'([\+\-]?\d+\.?\d*(?:[eE][\+\-]?\d+)?)',
    str: r'(.*)\s*$'
}


def read_key(line, key_dict=None):
    """Read one `key = value` line from an LTA file

    Parameters
    ----------
    line : str
    format : type in {int, float, str}

    Returns
    -------
    object or tuple or None

    """
    key_dict = key_dict or dict()

    line = line.split('\r\n')[0]  # remove eol (windows)
    line = line.split('\n')[0]    # remove eol (unix)
    line = line.split('#')[0]     # remove hanging comments
    pattern = r'^\s*(\S+)\s*=\s*(.*)$'
    match = re.match(pattern, line)
    if not match:
        warn(f'cannot read line: "{line}"', RuntimeWarning)
        return None, None

    key = match.group(1)
    value = match.group(2).rstrip()
    if key in key_dict:
        format = key_dict[key]
        if isinstance(format, type):
            pattern = patterns[format]
        else:
            pattern = r'\s*'.join([patterns[fmt] for fmt in format])
        match = re.match(pattern, value)
        if match:
            if isinstance(format, type):
                value = format(match.group(1))
            else:
                value = tuple(fmt(v) for v, fmt in zip(match.groups(), format))
        else:
            warn(f'cannot parse value: "{value}"', RuntimeWarning)
    return key, value


def read_values(line, format):
    """Read one `*values` line from an LTA file

    Parameters
    ----------
    line : str
    format : [sequence of] type
        One of {int, float, str}

    Returns
    -------
    object or tuple or None

    """
    line = line.split('\r\n')[0]  # remove eol (windows)
    line = line.split('\n')[0]    # remove eol (unix)
    line = line.split('#')[0]     # remove hanging comments
    pattern = r'\s*'
    if isinstance(format, type):
        reformat = [patterns[format]]
    else:
        reformat = [patterns[fmt] for fmt in format]
    pattern += r'\s*'.join(reformat)
    value = re.match(pattern, line)
    if value:
        if isinstance(format, type):
            value = format(value.group(1))
        else:
            value = tuple(fmt(v) for v, fmt in zip(value.groups(), format))
    else:
        warn(f'cannot read line: "{line}"', RuntimeWarning)
    return value


def write_key(key, value):
    """Write a `key = value` line in an LTA file.

    Parameters
    ----------
    key : str
    value : int or float or str

    Returns
    -------
    str

    """
    if isinstance(value, (str, int, float)):
        return f'{key:9s} = {value}'
    else:
        return f'{key:9s} = ' + ' '.join([str(v) for v in value])


def write_values(value):
    """Write a `*values` line in an LTA file.

    Parameters
    ----------
    value : [sequence of] int or float or str

    Returns
    -------
    str

    """
    if isinstance(value, (str, int, float)):
        return str(value)
    else:
        return ' '.join([str(v) for v in value])


def fs_to_affine(shape, voxel_size=1., x=None, y=None, z=None, c=0.,
                 source='voxel', dest='ras'):
    """Transform FreeSurfer orientation parameters into an affine matrix.

    The returned matrix is effectively a "<source> to <dest>" transform.

    Parameters
    ----------
    shape : sequence of int
    voxel_size : [sequence of] float, default=1
    x : [sequence of] float, default=[1, 0, 0]
    y: [sequence of] float, default=[0, 1, 0]
    z: [sequence of] float, default=[0, 0, 1]
    c: [sequence of] float, default=0
    source : {'voxel', 'physical', 'ras'}, default='voxel'
    dest : {'voxel', 'physical', 'ras'}, default='ras'

    Returns
    -------
    affine : (4, 4) tensor

    """
    dim = len(shape)
    shape, voxel_size, x, y, z, c \
        = utils.to_max_backend(shape, voxel_size, x, y, z, c)
    backend = dict(dtype=shape.dtype, device=shape.device)
    voxel_size = utils.make_vector(voxel_size, dim)
    if x is None:
        x = [1, 0, 0]
    if y is None:
        y = [0, 1, 0]
    if z is None:
        z = [0, 0, 1]
    x = utils.make_vector(x, dim)
    y = utils.make_vector(y, dim)
    z = utils.make_vector(z, dim)
    c = utils.make_vector(c, dim)

    shift = shape / 2.
    shift = -shift * voxel_size
    vox2phys = Orientation(shift, voxel_size).affine()
    phys2ras = XYZC(x, y, z, c).affine()

    affines = []
    if source.lower().startswith('vox'):
        affines.append(vox2phys)
        middle_space = 'phys'
    elif source.lower().startswith('phys'):
        if dest.lower().startswith('vox'):
            affines.append(affine_inv(vox2phys))
            middle_space = 'vox'
        else:
            affines.append(phys2ras)
            middle_space = 'ras'
    elif source.lower() == 'ras':
        affines.append(affine_inv(phys2ras))
        middle_space = 'phys'
    else:
        # We need a matrix to switch orientations
        affines.append(layout_matrix(source, **backend))
        middle_space = 'ras'

    if dest.lower().startswith('phys'):
        if middle_space == 'vox':
            affines.append(vox2phys)
        elif middle_space == 'ras':
            affines.append(affine_inv(phys2ras))
    elif dest.lower().startswith('vox'):
        if middle_space == 'phys':
            affines.append(affine_inv(vox2phys))
        elif middle_space == 'ras':
            affines.append(affine_inv(phys2ras))
            affines.append(affine_inv(vox2phys))
    elif dest.lower().startswith('ras'):
        if middle_space == 'phys':
            affines.append(phys2ras)
        elif middle_space.lower().startswith('vox'):
            affines.append(vox2phys)
            affines.append(phys2ras)
    else:
        if middle_space == 'phys':
            affines.append(affine_inv(phys2ras))
        elif middle_space == 'vox':
            affines.append(vox2phys)
            affines.append(phys2ras)
        layout = layout_matrix(dest, **backend)
        affines.append(affine_inv(layout))

    affine, *affines = affines
    for aff in affines:
        affine = affine_matmul(aff, affine)
    return affine


def affine_to_fs(affine, shape, source='voxel', dest='ras'):
    """Convert an affine matrix into FS parameters (vx/cosine/shift)

    Parameters
    ----------
    affine : (4, 4) tensor
    shape : (int, int, int)
    source : {'voxel', 'physical', 'ras'}, default='voxel'
    dest : {'voxel', 'physical', 'ras'}, default='ras'

    Returns
    -------
    voxel_size : (float, float, float)
    x : (float, float, float)
    y : (float, float, float)
    z: (float, float, float)
    c : (float, float, float)

    """

    affine = torch.as_tensor(affine)
    backend = dict(dtype=affine.dtype, device=affine.device)
    vx = get_voxel_size(affine)
    shape = torch.as_tensor(shape, **backend)
    source = source.lower()[0]
    dest = dest.lower()[0]

    shift = shape / 2.
    shift = -shift * vx
    vox2phys = Orientation(shift, vx).affine()

    if (source, dest) in (('v', 'p'), ('p', 'v')):
        phys2ras = torch.eye(4, **backend)

    elif (source, dest) in (('v', 'r'), ('r', 'v')):
        if source == 'r':
            affine = affine_inv(affine)
        phys2vox = affine_inv(vox2phys)
        phys2ras = affine_matmul(affine, phys2vox)

    else:
        assert (source, dest) in (('p', 'r'), ('r', 'p'))
        if source == 'r':
            affine = affine_inv(affine)
        phys2ras = affine

    phys2ras = HomogeneousAffineMatrix(phys2ras)
    return (vx.tolist(), phys2ras.xras().tolist(), phys2ras.yras().tolist(),
            phys2ras.zras().tolist(), phys2ras.cras().tolist())



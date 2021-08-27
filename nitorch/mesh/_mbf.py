import re
import torch
from nitorch.core import py, utils, math
from nitorch import spatial
from ._inclusion import is_inside
import copy


def read_markers(fname, vx, split_groups=False):
    """Read marker coordinates file into freesurfer-compatible json format

    Parameters are returned in voxel space. The image that was used to
    create the coordinates must be known to convert them to some sort of
    RAS space.

    Parameters
    ----------
    fname : str
        Path to marker file
    vx : sequence[float], defualt=1
        Voxel size
    split_groups : bool, default=False
        If True, return one dictionary per marker type.

    Returns
    -------
    coord : dict (or list of dict)
        A dictionary that can be dumped as json

    """
    patterns = {
        int: r'(\d+)',
        float: r'([\+\-]?\d+\.?\d*(?:[eE][\+\-]?\d+)?)',
        str: r'(.*)\s*$'
    }
    vx = py.make_list(vx, 3)

    coordinates = []
    groups = []
    with open(fname) as f:
        f.readline()  # skip first line
        i = 0
        for line in f:
            i += 1
            line = line.rstrip()
            print(f'{i} | {line}', end='\r')
            match = re.search(f'(?P<group>{patterns[int]})\s+'
                              f'(?P<x>{patterns[float]})\s+'
                              f'(?P<y>{patterns[float]})\s+'
                              f'(?P<z>{patterns[float]})\s+'
                              f'(?P<file>{patterns[float]})\s*', line)
            if not match:
                continue
            groups.append(int(match.group('group')))
            coordinates.append([
                float(match.group('x')),
                float(match.group('y')),
                float(match.group('z')),
            ])
    print('')

    pointset = dict()
    pointset['data_type'] = 'fs_pointset'
    pointset['points'] = list()
    for (x, y, z), g in zip(coordinates, groups):
        pointset['points'].append(dict())
        pointset['points'][-1]['coordinates'] = dict(x=+x * 1e-3 / vx[0],
                                                     y=-y * 1e-3 / vx[1],
                                                     z=-z * 1e-3 / vx[2])
        pointset['points'][-1]['legacy_stat'] = 1
        pointset['points'][-1]['statistics'] = dict(group=g)
    pointset['vox2ras'] = 'scanner_ras'  # ?

    if not split_groups:
        return pointset

    all_groups = set(groups)
    subsets = []
    for g in all_groups:
        subset = dict()
        subset['data_type'] = 'fs_pointset'
        subset['points'] = [p for p in pointset['points'] if
                            p['statistics']['group'] == g]
        subset['vox2ras'] = 'scanner_ras'  # ?
        subsets.append(subset)

    return subsets


def _gather_points(pointset, dtype=None, device=None):
    points = []
    for point in pointset['points']:
        points.append([point['coordinates']['x'],
                       point['coordinates']['y'],
                       point['coordinates']['z']])
    points = torch.as_tensor(points, dtype=dtype, device=device)
    return points


def _update_points(pointset, points):
    pointset = copy.deepcopy(pointset)
    for p, point in zip(points, pointset['points']):
        point['coordinates']['x'] = p[0].item()
        point['coordinates']['y'] = p[1].item()
        point['coordinates']['z'] = p[2].item()
    return pointset


def transform_pointset_affine(pointset, affine):
    points = _gather_points(pointset, dtype=affine.dtype, device=affine.device)
    points = spatial.affine_matvec(affine, points)
    pointset = _update_points(pointset, points)
    return pointset


def transform_pointset_dense(points, grid, type='grid', bound='dct2'):
    """Transform a pointset

    Points must already be expressed in "grid voxels" coordinates.

    Parameters
    ----------
    points : (n, dim) tensor
        Set of coordinates, in voxel space
    grid : (*spatial, dim) tensor
        Dense transformation or displacement grid, in voxel space
    type : {'grid', 'disp'}, defualt='grid'
        Transformation or displacement
    bound : str, default='dct2'
        Boundary conditions for out-of-bounds data

    Returns
    -------
    points : (n, dim) tensor
        Transformed coordinates

    """

    dim = grid.shape[-1]
    points = utils.unsqueeze(points, 0, dim)
    grid = utils.movedim(grid, -1, 0)[None]
    delta = spatial.grid_pull(grid, points, bound=bound, extrapolate=True)
    delta = utils.movedim(delta, 1, -1)
    if type == 'disp':
        points = points + delta
    else:
        points = delta
    points = utils.squeeze(points, -2, dim-1).squeeze(0)
    return points


def read_asc(fname, vx=1):
    """Read a MicroBrightField asc file that contains vertices of
    closed regions of interest.

    Parameters
    ----------
    fname : str
        Path to the file
    vx : sequence[float], default=1
        Voxel size

    Returns
    -------
    asc : dict
        Dictionary with parsed ROIs.
        Voxel coordinates are returned

    """

    # NOTE: 27 May 2021
    #   All z coordinates seem to fall between two integers
    #   (say 174.5 instead of 174), so I guess there is a half voxel shift
    #   somewhere. Maybe (0, 0, 0) is the corner of the first voxel instead
    #   of the center? I am now making this assumption and tweaking the code
    #   to add 0.5 in each direction when converting um to vox.

    vx = py.make_list(vx, 3)
    vx = [v*1e3 for v in vx]

    patterns = {
        int: r'\d+',
        float: r'[\+\-]?\d+\.?\d*(?:[eE][\+\-]?\d+)?',
        str: r'\w+'
    }

    def strip_line(line):
        return line.split(';')[0].strip()

    def is_closing(line):
        closed = ')' in line
        if closed:
            line = line[:line.index(')')].rstrip()
        return line, closed

    def parse_section(line):
        p = f'(?P<key>{patterns[str]})\s+"(?P<name>[^"]*)"\s' \
            f'+(?P<z>{patterns[float]})\s+' \
            f'(?P<y>{patterns[float]})\s+' \
            f'(?P<x>{patterns[float]})\s*'
        match = re.match(p, line)
        if match:
            key = match.group('key')
            val = dict(name=match.group('name'),
                       x=float(match.group('x'))/vx[0],
                       y=float(match.group('y'))/vx[1],
                       z=-float(match.group('z'))/vx[2])
            return key, val
        else:
            return None, None

    def parse_point(line):
        p = f'(?P<x>{patterns[float]})\s+' \
            f'(?P<y>{patterns[float]})\s+' \
            f'(?P<z>{patterns[float]})\s+' \
            f'(?P<unknown>{patterns[float]})\s+' \
            f'(?P<section>{patterns[str]})\s*'
        match = re.match(p, line)
        if match:
            val = dict(
                x=float(match.group('x'))/vx[0],
                y=-float(match.group('y'))/vx[1],
                z=-float(match.group('z'))/vx[2],
                unknown=float(match.group('unknown')),
                section=str(match.group('section')),
            )
            return val
        return None

    asc = dict(regions=dict())
    with open(fname) as f:

        while True:
            line = f.readline()
            if not line:
                # end of file
                break
            line = strip_line(line)
            if line.startswith('(Description'):
                line, closed = is_closing(line[12:].lstrip())
                description = line
                while True:
                    if closed:
                        break
                    line, closed = is_closing(strip_line(f.readline()))
                    description += line + ''
                asc['description'] = description.strip()
            elif line.startswith('(Sections'):
                sections = dict()
                line, closed = is_closing(line[9:].lstrip())
                if line:
                    key, val = parse_section(line)
                    if key:
                        sections[key] = val
                while True:
                    if closed:
                        break
                    line, closed = is_closing(strip_line(f.readline()))
                    if line:
                        key, val = parse_section(line)
                        if key:
                            sections[key] = val
                asc['sections'] = sections
            elif line.startswith('("'):
                line = line[2:]
                last = line.index('"')
                name = line[:last]
                shape = dict(points=[])
                while True:
                    line = strip_line(f.readline())
                    if ')' in line and '(' not in line:
                        break
                    line = line[line.index('(')+1:line.index(')')].strip()
                    if not line.startswith(tuple([str(n) for n in range(10)] + ['-'])):
                        continue
                    value = parse_point(line)
                    if value:
                        shape['points'].append(value)
                if name not in asc['regions']:
                    asc['regions'][name] = []
                asc['regions'][name].append(shape)
            elif line.startswith('('):
                line, closed = is_closing(line[1:].lstrip())
                while True:
                    if closed:
                        break
                    line, closed = is_closing(strip_line(f.readline()))
    return asc


def voxelize_rois(rois, shape, roi_to_vox=None, device=None):
    """Create a volume of labels from a parametric ROI.

    Parameters
    ----------
    rois : dict
        Object returned by `read_asc`
    shape : sequence[int]
    roi_to_vox : (d+1, d+1) tensor

    Returns
    -------
    roi : (*shape) tensor[int]
    names : list[str]

    """
    out = torch.empty(shape, dtype=torch.long)
    grid = spatial.identity_grid(shape[:2], device=device)
    if roi_to_vox is not None:
        roi_to_vox = roi_to_vox.to(device=device)

    names = list(rois['regions'].keys())

    for l, (name, shapes) in enumerate(rois['regions'].items()):
        print(name)
        label = l + 1
        for i, shape in enumerate(shapes):
            print(i+1, '/', len(shapes), end='\r')
            vertices = [[p['x'], p['y'], p['z']] for p in shape['points']]
            vertices = torch.as_tensor(vertices, device=device)
            if roi_to_vox is not None:
                vertices = spatial.affine_matvec(roi_to_vox, vertices)
            z = math.round(vertices[0, 2]).int().item()
            vertices = vertices[:, :2]
            faces = [(i, i+1 if i+1 < len(vertices) else 0)
                     for i in range(len(vertices))]

            mask = is_inside(grid, vertices, faces).cpu()
            out[..., z][mask] = label
        print('')

    return out, names


def roi_closing(label, radius=10, dim=None):
    """Performs a multi-label morphological closing.

    Parameters
    ----------
    label : (..., *spatial) tensor[int]
        Volume of labels.
    radius : float, default=1
        Radius of the structuring element (in voxels)
    dim : int, default=label.dim()
        Number of spatial dimensions

    Returns
    -------
    closed_label : tensor[int]

    """
    from scipy.ndimage import distance_transform_edt, binary_closing

    dim = dim or label.dim()
    closest_label = torch.zeros_like(label)
    closest_dist = label.new_full(label.shape, float('inf'), dtype=torch.float)
    dist = torch.empty_like(closest_dist)

    for l in label.unique():
        if l == 0:
            continue
        if label.dim() == dim:
            dist = torch.as_tensor(distance_transform_edt(label != l))
        elif label.dim() == dim + 1:
            for z in range(len(dist)):
                dist[z] = torch.as_tensor(distance_transform_edt(label[z] != l))
        else:
            raise NotImplementedError
        closest_label[dist < closest_dist] = l
        closest_dist = torch.min(closest_dist, dist)

    struct = spatial.identity_grid([2*radius+1]*dim).sub_(radius)
    struct = struct.square().sum(-1).sqrt() <= radius
    struct = utils.unsqueeze(struct, 0, label.dim() - dim)
    mask = binary_closing(label > 0, struct)
    mask = torch.as_tensor(mask).bitwise_not_()
    closest_label[mask] = 0

    return closest_label



# NOTES:
#
# rois = read_asc(aname, vx=3.3e-3)
# roi_to_vox = ni.spatial.affine_matmul(ni.spatial.affine_inv(f9.affine), f3.affine)
# labels, names = voxelize_rois(rois, f9.shape[:2], roi_to_vox)
# closed_labels = roi_closing(labels, 10)


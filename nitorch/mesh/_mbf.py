import re
import torch
from nitorch.core import py
from nitorch import spatial
from ._inclusion import is_inside


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


def voxelize_rois(rois, shape, roi_to_vox=None):
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
    single_plane = len(shape) == 2
    out = torch.empty(shape, dtype=torch.long)
    grid = spatial.identity_grid(shape)

    names = list(rois['regions'].keys())

    for l, (name, shapes) in enumerate(rois['regions'].items()):
        label = l + 1
        for shape in shapes:
            vertices = [[p['x'], p['y'], p['z']] for p in shape['points']]
            vertices = torch.as_tensor(vertices)
            if roi_to_vox is not None:
                vertices = spatial.affine_matvec(roi_to_vox, vertices)
            if single_plane:
                vertices = vertices[:, :2]
            faces = [(i, i+1 if i+1 < len(vertices) else 0)
                     for i in range(len(vertices))]
            mask = is_inside(grid, vertices, faces)
            out[mask] = label

    return out, names


def roi_closing(label, radius=1):
    from scipy.ndimage import distance_transform_edt, binary_closing

    closest_label = torch.zeros_like(label)
    closest_dist = label.new_full(label.shape, float('inf'), dtype=torch.float)

    for l in label.unique():
        if l == 0:
            continue
        dist = torch.as_tensor(distance_transform_edt(label != l))
        closest_label[dist < closest_dist] = l
        closest_dist = torch.min(closest_dist, dist)

    struct = spatial.identity_grid([2*radius+1]*label.dim()).sub_(radius)
    struct = struct.square().sum(-1).sqrt() <= radius
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


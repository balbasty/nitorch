import torch
from nitorch.core import utils, py, constants, linalg
import math as pymath
from typing import List


def is_inside(points, vertices, faces=None):
    """Test if a point is inside a (linear) polygon/surface.

    The polygon or surface *must* be closed.

    Parameters
    ----------
    points : (..., dim) tensor
        Coordinates of points to test
    vertices : (nv, dim) tensor
        Vertex coordinates
    faces : (nf, dim) tensor[int]
        Faces are encoded by the indices of its vertices.
        By default, assume that vertices are ordered and define a closed curve

    Returns
    -------
    check : (...) tensor[bool]

    """
    # This function uses a ray-tracing technique:
    #
    #   A half-line is started in each point. If it crosses an even
    #   number of faces, it is inside the shape. If it crosses an even
    #   number of faces, it is not.
    #
    #   In practice, we loop through faces (as we expect there are much
    #   less vertices than voxels) and compute intersection points between
    #   all lines and each face in a batched fashion. We only want to
    #   send these rays in one direction, so we keep aside points whose
    #   intersection have a positive coordinate along the ray.

    points = torch.as_tensor(points)
    vertices = torch.as_tensor(vertices)
    if faces is None:
        faces = [(i, i+1) for i in range(len(vertices)-1)]
        faces += [(len(vertices)-1, 0)]
        faces = utils.as_tensor(faces, dtype=torch.long)

    points, vertices = utils.to_max_dtype(points, vertices)
    points, vertices, faces = utils.to_max_device(points, vertices, faces)
    backend = utils.backend(points)
    batch = points.shape[:-1]
    dim = points.shape[-1]
    eps = constants.eps(points.dtype)
    cross = points.new_zeros(batch, dtype=torch.long)

    ray = torch.randn(dim, **backend)

    for face in faces:
        face = vertices[face]

        # compute normal vector
        origin = face[0]
        if dim == 3:
            u = face[1] - face[0]
            v = face[2] - face[0]
            norm = torch.stack([u[1] * v[2] - u[2] * v[1],
                                u[2] * v[0] - u[0] * v[2],
                                u[0] * v[1] - u[1] * v[0]])
        else:
            assert dim == 2
            u = face[1] - face[0]
            norm = torch.stack([-u[1], u[0]])

        # check co-linearity between face and ray
        colinear = linalg.dot(ray, norm).abs() / (ray.norm()*norm.norm()) < eps
        if colinear:
            continue

        # compute intersection between ray and plane
        #   plane: <norm, x - origin> = 0
        #   line: x = p + t*u
        #   => <norm, p + t*u - origin> = 0
        intersection = linalg.dot(norm, points - origin)
        intersection /= linalg.dot(norm, ray)
        halfmask = intersection >= 0  # we only want to shoot in one direction
        intersection = intersection[halfmask]
        halfpoints = points[halfmask]
        intersection = intersection[..., None] * (-ray)
        intersection += halfpoints

        # check if the intersection is inside the face
        #   first, we project it onto a frame of dimension `dim-1`
        #   defined by (origin, (u, v))
        intersection -= origin
        if dim == 3:
            interu = linalg.dot(intersection, u)
            interv = linalg.dot(intersection, v)
            intersection = (interu >= 0) & (interv > 0) & (interu + interv < 1)
        else:
            intersection = linalg.dot(intersection, u)
            intersection /= u.norm().square_()
            intersection = (intersection >= 0) & (intersection < 1)

        cross[halfmask] += intersection

    # check that the number of crossings is even
    cross = cross.bitwise_and_(1).bool()
    return cross


# ================================================================
# Compute mask of interior points by ray tracing
# (since we must loop about vertices, I use torch.jit to (try to)
# make it a bit faster)
# ================================================================


@torch.jit.script
def _is_valid_root(t, px, p1: float, p2: float, p3: float):
    bx = (1 - t).square() * p1 + 2 * t * (1 - t) * p2 + t.square() * p3
    return (0 <= t) & (t <= 1) & (px < bx)


@torch.jit.script
def _is_valid_root0(t: float, px, p1: float, p2: float, p3: float):
    bx = (1 - t) ** 2 * p1 + 2 * t * (1 - t) * p2 + t * t * p3
    if (t < 0) or (t > 1):
        return torch.zeros_like(px, dtype=torch.bool)
    else:
        return px < bx


@torch.jit.script
def quad_count(p1: List[float], p2: List[float], p3: List[float], points):
    """Count intersections (0, 1 or 2) between one quadratic curve and each ray+"""

    if abs(p2[1] - 0.5 * (p1[1] + p3[1])) < 1e-5:
        t = (p1[1] - points[..., 1]) / (p1[1] - p3[1])
        mask1 = _is_valid_root(t, points[..., 0], p1[0], p2[0], p3[0]).int()
        return mask1

    a = (p3[1] + p1[1] - 2 * p2[1])
    b = p2[1] - p1[1]
    c = p1[1] - points[..., 1]
    delta = b ** 2 - a * c

    t0 = -b / a
    t1 = (-b - delta.sqrt()) / a
    t2 = (-b + delta.sqrt()) / a
    t0 = _is_valid_root0(t0, points[..., 0], p1[0], p2[0], p3[0]).int()
    t1 = _is_valid_root(t1, points[..., 0], p1[0], p2[0], p3[0]).int()
    t2 = _is_valid_root(t2, points[..., 0], p1[0], p2[0], p3[0]).int()

    mask1 = (t1 + t2).masked_fill(delta <= 0, 0)
    mask1 += (delta == 0) * t0

    return mask1


@torch.jit.script
def is_inside_quad_jit(points, vertices: List[List[float]]):
    mask = points.new_zeros(points.shape[:-1], dtype=torch.int)
    npoints = len(vertices) // 2

    for n in range(npoints - 1):
        p1, p2, p3 = vertices[n * 2:n * 2 + 3]
        mask += quad_count(p1, p2, p3, points)

    p1, p2 = vertices[-2:]
    p3 = vertices[0]
    mask += quad_count(p1, p2, p3, points)

    return mask.bitwise_and(1) > 0


def is_inside_quad(points, vertices):
    """Interior mask of a closed curve encoded by quadratic Bezier splines.

    Parameters
    ----------
    points : (*batch, dim) tensor
        Coordinates of points to test
    vertices : (nv, dim) tensor
        Vertex and control points coordinates
        Sorted as (v1, c1, v2, c2, ...)

    Returns
    -------
    mask : (*batch) tensor[bool]

    References
    ----------
    ..[1]   "Random-Access Rendering of General Vector Graphics"
            Diego Nehab and Hugues Hoppe
            ACM Transactions on Graphics (2008)
            https://hhoppe.com/ravg.pdf

    """
    if torch.is_tensor(vertices):
        vertices = vertices.tolist()
    else:
        vertices = [[x, y] for x, y in vertices]
    mask = is_inside_quad_jit(points, vertices)
    return mask


# ================================================================
# Approximate cubic Bezier splines using several quadratic splines
# ================================================================

def cubic_to_quadratic(cubics):
    """
    Convert closed-path of cubic Bezier splines
    into closed path of quadratic Bezier splines
    """

    # https://stackoverflow.com/questions/2009160

    def c2q(a, b, c, d):
        """down-elevated cubic to quadratic"""
        ax, ay = a
        bx, by = b
        cx, cy = c
        dx, dy = d
        x = -0.25 * (ax + dx) + 0.75 * (bx + cx)
        y = -0.25 * (ay + dy) + 0.75 * (by + cy)
        return a, (x, y), d

    def c2qs(a, b, c, d):
        """Cubic to quadratics"""
        cubic = (a, b, c, d)
        quadratics = []
        t = [*critical_points(*cubic)]
        tprev = 0
        for t1 in t:
            if t1 < 0.01 or (1 - t1) < 0.01:
                continue
            t1, tprev = (t1 - tprev) / (1 - tprev), t1
            if t1 < 0.01 or (1 - t1) < 0.01:
                continue
            (p10x, p10y), (p20x, p20y), (p30x, p30y), (p40x, p40y) = cubic
            p11x, p11y = p10x + (p20x - p10x) * t1, p10y + (p20y - p10y) * t1
            p21x, p21y = p20x + (p30x - p20x) * t1, p20y + (p30y - p20y) * t1
            p31x, p31y = p30x + (p40x - p30x) * t1, p30y + (p40y - p30y) * t1
            p12x, p12y = p11x + (p21x - p11x) * t1, p11y + (p21y - p11y) * t1
            p22x, p22y = p21x + (p31x - p21x) * t1, p21y + (p31y - p21y) * t1
            p13x, p13y = p12x + (p22x - p12x) * t1, p12y + (p22y - p12y) * t1

            cubic = (p10x, p10y), (p11x, p11y), (p12x, p12y), (p13x, p13y)
            quadratics.extend(c2q(*cubic)[:-1])
            cubic = (p13x, p13y), (p22x, p22y), (p31x, p31y), (p40x, p40y)
        quadratics.extend(c2q(*cubic)[:-1])
        return quadratics

    cubics = list(cubics)
    npoints = len(cubics) // 3

    quadratics = []
    for n in range(npoints - 1):
        cubic = cubics[3 * n:3 * n + 4]
        quadratics.extend(c2qs(*cubic))
    cubic = cubics[-3:] + [cubics[0]]
    quadratics.extend(c2qs(*cubic))
    return quadratics


def critical_points(p1, c1, c2, p2):
    """Compute critical point of a 2D cubic Bezier curve"""

    p1x, p1y = p1
    c1x, c1y = c1
    c2x, c2y = c2
    p2x, p2y = p2
    t = []

    a = (c2x - 2 * c1x + p1x) - (p2x - 2 * c2x + c1x)
    if abs(a) > 1e-5:
        b = 2 * (c1x - p1x) - 2 * (c2x - c1x)
        c = p1x - c1x
        delta = b * b - 4 * a * c
        t1 = t2 = None
        if delta == 0:
            t1 = -b / (2 * a)
        elif delta > 0:
            delta = pymath.sqrt(b * b - 4 * a * c)
            t1 = (-b + delta) / (2 * a)
            t2 = (-b - delta) / (2 * a)
        if t1 is not None and 0 <= t1 <= 1 and t1 not in t:
            t.append(t1)
        if t2 is not None and 0 <= t2 <= 1 and t2 not in t:
            t.append(t2)

    a = (c2y - 2 * c1y + p1y) - (p2y - 2 * c2y + c1y)
    if abs(a) > 1e-5:
        b = 2 * (c1y - p1y) - 2 * (c2y - c1y)
        c = p1y - c1y
        delta = b * b - 4 * a * c
        t1 = t2 = None
        if delta == 0:
            t1 = -b / (2 * a)
        elif delta > 0:
            delta = pymath.sqrt(b * b - 4 * a * c)
            t1 = (-b + delta) / (2 * a)
            t2 = (-b - delta) / (2 * a)
        if t1 is not None and 0 <= t1 <= 1 and t1 not in t:
            t.append(t1)
        if t2 is not None and 0 <= t2 <= 1 and t2 not in t:
            t.append(t2)

    t += inflection_points((p1x, p1y), (c1x, c1y), (c2x, c2y), (p2x, p2y))

    t = list(sorted(t))
    return t or [0.5]


def inflection_points(p1, p2, p3, p4):
    """Compute inflection point of a 2D cubic Bezier curve"""

    p1x, p1y = p1
    p2x, p2y = p2
    p3x, p3y = p3
    p4x, p4y = p4

    ax = -p1x + 3 * p2x - 3 * p3x + p4x
    bx = 3 * p1x - 6 * p2x + 3 * p3x
    cx = -3 * p1x + 3 * p2x

    ay = -p1y + 3 * p2y - 3 * p3y + p4y
    by = 3 * p1y - 6 * p2y + 3 * p3y
    cy = -3 * p1y + 3 * p2y
    a = 3 * (ay * bx - ax * by)
    b = 3 * (ay * cx - ax * cy)
    c = by * cx - bx * cy
    r2 = b * b - 4 * a * c

    if r2 >= 0 and a != 0:
        r = pymath.sqrt(r2)
        t = list(sorted([(-b + r) / (2 * a), (-b - r) / (2 * a)]))
        if 0 <= t[0] <= t[1] <= 1:
            if t[1] - t[0] < 1e-5:
                return [t[0]]
            return t
        elif 0 <= t[0] <= 1:
            return [t[0]]
        elif 0 <= t[1] <= 1:
            return [t[1]]
        else:
            return []
    return []


# =============================
# Utility to plot bezier curves
# =============================

def plot_quad(vertices, show=True):
    import matplotlib.pyplot as plt

    vertices = torch.as_tensor(vertices)
    npoints = len(vertices) // 2

    xx = []
    yy = []
    for n in range(npoints - 1):
        p1, p2, p3 = vertices[n * 2:n * 2 + 3].unbind(0)
        t = torch.linspace(0, 1, 128)
        x = p1[0] * (1 - t).square() + 2 * p2[0] * t * (1 - t) + p3[
            0] * t.square()
        y = p1[1] * (1 - t).square() + 2 * p2[1] * t * (1 - t) + p3[
            1] * t.square()
        xx += [x]
        yy += [y]

    p1, p2 = vertices[-2:].unbind(0)
    p3 = vertices[0]
    x = p1[0] * (1 - t).square() + 2 * p2[0] * t * (1 - t) + p3[0] * t.square()
    y = p1[1] * (1 - t).square() + 2 * p2[1] * t * (1 - t) + p3[1] * t.square()
    xx += [x]
    yy += [y]

    xx = torch.cat(xx)
    yy = torch.cat(yy)
    plt.plot(xx, yy)
    if show:
        plt.show()


def plot_cubic(vertices, show=True):
    import matplotlib.pyplot as plt

    vertices = torch.as_tensor(vertices)
    npoints = len(vertices) // 3

    xx = []
    yy = []
    for n in range(npoints - 1):
        p1, p2, p3, p4 = vertices[n * 3:n * 3 + 4].unbind(0)
        t = torch.linspace(0, 1, 128)
        x = p1[0] * (1 - t).pow(3) + 3 * p2[0] * t * (1 - t).square() \
          + 3 * p3[0] * t.square() * (1 - t) + p4[0] * t.pow(3)
        y = p1[1] * (1 - t).pow(3) + 3 * p2[1] * t * (1 - t).square() \
          + 3 * p3[1] * t.square() * (1 - t) + p4[1] * t.pow(3)
        xx += [x]
        yy += [y]

    p1, p2, p3 = vertices[-3:].unbind(0)
    p4 = vertices[0]
    x = p1[0] * (1 - t).pow(3) + 3 * p2[0] * t * (1 - t).square() \
      + 3 * p3[0] * t.square() * (1 - t) + p4[0] * t.pow(3)
    y = p1[1] * (1 - t).pow(3) + 3 * p2[1] * t * (1 - t).square() \
      + 3 * p3[1] * t.square() * (1 - t) + p4[1] * t.pow(3)
    xx += [x]
    yy += [y]

    xx = torch.cat(xx)
    yy = torch.cat(yy)
    plt.plot(xx, yy)
    if show:
        plt.show()
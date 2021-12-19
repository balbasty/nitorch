import torch
from nitorch.core import utils, py, constants, linalg


def is_inside(points, vertices, faces=None):
    """Test if a point is inside a polygon/surface.

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
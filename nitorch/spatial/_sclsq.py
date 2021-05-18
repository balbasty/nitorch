"""Integrate stationary velocity fields."""

from ._grid import grid_pull, identity_grid
from ._regularisers import regulariser_grid
from ._shoot import greens, greens_apply
from nitorch.core import utils, py, linalg
import torch

__all__ = ['exp']


def exp(vel, inverse=False, steps=8, interpolation='linear', bound='dft',
        displacement=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    inverse : bool, default=False
        Generate the inverse transformation instead of the forward.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    displacement : bool, default=False
        Return a displacement field rather than a transformation field

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated tranformation

    """

    vel = -vel if inverse else vel.clone()

    # Precompute identity + aliases
    dim = vel.shape[-1]
    spatial = vel.shape[-1-dim:-1]
    id = identity_grid(spatial, **utils.backend(vel))
    opt = {'interpolation': interpolation, 'bound': bound}

    if vel.requires_grad:
        iadd = lambda x, y: x.add(y)
    else:
        iadd = lambda x, y: x.add_(y)

    vel /= (2**steps)
    for i in range(steps):
        vel = iadd(vel, _pull_vel(vel, id + vel, **opt))

    if not displacement:
        vel += id
    return vel


def _pull_vel(vel, grid, *args, **kwargs):
    """Interpolate a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial, ndim) tensor
        Velocity
    grid : ([batch], *spatial, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial, ndim) tensor
        Velocity

    """
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    dim = vel.shape[-1]
    vel = utils.movedim(vel, -1, -dim-1)
    vel_no_batch = vel.dim() == dim + 1
    grid_no_batch = grid.dim() == dim + 1
    if vel_no_batch:
        vel = vel[None]
    if grid_no_batch:
        grid = grid[None]
    vel = grid_pull(vel, grid, *args, **kwargs)
    vel = utils.movedim(vel, -dim-1, -1)
    if vel_no_batch and grid_no_batch:
        vel = vel[0]
    return vel


def _build_matrix(observed, latent):
    """

    Parameters
    ----------
    observed : list[(int, int)]
    latent : list[(int, int)]

    Returns
    -------
    mat : list[{(int, int): float}]

    """
    def explore(src, dst, path, remaining):
        remaining = list(remaining)
        explored = []
        while remaining:
            lat = remaining.pop(0)
            if src in lat:
                path0 = path
                src0 = src
                path = {**path0, lat: 1 if lat[0] == src else -1}
                src = lat[0] if lat[0] != src else lat[1]
                if dst in lat:
                    return path
                path = explore(src, dst, path, explored + remaining)
                if path and dst in list(path.keys())[-1]:
                    return path
                path = path0
                src = src0
            explored.append(lat)
        return []

    mat = []
    for src, dst in observed:
        path = explore(src, dst, {}, latent)
        if not path:
            raise ValueError(f'Could not find path from {src} to {dst}')
        mat.append(path)
    return mat


def _svf_graph_refinement(velocities, latent=None, distances=None, modalities=None,
                          likelihood='l2', penalty=None, voxel_size=1):
    """Refine a graph of velocity fields

    References
    ----------
    ..[1] "Robust joint registration of multiple stains and MRI for
           multimodal 3D histology reconstruction: Application to the
           Allen human brain atlas"
          Adrià Casamitjana, Marco Lorenzi, Sebastiano Ferraris,
          Loc Peter, Marc Modat, Allison Stevens, Bruce Fischl,
          Tom Vercauteren, Juan Eugenio Iglesias
          https://arxiv.org/abs/2104.14873

    Parameters
    ----------
    velocities : list of {((int, int), tensor)}
        List of (key, velocites) pairs.
        Keys are tuples of two indices that encode the nodes connected
        by each velocity and their direction:
        `(3, 5)` maps from node 3 to node 5 (and its negative maps from
        node 5 to node 3).
    latent : list of (int, int)
        List of latent velocities to estimate. It should form an
        acyclic graph that contains all the nodes.
    distances : dict of {(int, int): float}
        Dictionary of distances. The distance describes how "far" from
        each other two nodes are.
    modalities : list of (int, int), optional
        Each element encodes the pair of modalities linked by an
        observed velocity.
    likelihood : {'l1', 'l2'}, default='l2'
        Model used to define the likelihood of a measured SVF conditioned
        on a latent SVF.
            'l2' : use Gaussian noise (in feature space)
            'l1': use Laplace noise (in feature space)
    penalty : dict, optional
        If provided, transform velocities into momenta with respect to some
        energy. Possible keywords are 'absolute', 'membrane', 'bending' and 'lame'.

    Returns
    -------
    latent : dict of {tuple[int]: tensor}
        Dictionary of latent velocities.

    """
    def pair(*a):
        """Alias for unordered pairs"""
        return frozenset(a)

    def offdiag(keys):
        """Return indices of off-diagonal elements of a symmetric matrix.
           Indices are hashable keys. """
        keys = list(keys)
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                yield pair(keys[i], keys[j])

    def upper(keys):
        """Return indices of all elements of a symmetric matrix
           Indices are hashable keys."""
        keys = list(keys)
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                yield pair(keys[i], keys[j])

    def values(x):
        return [e[1] for e in x]

    def keys(x):
        return [e[0] for e in x]

    if isinstance(velocities, dict):
        velocities = [(k, v) for k, v in velocities.items()]
    velocities = list(velocities)

    # velocity to momentum
    backend = utils.backend(values(velocities)[0])
    if penalty:
        shape = values(velocities)[0].shape[:-1]
        penalty['voxel_size'] = voxel_size
        kernel = greens(shape, **penalty, **backend)
        momenta = []
        for key, vel in velocities:
            momenta.append((key, regulariser_grid(vel, **penalty)))
    else:
        momenta = velocities

    # get all nodes
    nodes = py.flatten(keys(velocities)) + py.flatten(list(latent or []))
    nodes = list(set(nodes))

    # choose arbitrary acyclic graph that covers all nodes if not provided
    latent = latent or [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
    latent = {key: None for key in latent}

    # same distance everywhere if not provided
    distances = distances or {key: 1 for key in offdiag(nodes)}
    distances = {pair(*key): value for key, value in distances.items()}

    # same modality if not provided
    modalities = modalities or [(1, 1)] * len(velocities)
    all_modalities = list(set(py.flatten(modalities)))

    # initialize variances
    sigma_mod = {key: 1 for key in offdiag(all_modalities)}
    sigma_dist = {key: 1 for key in upper(all_modalities)}
    sigma_comb = [sigma_mod.get(pair(*modalities[i]), 0) +
                  sigma_dist[pair(*modalities[i])] *
                  distances[pair(a, b)]
                  for i, (a, b) in enumerate(keys(velocities))]

    # compute W
    connections = _build_matrix(keys(velocities), latent.keys())

    for n_iter_em in range(100):

        # update momenta
        den = sum(len(connections[i]) / sigma_comb[i]
                  for i in range(len(momenta)))
        for lat in latent.keys():
            value = 0
            for i, (obs, mom) in enumerate(momenta):
                if lat in connections[i]:
                    value += connections[i][lat] * mom
            value /= den
            latent[lat] = value
        del value

        # compute errors
        errors = torch.zeros(len(momenta), **backend)
        for i in range(len(momenta)):
            vel = values(velocities)[i]
            mom = values(momenta)[i]
            recon = 0
            for lat, weight in connections[i].items():
                recon += weight * latent[lat]
            err = (vel*mom).sum()
            err -= 2 * (vel*recon).sum()
            if penalty:
                err += (recon * greens_apply(recon, kernel, voxel_size)).sum()
            else:
                err += recon.square().sum()
            err /= vel.numel()
            errors[i] = err

        # update variances
        mat = torch.zeros(len(errors), len(sigma_mod) + len(sigma_dist), **backend)
        for i, obs in enumerate(keys(momenta)):
            mod = pair(modalities[i][0], modalities[i][1])
            for j, key in enumerate(sigma_mod.keys()):
                if key == mod:
                    mat[i, j] = 1
            for j, key in enumerate(sigma_dist.keys()):
                if key == mod:
                    mat[i, j + len(sigma_mod)] = distances[pair(*obs)]
        prm = linalg.lmdiv(mat, errors[:, None], method='pinv')[:, 0]

        for j, key in enumerate(sigma_mod.keys()):
            sigma_mod[key] = prm[j]
        for j, key in enumerate(sigma_dist.keys()):
            sigma_dist[key] = prm[j+len(sigma_mod)]

    if penalty:
        for key, val in latent.items():
            latent[key] = greens_apply(latent[key], kernel, voxel_size)

    return latent, sigma_mod, sigma_dist

"""
This is an attempt at implementing registration refinement based on a
graph os stationary velocity fields.

References
----------
..[1] "Robust joint registration of multiple stains and MRI for
       multimodal 3D histology reconstruction: Application to the
       Allen human brain atlas"
      Adrià Casamitjana, Marco Lorenzi, Sebastiano Ferraris,
      Loic Peter, Marc Modat, Allison Stevens, Bruce Fischl,
      Tom Vercauteren, Juan Eugenio Iglesias
      https://arxiv.org/abs/2104.14873
"""

import torch
from nitorch.core import py, utils, linalg
from nitorch import spatial
from . import _prototype_svf, losses, phantoms


def _build_matrix(observed, latent):
    """Build graph matrix

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


def register_pairs(images=None, register=None, loss='mse'):

    if not register:
        prm = dict(lame=(0.05, 1e-4), lam=0.1, plot=True, max_iter=10)
        register = lambda x, y: _prototype_svf.register(x, y, loss=loss, **prm)

    # If no inputs provided: demo "squircles"
    if images is None:
        images = phantoms.demo_atlas(batch=8)

    # pairwise registrations
    vels = []
    for i in range(len(images) - 1):
        vel = register(images[i], images[i+1])
        vels.append(((i+1, i+2), vel))
    vel = register(images[-1], images[0])
    vels.append(vel)

    return vels


def atlas(images=None, register=None, vels=None, loss='mse'):
    """Generate an atlas from pairwise-registered images

    Parameters
    ----------
    images : sequence of (K, *spatial) tensor
        Input images to register
    register : callable(tensor, tensor), default=svf.register
        Function used to perform pairwise-registration
    vels : sequence of [*spatial, dim] tensor, optional
        Pre-computed SVFs.
    loss : {'mse', 'cat'}, default='mse'

    Returns
    -------
    template : (K, *spatial) tensor
        Inferred template.

    """

    if not register:
        register = lambda x, y: _prototype_svf.register(x, y, loss=loss)

    # If no inputs provided: demo "squircles"
    if images is None:
        images = phantoms.demo_atlas(batch=8)

    if vels is None:
        vels = register_pairs(images, register, loss)

    # latent velocities
    vels = graph_atlas(vels).neg_()

    # compute template
    template = _prototype_svf.init_template(images, loss, velocities=vels)

    return template


def graph_atlas(velocities, nodes=None, latent=None):
    """Infer template-to-image SVFs from pairwise SVFs..

    References
    ----------
    ..[1] "Robust joint registration of multiple stains and MRI for
           multimodal 3D histology reconstruction: Application to the
           Allen human brain atlas"
          Adrià Casamitjana, Marco Lorenzi, Sebastiano Ferraris,
          Loic Peter, Marc Modat, Allison Stevens, Bruce Fischl,
          Tom Vercauteren, Juan Eugenio Iglesias
          https://arxiv.org/abs/2104.14873

    Parameters
    ----------
    velocities : (n, *spatial, dim) tensor
    nodes : n-sequence of (int, int), default=[(1, 2), (2, 3), ..., (N, 1)]
    latent : k-sequence of (int, int) tensor, default=[(0, 1), ..., (0, N)]

    Returns
    -------
    latent : sequence of (k, *spatial, dim) tensor

    """

    def default_nodes(n):
        nodes = []
        for n in range(1, n):
            nodes += [(n, n+1)]
        nodes += [(n, 1)]
        return nodes

    def default_latent(n):
        nodes = [(0, n) for n in range(1, n+1)]
        return nodes

    velocities = utils.as_tensor(velocities)
    backend = utils.backend(velocities)

    # defaults
    observed_nodes = list(nodes or default_nodes(len(velocities)))
    latent_nodes = list(latent or default_latent(len(velocities)))

    # compute W
    connections = _build_matrix(observed_nodes, latent_nodes)
    coordinates = [(latent_nodes.index(j), i)
                   for i, c in enumerate(connections) for j in c]
    values = [v for c in connections for v in c.values()]
    values = torch.as_tensor(values, dtype=torch.float)
    coordinates = torch.as_tensor(coordinates, dtype=torch.long).T
    w = torch.sparse_coo_tensor(coordinates, values,
                                [len(latent_nodes), len(observed_nodes)],
                                **backend)

    # re-parameterise last velocity to enforce `sum of velocities = 0`
    w = w.to_dense()
    wlast = w[-1:, :]
    w = w[:-1, :]
    w -= wlast

    velocities = utils.movedim(velocities, 0, -1)[..., None]
    latent = w.transpose(-1, -2).pinverse().matmul(velocities)[..., 0]
    latent = utils.movedim(latent, -1, 0)
    latent = torch.cat([latent, -latent.sum(0, keepdim=True)], dim=0)

    return latent


# def graph(velocities, nodes=None, latent=None, distances=None, modalities=None,
#           likelihood='l2', penalty=None, voxel_size=1):
#     """Refine a graph of velocity fields
#
#     References
#     ----------
#     ..[1] "Robust joint registration of multiple stains and MRI for
#            multimodal 3D histology reconstruction: Application to the
#            Allen human brain atlas"
#           Adrià Casamitjana, Marco Lorenzi, Sebastiano Ferraris,
#           Loic Peter, Marc Modat, Allison Stevens, Bruce Fischl,
#           Tom Vercauteren, Juan Eugenio Iglesias
#           https://arxiv.org/abs/2104.14873
#
#     Parameters
#     ----------
#     velocities : sequence of (*spatial, dim) tensor
#     nodes : list of (int, int), default=[(1, 2), (2, 3), ..., (N, 1)]
#         Nodes connected by each velocity. The graph is directed so
#         order of node IDs matters!
#         It should have the same length as `velocities`.
#     latent : list of (int, int), default=[(0, 1), (0, 2), ..., (0, N)]
#         List of latent velocities to estimate. It should form an
#         acyclic graph that contains all the nodes.
#     distances : list[float], default=1
#         Physical distance between each pair of nodes corresponding to
#         each velocity.
#         It should have the same length as `velocities`.
#     modalities : list of (int, int), default=1
#         Modality of each node in each pair of nodes.
#         It should have the same length as `velocities`.
#     likelihood : {'l1', 'l2'}, default='l2'
#         Model used to define the likelihood of a measured SVF conditioned
#         on a latent SVF.
#             'l2' : use Gaussian noise (in feature space)
#             'l1': use Laplace noise (in feature space)
#     penalty : dict, optional
#         If provided, transform velocities into momenta with respect to some
#         energy. Possible keywords are 'absolute', 'membrane', 'bending' and 'lame'.
#
#     Returns
#     -------
#     latent : dict of {tuple[int]: tensor}
#         Dictionary of latent velocities.
#
#     """
#     def pair(*a):
#         """Alias for unordered pairs"""
#         return frozenset(a)
#
#     def offdiag(keys):
#         """Return indices of off-diagonal elements of a symmetric matrix.
#            Indices are hashable keys. """
#         keys = list(keys)
#         for i in range(len(keys)):
#             for j in range(i+1, len(keys)):
#                 yield pair(keys[i], keys[j])
#
#     def upper(keys):
#         """Return indices of all elements of a symmetric matrix
#            Indices are hashable keys."""
#         keys = list(keys)
#         for i in range(len(keys)):
#             for j in range(i, len(keys)):
#                 yield pair(keys[i], keys[j])
#
#     def values(x):
#         return [e[1] for e in x]
#
#     def keys(x):
#         return [e[0] for e in x]
#
#     velocities = torch.as_tensor(velocities)
#     backend = utils.backend(velocities)
#
#     # defaults
#     observed_nodes = list(nodes or default_nodes())
#     latent_nodes = list(latent or default_latent())
#     latent = velocities.new_empty([len(latent), velocities.shape[1:]])
#     all_nodes = list(set([*nodes, *latent]))
#     distances = py.make_list(distances or [1], len(velocities))
#     modalities = py.make_list(modalities or [(1, 1)], len(velocities))
#     all_modalities = list(set(py.flatten(modalities)))
#
#     # # initialize variances
#     # sigma_mod = {key: 1 for key in offdiag(all_modalities)}
#     # sigma_dist = {key: 1 for key in upper(all_modalities)}
#     # sigma_comb = [sigma_mod.get(pair(*modalities[i]), 0) +
#     #               sigma_dist[pair(*modalities[i])] *
#     #               distances[pair(a, b)]
#     #               for i, (a, b) in enumerate(keys(velocities))]
#
#     # compute W
#     connections = _build_matrix(observed_nodes, latent_nodes)
#     coordinates = [(latent_nodes.index(j), i)
#                    for i, c in enumerate(connections) for j in c]
#     values = [v for c in connections for v in c.values()]
#     w = torch.sparse_coo_tensor(coordinates, values,
#                                 [len(latent_nodes), len(observed_nodes)],
#                                 **backend)
#     print(connections)
#
#     for n_iter_em in range(100):
#
#         # update velocities
#         #   1) build linear systemn to invert
#         mat = torch.zeros(len(latent), len(latent))
#         for connection, in connections:
#             for ipath, ival in connection.items():
#                 i = list(latent.keys()).index(ipath)
#                 for jpath, jval in connection.items():
#                     j = list(latent.keys()).index(jpath)
#                     mat[i, j] += ival * jval
#         mat = torch.pinverse(mat)
#
#         for i in range(len(latent)):
#             latent[i] = 0.
#         for connection, vel in zip(connections, velocities):
#             for lat, val in connection.items():
#                 i = list(latent.keys()).index(lat)
#                 latent[i] += val * vel
#             value = 0
#             for i, (obs, mom) in enumerate(momenta):
#                 if lat in connections[i]:
#                     value += connections[i][lat] * mom
#             value /= den
#             latent[lat] = value
#         del value
#
#         # compute errors
#         errors = torch.zeros(len(momenta), **backend)
#         for i in range(len(momenta)):
#             vel = values(velocities)[i]
#             mom = values(momenta)[i]
#             recon = 0
#             for lat, weight in connections[i].items():
#                 recon += weight * latent[lat]
#             err = (vel*mom).sum()
#             err -= 2 * (vel*recon).sum()
#             if penalty:
#                 err += (recon * spatial.greens_apply(recon, kernel, voxel_size)).sum()
#             else:
#                 err += recon.square().sum()
#             err /= vel.numel()
#             errors[i] = err
#         print(sum(errors).item(), end='\r')
#
#         # update variances
#         mat = torch.zeros(len(errors), len(sigma_mod) + len(sigma_dist), **backend)
#         for i, obs in enumerate(keys(momenta)):
#             mod = pair(modalities[i][0], modalities[i][1])
#             for j, key in enumerate(sigma_mod.keys()):
#                 if key == mod:
#                     mat[i, j] = 1
#             for j, key in enumerate(sigma_dist.keys()):
#                 if key == mod:
#                     mat[i, j + len(sigma_mod)] = distances[pair(*obs)]
#         prm = linalg.lmdiv(mat, errors[:, None], method='pinv')[:, 0]
#
#         for j, key in enumerate(sigma_mod.keys()):
#             sigma_mod[key] = prm[j]
#         for j, key in enumerate(sigma_dist.keys()):
#             sigma_dist[key] = prm[j+len(sigma_mod)]
#
#     if penalty:
#         for key, val in latent.items():
#             latent[key] = spatial.greens_apply(latent[key], kernel, voxel_size)
#
#     return latent, sigma_mod, sigma_dist

"""Same as the itertools python package, but in pytorch."""

import torch


combinations = torch.combinations


def product(*inputs, r=1, **backend):
    """Cartesian product of a set.

    Parameters
    ----------
    inputs : iterable of tensor_like
        Input sets (tensors are flattened is not vectors)
        with `n[i]` elements each.
    r : int, default=1
        Repeats.
        .. warning:: keyword argument only.

    Returns
    -------
    output : (prod(n)**r, r) tensor
        Cartesian product

    """
    inputs = [torch.as_tensor(input, **backend) for input in inputs]
    return torch.cartesian_prod(*(inputs * r))


def permutations(input, r=None):
    """All possible r-length permutations of a set.

    .. note:: This function loops over the number of input elements,
              It can therefore be slow if this number is large.

    Parameters
    ----------
    input : tensor_like
        Input vector (tensor is flattened is not a vector)
        with `n` elements.
    r : int, default=n
        Length of the permutation vectors

    Returns
    -------
    output : (k, r) tensor
        All possible `r`-length permutations of `input`.
        `k = n! * (n-r)!`

    """
    # Ensure tensor
    input = torch.as_tensor(input).flatten()
    device = input.device
    n = input.numel()

    # Default repeats
    if r is None:
        r = n

    # Cartesian product of indices
    indices = torch.arange(input.numel(), dtype=torch.long, device=device)
    indices = product(indices, r=r)
    # -> indices.shape = (n**r, r)

    # Select only permutations (= discard rows with repeats)
    mask = indices.new_ones((indices.shape[0]), dtype=torch.bool)
    for r in range(n):
        mask &= (indices == r).sum(dim=1) <= 1
    indices = indices[mask, :]

    # Use indices to extract permutations of elements from input
    perms = input[indices]

    return perms

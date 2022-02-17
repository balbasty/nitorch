import torch


def make_list(x, n=None):
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    if n and x:
        x = x + max(0, n - len(x)) * x[-1:]
    return x


def vector_to_list(x, dtype=None, n=None):
    if torch.is_tensor(x):
        x = x.double().tolist()
    x = make_list(x, n)
    if dtype:
        x = list(map(dtype, x))
    return x


if hasattr(torch, 'movedim'):
    movedim = torch.movedim
else:
    def movedim(input, source, destination):
        """Move the position of exactly one dimension"""
        dim = input.dim()

        source = dim + source if source < 0 else source
        destination = dim + destination if destination < 0 else destination
        permutation = list(range(dim))
        del permutation[source]
        permutation.insert(destination, source)
        return input.permute(*permutation)


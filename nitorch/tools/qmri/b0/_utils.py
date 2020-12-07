import torch


def first_to_last(x):
    perm = list(range(1, x.dim())) + [0]
    return x.permute(perm)


def last_to_first(x):
    perm = [-1] + list(range(x.dim()-1))
    return x.permute(perm)


def complexmean(tensor, weighted=True):
    dtype = tensor.dtype
    abs = tensor.abs()
    if weighted:
        sumabs = abs.sum(dtype=torch.double)
    else:
        sumabs = float(abs.numel())
    angle = tensor.angle()
    sumsin = torch.sin(angle)
    if weighted:
        sumsin *= abs
    sumsin = sumsin.sum(dtype=torch.double)
    sumcos = torch.cos(angle)
    if weighted:
        sumcos *= abs
    sumcos = sumcos.sum(dtype=torch.double)
    sumsin /= sumabs
    sumcos /= sumabs
    angle = torch.atan2(sumsin, sumcos)
    mag = abs.mean().log()
    return mag.to(dtype) + 1j * angle.to(dtype)

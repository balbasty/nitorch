import torch
from nitorch.core.linalg import matvec
from nitorch.core.utils import min_intensity_step
from .multifit2 import Tissue, sin, cos


def despot1(dat, init=None):

    tissue = init or Tissue()
    if tissue.B1 is None:
        tissue.B1 = 1

    for dat1 in dat:
        dat1.step = min_intensity_step(dat1.dat)
        dat1.dat = torch.rand_like(dat1.dat).mul_(dat1.step).add_(dat1.dat)

    FAs = list(set([dat1.FA for dat1 in dat]))
    TRs = list(set([dat1.TR for dat1 in dat]))
    if len(TRs) > 1:
        raise ValueError('All images should have the same TR')

    # 1) Estimate R2* and TE=0 intercepts
    x = torch.stack([dat1.dat for dat1 in dat])
    x = x.log()
    mask = ((x >= -128) & (x <= 128) & x.isfinite()).all(0)
    x[:, ~mask] = 0

    mat = torch.zeros([len(FAs) + 1, len(dat)])
    for i, dat1 in enumerate(dat):
        mat[FAs.index(dat1.FA), i] = 1
        mat[-1, i] = -dat1.TE
    mat = mat.pinverse()

    x = matvec(mat.T, x.movedim(0, -1)).movedim(-1, 0)
    tissue.R2star = x[-1].clone()
    x = x[:-1].exp_()

    # 2) Estimate R1 and PD
    y = x.clone()
    for i, FA in enumerate(FAs):
        FA = FA * tissue.B1
        y[i] /= sin(FA)
        x[i] *= cos(FA) / sin(FA)

    alpha, beta = simple_linreg(x, y)

    tissue.R1 = beta.log() / (-dat[0].TR)
    tissue.PD = alpha / (1 - beta)

    tissue.PD[~mask] = 0
    tissue.R2star[~mask] = 0
    tissue.R1[~mask] = 0
    return tissue


def despot2(dat, init):

    tissue = init or Tissue()
    if tissue.B1 is None:
        tissue.B1 = 1

    FAs = [dat1.FA for dat1 in dat]
    TRs = [dat1.TR for dat1 in dat]
    if len(set(TRs)) > 1:
        raise ValueError('All images should have the same TR')
    TR = TRs[0]

    x = torch.stack([dat1.dat for dat1 in dat])
    y = x.clone()
    for i, FA in enumerate(FAs):
        FA = FA * tissue.B1
        y[i] /= sin(FA)
        x[i] *= cos(FA) / sin(FA)

    alpha, beta = simple_linreg(x, y)

    E1 = (-TR * tissue.R1).exp_()
    tissue.R2 = ((beta - E1)/(beta*E1 - 1)).log() / (-TR)

    E2 = (-TR * tissue.R2).exp_()
    tissue.PD = -alpha * (E1 * E2 - 1) / (1 - E1)
    tissue.PD /= E2.sqrt()

    return tissue


def simple_linreg(x, y):
    mean_y = y.mean(0)
    mean_x = x.mean(0)
    mean_x2 = x.square().mean(0)
    mean_xy = (y * x).mean(0)
    var_x = mean_x2 - mean_x.square()
    cov_xy = mean_xy - mean_x * mean_y

    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x

    return alpha, beta
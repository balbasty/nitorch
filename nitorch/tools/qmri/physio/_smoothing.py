from nitorch.spatial import smooth, identity_grid, conv
from nitorch.core.constants import pi
import torch
import matplotlib.pyplot as plt
import math


def gauss_kernel(f, dim):
    s = f/math.sqrt(8.*math.log(2.)) + 1E-7
    shape = math.ceil(4*s)
    shape = shape + (shape % 2 == 0)
    g = identity_grid([shape] * dim)
    g -= shape / 2
    g = g.square_().sum(-1)
    g *= (-0.5/(s**2))
    g.exp_()
    g /= g.sum()
    return g


def smooth_iid():

    shape = [128, 128]
    center = tuple([s//2 for s in shape])
    nrep = 100
    dim = len(shape)

    sig = [0.1, 0.5, 1, 2, 4, 8, 16] # [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
    fwhm = [2.355 * s for s in sig]
    nc = [(2*pi*(s**2))**(-dim/2) if s > 0 else 1 for s in sig]
    yb = [(4*pi*(s**2))**(-dim/2) if s > 0 else 1 for s in sig]

    dat = torch.randn([nrep, *shape])
    dat = smooth(dat, fwhm=1.5, basis=1, dim=2)

    var0 = dat.var(0)[64, 64].item()
    varb0 = []
    varb1 = []
    varbd = []
    for f in fwhm:
        sdat = smooth(dat, fwhm=f, basis=0, dim=2)
        varb0.append(sdat.var(0)[center].item()/var0)
        sdat = smooth(dat, fwhm=f, basis=1, dim=2)
        varb1.append(sdat.var(0)[center].item()/var0)
        kernel = gauss_kernel(f, dim)
        sdat = conv(dim, dat, kernel, padding='auto', bound='dct2')
        varbd.append(sdat.var(0)[center].item()/var0)

    for fn in (plt.loglog, plt.semilogy, plt.plot):
        fn(sig, nc, 'k:')
        fn(sig, yb, 'k--')
        fn(sig, varb0, 'r-+')
        fn(sig, varb1, 'b-+')
        fn(sig, varbd, 'g-+')
        plt.xlabel('Smoothing sigma')
        plt.ylabel('Variance')
        plt.show()

    fn = plt.plot
    fn(sig[2:], nc[2:], 'k:')
    fn(sig[2:], yb[2:], 'k--')
    fn(sig[2:], varb0[2:], 'r-+')
    fn(sig[2:], varb1[2:], 'b-+')
    fn(sig[2:], varbd[2:], 'g-+')
    plt.xlabel('Smoothing sigma')
    plt.ylabel('Variance')
    plt.show()


    t = lambda v: v**(-1/dim)
    varb0 = list(map(t, varb0))
    varb1 = list(map(t, varb1))
    varbd = list(map(t, varbd))
    fn = plt.plot
    fn(sig, sig, 'k--')
    fn(sig[2:], varb0[2:], 'r-+')
    fn(sig[2:], varb1[2:], 'b-+')
    fn(sig[2:], varbd[2:], 'g-+')
    plt.xlabel('Smoothing sigma')
    plt.ylabel('Variance')
    plt.show()

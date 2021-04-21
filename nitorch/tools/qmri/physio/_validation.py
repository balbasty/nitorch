import torch
import matplotlib.pyplot as plt
from ._sample import physio_sample
from ._fit import patch_and_fit_physio

# default parameters
sp = 0.008
lp = 0.4
s0 = 2.
lr = 0.2

# ranges explored
range_sp = torch.logspace(-4, 0, 20)
range_lp = torch.logspace(-3, 1, 20)
range_s0 = torch.logspace(-2, 1, 20)
range_lr = torch.logspace(-3, 1, 20)

# patches explores
patches = [3, 5, 7, 9, 15, 30, None]

do_sp = True
do_lp = True
do_s0 = True
do_lr = True


def validate(shape=32, dim=2):

    shape0 = shape
    shape = [shape] * dim

    # sp
    if do_sp:
        fits = []
        for sp1 in range_sp:
            time, rep = physio_sample(shape, sp1.item(), lp, s0, 1., lr, repeats=100)
            fsp, _, _, _ = patch_and_fit_physio(time, rep, patch=shape0, verbose=1)
            fits.append(fsp)

        plt.loglog(range_sp, range_sp, 'k-')
        plt.loglog(range_sp, fits, 'r+')
        plt.title('sigma_p')
        plt.show()

    # lp
    if do_lp:
        fits = []
        for lp1 in range_lp:
            time, rep = physio_sample(shape, sp, lp1.item(), s0, 1., lr, repeats=100)
            _, flp, _, _ = patch_and_fit_physio(time, rep, patch=shape0, verbose=1)
            fits.append(flp)

        plt.loglog(range_lp, range_lp, 'k-')
        plt.loglog(range_lp, fits, 'r+')
        plt.title('lambda_p')
        plt.show()

    # s0
    if do_s0:
        fits = []
        for s01 in range_s0:
            time, rep = physio_sample(shape, sp, lp, s01.item(), 1., lr, repeats=100)
            _, _, fs0, _ = patch_and_fit_physio(time, rep, patch=shape0, verbose=1)
            fits.append(fs0)

        plt.loglog(range_s0, range_s0, 'k-')
        plt.loglog(range_s0, fits, 'r+')
        plt.title('sigma_0')
        plt.show()

    # lr
    if do_lr:
        fits = []
        for lr1 in range_lr:
            time, rep = physio_sample(shape, sp, lp, s0, 1., lr1.item(), repeats=100)
            _, _, _, flr = patch_and_fit_physio(time, rep, patch=shape0, verbose=1)
            fits.append(flr)

        plt.loglog(range_lr, range_lr, 'k-')
        plt.loglog(range_lr, fits, 'r+')
        plt.title('lambda_r')
        plt.show()

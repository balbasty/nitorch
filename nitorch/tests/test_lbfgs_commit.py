"""L-BFGS with a Wolfe line search commits the side effects of the
accepted point (#155).

Line-search trials do not keep their side effects (cached transforms,
loss states, printed and recorded objective), so `LBFGS.step` evaluates
the accepted point normally. It used to do so before applying the step,
i.e. at the previous iterate.
"""
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import MSE, MAD
from nitorch.tools.registration.pairwise_impl import PairwiseRegister
from nitorch.tools.registration.pairwise_makeobj import (
    make_affine_optim, make_nonlin_optim)

dt = torch.double


def _images(shape=(10, 10, 10)):
    g = torch.Generator().manual_seed(0)
    x = torch.rand([1, *shape], dtype=dt, generator=g)
    for _ in range(2):
        for d in range(1, 4):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    aff = spatial.affine_default(shape, 1., dtype=dt)
    return objects.Image(x, affine=aff), objects.Image(x.roll(1, 3), affine=aff)


@pytest.mark.parametrize('kind', ['affine', 'nonlin'])
@pytest.mark.parametrize('loss', ['mse', 'mad'])
def test_lbfgs_commits_at_accepted_point(kind, loss):
    # `pairwise_postproc` writes the outputs from the cached transforms
    # (`exp(recompute=False)`): they must be those of the final
    # parameters, not of the previous iterate
    fix, mov = _images()
    lossobj = MSE(lam=1, dim=3) if loss == 'mse' else MAD(dim=3)
    sim = objects.Similarity(lossobj, mov, fix)
    affine = nonlin = None
    if kind == 'affine':
        affine = objects.AffineModel('rigid', position='sym',
                                     penalty=[1e-5] * 12)
        affine.frozen = False
        optim = make_affine_optim('lbfgs', max_iter=3)
    else:
        nonlin = objects.SVFModel(steps=4, penalty=dict(
            absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2),
            factor=1.))
        nonlin.set_dat(fix.shape, affine=fix.affine, dtype=dt)
        nonlin.frozen = False
        optim = make_nonlin_optim('lbfgs', max_iter=3, nonlin=nonlin)
    optim.iter.tol = 0
    PairwiseRegister([sim], affine, nonlin, optim, verbose=0).fit()
    model = affine if kind == 'affine' else nonlin
    assert float(model.dat.dat.abs().max()) > 0      # it did move
    assert torch.equal(model.exp(recompute=False), model.exp())

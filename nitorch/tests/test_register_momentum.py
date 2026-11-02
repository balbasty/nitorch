"""`@@optim nes|ogm -o/--output`: momentum maps of the nonlinear optimizer.

The option used to be parsed and passed to the optimizer constructor,
where it was silently swallowed: nothing was ever written.
"""
import glob
import os
import warnings
import numpy as np
import torch
import pytest

nib = pytest.importorskip('nibabel')

from nitorch import io
from nitorch.cli.registration.register.parser import parser
from nitorch.cli.registration.register.cli import _cli
from nitorch.tools.registration import optim as opt
from nitorch.tools.registration.pairwise_makeobj import make_nonlin_optim
from nitorch.tools.registration.pairwise_postproc import MomentumWriter


BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii']


class Collect:
    def __init__(self):
        self.states = []

    def __call__(self, delta):
        self.states.append(delta.clone())


def quadratic(x, grad=False, hess=False):
    """f(x) = |A x - b|^2 / 2"""
    A = torch.tensor([[3., 1.], [1., 2.]], dtype=torch.double)
    b = torch.tensor([1., -1.], dtype=torch.double)
    r = A @ x - b
    loss = r.square().sum() / 2
    if not grad:
        return loss
    return loss, A.T @ r


# ----------------------------------------------------------------------
#   optimizer hook
# ----------------------------------------------------------------------

@pytest.mark.parametrize('cls', [opt.Nesterov, opt.OGM, opt.Momentum])
def test_output_called_after_each_step(cls):
    collect = Collect()
    optim = cls(0.1, output=collect, auto_restart=False) \
        if cls is not opt.Momentum else cls(0.1, output=collect)
    optim.iter = opt.OptimIterator(max_iter=4, tol=0)
    x = torch.zeros(2, dtype=torch.double)
    optim.iter(x, quadratic)
    assert len(collect.states) == 4
    # the last state written is the optimizer's final momentum
    assert torch.equal(collect.states[-1], optim.delta)
    # the first one is a plain gradient step from x = 0
    _, g0 = quadratic(torch.zeros(2, dtype=torch.double), grad=True)
    assert torch.allclose(collect.states[0], -0.1 * g0)


def test_output_in_interleaved_iterator():
    """Each sub-optimizer writes after each of its own steps"""
    collect = Collect()
    first = opt.GradientDescent(0.1)
    first.iter = opt.OptimIterator(max_iter=2, tol=0)
    second = opt.Nesterov(0.1, output=collect)
    second.iter = opt.OptimIterator(max_iter=3, tol=0)
    joint = opt.InterleavedOptimIterator([first, second], max_iter=2, tol=0)
    x = [torch.zeros(2, dtype=torch.double), torch.zeros(2, dtype=torch.double)]
    joint.iter(x, [quadratic, quadratic])
    assert len(collect.states) == 2 * 3


def test_output_must_be_callable():
    # the CLI template used to reach the constructor as a plain string
    with pytest.raises(TypeError):
        opt.Nesterov(0.1, output='{dir}{sep}momentum.{iter}.nii.gz')
    assert opt.Nesterov(0.1).output is None
    assert opt.Nesterov(0.1, output='').output is None


def test_parser_default():
    options = parser.parse(BASE + ['@nonlin', '@@optim', 'nes'])
    assert not options.nonlin.optim.output
    options = parser.parse(BASE + ['@nonlin', '@@optim', 'ogm', '-o'])
    assert options.nonlin.optim.output == '{dir}{sep}momentum.{iter}.nii.gz'


def test_make_nonlin_optim_forwards_output():
    writer = Collect()
    optim = make_nonlin_optim('nes', 1, output=writer)
    assert optim.output is writer


class FakeNonlin:
    affine = torch.tensor([[2., 0, 0, -3], [0, 2., 0, 1], [0, 0, 2., 5],
                           [0, 0, 0, 1]], dtype=torch.double)


def test_momentum_writer(tmp_path):
    writer = MomentumWriter('{dir}{sep}{name}-mom.{iter:02d}.nii.gz',
                            FakeNonlin(), odir=str(tmp_path), name='svf')
    delta = torch.randn([4, 5, 6, 3], dtype=torch.double)
    f1 = writer(delta)
    f2 = writer(2 * delta)
    assert f1 == os.path.join(str(tmp_path), 'svf-mom.01.nii.gz')
    assert f2 == os.path.join(str(tmp_path), 'svf-mom.02.nii.gz')
    for f, ref in ((f1, delta), (f2, 2 * delta)):
        dat = io.loadf(f, dtype=torch.double)
        assert dat.shape == ref.shape
        assert torch.allclose(dat, ref, atol=1e-5)
        assert torch.allclose(io.map(f).affine.double(), FakeNonlin.affine)


# ----------------------------------------------------------------------
#   end-to-end
# ----------------------------------------------------------------------

def _save(path, seed=0, shift=0.):
    g = np.random.default_rng(seed)
    x = np.stack(np.meshgrid(*[np.arange(10.)] * 3, indexing='ij'), -1)
    dat = np.exp(-((x - 4.5 - shift) ** 2).sum(-1) / 8)
    dat += 0.01 * g.random(dat.shape)
    nib.save(nib.Nifti1Image(dat.astype(np.float32), np.eye(4)), path)
    return path


@pytest.fixture
def pair(tmp_path):
    fix = _save(str(tmp_path / 'fix.nii.gz'), 0)
    mov = _save(str(tmp_path / 'mov.nii.gz'), 1, shift=0.7)
    odir = str(tmp_path / 'out')
    return fix, mov, odir


def _run(pair, *args, levels='0'):
    fix, mov, odir = pair
    _cli(['-o', odir, '@loss', 'mse', '@@fix', fix, '-o', 'false',
          '@@mov', mov, '-o', 'false', *args, '@pyramid', '-l', levels])
    return odir


@pytest.mark.parametrize('name', ['nes', 'ogm'])
def test_cli_writes_momentum_like_velocity(pair, name):
    odir = _run(pair, '@nonlin', 'svf', '@@optim', name, '-n', '3', '-o')
    files = sorted(glob.glob(os.path.join(odir, 'momentum.*.nii.gz')))
    assert [os.path.basename(f) for f in files] == \
        [f'momentum.{i}.nii.gz' for i in (1, 2, 3)]
    vel = io.map(os.path.join(odir, 'svf.nii.gz'))
    for f in files:
        mom = io.map(f)
        assert tuple(mom.shape) == tuple(vel.shape)
        assert torch.allclose(mom.affine, vel.affine)
        dat = mom.fdata()
        assert torch.isfinite(dat).all() and dat.abs().max() > 0


def test_cli_momentum_custom_path_and_pyramid(pair):
    """{iter} keeps counting across pyramid levels (no overwrite), and each
    file is written on the lattice of its level."""
    _, _, odir = pair
    out = os.path.join(odir, 'mom', '{name}_{iter:03d}.nii.gz')
    os.makedirs(os.path.dirname(out))
    _run(pair, '@nonlin', 'svf', '@@optim', 'nes', '-n', '2', '-o', out,
         levels='0:2')
    files = sorted(glob.glob(os.path.join(odir, 'mom', '*.nii.gz')))
    assert [os.path.basename(f) for f in files] == \
        [f'svf_{i:03d}.nii.gz' for i in (1, 2, 3, 4)]
    vel = io.map(os.path.join(odir, 'svf.nii.gz'))
    coarse, fine = io.map(files[0]), io.map(files[-1])
    assert tuple(fine.shape) == tuple(vel.shape)
    assert torch.allclose(fine.affine, vel.affine)
    assert coarse.shape[0] < fine.shape[0]


def test_cli_no_momentum_by_default(pair):
    odir = _run(pair, '@nonlin', 'svf', '@@optim', 'nes', '-n', '2')
    assert not glob.glob(os.path.join(odir, 'momentum.*'))
    assert os.path.exists(os.path.join(odir, 'svf.nii.gz'))


def test_cli_affine_momentum_ignored_with_warning(pair):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        odir = _run(pair, '@affine', 'translation',
                    '@@optim', 'nes', '-n', '2', '-o')
    assert any('momentum maps' in str(x.message) for x in w)
    assert not glob.glob(os.path.join(odir, 'momentum.*'))
    assert os.path.exists(os.path.join(odir, 'translation.lta'))

"""Powell optimiser follow-ups: the line-search tolerance follows the
outer tolerance, nonlinear fields raise a clear error, and verbose
iterations end with a newline."""
import torch
import pytest
from nitorch.cli.registration.register.cli import cli as register
from nitorch.cli.registration.affreg.cli import cli as affreg
from nitorch.tools.registration import optim
from nitorch.tests.test_register_optim import (
    make_pair, check_affine, SHIFT, ROT)


def quadratic(target):
    """Closure that counts its evaluations"""
    def closure(x, **kwargs):
        closure.count += 1
        return (x - target).square().sum() + 0.1 * (x[0] * x[1])
    closure.count = 0
    return closure


def run_powell(tol):
    target = torch.as_tensor([0.3, -0.7, 1.1, 0.2], dtype=torch.double)
    closure = quadratic(target)
    x = torch.zeros(4, dtype=torch.double)
    opt = optim.Powell(lr=0.1)
    opt.iter = optim.OptimIterator(max_iter=64, tol=tol)
    x, f = opt.iter(x, closure)
    return x, target, closure.count


def test_powell_line_tol_follows_tolerance():
    x_tight, target, n_tight = run_powell(1e-8)
    x_loose, _, n_loose = run_powell(1e-3)
    assert n_loose < n_tight, (n_loose, n_tight)
    assert torch.allclose(x_tight, target, atol=0.1)
    assert torch.allclose(x_loose, target, atol=0.2)


def test_powell_explicit_line_tol():
    opt = optim.Powell(lr=0.1, line_tol=1e-3)
    opt.iter = optim.OptimIterator(tol=1e-8)
    assert opt._get_line_tol() == 1e-3
    opt = optim.Powell(lr=0.1)
    opt.iter = optim.OptimIterator(tol=1e-4)
    assert abs(opt._get_line_tol() - 1e-2) < 1e-12


def test_powell_rejects_dense_field():
    opt = optim.Powell(lr=0.1)
    vel = torch.zeros([8, 8, 8, 3])
    with pytest.raises(ValueError, match='only supported for affine'):
        opt.step(vel, lambda x, **k: x.square().sum())


def test_register_nonlin_powell_error(tmp_path):
    f, _ = make_pair(tmp_path, shift=SHIFT, rot=ROT)
    with pytest.raises(ValueError, match='only supported for affine'):
        register(['@loss', 'mse', '@@fix', f['fix'], '-o', 'false',
                  '@@mov', f['mov'], '-o', 'false',
                  '@nonlin', 'svf', '-o', 'false', '@@optim', 'powell',
                  '@pyramid', '--levels', '1'])


def test_powell_verbose_newlines(tmp_path, capsys):
    f, _ = make_pair(tmp_path, shift=SHIFT, rot=ROT)
    register(['-v', '1', '@loss', 'mse', '@@fix', f['fix'], '-o', 'false',
              '@@mov', f['mov'], '-o', 'false',
              '@affine', 'rigid', '-o', f['out'], '@@optim', 'powell',
              '@pyramid', '--levels', '1'])
    out = capsys.readouterr().out
    lines = [line for line in out.split('\n') if '(affine) |' in line]
    # one line per outer iteration (not everything on one line)
    assert len(lines) > 2, out


def test_affreg_default_recovers_rigid(tmp_path):
    f, expected = make_pair(tmp_path, shift=SHIFT, rot=ROT)
    affreg(['-m', f['mov'], '-f', f['fix'], '-o', f['out'], '-x', 'false',
            '-p', '1'])
    check_affine(f['out'], expected, atol_lin=0.03, atol_shift=0.15)

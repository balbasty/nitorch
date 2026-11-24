"""`register @nonlin shoot --exact` (#171).

`spatial.shoot(fast=True)` (the default) transports the momentum one
step at a time, which does not converge to the geodesic as the number of
steps grows. `--exact` makes the shooting model use `fast=False`, saves
`"fast": false` in the JSON sidecar, and `reslice`/`compose` read it
back to re-exponentiate the velocity the same way. The default (fast)
must be unchanged, and the option is only valid for the shoot model."""
import json

import numpy as np
import pytest
import torch

from nitorch import spatial
from nitorch.core import py

PEN = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2))
REGISTER_BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii',
                 '@nonlin']


def make_vel(shape, amp, seed=0):
    dim = len(shape)
    factor = 1. / py.prod(shape)
    g = torch.Generator().manual_seed(seed)
    mom = torch.randn([*shape, dim], generator=g, dtype=torch.float64)
    greens = spatial.greens(shape, **PEN, factor=factor, dtype=torch.float64)
    vel = spatial.greens_apply(mom, greens, factor=factor)
    vel *= amp / vel.square().sum(-1).sqrt().max()
    return vel.float()


def _spy(monkeypatch, name='shoot'):
    calls = []
    func = getattr(spatial, name)

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return func(*args, **kwargs)
    monkeypatch.setattr(spatial, name, spy)
    return calls


def _write(path, dat, affine=None):
    nib = pytest.importorskip('nibabel')
    nib.save(nib.Nifti1Image(np.asarray(dat, dtype='float32'),
                             np.eye(4) if affine is None else affine),
             str(path))
    return str(path)


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------
@pytest.mark.parametrize('args,expected', [
    ([], False),
    (['--exact'], True),
    (['--exact', 'true'], True),
    (['--exact', 'false'], False),
    (['--exact', '0'], False),
    (['-s', '16', '--exact', '--inversion', 'push'], True),
])
def test_register_parse_exact(args, expected):
    from nitorch.cli.registration.register.parser import parser
    options = parser.parse(REGISTER_BASE + ['shoot', *args])
    assert options.nonlin.exact is expected


@pytest.mark.parametrize('model', ['svf', 'smalldef'])
def test_register_parse_exact_other_models(model):
    """`--exact` is a shoot option: rejected for svf/smalldef"""
    from nitorch.cli.registration.register.parser import parser
    options = parser.parse(REGISTER_BASE + [model])
    assert 'exact' not in options.nonlin
    with pytest.raises(Exception):
        parser.parse(REGISTER_BASE + [model, '--exact'])


@pytest.mark.parametrize('args', [
    ['--exact', 'maybe'],
    ['--fast', 'false'],
    ['--not-fast'],
])
def test_register_parse_exact_invalid(args):
    from nitorch.cli.registration.register.parser import parser
    with pytest.raises(Exception):
        parser.parse(REGISTER_BASE + ['shoot', *args])


# ----------------------------------------------------------------------
# ShootModel(fast=...)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('fast', [True, False])
def test_model_fast_is_forwarded(monkeypatch, fast):
    from nitorch.tools.registration import objects
    shape = [10, 10, 10]
    vel = make_vel(shape, 3.)
    kw = {} if fast else dict(fast=False)
    nl = objects.ShootModel(vel.clone(), factor=1, **kw)
    assert nl.fast is fast
    calls = _spy(monkeypatch)
    g, ig = nl.exp(), nl.iexp()
    g2, ig2 = nl.exp2()
    assert len(calls) == 3 and all(c['fast'] is fast for c in calls)
    penalty = dict(nl.penalty)
    factor = penalty.pop('factor')
    ref = spatial.shoot(vel, nl.kernel, steps=nl.steps,
                        factor=factor / py.prod(shape), **penalty,
                        voxel_size=nl.voxel_size, return_inverse=True,
                        displacement=True, fast=fast)
    for a, b in ((g, ref[0]), (g2, ref[0]), (ig, ref[1]), (ig2, ref[1])):
        assert torch.equal(a, b)
    # the pyramid copies keep the option
    for other in (nl.downsample(), nl.upsample()):
        assert other.fast is fast
        assert other.inversion == nl.inversion
    assert ('fast=False' in repr(nl)) is (not fast)


def test_model_exact_differs_from_fast():
    from nitorch.tools.registration import objects
    vel = make_vel([12, 12, 12], 4.)
    fast = objects.ShootModel(vel.clone(), factor=1).iexp()
    exact = objects.ShootModel(vel.clone(), factor=1, fast=False).iexp()
    assert not torch.allclose(fast, exact, atol=1e-4)


# ----------------------------------------------------------------------
# register -> JSON -> reslice / compose
# ----------------------------------------------------------------------
@pytest.mark.parametrize('exact', [False, True])
def test_register_exact_json_reslice_compose(tmp_path, monkeypatch, exact):
    pytest.importorskip('nibabel')
    from nitorch import io
    from nitorch.cli.registration.register.cli import _cli
    from nitorch.cli.registration.reslice import main as reslice_main
    from nitorch.cli.registration.reslice import parser as reslice_parser
    from nitorch.cli.registration.compose.cli import _cli as compose_cli

    rng = np.random.default_rng(0)
    fix = _write(tmp_path / 'fix.nii.gz', rng.random([12, 12, 12]))
    mov = _write(tmp_path / 'mov.nii.gz', rng.random([12, 12, 12]))
    out = str(tmp_path / 'nl.nii.gz')
    flag = ['--exact'] if exact else []
    calls = _spy(monkeypatch)
    _cli(['@loss', 'mse', '-s', '@@fix', fix, '-o', 'false', '@@mov', mov,
          '-o', 'false', '@nonlin', 'shoot', '-s', '3', *flag, '-o', out,
          '@@optim', 'gd', '-n', '1', '@optim', '-n', '1',
          '@pyramid', '-l', '0'])
    # the registration used the option
    assert calls and all(kw['fast'] is (not exact) for kw in calls)
    with open(tmp_path / 'nl.json') as f:
        prm = json.load(f)
    assert prm['fast'] is (not exact)
    assert prm['model'] == 'shoot'

    vel = io.volumes.loadf(out, dtype=torch.float32).reshape([12, 12, 12, 3])
    ref = spatial.shoot(vel[None], displacement=True, return_inverse=True,
                        voxel_size=[1., 1., 1.], fast=not exact,
                        **{k: prm[k] for k in ('absolute', 'membrane',
                                               'bending', 'lame', 'factor',
                                               'steps')})
    for inv in (False, True):
        calls.clear()
        options = reslice_parser.parse(
            [fix, '-iv' if inv else '-v', out, '-t', fix])
        trf = options.transformations[0]
        reslice_main.exponentiate_transforms([trf], dtype=torch.float32)
        assert len(calls) == 1 and calls[0]['fast'] is (not exact)
        assert torch.allclose(trf.dat, ref[1 if inv else 0][0], atol=1e-5)

        calls.clear()
        fout = str(tmp_path / f'composed{int(inv)}.nii.gz')
        compose_cli(['-iv' if inv else '-v', out, '-o', fout])
        assert len(calls) == 1 and calls[0]['fast'] is (not exact)
        composed = io.volumes.loadf(fout, dtype=torch.float32)
        assert torch.allclose(composed.reshape([12, 12, 12, 3]),
                              ref[1 if inv else 0][0], atol=1e-4)


def test_json_without_fast_is_unchanged(tmp_path, monkeypatch):
    """Old sidecars (no "fast" key) shoot with the default (fast)"""
    from nitorch.cli.registration import helpers
    shape = (12, 12, 12)
    vel = make_vel(shape, 3.)
    aff = torch.eye(4, dtype=torch.float64)
    old = dict(PEN, factor=1. / py.prod(shape), steps=4)
    old['lame'] = list(old['lame'])
    files = {}
    for name, extra in (('old', {}), ('fast', dict(fast=True)),
                        ('exact', dict(fast=False))):
        files[name] = str(tmp_path / f'{name}.json')
        with open(files[name], 'w') as f:
            json.dump(dict(old, **extra), f)
    calls = _spy(monkeypatch)
    for t0, t1 in ((0, 1), (1, 0), (0.25, 1)):
        calls.clear()
        a = helpers.exponentiate_velocity(vel, aff, json=files['old'],
                                          t0=t0, t1=t1)
        assert calls and all('fast' not in kw for kw in calls)
        b = helpers.exponentiate_velocity(vel, aff, json=files['fast'],
                                          t0=t0, t1=t1)
        c = helpers.exponentiate_velocity(vel, aff, json=files['exact'],
                                          t0=t0, t1=t1)
        assert torch.equal(a, b)
        assert not torch.allclose(a, c, atol=1e-4)

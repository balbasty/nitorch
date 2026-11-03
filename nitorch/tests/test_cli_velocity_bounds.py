"""`reslice`/`compose -v VEL [JSON] [T0 T1]` integrate a velocity from
time T0 to time T1 (#171).

Let `phi_t` be the transform applied by `-v` when the velocity is
scaled by `t` (`phi_1` = `-v vel`, `phi_0.5` = `-v2 vel`). Then
`-v vel T0 T1` applies the sampling map `x -> inv(phi_T0)(phi_T1(x))`,
such that `(mov o phi_T0) o result = mov o phi_T1`.
"""
import json
import os

import pytest
import torch

from nitorch import io, spatial
from nitorch.core import py
from nitorch.core.cli import ParseError
from nitorch.spatial import _shoot
from nitorch.cli.registration import helpers
from nitorch.cli.registration.compose.cli import _cli as compose_cli
from nitorch.cli.registration.compose.parser import parse as compose_parse
from nitorch.cli.registration.reslice import main as reslice_main
from nitorch.cli.registration.reslice import struct as reslice_struct
from nitorch.cli.registration.reslice.parser import parse as reslice_parse

SHAPE = [24, 24, 24]
PEN = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=[0.05, 0.2])
PRM = dict(PEN, factor=1. / py.prod(SHAPE), steps=8, fast=True)


# ----------------------------------------------------------------------
#   helpers
# ----------------------------------------------------------------------

def smooth_velocity(amplitude=6., seed=0):
    """A smooth velocity (v = K m), as `nitorch register` produces"""
    g = torch.Generator().manual_seed(seed)
    greens = _shoot.greens(SHAPE, **PEN, factor=PRM['factor'])
    mom = torch.randn([1, *SHAPE, 3], generator=g)
    vel = _shoot.greens_apply(mom, greens, factor=PRM['factor'])[0]
    return vel * (amplitude / vel.square().sum(-1).sqrt().max())


@pytest.fixture
def velocity(tmp_path):
    """Velocity file (unit voxels) + its shooting sidecar `<vel>.json`"""
    fvel = str(tmp_path / 'vel.nii.gz')
    io.volumes.savef(smooth_velocity(), fvel, affine=torch.eye(4))
    fjson = str(tmp_path / 'vel.json')
    with open(fjson, 'w') as f:
        json.dump(PRM, f)
    return fvel, fjson


def load_vel(fvel):
    return io.volumes.loadf(fvel, dtype=torch.float32).reshape([*SHAPE, 3])


def legacy_exponentiate(vel, affine, json=None, square=False, inv=False):
    """`helpers.exponentiate_velocity` before time bounds (verbatim)"""
    vel = vel.clone()
    if square:
        vel.mul_(0.5)
    if json:
        prm = helpers.load_shoot_parameters(json)
        prm['voxel_size'] = spatial.voxel_size(affine)
        disp = spatial.shoot(vel[None], displacement=True,
                             return_inverse=inv, **prm)
        if inv:
            disp = disp[-1]
    else:
        disp = spatial.exp(vel[None], displacement=True, inverse=inv)
    return disp[0]


def reslice_exp(fvel, fjson, inv=False, square=False, t0=0, t1=1):
    """Displacement applied by reslice for one velocity"""
    trf = reslice_struct.Velocity()
    trf.file, trf.json = fvel, fjson
    trf.inv, trf.square, trf.t0, trf.t1 = inv, square, t0, t1
    reslice_main.exponentiate_transforms([trf], dtype=torch.float32)
    return trf.dat.reshape([*SHAPE, 3])


def parsed_exp(argv, fvel):
    """Displacement applied by reslice for the velocity of `argv`"""
    trf = reslice_parse(['in.nii', *argv]).transformations[0]
    return reslice_exp(fvel, trf.json, trf.inv, trf.square, trf.t0, trf.t1)


def run_compose(tmp_path, args, name='out.nii.gz'):
    out = str(tmp_path / name)
    compose_cli([*args, '-o', out])
    return io.volumes.loadf(out, dtype=torch.float32).reshape([*SHAPE, 3])


def compose_disp(a, b):
    """Voxel displacement of the sampling map x -> a(b(x))"""
    return b + helpers.pull_grid(a, spatial.add_identity_grid(b))


def norm(disp):
    return disp.square().sum(-1).sqrt()


# ----------------------------------------------------------------------
#   parsing
# ----------------------------------------------------------------------

def test_time_bounds_of_shorthands():
    tb = helpers.velocity_time_bounds
    assert tb() == (0, 1)
    assert tb(square=True) == (0, 0.5)
    assert tb(inv=True) == (1, 0)
    assert tb(square=True, inv=True) == (0.5, 0)
    # shorthands compose with explicit bounds
    assert tb(0.2, 0.6, inv=True) == (0.6, 0.2)
    assert tb(0.2, 0.6, square=True) == (0.1, 0.3)


@pytest.mark.parametrize('argv, fjson, bounds', [
    (['-v', 'vel.nii'], None, (0, 1)),
    (['-v', 'vel.nii', 'p.json'], 'p.json', (0, 1)),
    (['-v', 'vel.nii', '0', '0.5'], None, (0, 0.5)),
    (['-v', 'vel.nii', 'p.json', '1', '0.5'], 'p.json', (1, 0.5)),
    (['-v', 'vel.nii', '-0.5', '1.5'], None, (-0.5, 1.5)),
    (['--velocity', 'vel.nii', '.25', '1e-1'], None, (0.25, 0.1)),
])
@pytest.mark.parametrize('cmd', ['reslice', 'compose'])
def test_parse_velocity_values(cmd, argv, fjson, bounds):
    if cmd == 'reslice':
        opt = reslice_parse(['in.nii', *argv, '-t', 'fix.nii'])
        assert opt.files == ['in.nii'] and opt.target == 'fix.nii'
    else:
        opt = compose_parse([*argv, '-t', 'fix.nii'])
        assert opt.target == 'fix.nii'
    trf, = opt.transformations
    assert trf.file == 'vel.nii'
    assert trf.json == fjson
    assert (trf.t0, trf.t1) == bounds
    assert not trf.inv and not trf.square


@pytest.mark.parametrize('tag, inv, square', [
    ('-iv', True, False), ('-v2', False, True), ('-iv2', True, True),
    ('--velocity-inverse-square', True, True),
])
def test_parse_shorthands_keep_flags(tag, inv, square):
    trf, = reslice_parse(['in.nii', tag, 'vel.nii', '0', '0.4'])\
        .transformations
    assert (trf.inv, trf.square, trf.t0, trf.t1) == (inv, square, 0, 0.4)
    trf, = compose_parse([tag, 'vel.nii']).transformations
    assert (trf.inv, trf.square, trf.t0, trf.t1) == (inv, square, 0, 1)


@pytest.mark.parametrize('argv, match', [
    (['-v', 'vel.nii', '0.5'], 'two time bounds'),
    (['-v', 'vel.nii', 'p.json', '0.5', '-t', 'f.nii'], 'two time bounds'),
    (['-v', 'vel.nii', 'nan', '1'], 'finite'),
    (['-v', 'vel.nii', '0', 'inf'], 'finite'),
])
def test_parse_bad_bounds(argv, match):
    with pytest.raises(ParseError, match=match):
        reslice_parse(['in.nii', *argv])
    with pytest.raises(ParseError, match=match):
        compose_parse(argv)


def test_parse_json_must_come_before_bounds():
    with pytest.raises(ParseError):
        reslice_parse(['in.nii', '-v', 'vel.nii', '0', '1', 'p.json'])
    with pytest.raises(ParseError):
        compose_parse(['-v', 'vel.nii', '0', '1', 'p.json'])


def test_parse_sidecar_autodetected_with_bounds(velocity):
    fvel, fjson = velocity
    trf, = reslice_parse(['in.nii', '-v', fvel, '0', '0.5'])\
        .transformations
    assert trf.json == fjson and (trf.t0, trf.t1) == (0, 0.5)


def test_parse_bounds_then_more_transforms():
    opt = reslice_parse(['in.nii', '-v', 'a.nii', '1', '0.5',
                         '-il', 'b.lta', '-iv', 'c.nii', '0.5', '1'])
    a, b, c = opt.transformations
    assert (a.t0, a.t1, a.inv) == (1, 0.5, False)
    assert b.file == 'b.lta' and b.inv
    assert (c.t0, c.t1, c.inv) == (0.5, 1, True)


def test_reslice_double_dash():
    # `--`: what follows are input files, bounds stay with `-v`
    opt = reslice_parse(['in.nii', '-v', 'vel.nii', '0', '0.5',
                         '--', '-odd.nii', '0.5'])
    trf, = opt.transformations
    assert (trf.json, trf.t0, trf.t1) == (None, 0, 0.5)
    assert opt.files == ['in.nii', '-odd.nii', '0.5']


def test_compose_double_dash():
    # `--`: what follows completes the last option, values included
    trf, = compose_parse(['-v', '--', '-vel.nii', '-p.json', '1', '0.5'])\
        .transformations
    assert (trf.file, trf.json, trf.t0, trf.t1) == \
        ('-vel.nii', '-p.json', 1, 0.5)
    trf, = compose_parse(['-v', 'vel.nii', '--', '-p.json']).transformations
    assert (trf.file, trf.json, trf.t0, trf.t1) == \
        ('vel.nii', '-p.json', 0, 1)
    # the JSON comes before the bounds
    with pytest.raises(ParseError, match="after '--'"):
        compose_parse(['-v', 'vel.nii', '0', '0.5', '--', '-p.json'])
    with pytest.raises(ParseError, match="after '--'"):
        compose_parse(['-v', 'vel.nii', '0', '0.5', '--', 'x'])
    with pytest.raises(ParseError, match="after '--'"):
        compose_parse(['-l', 'a.lta', '--', 'x'])


# ----------------------------------------------------------------------
#   default bounds and shorthands are exactly today's results
# ----------------------------------------------------------------------

LEGACY = [  # (tag, inv, square, equivalent bounds)
    ('-v', False, False, ['0', '1']),
    ('-v2', False, True, ['0', '0.5']),
    ('-iv', True, False, ['1', '0']),
    ('-iv2', True, True, ['0.5', '0']),
]


@pytest.mark.parametrize('shoot', [True, False], ids=['shoot', 'svf'])
@pytest.mark.parametrize('tag, inv, square, bounds', LEGACY)
def test_shorthands_unchanged(velocity, shoot, tag, inv, square, bounds):
    fvel, fjson = velocity
    if not shoot:
        os.remove(fjson)
        fjson = None
    ref = legacy_exponentiate(load_vel(fvel), torch.eye(4), fjson,
                              square=square, inv=inv)
    # bitwise identical to the previous implementation...
    assert torch.equal(parsed_exp([tag, fvel], fvel), ref)
    # ... and to the equivalent bounds
    assert torch.equal(parsed_exp(['-v', fvel, *bounds], fvel), ref)


def test_inverted_bounds_for_streamlines(velocity):
    # streamlines are deformed by the inverse chain: bounds are swapped
    fvel, fjson = velocity
    trf, = reslice_parse(['in.nii', '-v', fvel, '0.25', '0.5'])\
        .transformations
    itrf, = reslice_main.invert_transforms([trf])
    disp = reslice_exp(fvel, fjson, itrf.inv, itrf.square, itrf.t0, itrf.t1)
    assert torch.equal(disp, reslice_exp(fvel, fjson, t0=0.5, t1=0.25))


# ----------------------------------------------------------------------
#   stationary velocity fields: exp((T1 - T0) * v)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('t0, t1, t', [
    (0.25, 0.75, 0.5), (0.75, 0.25, -0.5), (0.5, 0.5, 0.),
    (0, 1.5, 1.5), (-0.5, 0.5, 1.),
])
def test_svf_bounds(velocity, tmp_path, t0, t1, t):
    fvel, fjson = velocity
    os.remove(fjson)
    vel = load_vel(fvel)
    disp = parsed_exp(['-v', fvel, str(t0), str(t1)], fvel)
    ref = spatial.exp((vel * abs(t))[None], displacement=True,
                      inverse=t < 0)[0]
    assert torch.allclose(disp, ref, atol=1e-5)
    # compose: same transform
    out = run_compose(tmp_path, ['-v', fvel, str(t0), str(t1)])
    assert torch.allclose(out, ref, atol=1e-4)


# ----------------------------------------------------------------------
#   geodesic shooting
# ----------------------------------------------------------------------

def test_shoot_bounds_from_zero(velocity):
    # φ_t = shoot(t v): `-v vel 0 t` is exactly shoot(t v)
    fvel, fjson = velocity
    vel = load_vel(fvel)
    for t in (0.5, 0.3, 1.5):
        disp = parsed_exp(['-v', fvel, '0', str(t)], fvel)
        ref = spatial.shoot((vel * t)[None], displacement=True,
                            voxel_size=[1.] * 3, **PRM)[0]
        assert torch.equal(disp, ref)
    # `-v vel 0.5 0` is exactly the inverse returned by shoot(v / 2)
    disp = parsed_exp(['-v', fvel, '0.5', '0'], fvel)
    ref = spatial.shoot((vel * 0.5)[None], displacement=True,
                        return_inverse=True, voxel_size=[1.] * 3, **PRM)
    assert torch.equal(disp, ref[-1][0])


def test_shoot_general_bounds_definition(velocity):
    # x -> inv(φ_T0)(φ_T1(x))
    fvel, fjson = velocity
    disp = parsed_exp(['-v', fvel, '0.25', '0.75'], fvel)
    ref = compose_disp(reslice_exp(fvel, fjson, t0=0.25, t1=0),
                       reslice_exp(fvel, fjson, t0=0, t1=0.75))
    assert torch.equal(disp, ref)
    # identical bounds: identity
    disp = parsed_exp(['-v', fvel, '0.5', '0.5'], fvel)
    assert not disp.any()


def test_shoot_bounds_chain(velocity, tmp_path):
    # (0 -> 0.5) then (0.5 -> 1) is (0 -> 1), within the accuracy of
    # the inverse returned by shoot. It is better than `-v2 -v2`, which
    # assumes that the geodesic is a one-parameter subgroup.
    fvel, fjson = velocity
    full = run_compose(tmp_path, ['-v', fvel], 'full.nii.gz')
    chain = run_compose(tmp_path, ['-v', fvel, '0', '0.5',
                                   '-v', fvel, '0.5', '1'], 'chain.nii.gz')
    square = run_compose(tmp_path, ['-v2', fvel, '-v2', fvel], 'sq.nii.gz')
    half = reslice_exp(fvel, fjson, t0=0, t1=0.5)
    ihalf = reslice_exp(fvel, fjson, t0=0.5, t1=0)
    inverse_error = norm(compose_disp(ihalf, half)).max()
    chain_error = norm(chain - full)
    square_error = norm(square - full)
    assert chain_error.max() < 1.5 * inverse_error
    assert chain_error.max() < 0.5 * square_error.max()
    assert chain_error.mean() < 0.5 * square_error.mean()


def test_shoot_midpoint(velocity, tmp_path):
    # Midpoint of the path: `-v vel 0 0.5` on the moving image and
    # `-v vel 1 0.5` on the fixed image agree (within the accuracy of
    # the inverse returned by shoot), better than with `-iv2`.
    fvel, fjson = velocity
    phi1 = reslice_exp(fvel, fjson)
    half = parsed_exp(['-v', fvel, '0', '0.5'], fvel)
    back = parsed_exp(['-v', fvel, '1', '0.5'], fvel)
    ihalf = parsed_exp(['-iv2', fvel], fvel)
    # transforms: fix = mov o φ_1, so fix o back should be mov o φ_0.5
    err_new = norm(compose_disp(phi1, back) - half)
    err_old = norm(compose_disp(phi1, ihalf) - half)
    inverse_error = norm(compose_disp(phi1, reslice_exp(fvel, fjson,
                                                        inv=True)))
    assert err_new.mean() < 0.9 * err_old.mean()
    assert err_new.max() < err_old.max()
    assert err_new.mean() < 1.1 * inverse_error.mean()

    # images, end-to-end
    g = torch.Generator().manual_seed(1)
    mov = torch.randn([1, *SHAPE], generator=g)
    mov = spatial.smooth(mov, fwhm=8, dim=3, bound='dct2')[0]
    fmov = str(tmp_path / 'mov.nii.gz')
    io.volumes.savef(mov, fmov, affine=torch.eye(4))

    def reslice(fname, *trf, out):
        out = str(tmp_path / out)
        reslice_main._reslice([fname, *trf, '-i', '3', '-x', '-o', out])
        return io.volumes.loadf(out, dtype=torch.float32)

    reslice(fmov, '-v', fvel, out='fix.nii.gz')
    ffix = str(tmp_path / 'fix.nii.gz')
    mid_mov = reslice(fmov, '-v', fvel, '0', '0.5', out='mid_mov.nii.gz')
    mid_fix = reslice(ffix, '-v', fvel, '1', '0.5', out='mid_fix.nii.gz')
    mid_fix_old = reslice(ffix, '-iv2', fvel, out='mid_fix_old.nii.gz')
    crop = (slice(4, -4),) * 3
    rmse_new = (mid_fix - mid_mov)[crop].square().mean().sqrt()
    rmse_old = (mid_fix_old - mid_mov)[crop].square().mean().sqrt()
    assert rmse_new < 0.95 * rmse_old


def test_shoot_bounds_compose_matches_reslice(velocity, tmp_path):
    fvel, fjson = velocity
    for bounds in (['0.25', '0.75'], ['1', '0.5'], ['0', '0.5']):
        out = run_compose(tmp_path, ['-v', fvel, *bounds])
        ref = parsed_exp(['-v', fvel, *bounds], fvel)
        assert torch.allclose(out, ref, atol=1e-4)
    # explicit JSON + bounds
    fjson2 = str(tmp_path / 'shoot.json')
    os.rename(fjson, fjson2)
    out = run_compose(tmp_path, ['-v', fvel, fjson2, '1', '0.5'])
    ref = reslice_exp(fvel, fjson2, t0=1, t1=0.5)
    assert torch.allclose(out, ref, atol=1e-4)

"""`nitorch compose` must shoot velocities that have a shooting JSON
sidecar (like `nitorch reslice`), and exponentiate SVFs otherwise."""
import json
import os

import numpy as np
import pytest
import torch

from nitorch import io, spatial
from nitorch.cli.registration.compose.cli import _cli as compose_cli
from nitorch.cli.registration.compose.parser import parse as compose_parse
from nitorch.cli.registration.reslice import main as reslice_main
from nitorch.cli.registration.reslice import struct as reslice_struct

SHAPE = [10, 10, 10]
PRM = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=[0.05, 0.2],
           factor=1, steps=4, fast=True)
AFFINE = torch.diag(torch.as_tensor([2., 2., 2., 1.]))


def make_velocity(tmp_path, with_json=True, name='vel'):
    g = torch.Generator().manual_seed(0)
    vel = torch.randn([*SHAPE, 3], generator=g)
    vel = spatial.smooth(vel.movedim(-1, 0), fwhm=3, dim=3).movedim(0, -1)
    vel = vel * (1.5 / vel.abs().max())
    fvel = str(tmp_path / f'{name}.nii.gz')
    io.volumes.savef(vel, fvel, affine=AFFINE)
    fjson = None
    if with_json:
        fjson = str(tmp_path / f'{name}.json')
        with open(fjson, 'w') as f:
            json.dump(dict(PRM, voxel_size=[2., 2., 2.]), f)
    return fvel, fjson


def reference(fvel, fjson, square=False, inv=False):
    trf = reslice_struct.Velocity()
    trf.file = fvel
    trf.json = fjson
    trf.inv = inv
    trf.square = square
    reslice_main.exponentiate_transforms([trf], dtype=torch.float32)
    return trf.dat


def manual_shoot(fvel, square=False, inv=False):
    vel = io.volumes.loadf(fvel, dtype=torch.float32).reshape([*SHAPE, 3])
    if square:
        vel = vel * 0.5
    out = spatial.shoot(vel[None], displacement=True, return_inverse=inv,
                        voxel_size=[2., 2., 2.], **PRM)
    return (out[-1] if inv else out)[0]


def run_compose(args, out):
    compose_cli([*args, '-o', out])
    return io.volumes.loadf(out, dtype=torch.float32).reshape([*SHAPE, 3])


@pytest.mark.parametrize('tag, square, inv', [
    ('-v', False, False), ('-v2', True, False),
    ('-iv', False, True), ('-iv2', True, True),
])
def test_compose_explicit_json_shoots(tmp_path, tag, square, inv):
    fvel, fjson = make_velocity(tmp_path)
    # rename the sidecar so that only the explicit path can find it
    fjson2 = str(tmp_path / 'shoot_params.json')
    os.rename(fjson, fjson2)
    out = run_compose([tag, fvel, fjson2], str(tmp_path / 'out.nii.gz'))
    ref = manual_shoot(fvel, square, inv)
    assert torch.allclose(out, ref, atol=1e-4)
    assert torch.allclose(out, reference(fvel, fjson2, square, inv),
                          atol=1e-4)
    # and it differs from the stationary exponential
    svf = reference(fvel, None, square, inv)
    assert not torch.allclose(out, svf, atol=1e-3)


@pytest.mark.parametrize('tag, square, inv', [
    ('-v', False, False), ('-v2', True, False), ('-iv', False, True),
])
def test_compose_autodetects_json(tmp_path, tag, square, inv):
    fvel, fjson = make_velocity(tmp_path)
    options = compose_parse([tag, fvel, '-o', 'x.nii.gz'])
    assert options.transformations[0].json is None  # resolved in main
    out = run_compose([tag, fvel], str(tmp_path / 'out.nii.gz'))
    assert torch.allclose(out, manual_shoot(fvel, square, inv), atol=1e-4)


def test_compose_ignores_non_shooting_json(tmp_path):
    fvel, _ = make_velocity(tmp_path, with_json=False)
    with open(str(tmp_path / 'vel.json'), 'w') as f:
        json.dump({'foo': 1}, f)
    out = run_compose(['-v', fvel], str(tmp_path / 'out.nii.gz'))
    assert torch.allclose(out, reference(fvel, None), atol=1e-4)


@pytest.mark.parametrize('tag, square, inv', [
    ('-v', False, False), ('-v2', True, False), ('-iv', False, True),
])
def test_compose_svf_without_json_uses_exp(tmp_path, tag, square, inv):
    fvel, _ = make_velocity(tmp_path, with_json=False)
    out = run_compose([tag, fvel], str(tmp_path / 'out.nii.gz'))
    vel = io.volumes.loadf(fvel, dtype=torch.float32).reshape([*SHAPE, 3])
    if square:
        vel = vel * 0.5
    ref = spatial.exp(vel[None], displacement=True, inverse=inv)[0]
    assert torch.allclose(out, ref, atol=1e-4)


def test_compose_explicit_json_followed_by_other_options(tmp_path):
    fvel, fjson = make_velocity(tmp_path)
    options = compose_parse(['-v', fvel, fjson, '-ou', 'mm', '-o', 'x.nii'])
    assert options.transformations[0].json == fjson
    assert options.output_unit == 'mm'
    assert options.output == 'x.nii'

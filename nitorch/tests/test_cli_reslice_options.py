"""`nitorch reslice`: channels, prefilter, shooting, chunking, log/logit."""
import json

import numpy as np
import pytest

from nitorch import spatial
from nitorch.cli.registration.reslice.main import reslice
from nitorch.cli.registration.reslice.parser import parse

nib = pytest.importorskip('nibabel')

AFFINE = np.diag([2., 3., 4., 1.])


def save(fname, dat, affine=AFFINE):
    nib.save(nib.Nifti1Image(np.asarray(dat, dtype='float32'), affine),
             str(fname))
    return str(fname)


def load(fname):
    return np.asarray(nib.load(str(fname)).dataobj).squeeze()


# ----------------------------------------------------------------------
# -c / --channels
# ----------------------------------------------------------------------
@pytest.mark.parametrize('chunk', [[], ['-k', '4']])
@pytest.mark.parametrize('spec,index', [
    (['0:2'], [0, 1]),
    ([':2'], [0, 1]),
    (['1:'], [1, 2]),
    (['::2'], [0, 2]),
    (['2', '0'], [2, 0]),
    (['0:2', '2'], [0, 1, 2]),
])
def test_channel_ranges(tmp_path, spec, index, chunk):
    dat = np.random.RandomState(0).rand(8, 8, 8, 3)
    img = save(tmp_path / 'img.nii.gz', dat)
    out = tmp_path / 'out.nii.gz'
    reslice([img, '-t', img, '-c', *spec, *chunk, '-o', str(out)])
    np.testing.assert_allclose(load(out), dat[..., index].squeeze(),
                               atol=1e-4)


# ----------------------------------------------------------------------
# -p / --prefilter
# ----------------------------------------------------------------------
@pytest.mark.parametrize('args,expected', [
    (['-p'], True),
    (['-p', '-x'], True),
    (['-p', 'no'], False),
    (['-p', 'yes'], True),
    (['-p', '0'], False),
    (['--prefilter'], True),
])
def test_prefilter_value_is_optional(args, expected):
    options = parse(['img.nii.gz', *args])
    assert options.prefilter is expected


def test_bare_prefilter_runs(tmp_path):
    dat = np.random.RandomState(0).rand(8, 8, 8)
    img = save(tmp_path / 'img.nii.gz', dat)
    out = tmp_path / 'out.nii.gz'
    reslice([img, '-t', img, '-i', '3', '-o', str(out), '-p'])
    assert out.exists()
    np.testing.assert_allclose(load(out), dat, atol=1e-4)


# ----------------------------------------------------------------------
# -v with JSON shooting parameters
# ----------------------------------------------------------------------
@pytest.mark.parametrize('explicit_json', [False, True])
def test_shooting_parameters_reach_shoot(tmp_path, monkeypatch,
                                         explicit_json):
    img = save(tmp_path / 'img.nii.gz',
               np.random.RandomState(0).rand(8, 8, 8))
    vel = save(tmp_path / 'vel.nii.gz', np.zeros([8, 8, 8, 3]))
    # what `nitorch register` writes next to a shooting velocity
    prm = {'absolute': 0.001, 'membrane': 0.01, 'bending': 0.5,
           'lame': [0.1, 0.3], 'factor': 0.25}
    with open(tmp_path / 'vel.json', 'w') as f:
        json.dump(prm, f)

    calls = []
    shoot = spatial.shoot

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return shoot(*args, **kwargs)

    monkeypatch.setattr(spatial, 'shoot', spy)
    out = tmp_path / 'out.nii.gz'
    args = ['-v', vel] + ([str(tmp_path / 'vel.json')] if explicit_json else [])
    reslice([img, *args, '-t', img, '-o', str(out)])

    assert len(calls) == 1
    kwargs = calls[0]
    for key in ('absolute', 'membrane', 'bending', 'factor'):
        assert kwargs[key] == pytest.approx(prm[key])
    assert list(kwargs['lame']) == pytest.approx(prm['lame'])
    assert list(kwargs['voxel_size']) == pytest.approx([2., 3., 4.])
    # zero velocity -> identity
    np.testing.assert_allclose(load(out), load(img), atol=1e-4)


# ----------------------------------------------------------------------
# -k / --chunk
# ----------------------------------------------------------------------
@pytest.mark.parametrize('target_shape,chunk', [
    ((11, 6, 9), '4'), ((5, 5, 5), '4'), ((5, 5, 5), '16')])
def test_chunk_uses_output_grid(tmp_path, target_shape, chunk):
    dat = np.random.RandomState(0).rand(8, 8, 8, 2)
    img = save(tmp_path / 'img.nii.gz', dat)
    taff = np.diag([1.5, 3.5, 3., 1.])
    taff[:3, -1] = [-1., 2., 0.5]
    tgt = save(tmp_path / 'tgt.nii.gz', np.zeros(target_shape), taff)
    ref = tmp_path / 'ref.nii.gz'
    out = tmp_path / 'out.nii.gz'
    reslice([img, '-t', tgt, '-o', str(ref)])
    reslice([img, '-t', tgt, '-k', chunk, '-o', str(out)])
    assert load(out).shape == (*target_shape, 2)
    np.testing.assert_allclose(load(out), load(ref), atol=1e-5)


# ----------------------------------------------------------------------
# --log / --logit
# ----------------------------------------------------------------------
@pytest.mark.parametrize('chunk', [[], ['-k', '4']])
def test_log_identity(tmp_path, chunk):
    dat = np.random.RandomState(0).rand(8, 8, 8) + 0.5
    img = save(tmp_path / 'img.nii.gz', dat)
    out = tmp_path / 'out.nii.gz'
    reslice([img, '-t', img, '--log', *chunk, '-o', str(out)])
    np.testing.assert_allclose(load(out), dat, rtol=1e-4)


def test_log_chunk_matches_no_chunk(tmp_path):
    dat = np.random.RandomState(0).rand(8, 8, 8) + 0.5
    img = save(tmp_path / 'img.nii.gz', dat)
    ref = tmp_path / 'ref.nii.gz'
    out = tmp_path / 'out.nii.gz'
    reslice([img, '-t', img, '--log', '-o', str(ref)])
    reslice([img, '-t', img, '--log', '-k', '4', '-o', str(out)])
    np.testing.assert_allclose(load(out), load(ref), rtol=1e-5)


@pytest.mark.parametrize('chunk', [[], ['-k', '4']])
def test_logit_identity(tmp_path, chunk):
    p = np.random.RandomState(0).rand(8, 8, 8) * 0.9 + 0.05
    dat = np.stack([p, 1 - p], -1)
    img = save(tmp_path / 'prob.nii.gz', dat)
    out = tmp_path / 'out.nii.gz'
    reslice([img, '-t', img, '--logit', *chunk, '-o', str(out)])
    np.testing.assert_allclose(load(out), dat, atol=1e-4)


@pytest.mark.parametrize('chunk', [[], ['-k', '4']])
def test_logit_implicit_identity(tmp_path, chunk):
    # one explicit class; the background (1 - p) is implicit
    p = np.random.RandomState(0).rand(8, 8, 8, 1) * 0.9 + 0.05
    img = save(tmp_path / 'prob.nii.gz', p)
    out = tmp_path / 'out.nii.gz'
    reslice([img, '-t', img, '--logit', 'implicit', *chunk, '-o', str(out)])
    np.testing.assert_allclose(load(out), p.squeeze(), atol=1e-4)


# ----------------------------------------------------------------------
# -vx / --voxel-size: output grid uses the 'first' anchor rule
# floor((N-1)*f) + 1, so the last output voxel stays inside the FOV
# ----------------------------------------------------------------------
@pytest.mark.parametrize('use_target', [False, True])
@pytest.mark.parametrize('vx,expected', [
    ([1., 1.5, 2.], (15, 15, 15)),   # f = 2 -> (8-1)*2+1
    ([4., 6., 8.], (4, 4, 4)),       # f = 1/2 -> floor(3.5)+1
    ([6., 9., 12.], (3, 3, 3)),      # f = 1/3 -> floor(2.33)+1
])
def test_voxel_size_output_shape(tmp_path, use_target, vx, expected):
    dat = np.random.RandomState(0).rand(8, 8, 8)
    img = save(tmp_path / 'img.nii.gz', dat)
    out = tmp_path / 'out.nii.gz'
    args = [img] + (['-t', img] if use_target else [])
    reslice(args + ['-vx', *map(str, vx), '-o', str(out)])
    res = nib.load(str(out))
    assert res.shape[:3] == expected
    # first voxel preserved, last voxel inside the input FOV
    assert np.allclose(res.affine[:3, 3], AFFINE[:3, 3])
    last = res.affine[:3, :3] @ (np.array(expected) - 1)
    in_last = AFFINE[:3, :3] @ (np.array([8, 8, 8]) - 1)
    assert np.all(last <= in_last + 1e-5)


# ----------------------------------------------------------------------
# minimal reslicing (no -t): all linear transforms are absorbed in the
# output orientation matrix, wherever they sit in the chain
# ----------------------------------------------------------------------
def _rigid(shift, rot):
    import torch
    prm = torch.as_tensor([*shift, *rot], dtype=torch.double)
    return spatial.affine_matrix_classic(prm, dim=3).numpy()


@pytest.mark.parametrize('chunk', [[], ['-k', '6']])
@pytest.mark.parametrize('chain', ['field-lin', 'lin-field-lin', 'ilin-field'])
def test_minimal_reslice_keeps_all_linear(tmp_path, chain, chunk):
    from nitorch import io
    dat = np.random.RandomState(0).rand(12, 10, 8)
    img = save(tmp_path / 'img.nii.gz', dat)
    # zero displacement field (identity), on a grid that does not
    # match the image
    field_aff = np.diag([3., 3., 3., 1.])
    field_aff[:3, -1] = [-5., 2., 1.]
    field = save(tmp_path / 'disp.nii.gz', np.zeros([6, 6, 6, 3]), field_aff)
    # large shifts: more than the field of view
    lin1 = _rigid([40., -35., 50.], [0.1, -0.05, 0.08])
    lin2 = _rigid([-20., 60., 15.], [-0.02, 0.07, 0.03])
    io.transforms.savef(lin1.astype('float32'), str(tmp_path / 'lin1.lta'),
                        type='ras')
    io.transforms.savef(lin2.astype('float32'), str(tmp_path / 'lin2.lta'),
                        type='ras')
    l1, l2 = str(tmp_path / 'lin1.lta'), str(tmp_path / 'lin2.lta')
    if chain == 'field-lin':
        args, total = ['-d', field, '-l', l1], lin1
    elif chain == 'lin-field-lin':
        args, total = ['-l', l1, '-d', field, '-l', l2], lin1 @ lin2
    else:
        args, total = ['-il', l1, '-d', field], np.linalg.inv(lin1)

    out = tmp_path / 'out.nii.gz'
    reslice([img, *args, *chunk, '-o', str(out)])
    np.testing.assert_allclose(nib.load(str(out)).affine,
                               np.linalg.inv(total) @ AFFINE, atol=1e-3)
    np.testing.assert_allclose(load(out), dat, atol=1e-3)

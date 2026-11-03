"""`orient` transform option and `vopt` defaults (issue #171)."""
import os

import numpy as np
import pytest

from nitorch.cli.registration.orient.cli import _cli as orient
from nitorch.cli.registration.vopt import cli as vopt

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.arange(4 * 6 * 8, dtype='float32').reshape([4, 6, 8])
    nib.save(nib.Nifti1Image(dat, np.diag([2., 3., 4., 1.])), fname)
    return fname


def ltas(tmp_path):
    return sorted(f for f in os.listdir(tmp_path) if f.endswith('.lta'))


def test_orient_no_transform_by_default(tmp_path, volume):
    orient([volume, '-l', 'LPS'])
    assert os.path.exists(tmp_path / 'img.LPS.nii.gz')
    assert ltas(tmp_path) == []


def test_orient_short_transform_tag(tmp_path, volume):
    orient([volume, '-l', 'LPS', '-t'])
    assert ltas(tmp_path) == ['img_to_LPS.lta']


def test_orient_transform_name(tmp_path, volume):
    tname = str(tmp_path / 'my.lta')
    orient([volume, '-l', 'LPS', '-t', tname])
    assert ltas(tmp_path) == ['my.lta']


@pytest.fixture
def svfs(tmp_path):
    aff = np.diag([2., 3., 4., 1.])
    names = {}
    for fix, mov in (('a', 'b'), ('b', 'c'), ('c', 'a')):
        fname = str(tmp_path / f'{mov}_to_{fix}.nii.gz')
        dat = np.zeros([4, 5, 6, 3], dtype='float32')
        nib.save(nib.Nifti1Image(dat, aff), fname)
        names[(fix, mov)] = fname
    return names


def test_vopt_default_output_is_volume():
    _, _, output, _ = vopt.parse(['-i', 'a', 'b', 'a_to_b.nii.gz'])
    assert not output.endswith('.lta')
    assert output.endswith('.nii.gz')


def test_vopt_voxel_size_and_outputs(tmp_path, svfs, monkeypatch):
    seen = {}

    def fake_optimal_svf(inputs, weights=None, **prm):
        seen.update(prm)
        shape = list(next(iter(inputs.values())).shape)
        return [np.zeros(shape, dtype='float32') for _ in range(3)]

    monkeypatch.setattr(vopt, 'optimal_svf', fake_optimal_svf)
    args = []
    for (fix, mov), fname in svfs.items():
        args += ['-i', fix, mov, fname]
    vopt._cli(args)
    assert seen['voxel_size'] == pytest.approx([2., 3., 4.])
    outs = sorted(f for f in os.listdir(tmp_path) if 'optimal' in f)
    assert outs == ['a_optimal.nii.gz', 'b_optimal.nii.gz',
                    'c_optimal.nii.gz']


def test_orient_transform_is_ras_to_ras(tmp_path, volume):
    import torch
    from nitorch import io
    from nitorch.spatial import affine_rmdiv
    orient([volume, '-l', 'LPS', '-t'])
    f = io.transforms.map(str(tmp_path / 'img_to_LPS.lta'))
    assert f.type() == ('ras', 'ras')
    aff0 = io.volumes.map(volume).affine.double()
    aff1 = io.volumes.map(str(tmp_path / 'img.LPS.nii.gz')).affine.double()
    expected = affine_rmdiv(aff1, aff0)
    assert not torch.allclose(expected, torch.eye(4, dtype=torch.double))
    assert torch.allclose(f.fdata(dtype=torch.double).reshape([4, 4]),
                          expected, atol=1e-5)
    assert torch.allclose(f.data(dtype=torch.double).reshape([4, 4]),
                          expected, atol=1e-5)


def test_orient_like_without_value():
    from nitorch.cli.registration.orient.parser import parse
    from nitorch.core.cli import ParseError
    with pytest.raises(ParseError, match='-k'):
        parse(['a.nii', '-k'])


def test_orient_like_followed_by_option():
    from nitorch.cli.registration.orient.parser import parse
    from nitorch.core.cli import ParseError
    with pytest.raises(ParseError, match='-k'):
        parse(['a.nii', '-k', '-l', 'RAS'])


def test_orient_like_with_value():
    from nitorch.cli.registration.orient.parser import parse
    opt = parse(['a.nii', '-k', 'ref.nii', '-l', 'RAS'])
    assert opt.like == 'ref.nii'
    assert opt.layout == 'RAS'

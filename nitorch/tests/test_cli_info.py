"""`info` command: `-s` boolean parsing and field-of-view centre."""
import numpy as np
import pytest

from nitorch.cli.misc.info.cli import _cli as info
from nitorch.cli.misc.info.parser import parse

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.arange(4 * 6 * 8, dtype='float32').reshape([4, 6, 8])
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname


@pytest.mark.parametrize('val,expected', [
    ('no', False), ('n', False), ('false', False), ('0', False),
    ('yes', True), ('true', True), ('1', True),
])
def test_stat_bool(volume, val, expected):
    assert parse([volume, '-s', val]).stat is expected


def test_stat_default(volume):
    assert parse([volume]).stat is False
    assert parse([volume, '-s']).stat is True


def test_info_stat_no(volume, capsys):
    info([volume, '-s', 'no'])
    out = capsys.readouterr().out
    assert 'shape' in out and 'mean' not in out


def test_info_stat_default(volume, capsys):
    info([volume, '-s'])
    out = capsys.readouterr().out
    assert 'mean' in out


def test_info_center(volume, capsys):
    # voxel centres span 0..(n-1), so the FOV centre is at (n-1)/2
    info([volume])
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith('center'))
    assert '(3.0, 5.0, 7.0)' in line

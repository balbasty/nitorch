"""`-t` (output transform) option of the image utilities."""
import os

import numpy as np
import pytest

from nitorch.cli.misc.chunk.cli import _cli as chunk
from nitorch.cli.misc.unstack.cli import _cli as unstack
from nitorch.cli.misc.extract_patches.cli import _cli as extract_patches
from nitorch.cli.registration.reorient.cli import _cli as reorient

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.arange(4 * 6 * 8, dtype='float32').reshape([4, 6, 8])
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname


def ltas(tmp_path):
    return sorted(f for f in os.listdir(tmp_path) if f.endswith('.lta'))


def test_chunk_short_transform_tag(tmp_path, volume):
    chunk([volume, '-n', '4', '-d', '2', '-t'])
    assert len(ltas(tmp_path)) == 2


def test_unstack_short_transform_tag(tmp_path, volume):
    unstack([volume, '-d', '2', '-t'])
    assert len(ltas(tmp_path)) == 8


def test_unstack_dim(tmp_path, volume):
    unstack([volume, '-d', '0'])
    outs = sorted(f for f in os.listdir(tmp_path)
                  if f.endswith('.nii.gz') and f != 'img.nii.gz')
    assert len(outs) == 4


def test_extract_patches_short_transform_tag(tmp_path, volume):
    extract_patches([volume, '-n', '2', '-t'])
    assert len(ltas(tmp_path)) > 0


def test_reorient_short_transform_tag(tmp_path, volume):
    reorient([volume, '-l', 'LPS', '-t'])
    assert ltas(tmp_path) == ['img_2_LPS.lta']
    assert os.path.exists(tmp_path / 'img.LPS.nii.gz')


def test_reorient_transform_name(tmp_path, volume):
    tname = str(tmp_path / 'my.lta')
    reorient([volume, '-l', 'LPS', '-t', tname])
    assert ltas(tmp_path) == ['my.lta']

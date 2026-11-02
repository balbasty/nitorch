"""`inpaint`: >4D inputs and the RLS convergence test."""
import re

import numpy as np
import pytest
import torch

from nitorch.cli.misc.inpaint.cli import _cli as inpaint_cli
from nitorch.cli.misc.inpaint.main import inpaint

nib = pytest.importorskip('nibabel')


def _volume(shape=(8, 8, 8)):
    x = np.linspace(0, 1, shape[0])[:, None, None]
    y = np.linspace(0, 1, shape[1])[None, :, None]
    z = np.linspace(0, 1, shape[2])[None, None, :]
    dat = (x + 2 * y + 3 * z + (x > 0.5)).astype('float32')
    dat = np.broadcast_to(dat, shape).copy()
    dat[3:5, 3:5, 3:5] = np.nan
    return dat


def test_inpaint_5d_singleton(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    out = str(tmp_path / 'out.nii.gz')
    dat = _volume()[..., None, None]  # (8, 8, 8, 1, 1)
    nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)
    inpaint_cli([fname, '-o', out, '-v', '0', '-nrls', '2'])
    res = np.asarray(nib.load(out).dataobj)
    assert np.isfinite(res).all()


def _n_iter(capsys, **kwargs):
    dat = torch.as_tensor(_volume())[..., None]
    capsys.readouterr()
    out = inpaint(dat, verbose=1, **kwargs)
    log = capsys.readouterr().out
    n = len(re.findall(r'^\s*\d+ \|', log, flags=re.M)) - 1
    return n, 'Converged' in log, out


def test_inpaint_converges_on_tolerance(capsys):
    n_loose, conv_loose, out = _n_iter(capsys, max_iter_rls=30, tol_rls=1e-2)
    n_strict, conv_strict, _ = _n_iter(capsys, max_iter_rls=30, tol_rls=0)
    assert conv_loose and n_loose < 30
    assert not conv_strict and n_strict == 30
    assert torch.isfinite(out).all()

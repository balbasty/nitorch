"""`nitorch reslice`: command-line behaviour."""
import os

import numpy as np
import pytest

from nitorch.cli.registration.reslice.main import reslice

nib = pytest.importorskip('nibabel')


def test_reslice_default_output_name(tmp_path):
    # Without `-o`, the output is written next to the input, using the
    # default template `{dir}{sep}{base}.resliced{ext}`.
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.random.RandomState(0).rand(6, 7, 8).astype('float32')
    nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)

    reslice([fname, '-t', fname])

    oname = tmp_path / 'img.resliced.nii.gz'
    assert oname.exists()
    out = np.asarray(nib.load(str(oname)).dataobj).squeeze()
    assert out.shape == dat.shape
    np.testing.assert_allclose(out, dat, atol=1e-4)

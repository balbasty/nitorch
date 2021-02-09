from .. import map, load, loadf, save, savef
import os
import wget
import tempfile
import nibabel
import numpy as np

test_data = {
    'nifti': [],
    'minc': [],
    'mgh': [],
    'ecat': [],
    'parrec': [],
}

# from nitest-balls1
# LICENSE: Public Domain Dedication and License v1.0
root = 'https://github.com/yarikoptic/nitest-balls1/raw/2cd07d86e2cc2d3c612d5d4d659daccd7a58f126/'
test_data['nifti'] += [
    os.path.join(root, 'NIFTI/T1.nii.gz'),
    os.path.join(root, 'NIFTI/T2-interleaved.nii.gz'),
    os.path.join(root, 'NIFTI/T2.nii.gz'),
    os.path.join(root, 'NIFTI/T2_-interleaved.nii.gz'),
    os.path.join(root, 'NIFTI/T2_.nii.gz'),
    os.path.join(root, 'NIFTI/fieldmap.nii.gz'),
]
test_data['parrec'] += [
    [os.path.join(root, 'PARREC/DTI.PAR'), os.path.join(root, 'PARREC/DTI.REC')],
    [os.path.join(root, 'PARREC/NA.PAR'), os.path.join(root, 'PARREC/NA.REC')],
    [os.path.join(root, 'PARREC/T1.PAR'), os.path.join(root, 'PARREC/T1.REC')],
    [os.path.join(root, 'PARREC/T2-interleaved.PAR'), os.path.join(root, 'PARREC/T2-interleaved.REC')],
    [os.path.join(root, 'PARREC/T2.PAR'), os.path.join(root, 'PARREC/T2.REC')],
    [os.path.join(root, 'PARREC/T2_-interleaved.PAR'), os.path.join(root, 'PARREC/T2_-interleaved.REC')],
    [os.path.join(root, 'PARREC/T2_.PAR'), os.path.join(root, 'PARREC/T2_.REC')],
    [os.path.join(root, 'PARREC/fieldmap.PAR'), os.path.join(root, 'PARREC/fieldmap.REC')],
]

# from nitest-minc2
# LICENSE: Copyright (C) 1993-2004 Louis Collins, McConnell Brain
#          Permission to use, copy, modify, and distribute
root = 'https://github.com/matthew-brett/nitest-minc2/raw/c835bd43f40069d542f75386551ed0fd2377462a'
test_data['minc'] += [
    os.path.join(root, 'mincex_EPI-frame.mnc'),
    os.path.join(root, 'mincex_diff-B0.mnc'),
    os.path.join(root, 'mincex_diff-FA.mnc'),
    os.path.join(root, 'mincex_gado-contrast.mnc'),
    os.path.join(root, 'mincex_mask.mnc'),
    os.path.join(root, 'mincex_pd.mnc'),
    os.path.join(root, 'mincex_t1.mnc'),
]

# from nipy-ecattest
# LICENSE: CC0-1.0
root = 'https://github.com/effigies/nipy-ecattest/raw/9a0a592057bc16894c20c77b03ea1ebb5f8ca8f9'
test_data['ecat'] += [
    os.path.join(root, 'ECAT7_testcase_multiframe.v'),
    os.path.join(root, 'ECAT7_testcaste_neg_values.v'),
]

# from nitest-freesurfer
# LICENSE: Freesurfer license
root = 'https://bitbucket.org/nipy/nitest-freesurfer/raw/0d307865704df71c3b2248139714806aad47139d'
test_data['mgh'] += [
    os.path.join(root, 'fsaverage/mri/T1.mgz'),
    os.path.join(root, 'fsaverage/mri/aparc+aseg.mgz'),
    os.path.join(root, 'fsaverage/mri/aparc.a2005s+aseg.mgz'),
    os.path.join(root, 'fsaverage/mri/aparc.a2009s+aseg.mgz'),
    os.path.join(root, 'fsaverage/mri/aseg.mgz'),
    os.path.join(root, 'fsaverage/mri/brain.mgz'),
    os.path.join(root, 'fsaverage/mri/brainmask.mgz'),
    os.path.join(root, 'fsaverage/mri/lh.ribbon.mgz'),
    os.path.join(root, 'fsaverage/mri/mni305.cor.mgz'),
    os.path.join(root, 'fsaverage/mri/orig.mgz'),
    os.path.join(root, 'fsaverage/mri/p.aseg.mgz'),
    os.path.join(root, 'fsaverage/mri/rh.ribbon.mgz'),
    os.path.join(root, 'fsaverage/mri/ribbon.mgz'),
]


class TempFilename:
    """
    Generate a temporary file name (without creating it) and delete
    the file at the end (if it exists).
    """
    def __init__(self, *args, delete=True, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.delete = delete

    def __enter__(self):
        _, fname = tempfile.mkstemp(*self.args, **self.kwargs)
        os.remove(fname)
        self.fname = fname
        return fname

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.delete and os.path.exists(self.fname):
            os.remove(self.fname)


class DownloadedFile:
    """
    Generate a temporary filename and download the file at `url` to
    this location. The temporary file is deleted at the end (if it
    still exists).
    """
    def __init__(self, *url):
        self.url = url

    @staticmethod
    def base_ext(fname):
        base = os.path.basename(fname)
        base, ext = os.path.splitext(base)
        if ext == '.gz':
            base, ext = os.path.splitext(base)
            ext += '.gz'
        return base, ext

    def __enter__(self):
        fnames = []
        basenames = {}
        self.tmpfiles = []
        for fname in self.url:
            base, ext = self.base_ext(fname)
            if base in basenames:
                tmpbase = basenames[base]
            else:
                tmpbase = TempFilename(prefix=base, suffix='')
                self.tmpfiles.append(tmpbase)
                tmpbase = tmpbase.__enter__()
                basenames[base] = tmpbase
            tmpfile = tmpbase + ext
            fnames.append(tmpfile)
            wget.download(fname, tmpfile)
        if len(fnames) == 1:
            return fnames[0]
        else:
            return fnames

    def __exit__(self, exc_type, exc_val, exc_tb):
        for tmpfile in self.tmpfiles:
            tmpfile.__exit__(exc_type, exc_val, exc_tb)


def _test_nibabel_load(fname):
    """Format-agnostic test"""
    nib = nibabel.load(fname)
    fdata = nib.get_fdata()


def _test_full_load(fname):
    """Format-agnostic test"""
    nii = map(fname, 'r')
    nib = nibabel.load(fname)
    fdata = nii.fdata(numpy=True)
    assert np.allclose(fdata, nib.get_fdata()), "fdata full"
    data = nii.data(numpy=True)
    assert data.dtype == nib.dataobj.dtype, "mapped dtype"
    assert np.allclose(data, nib.dataobj.get_unscaled()), "data full"


def _test_partial_load(fname):
    nii = map(fname, 'r')
    fdata = nii.fdata(numpy=True)
    data = nii.data(numpy=True)

    slicer = (slice(None), 0, slice(None))
    perm = [1, 0]

    nii_slice = nii[slicer].permute(perm)
    fdata_slice = nii_slice.fdata(numpy=True)
    assert np.allclose(fdata_slice, fdata[slicer].transpose(perm)), "fdata slice"
    data_slice = nii_slice.data(numpy=True)
    assert np.allclose(data_slice, data[slicer].transpose(perm)), "data slice"


def _test_partial_load_more(fname):
    # compare symbolic slicing with numpy slicing
    nii = map(fname, 'r')

    # nii_slice = nii[:, 0, 5:-10][-40::-1, 30:31].permute([1, 0]).data(numpy=True)
    # dat_slice = nii.data(numpy=True)[:, 0, 5:-10][-40::-1, 30:31].transpose([1, 0])
    # assert np.allclose(nii_slice, dat_slice)
    nii_slice = nii.permute([2, 0, 1])[:, None, :, 5:-10].permute([1, 2, 0, 3]).data(numpy=True)
    dat_slice = nii.data(numpy=True).transpose([2, 0, 1])[:, None, :, 5:-10].transpose([1, 2, 0, 3])
    assert np.allclose(nii_slice, dat_slice)
    nii_slice = nii.permute([2, 0, 1])[:, None, -5::-2, 5:-10].permute([1, 2, 0, 3]).data(numpy=True)
    dat_slice = nii.data(numpy=True).transpose([2, 0, 1])[:, None, -5::-2, 5:-10].transpose([1, 2, 0, 3])
    assert np.allclose(nii_slice, dat_slice)


def _test_partial_save(fname):
    nii = map(fname, 'r+')
    slicer = (slice(None), 0, slice(None))
    perm = [1, 0]
    nii_slice = nii[slicer].permute(perm)

    rnd = np.random.randint(0, 3000, nii_slice.shape).astype(nii.dtype)
    nii_slice.set_data(rnd)
    data_slice = nii.data()[slicer].permute(perm)
    assert np.allclose(data_slice, rnd), "set data slice"


def _test_full_save(fname):
    dtype = np.int16
    dat = np.random.randint(0, 3000, [32, 32, 32]).astype(dtype)
    save(dat, fname)

    nii = map(fname, 'r')
    rdat = nii.data(numpy=True)
    assert np.allclose(rdat, dat), "save full data"


def test_nifti():
    url = test_data['nifti'][0]
    with DownloadedFile(url) as fname:
        _test_full_load(fname)
        _test_partial_load(fname)
        _test_partial_load_more(fname)
        _test_partial_save(fname)
    with TempFilename(suffix='.nii.gz') as fname:
        _test_full_save(fname)


def test_mgh():
    url = test_data['mgh'][0]
    with DownloadedFile(url) as fname:
        _test_full_load(fname)
        _test_partial_load(fname)
        _test_partial_save(fname)
    with TempFilename(suffix='.mgz') as fname:
        _test_full_save(fname)


# MINC needs a special implementation
# def test_minc():
#     url = test_data['minc'][0]
#     _test_partial_io(url)


# Even nibabel fails to read these files
# def test_ecat():
#     url = test_data['ecat'][1]
#     with DownloadedFile(url) as fname:
#         _test_nibabel_load(fname)


# PARREC needs a special implementation
# def test_parrec():
#     url = test_data['parrec'][0]
#     with DownloadedFile(*url) as fnames:
#         par, rec = fnames
#         _test_nibabel_load(par)
#         _test_full_load(par)

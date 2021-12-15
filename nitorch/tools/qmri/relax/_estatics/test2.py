from nitorch.tools.qmri import relax, io as qio
import os
import torch

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [120, 40]
plt.rcParams['figure.dpi'] = 10
plt.rcParams.update({'font.size': 52})

home = os.environ.get('HOME')
root = os.path.join(home, 'links', 'data', 'qMRI', 'hMRI', 'hmri_sample_dataset_with_maps')

t1dir = os.path.join(root, 't1w_mfc_3dflash_v1i_R4_0015_copy')
ft1w = [os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-00224-1.nii'),
        os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-00448-2.nii'),
        os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-00672-3.nii'),
        os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-00896-4.nii'),
        os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-01120-5.nii'),
        os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-01344-6.nii'),
        os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-01568-7.nii'),
        os.path.join(t1dir, 'anon_s2018-02-28_18-26-190921-00001-01792-8.nii'),]

pddir = os.path.join(root, 'pdw_mfc_3dflash_v1i_R4_0009_copy')
fpdw = [os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-00224-1.nii'),
        os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-00448-2.nii'),
        os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-00672-3.nii'),
        os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-00896-4.nii'),
        os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-01120-5.nii'),
        os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-01344-6.nii'),
        os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-01568-7.nii'),
        os.path.join(pddir, 'anon_s2018-02-28_18-26-185345-00001-01792-8.nii'),]

mtdir = os.path.join(root, 'mtw_mfc_3dflash_v1i_R4_0012_copy')
fmtw = [os.path.join(mtdir, 'anon_s2018-02-28_18-26-190132-00001-00224-1.nii'),
        os.path.join(mtdir, 'anon_s2018-02-28_18-26-190132-00001-00448-2.nii'),
        os.path.join(mtdir, 'anon_s2018-02-28_18-26-190132-00001-00672-3.nii'),
        os.path.join(mtdir, 'anon_s2018-02-28_18-26-190132-00001-00896-4.nii'),
        os.path.join(mtdir, 'anon_s2018-02-28_18-26-190132-00001-01120-5.nii'),
        os.path.join(mtdir, 'anon_s2018-02-28_18-26-190132-00001-01344-6.nii'),]

resdir = os.path.join(root, 'pdw_mfc_3dflash_v1i_R4_0009_copy', 'Results')
b1p_map = os.path.join(resdir, 'Supplementary', 'anon_s2018-02-28_18-26-184837-00001-00001-1_B1map.nii')
b1p_ref = os.path.join(resdir, 'Supplementary', 'anon_s2018-02-28_18-26-184837-00001-00001-1_B1ref.nii')
pdw_b1m_map = os.path.join(resdir, 'Supplementary', 'sensMap_HC_over_BC_division_PD.nii')
pdw_b1m_ref = fpdw[0]
t1w_b1m_map = os.path.join(resdir, 'Supplementary', 'sensMap_HC_over_BC_division_T1.nii')
t1w_b1m_ref = ft1w[0]
mtw_b1m_map = os.path.join(resdir, 'Supplementary', 'sensMap_HC_over_BC_division_MT.nii')
mtw_b1m_ref = fmtw[0]

t1w = qio.GradientEchoMulti.from_fnames(ft1w)
pdw = qio.GradientEchoMulti.from_fnames(fpdw)
mtw = qio.GradientEchoMulti.from_fnames(fmtw, mt=True)
readout = -2
ncoils = 21
noise = 306
for c in [t1w, pdw, mtw]:
    c.readout = readout
    c.ncoils = ncoils
    c.noise = noise

enable_reg = False
device = 'cuda'


if device == 'cuda':
    torch.cuda.empty_cache()
opt = relax.ESTATICSOptions()
opt.recon.space = 0
opt.recon.affine = 0
opt.recon.fov = 0
opt.regularization.norm = 'jtv'  # 'none', 'tkh', 'tv', 'jtv'
opt.regularization.factor = [1, 0.0005]
opt.backend.device = device
opt.preproc.register = enable_reg
opt.distortion.enable = True
opt.distortion.bending = 1e5
opt.verbose = 1
opt.plot = True
te0, r2s, *dist = relax.estatics([t1w], opt)

from nitorch.tools.qmri import relax, io as qio
import os
import torch
import glob

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [120, 40]
plt.rcParams['figure.dpi'] = 10
plt.rcParams.update({'font.size': 52})

# Demo dataset


home = os.environ.get('HOME')
root = os.path.join(home, 'links', 'data', 'qMRI', 'hires', 'I45')

fa10dir = os.path.join(root, 'FA10')
f10 = [os.path.join(fa10dir, 'echo0_whitened_rms_600mu.mgz'),
       os.path.join(fa10dir, 'echo1_whitened_rms_600mu.mgz'),
       os.path.join(fa10dir, 'echo2_whitened_rms_600mu.mgz'),
       os.path.join(fa10dir, 'echo3_whitened_rms_600mu.mgz'),]

resdir = os.path.join(root, 'Results')

fa10 = qio.GradientEchoMulti.from_fnames(f10)
fa10.readout = -3
fa10.noise = 1
fa10.ncoils = 32

# dat10 = fa10.volume
# fa10.volume = dat10[:, ::4, ::4, ::4]
# fa10.volume = fa10.volume.fdata()

# mask = fa10.fdata().square().sum(0).sqrt()
# mask = mask > 40
# fa10.mask = mask

enable_reg = False
device = 'cuda'

# Non-linear fit

if device == 'cuda':
    torch.cuda.empty_cache()
opt = relax.ESTATICSOptions()
opt.likelihood = 'chi'
opt.recon.space = 0
opt.recon.affine = 0
opt.recon.fov = 0
opt.regularization.norm = 'jtv'  # 'none', 'tkh', 'tv', 'jtv'
opt.regularization.factor = [1, 0.0005]
opt.backend.device = device
opt.preproc.register = enable_reg
opt.optim.max_iter_rls = 15
opt.distortion.enable = True
opt.distortion.factor = 1
opt.distortion.absolute = 0
opt.distortion.membrane = 0
opt.distortion.bending = 1e5
opt.distortion.model = 'smalldef'
opt.verbose = 1
opt.plot = 1
te0, r2s, *dist = relax.estatics([fa10], opt)
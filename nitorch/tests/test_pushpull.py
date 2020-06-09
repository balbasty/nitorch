import nibabel
import torch
import nitorch
import nitorch.spatial
import nitorch._C.spatial
from matplotlib import pyplot as plt

fname = '/scratch/mikael-python/data/T1w_iso.nii'
fmri = nibabel.load(fname)
mri = torch.tensor(fmri.get_fdata().astype('float'), dtype=torch.float)
mri = mri[None, None, ...]

# mri = mri[:,:,100:101,100:105,100:105]

id = nitorch.spatial.identity(mri.shape[4:1:-1])

inter = nitorch.spatial.InterpolationType.linear
bound = nitorch.spatial.BoundType.dct2

samp = nitorch._C.spatial.grid_pull(mri, id, [bound], [inter], True)
grad = nitorch._C.spatial.grid_grad(mri, id, [bound], [inter], True)

plt.subplot(1, 4, 1)
plt.imshow(samp[0, 0, 100, :, :])
plt.subplot(1, 4, 2)
plt.imshow(grad[0, 0, 100, :, :, 0])
plt.subplot(1, 4, 3)
plt.imshow(grad[0, 0, 100, :, :, 1])
plt.subplot(1, 4, 4)
plt.imshow(grad[0, 0, 100, :, :, 2])
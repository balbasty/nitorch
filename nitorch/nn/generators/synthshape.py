"""SynthShape generator

References
----------
..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
       MRI Scans of any Contrast and Resolution"
      Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
      Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
      2021
      https://arxiv.org/abs/2107.09559
..[2] "Learning image registration without images"
      Malte Hoffmann, Benjamin Billot, Juan Eugenio Iglesias,
      Bruce Fischl, Adrian V. Dalca
      ISBI 2021
      https://arxiv.org/abs/2004.10282
..[3] "Partial Volume Segmentation of Brain MRI Scans of any Resolution
       and Contrast"
      Benjamin Billot, Eleanor D. Robinson, Adrian V. Dalca, Juan Eugenio Iglesias
      MICCAI 2020
      https://arxiv.org/abs/2004.10221
..[4] "A Learning Strategy for Contrast-agnostic MRI Segmentation"
      Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl,
      Juan Eugenio Iglesias*, Adrian V. Dalca*
      MIDL 2020
      https://arxiv.org/abs/2003.01995
"""
import torch

from nitorch import spatial
from nitorch.core import utils
from ..base import Module
from ..preproc import LabelToOneHot, OneHotToLabel, AffineQuantiles
from .spatial import RandomDeform, RandomSmooth, RandomLowRes2D, RandomLowRes3D
from .field import HyperRandomBiasFieldTransform, HyperRandomMultiplicativeField
from .mixture import HyperRandomGaussianMixture
from .intensity import RandomGammaCorrection, HyperRandomChiNoise


buckner_left = []
buckner_right = []
buckner_skullstrip = []
buckner_csf = []
buckner_droppable = [
    buckner_skullstrip,                                  # skullstrip w/ csf
    [buckner_skullstrip, *buckner_csf],                  # skullstrip no csf
    [buckner_left, *buckner_skullstrip, *buckner_csf],   # hemi right
    [buckner_right, *buckner_skullstrip, *buckner_csf],  # hemi left
]
buckner_predicted = []


class AddBagClass(Module):

    def __init__(self, kernel=5, label=None):
        super().__init__()
        self.kernel = kernel
        self.label = label

    def forward(self, x):
        if x.dtype.is_floating_point:
            fwd = 1 - x[:, :1]
        else:
            fwd = (x == 0).bitwise_not_().float()
        fwd = spatial.smooth(fwd, fwhm=self.kernel, dim=x.dim()-2,
                             padding='same', bound='nearest')
        fwd = fwd > 0.1
        if x.dtype.is_floating_point:
            bag = x[:, :1] * fwd
            bg = x[:, :1] * fwd.bitwise_not_()
            return torch.cat([bg, x[1:], bag], dim=1)
        else:
            x = x.clone()
            bag = (x == 0).bitwise_and_(fwd)
            label = self.label or (x.max() + 1)
            x[bag] = label


class SynthMRI(Module):
    """Synthetic MRI generator

    The operations and hyper-parameters used in this class are largely
    inspired by those used in SynthSeg, with some changes:
    - The intensity variance is separated into two components: a smooth
      component inside the GMM (that models within-tissue variability)
      and an i.i.d. component in LR space (that models thermal noise)
    - Furthermore, the thermal component is modulated by a random g-factor
      map and follows a noncentral Chi distribution.
    - Motion artefacts are modeled by applying a small isotropic smoothing
      before the HR->LR transformation. Consequently, we do not model
      imperfect slice profiles, as it is simply merged with motion.
    - The HR->LR resolution is randomly chosen to model either a 2D
      acquisition (a single low-res dimension with varying slice thickness
      and slice gap) or a 3D acquisition (isotropic low-res, without slice
      gap)
    - We model all possible slice thickness in the range (0, resolution),
      instead of (1, resolution), although there are minimal differences
      between slice thicknesses in the range (0, 1).

    References
    ----------
    ..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
           MRI Scans of any Contrast and Resolution"
          Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
          Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
          2021
          https://arxiv.org/abs/2107.09559
    """

    def __init__(self,
                 classes,
                 channel=1,
                 vel_amplitude=3,
                 vel_fwhm=20,
                 translation=20,
                 rotation=15,
                 zoom=0.15,
                 shear=0.012,
                 gmm_fwhm=10,
                 bias_amplitude=0.5,
                 bias_fwhm=50,
                 gamma=0.6,
                 resolution=8,
                 noise=96,
                 gfactor_amplitude=0.5,
                 gfactor_fwhm=64,
                 gmm_cat='labels',
                 bag=0.5,
                 droppable_labels=None,
                 predicted_labels=None,
                 dtype=None):
        super().__init__()
        self.gmm_cat = gmm_cat[0].lower()
        dtype = dtype or torch.get_default_dtype()

        self.droppable_labels = droppable_labels
        self.predicted_labels = predicted_labels

        # label 2 one hot
        self.bag = AddBagClass() if bag else (lambda x: x)
        self.to_onehot = LabelToOneHot(dtype=dtype, nb_classes=classes)
        # deform
        self.deform = RandomDeform(
            amplitude='uniform',
            amplitude_exp=vel_amplitude/2,
            amplitude_scale=vel_amplitude/3.4,
            fwhm='dirac',
            fwhm_exp=vel_fwhm,
            translation='uniform',
            translation_scale=translation*2/3.4,
            rotation='uniform',
            rotation_scale=rotation*2/3.4,
            zoom='uniform',
            zoom_scale=zoom*2/3.4,
            shear='uniform',
            shear_scale=shear*2/3.4,
            image_bound='nearest',
        )
        # one hot 2 label
        self.to_label = OneHotToLabel()
        # gmm
        self.mixture = HyperRandomGaussianMixture(
            nb_classes=classes,
            nb_channels=channel,
            means='uniform',
            means_exp=125,
            means_scale=250/3.46,
            scales='uniform',
            scales_exp=13,
            scales_scale=24/3.46,
            fwhm='uniform',
            fwhm_exp=gmm_fwhm/2,
            fwhm_scale=gmm_fwhm/3.46,
            background_zero=True,
            dtype=dtype,
        )
        # gamma
        self.gamma = RandomGammaCorrection(
            factor='lognormal',
            factor_exp=1,
            factor_scale=gamma,
            vmin=0,
            vmax=255,
        )
        # bias field
        self.bias = HyperRandomBiasFieldTransform(
            amplitude='uniform',
            amplitude_exp=bias_amplitude/2,
            amplitude_scale=bias_amplitude/3.4,
            fwhm='dirac',
            fwhm_exp=bias_fwhm,
        )
        # smooth (iso)
        self.smooth = RandomSmooth(iso=True)
        # smooth and downsample
        self.lowres2d = RandomLowRes2D(
            resolution='uniform',
            resolution_exp=resolution/2+1,
            resolution_scale=resolution/3.46
        )
        self.lowres3d = RandomLowRes3D(
            resolution='uniform',
            resolution_exp=resolution/2+1,
            resolution_scale=resolution/3.46
        )
        # add noise
        gfactor = HyperRandomMultiplicativeField(
            amplitude='uniform',
            amplitude_exp=gfactor_amplitude/2,
            amplitude_scale=gfactor_amplitude/3.46,
            fwhm='dirac',
            fwhm_exp=gfactor_fwhm,
            dtype=dtype,
        )
        snoise = HyperRandomChiNoise(
            sigma='uniform',
            sigma_exp=noise/2,
            sigma_scale=noise/3.46,
        )
        self.noise = lambda x: snoise(x, gfactor(x.shape, device=x.device))
        # rescale
        self.rescale = AffineQuantiles()

    def forward(self, s):
        """

        Parameters
        ----------
        s : (batch, 1, *shape) tensor
            Tensor of labels

        Returns
        -------
        x : (batch, channel, *shape) tensor
            Multi-channel image

        """
        # s0 <- labels to predict
        # s  <- labels to model
        s0 = s

        if self.droppable_labels:
            # remove labels that are not modelled
            # E.g., this could be all non-brain labels (to model skull
            # stripping) or all left labels (to model hemi).
            ngroups = len(self.droppable_labels)
            group = torch.randint(ngroups+2, [])
            if group > 0:
                dropped_labels = self.droppable_labels[group]
                s[utils.isin(s, dropped_labels)] = 0
        if self.predicted_labels:
            # remove labels that are not predicted
            s0[utils.isin(s0, self.predicted_labels).bitwise_not_()] = 0

        s = self.bag(s)
        s = self.to_onehot(s)
        s, s0 = self.deform(s, s0)
        if self.gmm_cat == 'l':
            s = self.to_label(s)
        x = self.mixture(s)
        x.clamp_(0, 255)
        x = self.gamma(x)
        x = self.bias(x)
        x = self.smooth(x)
        e = self.noise(x)
        if torch.rand([]) > 0.5:
            x, vx = self.lowres2d(x, noise=e, return_resolution=True)
        else:
            x, vx = self.lowres3d(x, noise=e, return_resolution=True)
        x = self.rescale(x)
        return x, s0


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

# # WIP qmri synth
# from typing import Sequence
# from torch import Tensor
# from nitorch.tools import qmri

import torch
import nitorch as ni
from nitorch import spatial
from nitorch.core import utils
from nitorch.nn.generators.spatial import RandomPatch
from ..base import Module
from ..preproc import LabelToOneHot, OneHotToLabel, AffineQuantiles
from .spatial import RandomDeform, RandomSmooth, RandomLowRes2D, RandomLowRes3D
from .field import HyperRandomMultiplicativeField
from .mixture import HyperRandomGaussianMixture
from .intensity import RandomGammaCorrection, HyperRandomChiNoise, HyperRandomBiasFieldTransform
from ..datasets import lut


def buckner32_labels():
    central = [11, 12, 13]
    left = [1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 16, 17]
    right = [18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31]
    left_cb = [5, 6]
    right_cb = [22, 23]
    nonbrain = []
    csf = []

    drop_cb = [*left_cb, *right_cb]
    drop_hemi_left = [*right, *right_cb, *nonbrain, *csf]
    drop_hemi_left_nocb = [*drop_hemi_left, *right_cb]
    drop_hemi_right = [*left, *left_cb, *nonbrain, *csf]
    drop_hemi_right_nocb = [*drop_hemi_right, *left_cb]

    droppable = [
        drop_cb,
        drop_hemi_left,
        drop_hemi_left_nocb,
        drop_hemi_right,
        drop_hemi_right_nocb,
    ]
    predicted = list(range(1, 32))
    return droppable, predicted


def fs35_labels():
    from ..datasets.lut_fs import lut_fs35, fs35_hierarchy

    left = lut.ids_in_group('Left', lut_fs35, fs35_hierarchy)
    right = lut.ids_in_group('Right', lut_fs35, fs35_hierarchy)
    cerebellum = lut.ids_in_group('Cerebellum', lut_fs35, fs35_hierarchy)

    droppable = [
        cerebellum,
        left,
        right,
        {*left, *cerebellum},
        {*right, *cerebellum},
    ]

    discarded = [k for k, v in lut_fs35.items()
                 if 'Plexus' in v or 'Vessel' in v]
    predicted = {*lut_fs35.keys()} - {*discarded}
    return droppable, list(predicted)


def fs32_labels():
    central = [10, 11, 12]
    left = [1, 2, 3, 6, 7, 8, 9, 13, 14, 16, 17, 18]
    right = [19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    left_cb = [4, 5]
    right_cb = [22, 23]
    nonbrain = []
    csf = [15]

    drop_skullstrip = nonbrain
    drop_skullstrip_csf = [*nonbrain, *csf]
    drop_cb = [*left_cb, *right_cb]
    drop_hemi_left = [*right, *right_cb, *nonbrain, *csf]
    drop_hemi_left_nocb = [*drop_hemi_left, *left_cb]
    drop_hemi_right = [*left, *left_cb, *nonbrain, *csf]
    drop_hemi_right_nocb = [*drop_hemi_right, *right_cb]

    droppable = [
        drop_skullstrip,
        drop_skullstrip_csf,
        drop_cb,
        drop_hemi_left,
        drop_hemi_left_nocb,
        drop_hemi_right,
        drop_hemi_right_nocb,
    ]
    predicted = sorted(central + left_cb + right_cb + left[:-2] + right[:-2])
    return droppable, predicted


class AddBagClass(Module):

    def __init__(self, kernel=10, label=None):
        super().__init__()
        self.kernel = kernel
        self.label = label

    def forward(self, x):
        if x.dtype.is_floating_point:
            fwd = 1 - x[:, :1]
        else:
            fwd = (x == 0).bitwise_not_().float()
        fwd = spatial.smooth(fwd, fwhm=self.kernel, dim=x.dim()-2,
                             padding='same', bound='replicate')
        fwd = fwd > 1e-3
        if x.dtype.is_floating_point:
            bag = x[:, :1] * fwd
            bg = x[:, :1] * fwd.bitwise_not_()
            return torch.cat([bg, x[1:], bag], dim=1)
        else:
            x = x.clone()
            bag = (x == 0).bitwise_and_(fwd)
            label = self.label or (x.max() + 1)
            x[bag] = label
        return x


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
    - We model all possible slice thicknesses in the range (0, resolution),
      instead of (1, resolution), although there are minimal differences
      between slice thicknesses in the range (0, 1).
    - We sample a "bag" class obtained by dilating the brain. It can be used
      to roughly model the head or to model the embedding medium in ex vivo
      imaging.
    - We sample single hemispheres, as well as brains without the cerebellum,
      again in order to simulate ex vivo imaging.

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
                 motion_fwhm=3,
                 resolution=8,
                 noise=48,
                 gfactor_amplitude=0.5,
                 gfactor_fwhm=64,
                 gmm_cat='labels',
                 bag=0.5,
                 droppable_labels=None,
                 predicted_labels=None,
                 patch_size=None,
                 dtype=None):
        """

        Parameters
        ----------
        channel : int, default=1
            Number of channels
        vel_amplitude : float, default=3
            Amplitude of the velocity field (larger -> larger displacement)
        vel_fwhm : float, default=20
            Full-width at half-maximum of the velocity field
            (larger -> smoother deformation field)
        translation : float, default=20
            Maximum translation (in highres voxels)
        rotation : float, default=15
            Maximum rotation (in degrees)
        zoom : float, default=0.15
            Maximum scaling below/above 1
        shear : float, default=0.012
            Maximum shear factor
        gmm_fwhm : float, default=10
            Full-width at half-maximum of the within-tissue smoothing kernel
        bias_amplitude : float, default=0.5
            Amplitude of the bias field (larger -> stronger bias)
        bias_fwhm : float, default=50
            Full-width at half-maximum of the bias field
            (larger -> smoother bias field)
        gamma : float, default=0.6
            Standard deviation of the log-normal distribution from
            which the gamma exponent is sampled.
        motion_fwhm : float, default=3
            Maximum value of the full-width at half-maximum of the
            smoothing filter simulating slice profile + motion
        resolution : float, default=8
            Maximum voxel size in the thick-slice direction
        noise : float, default=48
            Maximum value of the noise standard deviation
            (assuming intensities in [0, 255])
        gfactor_amplitude, default=0.5
            Amplitude of the gfactor field (larger -> stronger bias)
            The gfactor maps modulates the noise std to simulate
            noise with non stationary variance (e.g. in parallel imaging).
        gfactor_fwhm, default=64
            Full-width at half-maximum of the gfactor field
            (larger -> smoother bias field)
        gmm_cat : {'labels', 'prob'}, default='labels'
            Use hard labels or soft probability to sample the GMM.
            Labels use less memory.
        bag : float, default=0.5
            Probability of sampling a bag/head class.
            The bag class is generated by randomly dilating a brain mask.
        droppable_labels : list[list[int]], optional
            Groups of labels that can be randomly dropped before
            synthesizing the image.
        predicted_labels : list[int], optional
            Labels that should be predicted.
            There can be more labels used for synthesis than predicted.
        patch_size : [list of] int, optional
            Extract a random patch from the image
        dtype : torch.dtype
        """
        super().__init__()
        self.gmm_cat = gmm_cat[0].lower()
        dtype = dtype or torch.get_default_dtype()

        self.droppable_labels = droppable_labels
        self.predicted_labels = predicted_labels

        # label 2 one hot
        self.pbag = bag
        self.bag = AddBagClass() if bag else (lambda x: x)
        self.to_onehot = LabelToOneHot(dtype=dtype)
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
        self.patch = RandomPatch(patch_size) if patch_size else None
        # one hot 2 label
        self.to_label = OneHotToLabel()
        # gmm
        self.mixture = HyperRandomGaussianMixture(
            nb_classes=None,
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
        self.smooth = RandomSmooth(
            iso=True,
            fwhm='uniform',
            fwhm_exp=motion_fwhm/2,
            fwhm_scale=motion_fwhm/3.46
        )
        # smooth and downsample
        self.lowres2d = RandomLowRes2D(
            resolution='uniform',
            resolution_exp=resolution/2+1,
            resolution_scale=resolution/3.46
        )
        self.lowres3d = RandomLowRes3D(
            resolution='uniform',
            resolution_exp=(resolution**0.33)/2+1,
            resolution_scale=(resolution**0.33)/3.46
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

    def preprocess_labels(self, s):
        # s0 <- labels to predict
        # s  <- labels to model
        s0 = s

        all_labels = set(s.unique().tolist())
        if self.droppable_labels:
            # remove labels that are not modelled
            # E.g., this could be all non-brain labels (to model skull
            # stripping) or all left labels (to model hemi).
            ngroups = len(self.droppable_labels)
            group = torch.randint(ngroups+1, [])
            if group > 0:
                dropped_labels = self.droppable_labels[group-1]
                s[utils.isin(s, dropped_labels)] = 0
                all_labels -= set(dropped_labels)
        s = utils.merge_labels(s, sorted(all_labels))
        nb_labels_sampled = len(all_labels)

        if self.predicted_labels:
            predicted_labels = self.predicted_labels
            if isinstance(self.predicted_labels, int):
                predicted_labels = list(range(predicted_labels+1))
            # remove labels that are not predicted
            s0[utils.isin(s0, predicted_labels).bitwise_not_()] = 0
            s0 = utils.merge_labels(s0, sorted([0, *predicted_labels]))
        else:
            predicted_labels = list(sorted(s0.unique().tolist()))[1:]
        nb_labels_predicted = len(predicted_labels) + 1

        return s, s0, nb_labels_sampled, nb_labels_predicted

    def forward(self, s, img=None, return_resolution=False):
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
        s, s0, n, n0 = self.preprocess_labels(s)

        if self.pbag and torch.rand([]) > 1-self.pbag:
            s = self.bag(s)
            n += 1
        if self.gmm_cat != 'l':
            s = self.to_onehot(s)
        if img is not None:
            s, s0, img = self.deform(s, s0, img)
            if self.patch:
                s, s0, img = self.patch(s, s0, img)
        else:
            s, s0 = self.deform(s, s0)
            if self.patch:
                s, s0 = self.patch(s, s0)
        # if self.gmm_cat == 'l':
        #     s = self.to_label(s)
        x = self.mixture(s, nb_classes=n)
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
        out = [x, s0]
        if img is not None:
            out += [img]
        if return_resolution:
            out += [vx]
        return tuple(out)



# default_params = {
#     'flair':
#     {
#         'te':0.02,
#         'tr':5,
#         'ti':1
#     },
#     'fse':
#     {
#         'te':0.02,
#         'tr':5
#     },
#     'mp2rage':
#     {
#         'tr':6.25,
#         'ti1':0.8,
#         'ti2':2.2,
#         'tx':None,
#         'te':None,
#         'fa':(4,5),
#         'pe_steps':160,
#         'inv_eff':0.96
#     },
#     'mprage':
#     {
#         'tr':2.3,
#         'ti':0.9,
#         'tx':6e-3,
#         'te':3e-3,
#         'fa':9,
#         'pe_steps':160,
#         'inv_eff':0.96
#     },
#     'spgr':
#     {
#         'te':0,
#         'tr':25e-3,
#         'fa':20
#     },
#     'gre':
#     {
#         'te':0,
#         'tr':25e-3,
#         'fa':20,
        
#     }
# }


# class SynthQMRI(SynthMRI):
#     """
#     Modified version of SynthMRI generator, where a set of labels of associated parameters (R1/R2s etc)
#     are used to generate synthetic QMRI maps which can then generate physically plausible synthetic scans.
#     """
#     def __init__(self, pd=None, mt=None, r1=None, r2s=None, sequence='mprage',
#                  receive_exp=None,
#                  receive_scale=None,
#                  te_exp=None,
#                  te_scale=None,
#                  tr_exp=None,
#                  tr_scale=None,
#                  ti_exp=None, # can be tuple for mp2rage
#                  ti_scale=None,
#                  fa_exp=None,
#                  fa_scale=None,
#                  transmit_exp=None,
#                  transmit_scale=None,
#                  tx_exp=None,
#                  tx_scale=None,
#                  pe_steps_exp=None,
#                  pe_steps_scale=None,
#                  inv_eff_exp=None,
#                  inv_eff_scale=None,
#                  channel=1,
#                  vel_amplitude=3,
#                  vel_fwhm=20,
#                  translation=20,
#                  rotation=15,
#                  zoom=0.15,
#                  shear=0.012,
#                  gmm_fwhm=10,
#                  bias_amplitude=0.5,
#                  bias_fwhm=50,
#                  gamma=0.01,
#                  motion_fwhm=3,
#                  resolution=8,
#                  noise=48,
#                  gfactor_amplitude=0.5,
#                  gfactor_fwhm=64,
#                  gmm_cat='labels',
#                  bag=0.5,
#                  droppable_labels=None,
#                  predicted_labels=None,
#                  dtype=None):

#         super().__init__(channel=1,
#                         vel_amplitude=vel_amplitude,
#                         vel_fwhm=vel_fwhm,
#                         translation=translation,
#                         rotation=rotation,
#                         zoom=zoom,
#                         shear=shear,
#                         gmm_fwhm=gmm_fwhm,
#                         bias_amplitude=bias_amplitude,
#                         bias_fwhm=bias_fwhm,
#                         gamma=gamma,
#                         motion_fwhm=motion_fwhm,
#                         resolution=resolution,
#                         noise=noise,
#                         gfactor_amplitude=gfactor_amplitude,
#                         gfactor_fwhm=gfactor_fwhm,
#                         gmm_cat=gmm_cat,
#                         bag=bag,
#                         droppable_labels=droppable_labels,
#                         predicted_labels=predicted_labels,
#                         dtype=dtype)

#         # now change self.mixture, either to HyperRandomMixture or to RandomMixture and add self.qmri generator from ni.tools.qmri...
#         # currently ditching sigma values and just using fixed mu with random fwhm 
#         self.sequence = sequence
#         self.mixture = None
#         self.mixture_pd = HyperRandomGaussianMixture(
#             nb_classes=None,
#             nb_channels=channel,
#             means='uniform',
#             means_exp=pd,
#             means_scale=1e-5*pd,
#             scales='uniform',
#             scales_exp=1e-1*pd,
#             scales_scale=1e-5*pd,
#             fwhm='uniform',
#             fwhm_exp=gmm_fwhm/2,
#             fwhm_scale=gmm_fwhm/3.46,
#             background_zero=True,
#             dtype=dtype,
#         )
#         self.mixture_mt = HyperRandomGaussianMixture(
#             nb_classes=None,
#             nb_channels=channel,
#             means='uniform',
#             means_exp=mt,
#             means_scale=1e-5*mt,
#             scales='uniform',
#             scales_exp=1e-1*mt,
#             scales_scale=1e-5*mt,
#             fwhm='uniform',
#             fwhm_exp=gmm_fwhm/2,
#             fwhm_scale=gmm_fwhm/3.46,
#             background_zero=True,
#             dtype=dtype,
#         )
#         self.mixture_r1 = HyperRandomGaussianMixture(
#             nb_classes=None,
#             nb_channels=channel,
#             means='uniform',
#             means_exp=r1,
#             means_scale=1e-5*r1,
#             scales='uniform',
#             scales_exp=1e-1*r1,
#             scales_scale=1e-5*r1,
#             fwhm='uniform',
#             fwhm_exp=gmm_fwhm/2,
#             fwhm_scale=gmm_fwhm/3.46,
#             background_zero=True,
#             dtype=dtype,
#         )
#         self.mixture_r2s = HyperRandomGaussianMixture(
#             nb_classes=None,
#             nb_channels=channel,
#             means='uniform',
#             means_exp=r2s,
#             means_scale=1e-5*r2s,
#             scales='uniform',
#             scales_exp=1e-1*r2s,
#             scales_scale=1e-5*r2s,
#             fwhm='uniform',
#             fwhm_exp=gmm_fwhm/2,
#             fwhm_scale=gmm_fwhm/3.46,
#             background_zero=True,
#             dtype=dtype,
#         )

#     def forward(self, s, img=None, return_resolution=False):
#         """

#         Parameters
#         ----------
#         s : (batch, 1, *shape) tensor
#             Tensor of labels

#         Returns
#         -------
#         x : (batch, channel, *shape) tensor
#             Multi-channel image

#         """
#         s, s0, n, n0 = self.preprocess_labels(s)
#         # if self.pbag and torch.rand([]) > 1-self.pbag:
#         #     s = self.bag(s)
#         #     n += 1
#         s = self.to_onehot(s)
#         if img is not None:
#             s, s0, img = self.deform(s, s0, img)
#         else:
#             s, s0 = self.deform(s, s0)
#         if self.gmm_cat == 'l':
#             s = self.to_label(s)
#         pd = self.mixture_pd(s, nb_classes=n)
#         mt = self.mixture_mt(s, nb_classes=n)
#         r1 = self.mixture_r1(s, nb_classes=n)
#         r2s = self.mixture_r2s(s, nb_classes=n)
#         ni.plot.show_slices(torch.cat([pd,mt,r1,r2s],1)[0].permute(1,2,3,0))
#         # if isinstance(self.sequence, list):
#         #     pick one randomly
#         # else:
#         #     sequence = self.sequence
#         sequence = self.sequence
#         if sequence == 'mprage':
#             x = qmri.generators.mprage(pd, r1, r2s,
#             # add here the random sampled parameters
#             )
#         elif sequence == 'mp2rage':
#             x = qmri.generators.mp2rage(pd, r1, r2s,
#             # add here the random sampled parameters
#             )
#         # elif sequence == 'gre':
#         #     x = qmri.gre(pd, r1, r2s, mt,
#         #     # add here the random sampled parameters
#         #     )
#         elif sequence == 'fse':
#             x = qmri.generators.fse(pd, r1, r2s, mt,
#             # add here the random sampled parameters
#             )
#         elif sequence == 'flair':
#             x = qmri.generators.flair(pd, r1, r2s, mt,
#             # add here the random sampled parameters
#             )
#         elif sequence == 'spgr' or sequence == 'flash':
#             x = qmri.generators.spgr(pd, r1, r2s, mt,
#             # add here the random sampled parameters
#             )
#         x.clamp_(0, 255)
#         x = self.gamma(x)
#         x = self.bias(x)
#         x = self.smooth(x)
#         e = self.noise(x)
#         if torch.rand([]) > 0.5:
#             x, vx = self.lowres2d(x, noise=e, return_resolution=True)
#         else:
#             x, vx = self.lowres3d(x, noise=e, return_resolution=True)
#         x = self.rescale(x)
#         out = [x, s0]
#         if img is not None:
#             out += [img]
#         if return_resolution:
#             out += [vx]
#         return tuple(out)

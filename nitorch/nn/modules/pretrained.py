import torch
from nitorch.core import py, utils
from nitorch import io, spatial
from .cnn import NeuriteUNet
from .norm import BatchNorm
from .conv import Conv
from .spatial import GridExp, GridResize, GridPull
from ..base import Module


class SynthUNet(NeuriteUNet):
    """Base for SynthStuff networks

    This architecture has been carefully designed and evaluated for
    the SynthSeg paper by B. Billot et al. If you use this architecture
    in your work, please cite their paper.

    References
    ----------
    ..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
           MRI Scans of any Contrast and Resolution"
          Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
          Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
          https://arxiv.org/abs/2107.09559
    """
    model_url = None

    def __init__(self,
                 dim=3,
                 in_channels=1,
                 out_channels=32,
                 nb_levels=5,
                 skip_decoder_levels=0,
                 kernel_size=3,
                 nb_feat=24,
                 feat_mult=2,
                 pool_size=2,
                 padding='same',
                 dilation_rate_mult=1,
                 activation='elu',
                 residual=False,
                 final_activation='softmax',
                 final_kernel_size=1,
                 nb_conv_per_level=2,
                 dropout=0,
                 batch_norm='batch',
                 verbose=False):
        super().__init__(
            dim, in_channels, out_channels, nb_levels, skip_decoder_levels,
            kernel_size, nb_feat, feat_mult, pool_size, padding,
            dilation_rate_mult, activation, residual, final_activation,
            final_kernel_size, nb_conv_per_level, dropout, batch_norm
        )
        self.verbose = verbose

    @classmethod
    def download_tf_weights(cls, url=None, output_path=None, verbose=False):
        if verbose:
            print('Downloading weights... ', end='', flush=True)
        url = url or cls.model_url
        if not output_path:
            import tempfile
            _, output_path = tempfile.mkstemp(suffix='.h5')
        try:
            import wget
        except ImportError:
            wget = None
        if not wget:
            raise ImportError('wget must be installed to download tf weights')
        output_path = wget.download(url, output_path)
        if verbose:
            print('done.', flush=True)
        return output_path

    def load_tf_weights(self, path_to_h5=None):
        def _set_weights(module, conv_keys, bn_keys, f, prefix='unet'):
            if isinstance(module, Conv):
                if conv_keys:
                    key = conv_keys.pop(0)
                else:
                    # we might have reached the final "feat 2 class" conv
                    key = 'unet_likelihood'
                try:
                    kernel = torch.as_tensor(f[key][key]['kernel:0'],
                                         **utils.backend(module.weight))
                except:
                    kernel = torch.as_tensor(f[key][key+'_1']['kernel:0'],
                                         **utils.backend(module.weight))
                kernel = utils.movedim(kernel, [-1, -2], [0, 1])
                module.weight.copy_(kernel)
                try:
                    bias = torch.as_tensor(f[key][key]['bias:0'],
                                       **utils.backend(module.bias))
                except:
                    bias = torch.as_tensor(f[key][key+'_1']['bias:0'],
                                       **utils.backend(module.bias))
                module.bias.copy_(bias)
            elif isinstance(module, BatchNorm):
                key = bn_keys.pop(0)
                try:
                    beta = torch.as_tensor(f[key][key]['beta:0'],
                                       **utils.backend(module.norm.bias))
                except:
                    beta = torch.as_tensor(f[key][key+'_1']['beta:0'],
                                       **utils.backend(module.norm.bias))
                module.norm.bias.copy_(beta)
                try:
                    gamma = torch.as_tensor(f[key][key]['gamma:0'],
                                        **utils.backend(module.norm.weight))
                except:
                    gamma = torch.as_tensor(f[key][key+'_1']['gamma:0'],
                                        **utils.backend(module.norm.weight))
                module.norm.weight.copy_(gamma)
                try:
                    mean = torch.as_tensor(f[key][key]['moving_mean:0'],
                                       **utils.backend(module.norm.running_mean))
                except:
                    mean = torch.as_tensor(f[key][key+'_1']['moving_mean:0'],
                                       **utils.backend(module.norm.running_mean))
                module.norm.running_mean.copy_(mean)
                try:
                    var = torch.as_tensor(f[key][key]['moving_variance:0'],
                                      **utils.backend(module.norm.running_var))
                except:
                    var = torch.as_tensor(f[key][key+'_1']['moving_variance:0'],
                                      **utils.backend(module.norm.running_var))
                module.norm.running_var.copy_(var)
            else:
                for name, child in module.named_children():
                    _set_weights(child, conv_keys, bn_keys, f, f'{prefix}.{name}')

        if not path_to_h5:
            path_to_h5 = self.download_tf_weights(verbose=self.verbose)

        if self.verbose:
            print('Loading weights... ', end='', flush=True)
        try:
            import h5py
        except ImportError:
            h5py = None
        if not h5py:
            raise ImportError('h5py must be installed to load tf weights')
        with h5py.File(path_to_h5, 'r') as f, torch.no_grad():
            if 'model_weights' in list(f.keys()):
                f = f['model_weights']
            # The SynthSeg unet only has conv and batch norm weights
            down_conv_keys = sorted([key for key in f if 'conv_downarm' in key])
            up_conv_keys = sorted([key for key in f if 'conv_uparm' in key])
            down_bn_keys = sorted([key for key in f if 'bn_down' in key])
            up_bn_keys = sorted([key for key in f if 'bn_up' in key])
            conv_keys = [*down_conv_keys, *up_conv_keys]
            bn_keys = [*down_bn_keys, *up_bn_keys]

            _set_weights(self, conv_keys, bn_keys, f)
        if self.verbose:
            print('done.', flush=True)


class SynthSegUNet(SynthUNet):
    """
    PyTorch port of [SynthSeg](https://github.com/BBillot/SynthSeg)

    If you use this network and/or the pretrained weights in your work,
    please cite the corresponding papers.

    References
    ----------
    ..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
           MRI Scans of any Contrast and Resolution"
          Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
          Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
          https://arxiv.org/abs/2107.09559
    ..[2] "A Learning Strategy for Contrast-agnostic MRI Segmentation"
          Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl,
          Juan Eugenio Iglesias*, Adrian V. Dalca*
          MIDL 2020
          https://arxiv.org/abs/2003.01995
    ..[3] "Partial Volume Segmentation of Brain MRI Scans of any Resolution
           and Contrast"
          Benjamin Billot, Eleanor D. Robinson, Adrian V. Dalca, Juan Eugenio Iglesias
          MICCAI 2020
          https://arxiv.org/abs/2004.10221

    """

    model_url = 'https://github.com/BBillot/SynthSeg/raw/master/models/SynthSeg.h5'
    lookup_url = ('https://raw.githubusercontent.com/BBillot/SynthSeg/'
                  '492453421020d66ebf0e11bf0cc266754d21b895/data/'
                  'labels%20table.txt')

    def __init__(self,
                 dim=3,
                 in_channels=1,
                 out_channels=32,
                 nb_levels=5,
                 kernel_size=3,
                 skip_decoder_levels=0,
                 nb_feat=24,
                 feat_mult=2,
                 pool_size=2,
                 padding='same',
                 dilation_rate_mult=1,
                 activation='elu',
                 residual=False,
                 final_activation='softmax',
                 final_kernel_size=1,
                 nb_conv_per_level=2,
                 dropout=0,
                 batch_norm='batch',
                 verbose=False):
        super().__init__(
            dim, in_channels, out_channels, nb_levels, skip_decoder_levels,
            kernel_size, nb_feat, feat_mult, pool_size, padding,
            dilation_rate_mult, activation, residual, final_activation,
            final_kernel_size, nb_conv_per_level, dropout, batch_norm, verbose
        )

    def relabel(self, s, lookup=None):
        """

        Parameters
        ----------
        s : tensor[int]

        Returns
        -------
        s : tensor[int]

        """
        if not lookup or isinstance(lookup, str):
            lookup = self.load_lookup(lookup)
        with torch.no_grad():
            out = torch.zeros_like(s)
            for i, o in enumerate(lookup):
                out[s == i] = o
        return out

    @classmethod
    def download_lookup(cls, url=None, output_path=None):
        """Download lookup table for SynthSeg labels"""
        url = url or cls.lookup_url
        if not output_path:
            import tempfile
            _, output_path = tempfile.mkstemp(suffix='.txt')

        import urllib
        with urllib.request.urlopen(url) as r:
            with open(output_path, 'w') as f:
                for line in r:
                    decoded_line = line.decode("utf-8")
                    f.write(decoded_line)
        return output_path

    @classmethod
    def load_lookup(cls, path_to_txt=None):
        """Load lookup table for SynthSeg from file"""
        # This assumes that the file is "labels table.txt"
        # and is really not robust
        path_to_txt = path_to_txt or cls.download_lookup()
        lookup = []
        with open(path_to_txt) as f:
            for _, _ in zip(range(9), f):
                # skip first 9 lines
                pass
            for line in f:
                if line:
                    lookup.append(int(line.split(' ')[0]))
        return lookup


class SynthPreproc:

    @classmethod
    def addnoise(cls, x):
        return cls.addnoise_(x.clone())

    @classmethod
    def addnoise_(cls, x):
        v = x.unique().sort().values
        if v.shape[0] > 1:
            v = v[1:] - v[:-1]
        v = utils.quantile(v, 0.005).item()
        mask = torch.isfinite(x).bitwise_not_().bitwise_or_(x == 0)
        x.masked_fill_(mask, 0)
        x.addcmul_(mask.to(x.dtype), torch.rand_like(x), value=v)
        return x

    @classmethod
    def preproc(cls, x):
        return cls.preproc_(x.clone())

    @classmethod
    def preproc_(cls, x):
        mn, mx = utils.quantile(x, [0.005, 0.995], keepdim=True).unbind(-1)
        x = x.max(mn).min(mx).sub_(mn).div_(mx - mn)
        return x

    @classmethod
    def crop(cls, x):
        crop = [s % 2**5 for s in x.shape]
        crop = tuple(slice(c//2, (c//2-c) or None) for c in crop)
        x = x[crop]
        return x, crop

    @classmethod
    def pad(cls, x, oshape, crop):
        if oshape == x.shape:
            return x
        ox = x.new_zeros(oshape)
        ox[tuple(crop)] = x
        return ox

    @classmethod
    def reorient(cls, x, affine, target='RAS'):
        affine, x = spatial.affine_reorient(affine, x, target)
        return x, affine

    @classmethod
    def resize(cls, x, affine, target_vx=1):
        target_vx = utils.make_vector(target_vx, x.dim(), **utils.backend(affine))
        vx = spatial.voxel_size(affine)
        factor = vx / target_vx
        fwhm = 0.25 * factor.reciprocal()
        fwhm[factor > 1] = 0
        x = spatial.smooth(x, fwhm=fwhm.tolist(), dim=3)
        x, affine = spatial.resize(x[None, None], factor.tolist(), affine=affine)
        x = x[0, 0]
        return x, affine


class SynthSegmenter(SynthSegUNet):
    """
    A high-level class that loads, preprocess and segments an MRI using
    SynthSeg.
    """

    def __init__(self, weights=None, tf_weights=None, verbose=True, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.eval()
        if weights:
            self.load_state_dict(weights)
        else:
            self.load_tf_weights(tf_weights)

    def forward(self, x, affine=None):
        """

        Parameters
        ----------
        x : (X, Y, Z) tensor or str
        affine : (4, 4) tensor, optional

        Returns
        -------
        seg : (32, oX, oY, oZ) tensor
            Segmentation
        resliced : (oX, oY, oZ) tensor
            Input resliced to 1 mm RAS
        affine : (4, 4) tensor
            Output orientation matrix

        """
        if self.verbose:
            print('Preprocessing... ', end='', flush=True)
        if isinstance(x, str):
            x = io.map(x)
        if isinstance(x, io.MappedArray):
            if affine is None:
                affine = x.affine
                x = x.fdata()
                x = x.reshape(x.shape[:3])
        x = SynthPreproc.addnoise(x)
        if affine is not None:
            affine, x = spatial.affine_reorient(affine, x, 'RAS')
            vx = spatial.voxel_size(affine)
            fwhm = 0.25*vx.reciprocal()
            fwhm[vx > 1] = 0
            x = spatial.smooth(x, fwhm=fwhm.tolist(), dim=3)
            x, affine = spatial.resize(x[None, None], vx.tolist(),
                                       affine=affine)
            x = x[0, 0]
        oshape = x.shape
        x, crop = SynthPreproc.crop(x)
        x = SynthPreproc.preproc(x)[None, None]
        if self.verbose:
            print('done.', flush=True)
            print('Segmenting... ', end='', flush=True)
        s, x = super().forward(x)[0], x[0, 0]
        if self.verbose:
            print('done.', flush=True)
            print('Postprocessing... ', end='', flush=True)
        s = self.relabel(s.argmax(0))
        x = SynthPreproc.pad(x, oshape, crop)
        s = SynthPreproc.pad(s, oshape, crop)
        if self.verbose:
            print('done.', flush=True)
        return s, x, affine


class SynthSRUNet(SynthUNet):
    """
    PyTorch port of [SynthSR](https://github.com/BBillot/SynthSR)

    If you use this network and/or the pretrained weights in your work,
    please cite both papers below. Ref [1] describes the super-resolution
    network, while ref [2] describes the strategy that allows a single
    contrast- and resolution- agnostic network to be trained.


    References
    ----------
    ..[1] "Joint super-resolution and synthesis of 1 mm isotropic MP-RAGE
           volumes from clinical MRI exams with scans of different orientation,
           resolution and contrast"
          Iglesias JE, Billot B, Balbastre Y, Tabari A, Conklin J,
          RG Gonzalez, Alexander DC, Golland P, Edlow B, Fischl B
          NeuroImage (2021)
          https://arxiv.org/abs/2012.13340
    ..[2] "SynthSeg: Domain Randomisation for Segmentation of Brain
           MRI Scans of any Contrast and Resolution"
          Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
          Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
          https://arxiv.org/abs/2107.09559
    """

    model_url = 'https://github.com/BBillot/SynthSR/raw/main/models/SynthSR_v10_210712.h5'

    def __init__(self,
                 dim=3,
                 in_channels=1,
                 out_channels=1,
                 nb_levels=5,
                 kernel_size=3,
                 skip_decoder_levels=0,
                 nb_feat=24,
                 feat_mult=2,
                 pool_size=2,
                 padding='same',
                 dilation_rate_mult=1,
                 activation='elu',
                 residual=False,
                 final_activation=None,
                 final_kernel_size=1,
                 nb_conv_per_level=2,
                 dropout=0,
                 batch_norm='batch',
                 verbose=False):
        super().__init__(
            dim, in_channels, out_channels, nb_levels, skip_decoder_levels,
            kernel_size, nb_feat, feat_mult, pool_size, padding,
            dilation_rate_mult, activation, residual, final_activation,
            final_kernel_size, nb_conv_per_level, dropout, batch_norm, verbose
        )


class SynthHyperFineUNet(SynthSRUNet):
    """HyperFine variant of SynthSR"""

    model_url = 'https://github.com/BBillot/SynthSR/raw/main/models/SynthSR_v10_210712_hyperfine.h5'


class SynthResolver(SynthSRUNet):
    """
    A high-level class that loads, preprocess and translates an MRI using
    SynthSR.
    """

    def __init__(self, weights=None, tf_weights=None, verbose=True):
        super().__init__(verbose=verbose)
        self.eval()
        if weights:
            self.load_state_dict(weights)
        else:
            self.load_tf_weights(tf_weights)

    def forward(self, x, affine=None):
        """

        Parameters
        ----------
        x : (X, Y, Z) tensor or str
        affine : (4, 4) tensor, optional

        Returns
        -------
        super : (oX, oY, oZ) tensor
            Super-resolve + synthesized T1w
        resliced : (oX, oY, oZ) tensor
            Input resliced to 1 mm RAS
        affine : (4, 4) tensor
            Output orientation matrix

        """
        if self.verbose:
            print('Preprocessing... ', end='', flush=True)
        if isinstance(x, str):
            x = io.map(x)
        if isinstance(x, io.MappedArray):
            if affine is None:
                affine = x.affine
                x = x.fdata()
                x = x.reshape(x.shape[:3])
        x = SynthPreproc.addnoise(x)
        if affine is not None:
            affine, x = spatial.affine_reorient(affine, x, 'RAS')
            vx = spatial.voxel_size(affine)
            fwhm = (0.25*vx)
            fwhm[vx > 0] = 0
            x = spatial.smooth(x, fwhm=fwhm.tolist(), dim=3)
            x, affine = spatial.resize(x[None, None], vx.tolist(),
                                       affine=affine, anchor='f')
            x = x[0, 0]
        oshape = x.shape
        x, crop = SynthPreproc.crop(x)
        x = SynthPreproc.preproc(x)[None, None]
        if self.verbose:
            print('done.', flush=True)
            print('Super-resolving... ', end='', flush=True)
        s, x = super().forward(x)[0, 0], x[0, 0]
        if self.verbose:
            print('done.', flush=True)
        s = SynthPreproc.pad(s, oshape, crop)
        x = SynthPreproc.pad(x, oshape, crop)
        return s, x, affine


SynthSegmenter.__doc__ += SynthSegUNet.__doc__
SynthResolver.__doc__ += SynthSRUNet.__doc__


_synthmorph_feat = (256,) * 10


class SynthMorphUNet(NeuriteUNet):
    """Base for SynthMorph networks

    This architecture has been carefully designed and evaluated for
    the SynthMorph paper by M. Hoffmann et al. If you use this architecture
    in your work, please cite their paper.

    References
    ----------
    ..[1] "SynthMorph: learning contrast-invariant registration without
           acquired images"
          Malte Hoffmann, Benjamin Billot, Douglas N Greve,
          Juan Eugenio Iglesias, Bruce Fischl, Adrian V Dalca
          https://arxiv.org/abs/2004.10282
    """
    model_url = None

    def __init__(self,
                 dim=3,
                 in_channels=2,
                 out_channels=None,
                 nb_levels=4,
                 skip_decoder_levels=0,
                 kernel_size=3,
                 nb_feat=_synthmorph_feat,
                 feat_mult=1,
                 pool_size=2,
                 padding='same',
                 dilation_rate_mult=1,
                 activation=torch.nn.LeakyReLU(0.2),
                 residual=False,
                 final_activation=None,
                 final_kernel_size=None,
                 nb_conv_per_level=1,
                 dropout=0,
                 batch_norm=None,
                 verbose=False):
        out_channels = out_channels or dim
        final_kernel_size = final_kernel_size or kernel_size
        super().__init__(
            dim, in_channels, out_channels, nb_levels, skip_decoder_levels,
            kernel_size, nb_feat, feat_mult, pool_size, padding,
            dilation_rate_mult, activation, residual, final_activation,
            final_kernel_size, nb_conv_per_level, dropout, batch_norm
        )
        self.verbose = verbose

    @classmethod
    def download_tf_weights(cls, url=None, output_path=None, verbose=False):
        if verbose:
            print('Downloading weights... ', end='', flush=True)
        url = url or cls.model_url
        if not output_path:
            import tempfile
            _, output_path = tempfile.mkstemp(suffix='.h5')
        try:
            import wget
        except ImportError:
            wget = None
        if not wget:
            raise ImportError('wget must be installed to download tf weights')
        output_path = wget.download(url, output_path)
        if verbose:
            print('done.', flush=True)
        return output_path

    def load_tf_weights(self, path_to_h5=None):
        def _set_weights(module, conv_keys, f, prefix='unet'):
            # print(prefix)
            if isinstance(module, Conv):
                if conv_keys:
                    key = conv_keys.pop(0)
                else:
                    # we might have reached the final "feat 2 class" conv
                    key = 'vxm_dense_flow'
                try:
                    kernel = torch.as_tensor(f[key][key]['kernel:0'],
                                         **utils.backend(module.weight))
                except:
                    kernel = torch.as_tensor(f[key][key+'_1']['kernel:0'],
                                         **utils.backend(module.weight))
                kernel = utils.movedim(kernel, [-1, -2], [0, 1])
                module.weight.copy_(kernel)
                try:
                    bias = torch.as_tensor(f[key][key]['bias:0'],
                                       **utils.backend(module.bias))
                except:
                    bias = torch.as_tensor(f[key][key+'_1']['bias:0'],
                                       **utils.backend(module.bias))
                module.bias.copy_(bias)
            else:
                for name, child in module.named_children():
                    _set_weights(child, conv_keys, f, f'{prefix}.{name}')

        if not path_to_h5:
            path_to_h5 = self.url
        if path_to_h5.startswith('http'):
            path_to_h5 = self.download_tf_weights(path_to_h5, verbose=self.verbose)

        if self.verbose:
            print('Loading weights... ', end='', flush=True)
        try:
            import h5py
        except ImportError:
            h5py = None
        if not h5py:
            raise ImportError('h5py must be installed to load tf weights')
        with h5py.File(path_to_h5, 'r') as f, torch.no_grad():
            f = f['model_weights']
            # The SynthSeg unet only has conv and batch norm weights
            down_conv_keys = sorted([key for key in f if 'enc_conv' in key and not key.endswith('activation')])
            up_conv_keys = sorted([key for key in f if 'dec_conv' in key and not key.endswith('activation')])
            final_conv_keys = sorted([key for key in f if 'final_conv' in key and not key.endswith('activation')])
            conv_keys = [*down_conv_keys, *up_conv_keys, *final_conv_keys]

            _set_weights(self, conv_keys, f)
        if self.verbose:
            print('done.', flush=True)


class SynthMorphNet(Module):
    """SynthMorph registration networks (UNet + Transformer)

    References
    ----------
    ..[1] "SynthMorph: learning contrast-invariant registration without
           acquired images"
          Malte Hoffmann, Benjamin Billot, Douglas N Greve,
          Juan Eugenio Iglesias, Bruce Fischl, Adrian V Dalca
          https://arxiv.org/abs/2004.10282
    """

    def __init__(self,
                 dim=3,
                 in_channels_src=1,
                 in_channels_tgt=1,
                 nb_levels=4,
                 steps=7,
                 vel_level=0,
                 int_level=1,
                 kernel_size=3,
                 nb_feat=_synthmorph_feat,
                 feat_mult=1,
                 pool_size=2,
                 padding='same',
                 dilation_rate_mult=1,
                 activation=torch.nn.LeakyReLU(0.2),
                 residual=False,
                 final_activation=None,
                 final_kernel_size=None,
                 nb_conv_per_level=1,
                 dropout=0,
                 batch_norm=None,
                 bound_image='dct2',
                 bound_vel='nearest',
                 verbose=False):
        super().__init__()
        self.verbose = verbose
        in_channels = in_channels_src + in_channels_tgt
        out_channels = dim
        skip_decoder_levels = vel_level
        final_kernel_size = final_kernel_size or kernel_size
        self.unet = SynthMorphUNet(
            dim, in_channels, out_channels, nb_levels, skip_decoder_levels,
            kernel_size, nb_feat, feat_mult, pool_size, padding,
            dilation_rate_mult, activation, residual, final_activation,
            final_kernel_size, nb_conv_per_level, dropout, batch_norm, verbose
        )

        if vel_level != int_level:
            self.resize_vel = GridResize(bound=bound_vel,
                                         factor=2**(vel_level-int_level),
                                         type='disp')
        self.exp = GridExp(steps=steps, bound=bound_vel, displacement=True)
        if int_level != 0:
            self.resize_grid = GridResize(bound=bound_vel,
                                          factor=2**int_level,
                                          type='disp')
        self.pull = GridPull(bound=bound_image)

    def forward(self, source, target, source_seg=None, target_seg=None,
                *, _loss=None, _metric=None):

        vel = self.unet(torch.cat([source, target], dim=1))
        if hasattr(self, 'resize_vel'):
            vel = self.resize_vel(vel)
        grid = self.exp(vel)
        if hasattr(self, 'resize_grid'):
            grid = self.resize_grid(vel, output_shape=source.shape[2:])
        grid = spatial.add_identity_grid_(grid)
        deformed_source = self.pull(source, grid)

        if source_seg is not None:
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = None

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     velocity=[vel],
                     segmentation=[deformed_source_seg, target_seg])

        if source_seg is None:
            return deformed_source, vel, grid
        else:
            return deformed_source, deformed_source_seg, vel, grid


class SynthMorphRegister(SynthMorphNet):
    """
    A high-level class that loads, preprocess and warps MRIs using
    SynthMorph.
    """
    url = None

    def __init__(self, weights=None, tf_weights=None, verbose=True):
        super().__init__(verbose=verbose)
        self.eval()
        if weights:
            self.unet.load_state_dict(weights)
        else:
            self.unet.load_tf_weights(tf_weights or self.url)

    def load(self, x, affine=None):
        if isinstance(x, str):
            x = io.map(x)
        if isinstance(x, io.MappedArray):
            if affine is None:
                affine = x.affine
                x = x.fdata()
                x = x.reshape(x.shape[:3])
        affine_original = affine
        x_original = x.shape
        if affine is not None:
            affine, x = spatial.affine_reorient(affine, x, 'RAS')
            vx = spatial.voxel_size(affine)
            x, affine = spatial.resize(x[None, None], vx.tolist(), affine=affine)
            x = x[0, 0]
        return x, affine, x_original, affine_original

    def forward(self, source, target, source_affine=None, target_affine=None):
        """

        Parameters
        ----------
        source : (sX, sY, sZ) tensor or str
        target : (tX, tY, tZ) tensor or str
        source_affine : (4, 4) tensor, optional
        target_affine : (4, 4) tensor, optional

        Returns
        -------
        warped : (tX, tY, tZ) tensor
            Source warped to target
        velocity : (vX, vY, vZ, 3) tensor
            Stationary velocity field
        affine : (4, 4) tensor, optional
            Affine of the velocity space

        """
        if self.verbose:
            print('Preprocessing... ', end='', flush=True)
        source, source_affine, source_orig, source_affine_orig \
            = self.load(source, source_affine)
        target, target_affine, target_orig, target_affine_orig \
            = self.load(target, target_affine)
        source = spatial.reslice(source, source_affine,
                                 target_affine, target.shape)
        if self.verbose:
            print('done.', flush=True)
            print('Registering... ', end='', flush=True)
        source = source[None, None]
        target = target[None, None]
        warped, vel, grid = super().forward(source, target)
        if self.verbose:
            print('done.', flush=True)
        del source, target, warped
        vel = vel[0]
        grid = grid[0]
        grid -= spatial.identity_grid(grid.shape[:-1],
                                      dtype=grid.dtype,
                                      device=grid.device)
        right_affine = target_affine.inverse() @ target_affine_orig
        right_affine = spatial.affine_grid(right_affine, target_orig.shape)
        grid = spatial.grid_pull(utils.movedim(grid, -1, 0), right_affine,
                                 bound='nearest', extrapolate=True)
        grid = utils.movedim(grid, 0, -1).add_(right_affine)
        left_affine = source_affine_orig.inverse() @ target_affine
        grid = spatial.affine_matvec(left_affine, grid)
        warped = spatial.grid_pull(source_orig, grid)
        return warped, vel, target_affine


class ShapeMorphRegister(SynthMorphRegister):
    url = ('https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/'
           'synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5')


class BrainMorphRegister(SynthMorphRegister):
    url = ('https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/'
           'synthmorph/brains-dice-vel-0.5-res-16-256f.h5')
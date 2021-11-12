import torch
from nitorch.core.optionals import try_import
from .. import generators, preproc
import os
import math
import gzip
import random
wget = try_import('wget')
appdirs = try_import('appdirs')
np = try_import('numpy')


def wget_check():
    if not wget:
        raise ImportError('wget needed to download dataset')
    return wget


class MNIST:
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    base_train_images = 'train-images-idx3-ubyte.gz'
    base_train_labels = 'train-labels-idx1-ubyte.gz'
    base_test_images = 't10k-images-idx3-ubyte.gz'
    base_test_labels = 't10k-labels-idx1-ubyte.gz'
    datadir = appdirs.user_cache_dir('nitorch') if appdirs else '.'

    def _load1(self, base, magic=None):
        fname = os.path.join(self.datadir, base)
        if not os.path.exists(fname):
            os.makedirs(self.datadir, exist_ok=True)
            wget_check().download(self.url_base + base, fname)
        if not os.path.exists(fname):
            raise ValueError('Something went wrong')
        with gzip.open(fname, 'rb') as f:
            check_magic = np.frombuffer(f.read(4), dtype='>i4').item()
            if magic and check_magic != magic:
                raise RuntimeError(f'Bad magic number: expected {magic} '
                                   f'but got {check_magic}')
            n = np.frombuffer(f.read(4), dtype='>i4').item()
            if check_magic == 2051:
                nr = np.frombuffer(f.read(4), dtype='>i4').item()
                nc = np.frombuffer(f.read(4), dtype='>i4').item()
            else:
                nr = nc = 1
            dat = np.frombuffer(f.read(n*nr*nc), dtype='uint8')
            if check_magic == 2051:
                dat = dat.reshape([n, nr, nc])
            else:
                dat = dat.reshape([n])
        return dat

    def _load(self, mode='train'):
        base = self.base_train_images if mode == 'train' else self.base_test_images
        dat = self._load1(base, 2051)
        dat = torch.as_tensor(dat, dtype=torch.float32)
        base = self.base_train_labels if mode == 'train' else self.base_test_labels
        lab = self._load1(base, 2049)
        lab = torch.as_tensor(lab, dtype=torch.float32)
        return dat, lab

    def _shuffle(self, dat, lab):
        idx = list(range(len(dat)))
        random.shuffle(idx)
        dat = dat[idx]
        lab = lab[idx]
        return dat, lab

    def __init__(self, mode='train', shuffle='once', batch=1, nmax=None):
        self.mode = mode
        self.shuffle = shuffle
        self.batch = batch
        self.nmax = nmax

        self._all_images, self._all_labels = self._load(mode)
        if shuffle == 'once':
            self._all_images, self._all_labels \
                = self._shuffle(self._all_images, self._all_labels)
        if shuffle != 'always' and nmax:
            self._all_images = self._all_images[:nmax]
            self._all_labels = self._all_labels[:nmax]

    def to(self, **backend):
        self._all_images.to(**backend)
        self._all_labels.to(device=backend.pop('device', None))
        return self

    def __len__(self):
        nmax = self.nmax or len(self._all_images)
        return math.ceil(nmax / self.batch)

    def __iter__(self):
        all_images = self._all_images
        all_labels = self._all_labels
        if self.shuffle == 'always':
            all_images, all_labels = self._shuffle(all_images, all_labels)

        for n in range(len(self)):
            images = all_images[n*self.batch:(n+1)*self.batch]
            labels = all_labels[n*self.batch:(n+1)*self.batch]
            yield images.unsqueeze(1), labels


class AugmentedMNIST:

    def __init__(self, mnist=None, flip=False, affine=True, nonlin=True,
                 bias=True, noise=True, gamma=True, rescale=True,
                 return_seg=False, return_class=True, **backend):
        self.mnist = mnist or MNIST()
        self.mnist = self.mnist.to(**backend)
        self.flip = flip
        self.affine = affine
        self.nonlin = nonlin
        self.bias = bias
        self.noise = noise
        self.gamma = gamma
        self.rescale = rescale
        self.return_seg = return_seg
        self.return_class = return_class

    def to(self, **backend):
        self.mnist = self.mnist.to(**backend)
        return self

    def __len__(self):
        return len(self.mnist)

    def __iter__(self):
        for vol, lab in self.mnist:

            if self.flip:
                vol = generators.RandomFlip()(vol)
            if self.affine or self.nonlin:
                opt = dict(translation=False, rotation=self.affine,
                           zoom=self.affine, shear=self.affine,
                           image_bound='nearest')
                if not self.nonlin:
                    opt['vel_amplitude'] = 0
                else:
                    opt['vel_amplitude'] = 3
                    opt['vel_fwhm'] = 14
                vol = generators.RandomDeform(**opt)(vol)
            seg = vol.clone().div_(255)
            if self.bias:
                vol = generators.BiasFieldTransform(amplitude=0.05, fwhm=32)(vol)
            if self.rescale:
                vol = preproc.AffineQuantiles(0.001, 0.999)(vol)
            if self.gamma:
                vol = generators.RandomGammaCorrection()(vol)
            if self.noise:
                vol = generators.RandomChiNoise(ncoils=32)(vol)
            vol = preproc.AffineQuantiles(0, 0.95)(vol)
            out = [vol]
            if self.return_seg:
                out.append(seg)
            if self.return_class:
                out.append(lab)
            if len(out) == 1:
                out = out[0]
            else:
                out = tuple(out)
            yield out

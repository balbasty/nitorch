# general class for VAE. TODO:
# bottleneck layer, with both deterministic and variational
# VAE / AE models (just encoder/decoder and bottleneck)
# beta-VAE - need to make loss modules for different VAE training
# VQ-VAE
# some kind of anomaly detecting VAE for brain pathology
# implement consistency reg: https://arxiv.org/pdf/2105.14859.pdf
# eventually figure out DDPM and VQ-GAN...

import torch
from torch import nn as tnn
from nitorch.core import py
from .conv import Conv, ConvBlock
from .cnn import Encoder, Decoder, NeuriteEncoder, NeuriteDecoder, NeuriteConv
from nitorch.nn.base import Module, Sequential, ModuleList
from .linear import Linear


class VAE(Module):
    """
    Plain VAE class of encoder-latent-decoder
    """
    def __init__(self, dim, img_size, in_channels, latent_dim, out_channels=None, 
                 encoder=(16, 32, 64, 128, 256), decoder=None, neurite=False,
                 final_activation='Tanh', **kwargs):
        super().__init__()
        self.dim = dim
        if isinstance(img_size, int):
            img_size = [img_size] * dim
        out_channels = out_channels or in_channels
        encoder = list(encoder)
        if not decoder:
            decoder = list(encoder)
            decoder.reverse()
        # TODO: add options for full stack like UNet
        self.decoder_in_channels = decoder[0]
        if neurite:
            self.encoder = NeuriteEncoder(dim, in_channels, encoder, **kwargs)
            self.decoder = NeuriteDecoder(dim, decoder[0], decoder[1:], **kwargs)
            self.final = NeuriteConv(dim, decoder[-1], out_channels, activation=final_activation)
        else:
            self.encoder = Encoder(dim, in_channels, encoder, **kwargs)
            self.decoder = Decoder(dim, decoder[0], decoder[1:], **kwargs)
            self.final = ConvBlock(dim, decoder[-1], out_channels, activation=final_activation)
        shape = torch.tensor(img_size).unsqueeze(0).unsqueeze(0)
        for layer in self.encoder:
            shape = layer.shape(shape)
        self.out_shape = shape[2:]
        self.latent_mu = Linear(encoder[-1] * py.prod(self.shape), latent_dim)
        self.latent_sigma = Linear(encoder[-1] * py.prod(self.shape), latent_dim)
        self.decoder_input = Linear(latent_dim, decoder[0]*4)

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        mu = self.latent_mu(x)
        sigma = self.latent_sigma(x)
        return mu, sigma

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, self.decoder_in_channels, *self.shape)
        z = self.decoder(z)
        z = self.final(z)
        return z

    def reparametrise(self, mu, sigma):
        std = (0.5 * sigma).exp()
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, return_all=False):
        mu, sigma = self.encode(x)
        z = self.reparametrise(mu, sigma)
        out = self.decode(z)
        if return_all:
            return out, x, z, mu, sigma
        else:
            return out

    def sample(self, nb_samples):
        z = torch.randn(nb_samples, self.latent_dim)
        z = self.decode(z)
        return z


class CVAE(Module):
    """
    Translation invariant convolutional VAE.
    """
    def __init__(self, dim, in_channels, latent_dim, out_channels=None,
                 encoder=(16, 32, 64, 128, 256), decoder=None, neurite=False,
                 final_activation='Tanh', **kwargs):
        kwargs.setdefault('activation', 'relu')
        kwargs.setdefault('kernel_size', 3)
        super().__init__()
        out_channels = out_channels or in_channels
        encoder = list(encoder)
        if not decoder:
            decoder = list(encoder)
            decoder.reverse()
        if neurite:
            encod = NeuriteEncoder(dim, in_channels, encoder, **kwargs)
            decod = NeuriteDecoder(dim, decoder[0], decoder[1:], **kwargs)
            final = NeuriteConv(dim, decoder[-1], out_channels,
                                kernel_size=1, activation=final_activation)
        else:
            encod = Encoder(dim, in_channels, encoder, **kwargs)
            decod = Decoder(dim, decoder[0], decoder[1:], **kwargs)
            final = ConvBlock(dim, decoder[-1], out_channels,
                              kernel_size=1, activation=final_activation)
        self.encoder = encod
        self.latent_mu = ConvBlock(dim, encoder[-1], latent_dim, kernel_size=1)
        self.latent_sigma = ConvBlock(dim, encoder[-1], latent_dim, kernel_size=1)
        self.decoder_input = ConvBlock(dim, latent_dim, decoder[0], kernel_size=1)
        self.decoder = decod
        self.final = final

        self.tags = ['mu', 'sigma', 'z', 'similarity', 'recon']

    def encode(self, x):
        x = self.encoder(x)
        mu = self.latent_mu(x)
        sigma = self.latent_sigma(x)
        sigma = (0.5 * sigma).exp()
        return mu, sigma

    def decode(self, z):
        z = self.decoder_input(z)
        z = self.decoder(z)
        z = self.final(z)
        return z

    def reparametrise(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return eps.mul_(sigma).add(mu)

    def forward(self, x, return_all=False, *, _loss=None, _metric=None):
        mu, sigma = self.encode(x)
        z = self.reparametrise(mu, sigma)
        out = self.decode(z)

        self.compute(_loss, _metric,
                     mu=[mu], sigma=[sigma], z=[z],
                     similarity=[x, out], recon=[out])

        if return_all:
            return out, x, z, mu, sigma
        else:
            return out

    def sample(self, nb_samples):
        z = torch.randn(nb_samples, self.latent_dim)
        z = self.decode(z)
        return z

    def board(self, tb, inputs=None, outputs=None, epoch=None, minibatch=None,
              mode=None, loss=None, losses=None, metrics=None, *args, **kwargs):

        import matplotlib.pyplot as plt
        if not hasattr(self, 'tbstep'):
            self.tbstep = dict()
        self.tbstep.setdefault(mode, 0)
        self.tbstep[mode] += 1

        x = inputs[0]
        dim = x.dim() - 2

        # --- generate samples ---
        if not self.training:
            fig_samples = plt.figure()
            z = torch.randn([16, self.latent_mu.out_channels] + [1] * dim,
                            dtype=x.dtype, device=x.device)
            y = self.decode(z)

            for b in range(16):
                plt.subplot(4, 4, b)
                y1 = y[b, 0]
                if dim == 3:
                    y1 = y1[..., y1.shape[-1]//2]
                plt.imshow(y1)
                plt.axis('off')

            tb.add_figure(f'samples/{mode}', fig_samples,
                          global_step=self.tbstep[mode])

        # --- show reconstructions ---
        fig_recon = plt.figure
        y = outputs[0]
        plt.subplot(1, 2, 1)
        x1 = x[0, 0]
        y1 = y[0, 0]
        if dim == 3:
            x1 = x1[..., x1.shape[-1]//2]
            y1 = y1[..., y1.shape[-1]//2]
        plt.imshow(x1)
        plt.axis('off')
        plt.imshow(y1)
        plt.axis('off')
        tb.add_figure(f'recon/{mode}', fig_recon,
                      global_step=self.tbstep[mode])


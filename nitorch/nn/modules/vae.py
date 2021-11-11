# general class for VAE. TODO:
# bottleneck layer, with both deterministic and variational
# VAE / AE models (just encoder/decoder and bottleneck)
# beta-VAE - need to make loss modules for different VAE training
# VQ-VAE
# some kind of anomaly detecting VAE for brain pathology
# implement consistency reg: https://arxiv.org/pdf/2105.14859.pdf
# eventually figure out DDPM and VQ-GAN...

import torch
import numpy as np
from .conv import Conv, ConvBlock
from .cnn import Encoder, Decoder, NeuriteEncoder, NeuriteDecoder, NeuriteConv
from nitorch.nn.base import Module, Sequential, ModuleList
from .linear import Linear


class VAE(Module):
    """
    Plain VAE class of encoder-latent-decoder
    """
    def __init__(self, dim, img_size, in_channels, latent_dim, out_channels=None, 
                 encoder=[16,32,64,128,256], decoder=None, neurite=False,
                 final_activation='Tanh', **kwargs):
        super().__init__()
        self.dim = dim
        if isinstance(img_size, int):
            img_size = [img_size] * dim
        if not out_channels:
            out_channels = in_channels
        if decoder is None:
            decoder = encoder.reverse()
        # TODO: add options for full stack like UNet
        self.decoder_in_channels = decoder[0]
        if neurite:
            self.encoder = NeuriteEncoder(dim, in_channels, encoder, **kwargs)
            self.decoder = NeuriteDecoder(dim, decoder[0], decoder[1:], **kwargs)
            self.final = NeuriteConv(dim, decoder[-1], out_channels, activation=final_activation)
        else:
            self.encoder = Encoder(dim, in_channels, encoder, **kwargs)
            self.decoder = Decoder(dim, decoder[0], decoder[1:], **kwargs)
            self.final = Conv(dim, decoder[-1], out_channels, activation=final_activation)
        shape = torch.tensor(img_size).unsqueeze(0).unsqueeze(0)
        for layer in self.encoder:
            shape = layer.shape(shape)
        self.out_shape = shape[2:]
        self.latent_mu = Linear(encoder[-1] * np.prod(self.shape), latent_dim)
        self.latent_sigma = Linear(encoder[-1] * np.prod(self.shape), latent_dim)
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

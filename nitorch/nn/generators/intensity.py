import torch
import torch.distributions as td
from nitorch.core.utils import unsqueeze
from nitorch.core import utils
from nitorch.nn.base import Module
from . import field


class RandomOperation(Module):
    """Sample a random tensor and applies it (according to some binary
    operator) to an image."""

    op = None

    def __init__(self, factor=None):
        """

        Parameters
        ----------
        factor : tensor_like or callable(sequence[int]) -> tensor
            Either a fixed factor or a function that returns a factor of
            a given shape. Note that the output factor gets singleton
            dimensions added to its *right* before being multiplied with
            the image.
        """
        super().__init__()
        self.factor = factor

    @classmethod
    def default_factor(cls, *b, **backend):
        zero = torch.tensor(0, **backend)
        one = torch.tensor(1, **backend)
        return td.Normal(zero, one).sample(b)

    def forward(self, image, **overload):
        factor = overload.get('factor', self.factor)
        if factor is None:
            factor = self.default_factor(len(image), **utils.backend(image))
        if callable(factor):
            factor = factor(image.shape[0])
        factor = torch.as_tensor(factor, **utils.backend(image))
        factor = unsqueeze(factor, -1, image.dim() - factor.dim())
        image = self.op(image, factor)
        return image


class RandomMultiplicative(RandomOperation):
    """Multiply an image with a random factor."""

    op = torch.mul

    @classmethod
    def default_factor(cls, *b, **backend):
        zero = torch.tensor(0, **backend)
        one = torch.tensor(1, **backend)
        return td.Normal(zero, one).sample(b).exp()


class RandomAdditive(RandomOperation):
    """Add a random value to an image."""
    op = torch.add


class RandomGammaCorrection(RandomOperation):
    """Perform a random Gamma correction"""

    def __init__(self, factor=None, vmin=None, vmax=None):
        """

        Parameters
        ----------
        factor : tensor_like or callable(sequence[int]) -> tensor
            Either a fixed factor or a function that returns a factor of
            a given shape. Note that the output factor gets singleton
            dimensions added to its *right* before being multiplied with
            the image.
        vmin : Value used as the "zero" fixed point, optional
        vmax : Value used as the "one" fixed point, optional
        """
        super().__init__(factor)
        self.vmin = vmin
        self.vmax = vmax

    @classmethod
    def default_factor(cls, *b, **backend):
        zero = torch.tensor(0, **backend)
        one = torch.tensor(1, **backend)
        return td.Normal(zero, one).sample(b).exp()

    def forward(self, image, **overload):
        factor = overload.get('factor', self.factor)
        vmin = overload.get('vmin', self.vmin)
        vmax = overload.get('vmax', self.vmax)
        if factor is None:
            factor = self.default_factor(len(image), **utils.backend(image))
        if callable(factor):
            factor = factor(image.shape[0])
        factor = torch.as_tensor(factor, **utils.backend(image))
        factor = unsqueeze(factor, -1, image.dim() - factor.dim())

        if vmin is None:
            vmin = image.reshape([image.shape[0], -1]).min(dim=-1).values
            vmax = image.reshape([image.shape[0], -1]).max(dim=-1).values
        vmin = torch.as_tensor(vmin, **utils.backend(image))
        vmin = unsqueeze(vmin, -1, image.dim() - vmin.dim())
        vmax = torch.as_tensor(vmax, **utils.backend(image))
        vmax = unsqueeze(vmax, -1, image.dim() - vmax.dim())

        image = (image - vmin) / (vmax - vmin)
        image = image.pow(factor)
        image = image * (vmax - vmin) + vmin
        return image


class RandomGaussianNoise(Module):

    def __init__(self, sigma=None, gfactor=False):
        super().__init__()
        self.sigma = sigma
        self.gfactor = gfactor

    @classmethod
    def default_sigma(cls, *b, **backend):
        sd = torch.tensor(1, **backend)
        mean = torch.tensor(-3.5, **backend)
        return td.Normal(mean, sd).sample(b).exp()

    def forward(self, image, **overload):
        backend = utils.backend(image)
        sigma = overload.get('sigma', self.sigma)
        gfactor = overload.get('gfactor', self.gfactor)

        # sample sigma
        if sigma is None:
            sigma = self.default_sigma(*image.shape[:2], **backend)
        if callable(sigma):
            sigma = sigma(image.shape[:2])
        sigma = torch.as_tensor(sigma, **backend)
        sigma = unsqueeze(sigma, -1, 2 - sigma.dim())

        # sample gfactor
        if gfactor is True:
            gfactor = field.RandomMultiplicativeField()
        if callable(gfactor):
            gfactor = gfactor(image.shape)

        # sample noise
        zero = torch.tensor(0, **backend)
        noise = td.Normal(zero, sigma).sample(image.shape[2:])
        noise = utils.movedim(noise, [-1, -2], [0, 1])

        if torch.is_tensor(gfactor):
            noise *= gfactor

        image = image + noise
        return image


class RandomChiNoise(Module):
    """Add random Chi-distributed noise."""

    def __init__(self, sigma=None, gfactor=False, ncoils=None):
        """

        Parameters
        ----------
        sigma : (batch, channel) tensor_like or callable, optional
            Base standard deviation of the noise.
            By default, sample from a Log-Normal distribution with mean 0.03.
        gfactor : tensor_like or callable, optional
            Nonstationary modulation of the noise.
            Do not use by default (False).
            If True, sample a smooth random multiplicative field.
        ncoils : (batch) tensor_like or callable, optional
            Number of coils = degrees of freedom / 2
            By default, sample uniformly in {1, 2, 4, 8, 16, 32, 64, 128}.
        """
        super().__init__()
        self.sigma = sigma
        self.gfactor = gfactor
        self.ncoils = ncoils

    @classmethod
    def default_sigma(cls, *b, **backend):
        sd = torch.tensor(0.1, **backend)
        mean = torch.tensor(-3.5, **backend)
        return td.Normal(mean, sd).sample(b).exp()

    @classmethod
    def default_ncoils(cls, *b, **backend):
        backend.pop('dtype', None)
        ncoils = torch.randint(0, 7, b, **backend)
        ncoils = 2 ** ncoils
        return ncoils

    def forward(self, image, **overload):
        backend = utils.backend(image)
        sigma = overload.get('sigma', self.sigma)
        gfactor = overload.get('gfactor', self.gfactor)
        ncoils = overload.get('ncoils', self.ncoils)

        # sample sigma
        if sigma is None:
            sigma = self.default_sigma(*image.shape[:2], **backend)
        if callable(sigma):
            sigma = sigma(image.shape[:2])
        sigma = torch.as_tensor(sigma, **backend)
        sigma = unsqueeze(sigma, -1, 2 - sigma.dim())

        # sample gfactor
        if gfactor is True:
            gfactor = field.RandomMultiplicativeField()
        if callable(gfactor):
            gfactor = gfactor(image.shape)

        # sample number coils
        if ncoils is None:
            ncoils = self.default_ncoils(*image.shape[:1], **backend)
        if callable(ncoils):
            ncoils = ncoils(image.shape[:1])
        ncoils = torch.as_tensor(ncoils, device=backend['device'], dtype=torch.int)

        # sample noise
        zero = torch.tensor(0, **backend)
        sampler = lambda: utils.movedim(td.Normal(zero, sigma).sample(image.shape[2:]).square_(), [-2, -1], [0, 1])
        noise = sampler()
        for n in range(2*ncoils.max()-1):
            tmp = sampler()
            tmp[2*ncoils + 1 >= n + 1, ...] = 0
            noise += tmp
        noise = noise.sqrt_()

        if torch.is_tensor(gfactor):
            noise *= gfactor

        image = image + noise
        return image
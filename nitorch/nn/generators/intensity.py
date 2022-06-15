import torch
import torch.distributions as td
from nitorch.core.utils import unsqueeze
from nitorch.core import utils
from nitorch.nn.base import Module
from . import field
from .distribution import _get_dist


class RandomBiasFieldTransform(Module):
    """Apply a random multiplicative bias field to an image."""

    def __init__(self, mean=1, amplitude=1, fwhm=48, sigmoid=False):
        """
        The geometry of a random field is controlled by three parameters:
            - `mean` controls the expected value of the field.
            - `amplitude` controls the voxel-wise variance of the field.
            - `fwhm` controls the smoothness of the field.

        Parameters
        ----------
        mean : float or (channel,) vector_like, default=0
            Log-Mean value.
        amplitude : float or (channel,) vector_like, default=1
            Amplitude of the squared-exponential kernel.
        fwhm : float or (channel,) vector_like, default=48
            Full-width at Half Maximum of the squared-exponential kernel.
        """
        super().__init__()
        self.bias = field.RandomMultiplicativeField(mean, amplitude, fwhm,
                                                    sigmoid=sigmoid)

    def forward(self, image, **overload):
        """

        Parameters
        ----------
        image : (batch, channel, *shape)
            Input tensor

        Returns
        -------
        transformed_image : (batch, channel, *shape)
            Bias-multiplied tensor

        """
        image = torch.as_tensor(image)
        overload['dtype'] = image.dtype
        overload['device'] = image.device
        bias = self.bias(image.shape, **overload)
        image = image * bias
        return image


class HyperRandomBiasFieldTransform(Module):
    """
    Apply a random multiplicative bias field to an image,
    with randomized hyper-parameters.
    """

    def __init__(self,
                 mean=None, mean_exp=1, mean_scale=0.1,
                 amplitude='lognormal', amplitude_exp=1, amplitude_scale=10,
                 fwhm='lognormal', fwhm_exp=48, fwhm_scale=16, sigmoid=False):
        """
        The geometry of a random field is controlled by three parameters:
            - `mean` controls the expected value of the field.
            - `amplitude` controls the voxel-wise variance of the field.
            - `fwhm` controls the smoothness of the field.

        Each of these parameter is sampled according to three hyper-parameters:
            - <param>       : distribution family
                              {'normal', 'lognormal', 'uniform', 'gamma', None}
            - <param>_exp   : expected value of the parameter
            - <param>_scale : standard deviation of the parameter

        Parameters
        ----------
        mean : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        mean_exp : float or (channel,) vector_like, default=1
        mean_scale : float or (channel,) vector_like, default=0.1
        amplitude : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        amplitude_exp : float or (channel,) vector_like, default=1
        amplitude_scale : float or (channel,) vector_like, default=10
        fwhm : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        fwhm_exp : float or (channel,) vector_like, default=48
        fwhm_scale : float or (channel,) vector_like, default=16
        """
        super().__init__()
        self.bias = field.HyperRandomMultiplicativeField(
            mean, mean_exp, mean_scale,
            amplitude, amplitude_exp, amplitude_scale,
            fwhm, fwhm_exp, fwhm_scale, sigmoid=sigmoid)

    def forward(self, image, **overload):
        """

        Parameters
        ----------
        image : (batch, channel, *shape)
            Input tensor

        Returns
        -------
        transformed_image : (batch, channel, *shape)
            Bias-multiplied tensor

        """
        image = torch.as_tensor(image)
        overload['dtype'] = image.dtype
        overload['device'] = image.device
        bias = self.bias(image.shape, **overload)
        image = image * bias
        return image


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


class RandomGammaCorrection(Module):
    """Perform a random Gamma correction"""

    def __init__(self, factor='lognormal', factor_exp=1, factor_scale=1,
                 vmin=None, vmax=None):
        """

        Parameters
        ----------
        factor : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        factor_exp : float or (channel,) vector_like, default=1
        factor_scale : float or (channel,) vector_like, default=1
        vmin : Value used as the "zero" fixed point, optional
        vmax : Value used as the "one" fixed point, optional
        """
        super().__init__()
        self.factor = _get_dist(factor)
        self.factor_exp = factor_exp
        self.factor_scale = factor_scale
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, x):

        backend = utils.backend(x)

        # compute intensity bounds
        vmin = self.vmin
        if vmin is None:
            vmin = x.reshape([*x.shape[:2], -1]).min(dim=-1).values
        vmax = self.vmax
        if vmax is None:
            vmax = x.reshape([*x.shape[:2], -1]).max(dim=-1).values
        vmin = torch.as_tensor(vmin, **backend).expand(x.shape[:2])
        vmin = unsqueeze(vmin, -1, x.dim() - vmin.dim())
        vmax = torch.as_tensor(vmax, **backend).expand(x.shape[:2])
        vmax = unsqueeze(vmax, -1, x.dim() - vmax.dim())

        # sample factor
        factor_exp = utils.make_vector(self.factor_exp, x.shape[1], **backend)
        factor_scale = utils.make_vector(self.factor_scale, x.shape[1], **backend)
        factor = self.factor(factor_exp, factor_scale)
        factor = factor.sample([len(x)])
        factor = unsqueeze(factor, -1, x.dim() - 2)

        # apply correction
        x = (x - vmin) / (vmax - vmin)
        x = x.pow(factor)
        x = x * (vmax - vmin) + vmin
        return x


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

    def __init__(self, sigma=1, ncoils=1):
        """

        Parameters
        ----------
        sigma : float or (channel,) vector_like, default=1
            Base standard deviation of the noise.
        ncoils : float or (channel,) vector_like, default=1
            Number of coils = degrees of freedom / 2
        """
        super().__init__()
        self.sigma = sigma
        self.ncoils = ncoils

    def forward(self, image, gfactor=None):
        backend = utils.backend(image)

        sigma = utils.make_vector(self.sigma, image.shape[1], **backend)
        ncoils = utils.make_vector(self.ncoils, image.shape[1],
                                   device=backend['device'], dtype=torch.int)

        zero = torch.tensor(0, **backend)
        def sampler():
            shape = [len(image), *image.shape[2:]]
            noise = td.Normal(zero, sigma).sample(shape).square_()
            return utils.movedim(noise, -1, 1)

        # sample noise
        noise = sampler()
        for n in range(2*ncoils.max()-1):
            tmp = sampler()
            tmp[:, 2*ncoils + 1 >= n + 1, ...] = 0
            noise += tmp
        noise = noise.sqrt_()
        noise /= ncoils

        if gfactor is not None:
            noise *= gfactor

        image = image + noise
        return image


class HyperRandomChiNoise(Module):

    def __init__(self,
                 sigma='gamma',
                 sigma_exp=1,
                 sigma_scale=2,
                 ncoils='duniform',
                 ncoils_exp=8,
                 ncoils_scale=4/3.464):
        super().__init__()
        self.sigma = _get_dist(sigma)
        self.sigma_exp = sigma_exp
        self.sigma_scale = sigma_scale
        self.ncoils = _get_dist(ncoils)
        self.ncoils_exp = ncoils_exp
        self.ncoils_scale = ncoils_scale

    def forward(self, x, gfactor=None):
        out = torch.empty_like(x)
        for b in range(len(x)):
            sigma = self.sigma(self.sigma_exp, self.sigma_scale, **utils.backend(x)).sample().clamp_min_(0)
            ncoils = self.ncoils(self.ncoils_exp, self.ncoils_exp, device=x.device).sample().clamp_min_(1)
            print('chi:', sigma.item(), ncoils.item())
            sampler = RandomChiNoise(sigma, ncoils)
            if gfactor is not None and gfactor.dim() == x.dim():
                gfactor1 = gfactor[b]
            else:
                gfactor1 = None
            out[b] = sampler(x[None, b], gfactor1)[0]

        return out

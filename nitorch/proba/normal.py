"""Normal (Gaussian) distribution and its generalizations

- Normal
- MultivariateNormal
- LogNormal             == Exp(Normal)
- FoldedNormal          == Abs(Normal)
- HalfNormal            == FoldedNormal(location=0)
"""
import torch
from torch import distributions
from nitorch import core
from nitorch.core.utils import ensure_shape
from .variables import RandomVariable
from .transforms import Exp, Abs
from .utils import max_backend, to
import math


class Normal(RandomVariable):

    def __init__(self, mean=0., variance=1., shape=None,
                 dtype=None, device=None, **kwargs):
        """

        Signatures
        ----------
        Normal(mean, variance, ...)
        Normal(mean, std=sigma, ...)
        Normal(mean, precision=lam, ...)

        Parameters
        ----------
        mean or location : tensor_like, default=0
            Expected value of the distribution.
        variance or precision or std or scale: tensor_like, default=1
            Variance (or precision) of the distribution.
            Only one of these keywords can be provided.
            - The precision is the inverse of the variance.
            - The standard deviation (or scale) is the square root of
              the variance.
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        # find common backend
        super().__init__()
        location = kwargs.get('location', None)
        precision = kwargs.get('precision', None)
        std = kwargs.get('std', None)
        scale = kwargs.get('scale', None)
        backend = max_backend(mean, location, variance, precision, std, scale)
        if dtype:
            backend['dtype'] = dtype
        if device:
            backend['device'] = device

        # ensure tensor
        if 'location' in kwargs:
            self.parameters['mean'] = to(location, **backend)
        else:
            self.parameters['mean'] = to(mean, **backend)
        if 'precision' in kwargs:
            self.parameters['precision'] = to(precision, **backend)
        elif 'std' in kwargs:
            self.parameters['variance'] = to(std, **backend).square()
        elif 'scale' in kwargs:
            self.parameters['variance'] = to(scale, **backend).square()
        else:
            self.parameters['variance'] = to(variance, **backend)

        # ensure shape
        shape = tuple(shape or [])
        if shape:
            for key, value in self.parameters:
                self.parameters[key] = ensure_shape(value, shape)

    @property
    def shape(self):
        if 'variance' in self.parameters:
            return core.utils.max_shape(self.mean.shape, self.variance.shape)
        else:
            return core.utils.max_shape(self.mean.shape, self.precision.shape)

    @property
    def batch_shape(self):
        return self.shape

    @property
    def event_shape(self):
        return tuple()

    @property
    def mean(self):
        """(*shape) tensor"""
        return self.parameters['mean']

    @property
    def variance(self):
        """(*shape) tensor"""
        if 'variance' in self.parameters:
            return self.parameters['variance']
        else:
            return self.precision.reciprocal()

    @property
    def std(self):
        """(*shape) tensor"""
        return self.variance.sqrt()

    @property
    def precision(self):
        """(*shape) tensor"""
        if 'variance' in self.parameters:
            return self.variance.reciprocal()
        else:
            return self.parameters['precision']

    @property
    def covariance(self):
        """(*shape, 1, 1) tensor"""
        return self.variance[..., None, None]

    @property
    def location(self):
        """(*shape) tensor"""
        return self.mean

    @property
    def scale(self):
        """(*shape) tensor"""
        return self.std

    @property
    def mode(self):
        """(*shape) tensor"""
        return self.mean

    def moment(self, order):
        """(*shape) tensor"""
        if order == 1:
            return self.mean
        elif order == 2:
            return self.mean.square() + self.variance
        else:
            raise NotImplementedError

    def pdf(self, value):
        """Evaluate the probability density function.

        Parameters
        ----------
        value : (*inshape) tensor_like
            Value at which to evaluate the PDF.

        Returns
        -------
        pdf : (*outshape) tensor
            Probability density function.
            The output shape is obtained by broadcasting `inshape`
            and `self.shape`.

        """
        twopi = torch.as_tensor(2 * core.constants.pi, **self.backend)
        p = -0.5 * (value - self.mean).square() * self.precision
        p = p.exp() * self.precision.sqrt()
        p = p / twopi.sqrt()
        return p

    def logpdf(self, value):
        """Evaluate the log probability density function.

        Parameters
        ----------
        value : (*inshape) tensor_like
            Value at which to evaluate the log-PDF.

        Returns
        -------
        pdf : (*outshape) tensor
            Log probability density function.
            The output shape is obtained by broadcasting `inshape`
            and `self.shape`.

        """
        twopi = torch.as_tensor(2 * core.constants.pi, **self.backend)
        p = -0.5 * (value - self.mean).square() * self.precision
        p = p + 0.5 * self.precision.log()
        p = p - 0.5 * twopi.log()
        return p

    def sample(self, shape=None, generator=None, out=None):
        """Generate random samples.

        Parameters
        ----------
        shape : sequence[int], optional
        generator : torch.Generator, optional
        out : tensor, optional

        Returns
        -------
        samp : (*shape, *self.shape) tensor
            Normal samples

        """
        mu = self.mean
        if isinstance(mu, RandomVariable):
            mu = mu.sample(generator=generator)
        sigma = self.std
        if isinstance(sigma, RandomVariable):
            sigma = sigma.sample(generator=generator)

        shape = tuple(shape or []) + self.shape
        mu = ensure_shape(mu, shape)
        sigma = ensure_shape(sigma, shape)
        out = torch.normal(mu, sigma, generator=generator, out=out)
        return out

    def __add__(self, other):
        if isinstance(other, RandomVariable):
            if isinstance(other, Normal):
                return Normal(mean=self.mean + other.mean,
                              variance=self.variance + other.variance)
            return super().__add__(self, other)
        else:
            return Normal(mean=self.mean + other, variance=self.variance)

    def __mul__(self, other):
        if isinstance(other, RandomVariable):
            return super().__mul__(self, other)
        else:
            return Normal(mean=self.mean * other,
                          variance=self.variance * (other ** 2))

    def __repr__(self):
        return f'Normal(mean={self.mean.__repr__()}, ' \
               f'variance={self.variance.__repr__()})'


Gaussian = Normal


class MultivariateNormal(RandomVariable):

    def __init__(self, mean=0., covariance=1., shape=None,
                 dtype=None, device=None, **kwargs):
        """

        Signatures
        ----------
        Normal(mean, covariance, ...)
        Normal(mean, precision=A, ...)
        Normal(mean, scale=L, ...)

        Parameters
        ----------
        mean or location : tensor_like, default=0
            Expected value of the distribution.
            Only one of these keywords can be provided.
        covariance or precision or scale : tensor_like, default=1
            Covariance (or precision or scale) matrix of the distribution.
            Only one of these keywords can be provided.
            - The precision matrix is the inverse of the covariance matrix.
            - The scale is the "square root" of the covariance: it is a matrix
              such that `L @ L.T = C`. Multiple matrices have that property,
              but the Cholesky decomposition of the covariance is typically
              used.
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape
            (including the number of channels).
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        # find common backend
        super().__init__()
        location = kwargs.get('location', None)
        precision = kwargs.get('precision', None)
        scale = kwargs.get('scale', None)
        prms = [mean, location, covariance, precision, scale]
        backend = core.utils.max_backend(*prms)
        if dtype:
            backend['dtype'] = dtype
        if device:
            backend['device'] = device

        # ensure tensor
        if 'location' in kwargs:
            self.parameters['mean'] = to(location, **backend)
        else:
            self.parameters['mean'] = to(mean, **backend)
        if 'precision' in kwargs:
            self.parameters['precision'] = to(precision, **backend)
        elif 'scale' in kwargs:
            self.parameters['scale'] = to(scale, **backend)
        else:
            self.parameters['covariance'] = to(covariance, **backend)

        # ensure shape
        shape = tuple(shape or [])
        if shape:
            self.parameters['mean'] = \
                ensure_shape(self.parameters['mean'], shape)
            if 'precision' in self.parameters:
                self.parameters['precision'] = \
                    ensure_shape(self.parameters['precision'], shape)
            if 'scale' in self.parameters:
                self.parameters['scale'] = \
                    ensure_shape(self.parameters['scale'], shape + shape[-1:])
            if 'covariance' in self.parameters:
                self.parameters['covariance'] = \
                    ensure_shape(self.parameters['covariance'], shape + shape[-1:])

    @property
    def shape(self):
        if 'covariance' in self.parameters:
            return core.utils.max_shape(self.mean.shape,
                                        self.covariance.shape[:-1])
        if 'scale' in self.parameters:
            return core.utils.max_shape(self.mean.shape,
                                        self.scale.shape[:-1])
        if 'precision' in self.parameters:
            return core.utils.max_shape(self.mean.shape,
                                        self.precision.shape[:-1])

    @property
    def batch_shape(self):
        return self.shape[:-1]

    @property
    def event_shape(self):
        return self.shape[-1:]

    @property
    def mean(self):
        """(*shape, C) tensor"""
        return self.parameters['mean']

    @property
    def variance(self):
        """(*shape) tensor"""
        return self.covariance.diagonal(-1, -2).sum(-1)

    @property
    def std(self):
        """(*shape) tensor"""
        return self.variance.sqrt()

    @property
    def precision(self):
        """(*shape, C, C) tensor"""
        if 'precision' in self.parameters:
            return self.parameters['precision']
        else:
            return core.linalg.inv(self.covariance, 'chol')

    @property
    def covariance(self):
        """(*shape, C, C) tensor"""
        if 'covariance' in self.parameters:
            return self.parameters['precision']
        elif 'scale' in self.parameters:
            return self.scale.matmul(self.scale.transpose(-1, -2))
        else:
            return core.linalg.inv(self.precision, 'chol')

    @property
    def location(self):
        """(*shape, C) tensor"""
        return self.mean

    @property
    def scale(self):
        """(*shape, C, C) tensor"""
        if 'scale' in self.parameters:
            return self.parameters['scale']
        else:
            return self.covariance.cholesky()

    @property
    def mode(self):
        """(*shape, C) tensor"""
        return self.mean

    def moment(self, order):
        """(*shape, ...) tensor"""
        if order == 1:
            return self.mean
        elif order == 2:
            return self.mean[..., :, None] * self.mean[..., None, :] \
                   + self.covariance
        else:
            raise NotImplementedError

    def pdf(self, value):
        """Evaluate the probability density function.

        Parameters
        ----------
        value : (*inshape) tensor_like
            Value at which to evaluate the PDF.

        Returns
        -------
        pdf : (*outshape) tensor
            Probability density function.
            The output shape is obtained by broadcasting `inshape`
            and `self.shape`.

        """
        twopi = torch.as_tensor(2 * core.constants.pi, **self.backend)
        p = -0.5 * (value - self.mean).square() * self.precision
        p = p.exp() * self.precision.sqrt()
        p = p / twopi.sqrt()
        return p

    def logpdf(self, value):
        """Evaluate the log probability density function.

        Parameters
        ----------
        value : (*inshape) tensor_like
            Value at which to evaluate the log-PDF.

        Returns
        -------
        pdf : (*outshape) tensor
            Log probability density function.
            The output shape is obtained by broadcasting `inshape`
            and `self.shape`.

        """
        twopi = torch.as_tensor(2 * core.constants.pi, **self.backend)
        p = -0.5 * (value - self.mean).square() * self.precision
        p = p + 0.5 * self.precision.log()
        p = p - 0.5 * twopi.log()
        return p

    def sample(self, shape=None, generator=None, out=None):
        """Generate random samples.

        Parameters
        ----------
        shape : sequence[int], optional
        generator : torch.Generator, optional
        out : tensor, optional

        Returns
        -------
        samp : (*shape, *self.shape) tensor
            Normal samples

        """
        mu = self.mean
        if isinstance(mu, RandomVariable):
            mu = mu.sample(generator=generator)
        sigma = self.covariance
        if isinstance(sigma, RandomVariable):
            sigma = sigma.sample(generator=generator)

        mvn = distributions.MultivariateNormal(mu, sigma)
        samp = mvn.sample(shape)
        if out is not None:
            out[...] = samp
        return out


class LogNormal(Exp):
    """Log-Normal distribution

    Note
    ----
    .. A log-Normal variable is obtained by taking the exponential of a
       Normal variable.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    def __init__(self, *args, **kwargs):
        """

        Signatures
        ----------
        LogNormal(location, scale, ...)
        LogNormal(base=x, ...)

        Parameters
        ----------
        location : tensor_like, default=0
            The location is the mean of the underlying Gaussian variable.
        scale : tensor_like, default=1
            The scale is the standard-deviation of the underlying
            Gaussian variable.
        base : RandomVariable
            The underlying Gaussian variable.
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        if 'base' not in kwargs:
            base = Normal(*args, **kwargs)
        else:
            if args or 'location' in kwargs or 'scale' in kwargs:
                raise TypeError("Cannot use both 'base' and "
                                "'location'/'scale'")
            base = kwargs['base'].expand(kwargs.get('shape', None))
            base = base.to(dtype=kwargs.get('dtype', None),
                           device=kwargs.get('device', None))
        super().__init__(base)

    @property
    def mean(self):
        """(*shape) tensor"""
        return (self.location + 0.5 * self.base.variance).exp()

    @property
    def location(self):
        return self.base.location

    @property
    def variance(self):
        """(*shape) tensor"""
        v = self.base.variance.exp() - 1
        v = v * (2 * self.location + self.base.variance).exp()
        return v

    @property
    def scale(self):
        """(*shape) tensor"""
        return self.base.scale

    @property
    def mode(self):
        """(*shape) tensor"""
        return (self.location - self.base.variance).exp()

    @property
    def entropy(self):
        """(*shape) tensor"""
        sqrt2pi = math.sqrt(2 * math.pi)
        h = self.scale * (self.location + 0.5).exp() * sqrt2pi
        return h.log()

    def log(self):
        return self.base


class FoldedNormal(Abs):
    """Folded Normal distribution

    Note
    ----
    .. A folded Normal variable is obtained by taking the absolute value
       of a Normal variable.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Folded_normal_distribution
    """

    def __init__(self, *args, **kwargs):
        """

        Signatures
        ----------
        FoldedNormal(location, scale, ...)
        FoldedNormal(base=x, ...)

        Parameters
        ----------
        location : tensor_like, default=0
            The location is the mean of the underlying Gaussian variable.
        scale : tensor_like, default=1
            The scale is the standard-deviation of the underlying
            Gaussian variable.
        base : RandomVariable
            The underlying Gaussian variable.
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        if 'base' not in kwargs:
            base = Normal(*args, **kwargs)
        else:
            if args or 'location' in kwargs or 'scale' in kwargs:
                raise TypeError("Cannot use both 'base' and "
                                "'location'/'scale'")
            base = kwargs['base'].expand(kwargs.get('shape', None))
            base = base.to(dtype=kwargs.get('dtype', None),
                           device=kwargs.get('device', None))
        super().__init__(base)

    @property
    def mean(self):
        """(*shape) tensor"""
        sqrt2opi = math.sqrt(2 / math.pi)
        e = self.scale * sqrt2opi
        e = e * (-0.5 * self.location.square()*self.base.precision).exp()
        e = e + self.location * (1 - 2 * self.base.cdf(-self.location/self.scale))
        return e

    @property
    def location(self):
        return self.base.location

    @property
    def variance(self):
        """(*shape) tensor"""
        return self.location.square() + self.base.variance - self.mean.square()

    @property
    def scale(self):
        """(*shape) tensor"""
        return self.base.scale

    @property
    def mode(self):
        """(*shape) tensor"""
        # No closed-form
        raise NotImplementedError

    @property
    def entropy(self):
        """(*shape) tensor"""
        sqrt2pi = math.sqrt(2 * math.pi)
        h = self.scale * (self.location + 0.5).exp() * sqrt2pi
        return h.log()

    def log(self):
        return self.base


class HalfNormal(FoldedNormal):
    """Half-Normal distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Half-normal_distribution
    """

    def __init__(self, *args, **kwargs):
        """

        Signatures
        ----------
        HalfNormal(scale, ...)

        Parameters
        ----------
        scale : tensor_like, default=1
            The scale is the standard-deviation of the underlying
            Gaussian variable.
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        super().__init__(0, *args, **kwargs)

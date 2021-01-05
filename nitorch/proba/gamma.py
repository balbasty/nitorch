"""Gamma distribution and its generalizations:

- Gamma
- InverseGamma
- Wishart
- InverseWishart
"""
import torch
import math
from torch import distributions
from nitorch import core
from nitorch.core.utils import ensure_shape
from .variables import RandomVariable
from .transforms import Reciprocal, MatrixInverse
from .utils import max_backend, to


class Gamma(RandomVariable):
    """Gamma distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gamma_distribution
    """

    def __init__(self, concentration=1., rate=1., shape=None,
                 dtype=None, device=None, **kwargs):
        """

        Signatures
        ----------
        Gamma(concentration, rate, ...)
        Gamma(concentration, scale=theta, ...)

        Parameters
        ----------
        concentration : tensor_like, default=1
            The concentration (or shape) `alpha` is the power of the
            polynomial term.
        rate or scale : tensor_like, default=1
            Only one of these keywords can be provided.
            - The rate `beta` is the exponential decay rate.
            - The scale `theta=1/beta` is the inverse of the rate.
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        super().__init__()
        # find common backend
        scale = kwargs.get('scale', None)
        backend = max_backend(concentration, rate, scale)
        if dtype:
            backend['dtype'] = dtype
        if device:
            backend['device'] = device

        # ensure tensor
        self.parameters['concentration'] = to(concentration, **backend)
        if 'scale' in kwargs:
            self.parameters['scale'] = to(scale, **backend)
        else:
            self.parameters['rate'] = to(rate, **backend)
        if len(self.parameters) != 2:
            raise RuntimeError('Too many or too few parameters')

        # ensure shape
        shape = tuple(shape or [])
        if shape:
            for key, value in self.parameters:
                self.parameters[key] = ensure_shape(value, shape)

    @property
    def shape(self):
        return core.utils.max_shape(*self.parameters.values())

    @property
    def batch_shape(self):
        return self.shape

    @property
    def event_shape(self):
        return tuple()

    @property
    def mean(self):
        """(*shape) tensor"""
        if 'rate' in self.parameters:
            return self.concentration / self.rate
        else:
            return self.concentration * self.scale

    @property
    def variance(self):
        if 'rate' in self.parameters:
            return self.concentration / self.rate.square()
        else:
            return self.concentration * self.scale.square()

    @property
    def std(self):
        """(*shape) tensor"""
        return self.variance.sqrt()

    @property
    def covariance(self):
        """(*shape, 1, 1) tensor"""
        return self.variance[..., None, None]

    @property
    def rate(self):
        """(*shape) tensor"""
        if 'rate' in self.parameters:
            return self.parameters['rate']
        else:
            return self.parameters['scale'].reciprocal()

    @property
    def scale(self):
        """(*shape) tensor"""
        if 'scale' in self.parameters:
            return self.parameters['scale']
        else:
            return self.parameters['rate'].reciprocal()

    @property
    def concentration(self):
        """(shape) tensor"""
        return self.parameters['concentration']

    @property
    def mode(self):
        """(*shape) tensor"""
        val = (self.concentration - 1) * self.scale
        nan = torch.as_tensor(core.constants.nan, **self.backend)
        val = torch.where(val < 0, nan, val)
        return val

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
        value = torch.as_tensor(value, **self.backend)
        concentration = self.concentration
        rate = self.rate
        p = torch.lgamma(concentration).exp()
        p = p * (value ** (concentration - 1))
        p = p * (value * rate).neg().exp()
        p = p * rate ** concentration
        return p

    def logpdf(self, value):
        """Evaluate the log probability density function.

        Parameters
        ----------
        value : (*inshape) tensor_like
            Value at which to evaluate the log-PDF.

        Returns
        -------
        logpdf : (*outshape) tensor
            Log probability density function.
            The output shape is obtained by broadcasting `inshape`
            and `self.shape`.

        """
        value = torch.as_tensor(value, **self.backend)
        concentration = self.concentration
        rate = self.rate
        p = torch.lgamma(concentration)
        p += (concentration - 1) * value.log()
        p -= value * rate
        p += concentration * rate.log()
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
        alpha = self.concentration
        if isinstance(alpha, RandomVariable):
            alpha = alpha.sample(generator=generator)
        beta = self.rate
        if isinstance(beta, RandomVariable):
            beta = beta.sample(generator=generator)

        gam = distributions.Gamma(alpha, beta)
        samp = gam.sample(shape)
        if out is not None:
            out = out.copy_(samp)
        return out

    def reciprocal(self):
        return InverseGamma(shape=self.shape, scale=self.rate,
                            **self.backend)

    def __mul__(self, other):
        if isinstance(other, RandomVariable):
            return super().__mul__(self, other)
        else:
            return Gamma(concentration=self.concentration,
                         scale=self.scale * other)


class InverseGamma(Reciprocal):
    """Inverse-Gamma distribution

    Note
    ----
    .. The inverse of a Gamma-distributed random variable is
       Inverse-Gamma-distributed.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """

    def __init__(self, *args, **kwargs):
        """

        Note
        ----
        .. The `scale` of the inverse-Gamma distribution is the `rate`
           of the corresponding Gamma distribution. I.e., if
           `x ~ Gamma(concentration=alpha, rate=beta)` then
           `1/x ~ InverseGamma(concentration=alpha, scale=beta)`.

        Signatures
        ----------
        InverseGamma(concentration, scale, ...)
        InverseGamma(concentration, rate=theta, ...)

        Parameters
        ----------
        concentration : tensor_like, default=1
            The concentration (or shape) `alpha` is the power of the
            polynomial term.
        rate or scale : tensor_like, default=1
            Only one of these keywords can be provided.
            - The scale `beta` is the exponential decay rate.
            - The rate `theta=1/beta` is the inverse of the rate.
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        if 'rate' in kwargs:
            kwargs['scale'] = kwargs['rate']
            del kwargs['rate']
        super().__init__(Gamma(*args, **kwargs))

    @property
    def mean(self):
        """(*shape) tensor"""
        nan = torch.as_tensor(core.constants.nan, **self.backend)
        e = self.scale / (self.concentration - 1)
        return torch.where(self.concentration <= 1, nan, e)

    @property
    def variance(self):
        """(*shape) tensor"""
        v = self.scale.square()
        v = v / ((self.concentration - 1).square() * (self.concentration - 2))
        nan = torch.as_tensor(core.constants.nan, **self.backend)
        return torch.where(self.concentration <= 2, nan, v)

    @property
    def rate(self):
        """(*shape) tensor"""
        return self.base.scale

    @property
    def scale(self):
        """(*shape) tensor"""
        return self.base.rate

    @property
    def concentration(self):
        """(shape) tensor"""
        return self.base.concentration

    @property
    def mode(self):
        """(*shape) tensor"""
        return self.scale / (self.concentration + 1)

    def reciprocal(self):
        return self.base

    def __mul__(self, other):
        if isinstance(other, RandomVariable):
            return super().__mul__(self, other)
        else:
            return InverseGamma(concentration=self.concentration,
                                scale=self.scale * other)


class NormalPrecisionPrior(Gamma):
    # TODO
    pass


class NormalVariancePrior(InverseGamma):
    # TODO
    pass


class Wishart(RandomVariable):
    """Wishart distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wishart_distribution
    """

    def __init__(self, scale, df, shape=None,
                 dtype=None, device=None, **kwargs):
        """

        Parameters
        ----------
        scale : (..., p, p) tensor_like
            Scale matrix
        df : tensor_like
            Degrees of freedom (must be greater than `p - 1`)
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        super().__init__()
        # find common backend
        backend = max_backend(scale, df)
        if dtype:
            backend['dtype'] = dtype
        if device:
            backend['device'] = device

        # ensure tensor
        self.parameters['scale'] = to(scale, **backend)
        self.parameters['df'] = to(df, **backend)

        # ensure shape
        shape = tuple(shape or [])
        if shape:
            self.parameters['scale'] = ensure_shape(self.parameters['scale'], shape)
            self.parameters['df'] = ensure_shape(self.parameters['df'], shape[:-2])

    @property
    def shape(self):
        return self.parameters['scale'].shape

    @property
    def batch_shape(self):
        return self.shape[:-2]

    @property
    def event_shape(self):
        return self.shape[-2:]

    @property
    def mean(self):
        return self.df * self.scale

    @property
    def variance(self):
        """
        (*shape) tensor
            Variance of each element in the matrix
        """
        diag = self.scale.diagonal(-1, -2)
        diag2 = diag[..., None].matmul(diag[..., None, :])
        return self.df * (self.scale.square() + diag2)

    @property
    def std(self):
        """
        (*shape) tensor
            Standard deviation of each element in the matrix
        """
        return self.variance.sqrt()

    @property
    def covariance(self):
        """
        (*batch_shape, *event_shape, *event_shape) tensor
            Covariance between elements of the matrix
        """
        raise NotImplementedError

    @property
    def scale(self):
        """
        (*shape) tensor
            Scale matrix
        """
        return self.parameters['scale']

    @property
    def rate(self):
        """
        (*shape) tensor
            Rate matrix (inverse of the scale matrix)
        """
        return self.parameters['scale'].inverse()

    @property
    def df(self):
        """
        (*batch_shape) tensor
            Degrees of freedom (> p - 1)
        """
        return self.parameters['df']

    @property
    def mode(self):
        """(*shape) tensor"""
        return (self.df * self.shape[-1] - 1) * self.scale

    def moment(self, order):
        """(*shape) tensor"""
        if order == 1:
            return self.mean
        raise NotImplementedError

    def pdf(self, value):
        """Evaluate the probability density function.

        Parameters
        ----------
        value : (*inshape, p, p) tensor_like
            Value at which to evaluate the PDF.

        Returns
        -------
        pdf : (*outshape) tensor
            Probability density function.
            The output shape is obtained by broadcasting `inshape`
            and `self.batch_shape`.

        """
        value = torch.as_tensor(value, **self.backend)
        scale = self.scale
        df = self.df
        size = self.shape[-1]
        p = 2. ** (-df * size / 2.)
        p = p * value.det() ** ((df - size - 1) / 2.)
        p = p / scale.det() ** (df/2.)
        p = p / (df/2.).mvlgamma(size).exp()
        p = p * (-0.5 * scale.inverse().matmul(value).diagonal(-1, -2).sum(-1)).exp()
        return p

    def logpdf(self, value):
        """Evaluate the log probability density function.

        Parameters
        ----------
        value : (*inshape, p, p) tensor_like
            Value at which to evaluate the log-PDF.

        Returns
        -------
        logpdf : (*outshape) tensor
            Log probability density function.
            The output shape is obtained by broadcasting `inshape`
            and `self.batch_shape`.

        """
        value = torch.as_tensor(value, **self.backend)
        scale = self.scale
        df = self.df
        size = self.shape[-1]
        p = -(df * size / 2.) * math.log(2.)
        p = p + 0.5 * (df - size - 1) * value.logdet()
        p = p - 0.5 * df * scale.logdet()
        p = p - (df/2.).mvlgamma(size)
        p = p - 0.5 * scale.inverse().matmul(value).diagonal(-1, -2).sum(-1)
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
        scale = self.scale
        if isinstance(scale, RandomVariable):
            scale = scale.sample(generator=generator)
        df = self.df
        if isinstance(df, RandomVariable):
            df = df.sample(generator=generator)

        nrm = distributions.MultivariateNormal(0, covariance_matrix=scale)
        samp = 0
        zero = torch.as_tensor(0, **self.backend)
        for n in range(df.max().item()):
            vec = nrm.sample(shape)
            vec2 = vec[..., None].matmul(vec[..., None, :].transpose(-1, -2))
            vec2 = torch.where(n < df[..., None, None], vec2, zero)
            samp = samp + vec2
        if out is not None:
            out = out.copy_(samp)
        return out

    def inverse(self):
        return InverseWishart(scale=self.scale.inverse(), df=self.df,
                              **self.backend)


class InverseWishart(MatrixInverse):

    def __init__(self, *args, **kwargs):
        """

        Note
        ----
        .. The `scale` of the inverse-Wishart distribution is the inverse
            of the `scale` of the corresponding Wishart distribution.
            I.e., if `X ~ Wishart(scale=V, df=n)` then
           `X.inverse() ~ InverseWishart(scale=V.inverse(), df=n)`.

        Parameters
        ----------
        scale or rate : (..., p, p) tensor_like
            Scale matrix (or its inverse)
        df : tensor_like
            Degrees of freedom (must be greater than `p - 1`)
        shape : sequence[int], optional
            Expand all parameters so that they have at least this shape.
        dtype : torch.dtype, optional
            Move all parameters to this data type
        device : torch.device or str, optional
            Move all parameters to this device

        """
        if args:
            args = list(args)
            args[0] = args[0].inverse()
        elif 'scale' in kwargs:
            kwargs['scale'] = kwargs['scale'].inverse()
        elif 'rate' in kwargs:
            kwargs['scale'] = kwargs['rate']
            del kwargs['rate']
        super().__init__(Wishart(*args, **kwargs))

    @property
    def mean(self):
        """(*shape) tensor"""
        nan = torch.as_tensor(core.constants.nan, **self.backend)
        e = self.scale / (self.df - self.shape[-1] - 1)
        p = self.shape[-1]
        return torch.where(self.df[..., None, None] <= p + 1, nan, e)

    @property
    def variance(self):
        # TODO: see wikipedia
        raise NotImplementedError

    @property
    def scale(self):
        """(*shape) tensor"""
        return self._base.scale.inverse()

    @property
    def rate(self):
        """(*shape) tensor"""
        return self._base.scale

    @property
    def df(self):
        """(*batch_shape) tensor"""
        return self._base.df

    @property
    def mode(self):
        """(*shape) tensor"""
        return self.scale / (self.df + self.shape[-1] + 1)

    def inverse(self):
        return self.base


class MultivariateNormalPrecisionPrior(Wishart):
    # TODO
    pass


class MultivariateNormalCovariancePrior(InverseWishart):
    # TODO
    pass

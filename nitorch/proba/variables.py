import torch
from nitorch import core


def as_parameter(x, dtype=None, device=None):
    """Convert the input data into a `Parameter` -- either a
    `DeterministicParameter` or a `RandomParameter` -- based on its
    type.

    Parameters
    ----------
    x : tensor_like or Parameter
    dtype : torch.dtype, optional
    device : torch.dtype, optional

    Returns
    -------
    Parameter

    """
    return Parameter(x, dtype=dtype, device=device)


class Parameter:
    """Base class for all parameters of a distribution"""

    def __init__(self, value=None, dtype=None, device=None):
        backend = dict(dtype=dtype, device=device)
        if isinstance(value, Parameter):
            self._value = value._value.to(**backend)
        elif isinstance(value, RandomVariable):
            self._value = value.to(**backend)
        else:
            self._value = torch.as_tensor(value, **backend)


class RandomVariable:
    """Base class for all random variables (i.e., probability distributions)"""

    def __init__(self):
        self.parameters = dict()

    @property
    def shape(self):
        """Shape of the variable.

        The shape of a random variable is the shape of a sample
        (or _realization_) of this variable.
        When the parameters of the random variable are batched, its shape
        is the concatenation of its `batch_shape` and `event_shape`.

        Returns
        -------
        tuple[int] or TensorSize

        """
        return self.batch_shape + self.event_shape

    @property
    def batch_shape(self):
        """Batch component of the shape"""
        return tuple()

    @property
    def event_shape(self):
        """Event component of the shape"""
        return tuple()

    @property
    def E(self):
        """Expected value of the distribution: `\int x p(x) dx`"""
        return self.expectation

    @property
    def V(self):
        """Variance of the distribution: `\int x^2 p(x) dx`
        If the variable is multi-variate (its event_shape is not empty)
        the variance is the trace of its covariance.
        """
        return self.variance

    @property
    def H(self):
        """Entropy: `-\int \log p(x) \times p(x) dx`"""
        return self.entropy

    @property
    def expectation(self):
        """Expected value of the distribution: `\int x p(x) dx`"""
        return self.mean

    @property
    def mean(self):
        """Expected value of the distribution: `\int x p(x) dx`"""
        return NotImplementedError

    @property
    def variance(self):
        """Variance of the distribution: `\int x^2 p(x) dx`
        If the variable is multi-variate (its event_shape is not empty)
        the variance is the trace of its covariance.
        """
        raise NotImplementedError

    @property
    def covariance(self):
        """Covariance of the distribution: `\int (x \times x^T) p(x) dx`"""
        raise NotImplementedError

    @property
    def mode(self):
        """Mode/maximum of the distribution"""
        raise NotImplementedError

    def moment(self, order):
        """Moment of the distribution: `\int x^d p(x) dx`"""
        return (self ** order).mean

    def pdf(self, value):
        """Probability density function"""
        raise NotImplementedError

    def logpdf(self, value):
        """Natural logarithm of the probability density function"""
        raise NotImplementedError

    def cdf(self, value):
        """Cumulative distribution function: `\int_{-\inf}^x p(y) dy`"""
        raise NotImplementedError

    def icdf(self, value):
        """Inverse cumulative distribution function"""
        raise NotImplementedError

    @property
    def entropy(self):
        """Entropy: `-\int \log p(x) \times p(x) dx`"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    #                         RANDOM SAMPLING
    # ------------------------------------------------------------------
    # TODO: `rsample` like in torch.distributions?

    def sample(self, shape=None, generator=None, out=None):
        """Generate random samples from the distribution

        Parameters
        ----------
        shape : tuple[int], optional
            If a shape is provided, as many samples as elements in
            `shape` will be generated.
        generator : torch.Generator, optional
            A random number generator
        out : tensor, optional
            A pre-allocated tensor to fill

        Returns
        -------
        samp : (*shape, *self.shape) tensor
            Random samples from the distribution

        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    #                              BACKEND
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        """Move all parameters to a given dtype and/or device.

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : str or torch.device, optional

        """
        for key, parameter in self.parameters.items():
            self.parameters[key] = parameter.to(*args, **kwargs)
        return self

    @property
    def dtype(self):
        for value in self.parameters.values():
            return value.dtype
        return None

    @property
    def device(self):
        for value in self.parameters.values():
            return value.device
        return None

    @property
    def backend(self):
        return dict(dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    #                              ALGEBRA
    # ------------------------------------------------------------------

    def __add__(self, other):
        from . import algebra, transforms
        if isinstance(other, RandomVariable):
            return algebra.Sum(self, other)
        else:
            return transforms.Shift(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        from . import algebra, transforms
        if isinstance(other, RandomVariable):
            return algebra.Product(self, other)
        else:
            return transforms.Scale(self, other)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1 * self

    def neg(self):
        return -self

    def __truediv__(self, other):
        from . import algebra, transforms
        if isinstance(other, RandomVariable):
            return algebra.Ratio(self, other)
        else:
            return transforms.Scale(self, 1./other)

    def __rtruediv__(self, other):
        from . import algebra, transforms
        if isinstance(other, RandomVariable):
            return algebra.Ratio(other, self)
        else:
            return transforms.Scale(self.reciprocal(), other)

    def reciprocal(self):
        from . import transforms
        return transforms.Reciprocal(self)

    def inverse(self):
        from . import transforms
        return transforms.MatrixInverse(self)

    def log(self):
        from . import transforms
        return transforms.Log(self)

    def exp(self):
        from . import transforms
        return transforms.Exp(self)

    def __pow__(self, power, modulo=None):
        # TODO: modulo?
        from . import transforms
        if isinstance(power, RandomVariable):
            return (power.log() * self).exp()
        else:
            return transforms.Power(self, power)

    def pow(self, order):
        return self ** order

    def square(self):
        return self ** 2

    def __matmul__(self, other):
        from . import algebra, transforms
        if isinstance(other, RandomVariable):
            return algebra.MatrixProduct(self, other)
        else:
            return transforms.MatrixScaleLeft(self, other)

    def __rmatmul__(self, other):
        from . import algebra, transforms
        if isinstance(other, RandomVariable):
            return algebra.MatrixProduct(other, self)
        else:
            return transforms.MatrixScaleRight(self, other)

    def matmul(self, other):
        return self.__matmul__(other)

    def mm(self, other):
        return self.__matmul__(other)

    def matvec(self, other):
        from . import algebra, transforms
        if isinstance(other, RandomVariable):
            return algebra.MatrixVectorProduct(self, other)
        else:
            return transforms.VectorScale(self, other)

    def mv(self, other):
        return self.matvec(other)

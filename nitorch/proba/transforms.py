"""Deterministic transformations of random variables.

This is almost a clone of torch.distributions.transforms.
"""
import nitorch.proba.variables as variables
from nitorch import core
import torch


def _sum_rightmost(x, n=1, keepdim=False):
    dim = list(range(-n, 0, -1))
    return x.sum(dim, keepdim=keepdim)


class TransformedVariable(variables.RandomVariable):
    """Base class for transformed variables."""

    event_dim = None

    def __init__(self, base):
        super().__init__()
        self._base = base

    @property
    def base(self):
        return self._base

    @property
    def event_shape(self):
        return self._base.event_shape

    @property
    def batch_shape(self):
        return self._base.batch_shape

    def transform(self, value):
        """Forward transformation"""
        raise NotImplementedError

    def inv_transform(self, value):
        """Inverse transformation"""
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        """Log absolute determinant of the forward deformation"""
        raise NotImplementedError

    def logpdf(self, value):
        # event_dim = max(event_dim_base, event_dim_transform)
        event_dim = len(self.event_shape)
        event_dim_base = len(self._base.event_shape)
        event_dim_trf = self.event_dim
        inv_value = self.inv_transform(value)
        ll = self._base.logpdf(inv_value)
        ll = _sum_rightmost(ll, event_dim - event_dim_base)
        ld = self.log_abs_det_jacobian(inv_value, value)
        ld = _sum_rightmost(ld, event_dim - event_dim_trf)
        return ll - ld

    def pdf(self, value):
        return self.logpdf(value).exp()

    def sample(self, shape=None, generator=None, out=None):
        samp = self._base.sample(shape, generator=None)
        samp = self.transform(samp)
        if out is not None:
            out = out.copy_(samp)
        return out


class Affine(TransformedVariable):
    """Affine transformed variable (scale and shift)"""

    def __init__(self, base, shift=0, scale=1):
        if (isinstance(shift, variables.RandomVariable) or
            isinstance(scale, variables.RandomVariable)):
            raise TypeError('Transformation parameters cannot be '
                            'random variables')

        super().__init__(base)
        self._shift = variables.as_parameter(shift, **base.backend)
        self._scale = variables.as_parameter(scale, **base.backend)

    event_dim = 0

    def transform(self, value):
        return value * self._scale + self._shift

    def inv_transform(self, value):
        return (value - self._shift) / self._scale

    def log_abs_det_jacobian(self, x, y):
        ld = self._scale.abs().log().reshape(x.shape)
        return ld

    @property
    def mean(self):
        return self.transform(self._base.mean)

    @property
    def variance(self):
        return self._base.variance * self._scale.square()


class Scale(Affine):

    def __init__(self, base, scale=1):
        super().__init__(base, scale=scale)


class Shift(Affine):

    def __init__(self, base, shift=0):
        super().__init__(base, shift=shift)


class Reciprocal(TransformedVariable):

    event_dim = 0

    def transform(self, value):
        return value.reciprocal()

    def inv_transform(self, value):
        return value.reciprocal()

    def log_abs_det_jacobian(self, x, y):
        return x.square().reciprocal().log()


class Exp(TransformedVariable):

    event_dim = 0

    def transform(self, value):
        return value.exp()

    def inv_transform(self, value):
        return value.log()

    def log_abs_det_jacobian(self, x, y):
        return x


class Log(TransformedVariable):

    event_dim = 0

    def transform(self, value):
        return value.log()

    def inv_transform(self, value):
        return value.exp()

    def log_abs_det_jacobian(self, x, y):
        return x


class Power(TransformedVariable):

    def __init__(self, base, order):
        if isinstance(order, variables.RandomVariable):
            raise TypeError('Transformation parameters cannot be '
                            'random variables')

        super().__init__(base)
        self._order = variables.as_parameter(order, **base.backend)

    event_dim = 0

    def transform(self, value):
        return value ** self._order

    def inv_transform(self, value):
        return value ** self._order.reciprocal()

    def log_abs_det_jacobian(self, x, y):
        return (self._order * y / x).abs().log()


class Abs(TransformedVariable):

    event_dim = 0

    def transform(self, value):
        return value.abs()

    def inv_transform(self, value):
        return value

    def log_abs_det_jacobian(self, x, y):
        return 1


class MatrixScaleLeft(TransformedVariable):

    def __init__(self, base, scale):
        if isinstance(scale, variables.RandomVariable):
            raise TypeError('Transformation parameters cannot be '
                            'random variables')

        super().__init__(base)
        self._scale = variables.as_parameter(scale, **base.backend)

    event_dim = 2

    def transform(self, value):
        return self._scale.matmul(value)

    def inv_transform(self, value):
        return self._scale.inverse().matmul(value)

    def log_abs_det_jacobian(self, x, y):
        return self._scale.logdet().abs()

    @property
    def mean(self):
        return self.transform(self._base.mean)


class MatrixScaleRight(TransformedVariable):

    def __init__(self, base, scale):
        if isinstance(scale, variables.RandomVariable):
            raise TypeError('Transformation parameters cannot be '
                            'random variables')

        super().__init__(base)
        self._scale = variables.as_parameter(scale, **base.backend)

    event_dim = 2

    def transform(self, value):
        return value.matmul(self._scale)

    def inv_transform(self, value):
        return value.matmul(self._scale.inverse())

    def log_abs_det_jacobian(self, x, y):
        return self._scale.logdet().abs()

    @property
    def mean(self):
        return self.transform(self._base.mean)


class VectorScale(TransformedVariable):

    def __init__(self, base, scale):
        if isinstance(scale, variables.RandomVariable):
            raise TypeError('Transformation parameters cannot be '
                            'random variables')

        super().__init__(base)
        self._scale = variables.as_parameter(scale, **base.backend)

    event_dim = 1

    def transform(self, value):
        return self._scale.matvec(value)

    def inv_transform(self, value):
        return self._scale.inverse().matmul(value)

    def log_abs_det_jacobian(self, x, y):
        return self._scale.logdet().abs()

    @property
    def mean(self):
        return self.transform(self._base.mean)


class MatrixInverse(TransformedVariable):

    event_dim = 2

    def transform(self, value):
        return value.inverse()

    def inv_transform(self, value):
        return value.inverse()

    def log_abs_det_jacobian(self, x, y):
        return x.logdet().square()

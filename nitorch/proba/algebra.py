"""Implements an algebra (sum and product operators) on the space of
random variables."""
import nitorch.proba.variables as variables
import nitorch.proba.dag as dag
from nitorch import core


class Sum(variables.RandomVariable):
    """Composite variable obtained by summing two random variables."""

    def __init__(self, left, right):
        super.__init__()
        try:
            core.utils.max_shape(self.left, self.right)
        except ValueError:
            raise ValueError('Shapes cannot be broadcasted together')
        self.left = left
        self.right = right

    @property
    def batch_shape(self):
        return core.utils.max_shape(self.left.batch_shape,
                                    self.right.batch_shape)

    @property
    def event_shape(self):
        return core.utils.max_shape(self.left.event_shape,
                                    self.right.event_shape)

    @property
    def mean(self):
        return self.left.mean + self.right.mean

    @property
    def variance(self):
        if dag.is_independent(self.left, self.right):
            return self.left.variance + self.right.variance
        else:
            raise RuntimeError('Variables in the sum are not independent and '
                               'I do not know their covariance.')

    def moment(self, order):
        if order == 1:
            return self.mean
        elif order == 2:
            return self.mean ** 2 + self.variance
        else:
            return (self ** order).mean

    def sample(self, shape=None, generator=None, out=None):
        x = self.left.sample(shape, generator=generator)
        y = self.right.sample(shape, generator=generator)
        samp = x + y
        if out is not None:
            out = out.copy_(samp)
        return out


class Product(variables.RandomVariable):
    """Composite variable obtained by multiplying two random variables."""

    def __init__(self, left, right):
        super.__init__()
        try:
            core.utils.max_shape(self.left, self.right)
        except ValueError:
            raise ValueError('Shapes cannot be broadcasted together')
        self.left = left
        self.right = right

    @property
    def batch_shape(self):
        return core.utils.max_shape(self.left.batch_shape,
                                    self.right.batch_shape)

    @property
    def event_shape(self):
        return core.utils.max_shape(self.left.event_shape,
                                    self.right.event_shape)

    @property
    def mean(self):
        if dag.is_independent(self.left, self.right):
            return self.left.mean * self.right.mean
        else:
            raise RuntimeError('Variables in the product are not independent '
                               'and I do not know their covariance.')

    @property
    def variance(self):
        if dag.is_independent(self.left, self.right):
            return (self.left.variance * self.right.variance +
                    self.left.variance * self.right.mean.square() +
                    self.right.variance * self.left.mean.square())
        else:
            raise RuntimeError('Variables in the sum are not independent and '
                               'I do not know their covariance.')

    def sample(self, shape=None, generator=None, out=None):
        x = self.left.sample(shape, generator=generator)
        y = self.right.sample(shape, generator=generator)
        samp = x * y
        if out is not None:
            out = out.copy_(samp)
        return out


class Ratio(Product):
    """Composite variable obtained by taking the ratio of two random variables.
    """

    def __init__(self, left, right):
        super.__init__(left, right.reciprocal())


class MatrixProduct(variables.RandomVariable):
    """Composite variable obtained by multiplying two random variables."""

    def __init__(self, left, right):
        super.__init__()
        try:
            core.utils.max_shape(self.left, self.right)
        except ValueError:
            raise ValueError('Shapes cannot be broadcasted together')
        if len(left.event_shape) != 2 or len(right.event_shape) != 2:
            raise ValueError('Expected two matrices as inputs')
        if left.event_shape[-1] != right.event_shape[0]:
            raise ValueError('Matrix shape not compatible for multiplication')
        self.left = left
        self.right = right

    @property
    def batch_shape(self):
        return core.utils.max_shape(self.left.batch_shape,
                                    self.right.batch_shape)

    @property
    def event_shape(self):
        return (self.left.event_shape[0], self.right.event_shape[-1])

    @property
    def mean(self):
        if dag.is_independent(self.left, self.right):
            return self.left.mean.matmul(self.right.mean)
        else:
            raise RuntimeError('Variables in the product are not independent '
                               'and I do not know their covariance.')

    def sample(self, shape=None, generator=None, out=None):
        x = self.left.sample(shape, generator=generator)
        y = self.right.sample(shape, generator=generator)
        samp = x.matmul(y)
        if out is not None:
            out = out.copy_(samp)
        return out


class MatrixVectorProduct(variables.RandomVariable):
    """Composite variable obtained by multiplying two random variables."""

    def __init__(self, left, right):
        super.__init__()
        try:
            core.utils.max_shape(self.left, self.right)
        except ValueError:
            raise ValueError('Shapes cannot be broadcasted together')
        if len(left.event_shape) != 2 or len(right.event_shape) != 1:
            raise ValueError('Expected a matrix and a vector as inputs')
        if left.event_shape[-1] != right.event_shape[0]:
            raise ValueError('Matrix shape not compatible for multiplication')
        self.left = left
        self.right = right

    @property
    def batch_shape(self):
        return core.utils.max_shape(self.left.batch_shape,
                                    self.right.batch_shape)

    @property
    def event_shape(self):
        return self.right.event_shape

    @property
    def mean(self):
        if dag.is_independent(self.left, self.right):
            return self.left.mean.matvec(self.right.mean)
        else:
            raise RuntimeError('Variables in the product are not independent '
                               'and I do not know their covariance.')

    def sample(self, shape=None, generator=None, out=None):
        x = self.left.sample(shape, generator=generator)
        y = self.right.sample(shape, generator=generator)
        samp = x.matmul(y[..., None])[..., 0]
        if out is not None:
            out = out.copy_(samp)
        return out

from nitorch.nn.base import Module
import torch


class RandomDistribution(Module, torch.distributions.Distribution):

    def __init__(self, distribution, *args, **kwargs):
        super().__init__()
        self.distribution = distribution
        self.args = args
        self.kwargs = kwargs

    def forward(self):
        args = list(self.args)
        kwargs = dict(self.kwargs)

        for i, value in args:
            if callable(value):
                args[i] = value()

        for i, (key, value) in kwargs.items():
            if callable(value):
                kwargs[key] = value()

        dist = self.distribution(*args, **kwargs)
        return dist


# utilities for random generators

class _Dist:
    def sample(self, shape=None):
        return self.dist.sample(shape or [])


class _Normal(_Dist):
    def __init__(self, mean, scale, **backend):
        mean = torch.as_tensor(mean, **backend)
        scale = torch.as_tensor(scale, **backend)
        self.dist = torch.distributions.Normal(mean, scale)


class _LogNormal(_Dist):
    def __init__(self, mean, scale, **backend):
        mean = torch.as_tensor(mean, **backend)
        scale = torch.as_tensor(scale, **backend)
        scale = (scale/mean).square().add(1).log()
        mean = mean.log() - scale.div(2)
        scale = scale.sqrt()
        self.dist = torch.distributions.LogNormal(mean, scale)


class _Uniform(_Dist):
    def __init__(self, mean, scale, **backend):
        mean = torch.as_tensor(mean, **backend)
        scale = torch.as_tensor(scale, **backend)
        width = 3.46410161514 * scale  # sqrt(12)
        vmin = mean - width/2
        vmax = mean + width/2
        self.dist = torch.distributions.Uniform(vmin, vmax)


class _Gamma(_Dist):
    def __init__(self, mean, scale, **backend):
        mean = torch.as_tensor(mean, **backend)
        scale = torch.as_tensor(scale, **backend)
        scale = scale.square()  # variance
        rate = mean / scale
        concentration = mean.square() / scale
        self.dist = torch.distributions.Gamma(concentration, rate)


class _Dirac(_Dist):
    def __init__(self, mean, scale=None, **backend):
        self.value = torch.as_tensor(mean, **backend)

    def sample(self, shape=None):
        value = self.value
        if shape:
            value = value.expand([*shape, *value.shape])
        value = value.clone()
        return value

class _DiscreteUniform(_Dist):
    def __init__(self, mean, scale, **backend):
        dtype = backend.pop('dtype', torch.long)
        backend['dtype'] = torch.get_default_dtype()
        mean = torch.as_tensor(mean, **backend)
        scale = torch.as_tensor(scale, **backend)
        scale = scale.square()  # variance
        width = 3.46410161514 * scale.add(1).sqrt()  # sqrt(12)
        vmin = (mean - width/2).ceil().to(dtype)
        vmax = (mean + width/2).floor().to(dtype)
        self.vmin = vmin
        self.vmax = vmax

    def sample(self, shape=None):
        shape = shape or []
        import itertools
        if self.vmin.shape:
            out = self.vmin.new_empty([*shape, *self.vmin.shape])
            for coord in itertools.product(range(s) for s in self.vmin.shape):
                vmin = self.vmin(coord)
                vmax = self.vmax(coord)
                out[(Ellipsis, *coord)] = torch.randint(vmin, vmax+1, shape)
        else:
            out = torch.randint(self.vmin, self.vmax+1, shape)
        return out


def _get_dist(name):
    if not isinstance(name, str):
        return name
    name = name.lower()
    if name == 'normal':
        return _Normal
    elif name == 'lognormal':
        return _LogNormal
    elif name == 'uniform':
        return _Uniform
    elif name == 'duniform':
        return _DiscreteUniform
    elif name == 'gamma':
        return _Gamma
    elif name == 'dirac' or name is None:
        return _Dirac
    raise ValueError(f'Unknown distribution: {name}')

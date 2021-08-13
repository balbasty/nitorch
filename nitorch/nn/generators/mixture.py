import torch
from nitorch.core.utils import expand, channel2last
from nitorch.core import py, utils, math
from nitorch.nn.base import Module
from .distribution import RandomDistribution
from .field import RandomFieldSpline


class RandomSmoothSimplexMap(Module):
    """Sample a smooth categorical prior"""

    def __init__(self, shape=None, nb_classes=None, amplitude=5, fwhm=15,
                 logits=False, implicit=False, device=None, dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int], optional
            Output shape
        nb_classes : int, optional
            Number of classes (excluding background)
        amplitude : float or callable or list, default=5
            Amplitude of the random field (per channel).
        fwhm : float or callable or list, default=15
            Full-width at half-maximum of the random field (per channel).
        logits : bool, default=False
            Input priors are log-odds instead of probabilities
        implicit : bool, default=False
            Input priors have an implicit K+1-th class.
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.
        """

        super().__init__()
        self.logits = logits
        self.implicit = implicit
        shape = py.make_list(shape)
        self.field = RandomFieldSpline(shape=shape, channel=nb_classes,
                                       amplitude=amplitude, fwhm=fwhm,
                                       device=device, dtype=dtype)

    nb_classes = property(lambda self: self.field.channel)
    shape = property(lambda self: self.field.shape)
    amplitude = property(lambda self: self.field.amplitude)
    fwhm = property(lambda self: self.field.fwhm)
    device = property(lambda self: self.field.device)
    dtype = property(lambda self: self.field.dtype)

    def to(self, *args, **kwargs):
        self.field.to(*args, **kwargs)
        super().to(*args, **kwargs)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        prob : (batch, nb_classes[+1], *shape) tensor
            Probabilities

        """

        # get arguments
        opt = {
            'shape': overload.get('shape', self.shape),
            'channel': overload.get('nb_classes', self.nb_classes),
            'amplitude': overload.get('amplitude', self.amplitude),
            'fwhm': overload.get('fwhm', self.fwhm),
            'dtype': overload.get('dtype', self.dtype),
            'device': overload.get('device', self.device),
        }
        implicit = overload.get('implicit', self.implicit)
        logits = overload.get('logits', self.logits)
        if not implicit:
            opt['channel'] = opt['channel'] + 1

        # preprocess amplitude
        # > RandomField broadcast amplitude to (channel, *shape), with
        #   padding from the left, which means that a 1d amplitude would
        #   be broadcasted to (1, ..., dim) instead of (dim, ..., 1)
        # > We therefore reshape amplitude to avoid left-side padding
        def preprocess(a):
            a = torch.as_tensor(a)
            a = utils.unsqueeze(a, dim=-1, ndim=opt['channel']+1-a.dim())
            return a
        amplitude = opt['amplitude']
        if callable(amplitude):
            amplitude_fn = amplitude
            amplitude = lambda *args, **kwargs: preprocess(amplitude_fn(*args, **kwargs))
        else:
            amplitude = preprocess(amplitude)
        opt['amplitude'] = amplitude

        sample = self.field(batch, **opt)
        if logits:
            sample = math.log_softmax(sample, 1, implicit=implicit)
        else:
            sample = math.softmax(sample, 1, implicit=implicit)

        return sample


class RandomSmoothLabelMap(Module):
    """Sample a smooth (i.e., contiguous) label map.

    This function first samples smooth log-probabilities from a
    smooth random field and then returns the maximum-probability labels.
    """

    def __init__(self, shape=None, nb_classes=None, amplitude=5, fwhm=15,
                 device=None, dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int], optional
            Output shape
        nb_classes : int, optional
            Number of classes (excluding background)
        amplitude : float or callable or list, default=5
            Amplitude of the random field (per channel).
        fwhm : float or callable or list, default=15
            Full-width at half-maximum of the random field (per channel).
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.
        """

        super().__init__()
        self.prob = RandomSmoothSimplexMap(
            shape=shape,
            nb_classes=nb_classes,
            amplitude=amplitude,
            fwhm=fwhm,
            device=device,
            dtype=dtype)

    nb_classes = property(lambda self: self.prob.nb_classes)
    shape = property(lambda self: self.prob.shape)
    amplitude = property(lambda self: self.prob.amplitude)
    fwhm = property(lambda self: self.prob.fwhm)
    device = property(lambda self: self.prob.device)
    dtype = property(lambda self: self.prob.dtype)

    def to(self, *args, **kwargs):
        self.field.to(*args, **kwargs)
        super().to(*args, **kwargs)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        vel : (batch, 1, dim) tensor[long]
            Labels

        """
        prob = self.prob(batch, **overload)
        return prob.argmax(dim=1, keepdim=True)


class RandomLabel(Module):
    """Sample from a categorical distribution."""

    def __init__(self, shape=None, logits=False, implicit=False):
        """

        Parameters
        ----------
        shape : sequence[int], optional
            Output shape
        logits : bool, default=False
            Input priors are log-odds instead of probabilities
        implicit : bool, default=False
            Input priors have an implicit K+1-th class.
        """

        super().__init__()
        self.shape = shape
        self.logits = logits
        self.implicit = implicit

    def forward(self, prior, **overload):
        """

        Parameters
        ----------
        prior : (batch, channel, *shape)
            Prior probabilities or log-odds of the Categorical distribution
        overload : dict
            All parameters defined at buildtime can be overridden at call time

        Returns
        -------
        sample : (batch, 1, *shape)

        """

        # read arguments
        shape = overload.get('shape', self.shape)
        logits = overload.get('logits', self.logits)
        implicit = overload.get('implicit', self.implicit)

        # call prior in case it is a random parameter
        prior = prior() if callable(prior) else torch.as_tensor(prior)

        # repeat prior if shape provided
        if shape is not None:
            if prior.dim() != 2:
                raise ValueError('Expected tensor with shape (batch, channel) '
                                 'but got {}'.format(prior.shape))
            prior = expand(prior, [*prior.shape, *shape], side='right')

        # add implicit class
        if implicit:
            shape = list(prior.shape)
            shape[1] = 1
            zero = torch.zeros(shape, dtype=prior.dtype, device=prior.device)
            prior = torch.cat((prior, zero), dim=1)

        # reshape in 2d
        batch, channel, *shape = prior.shape
        prior = channel2last(prior)  # make class dimension last
        kwargs = dict()
        kwargs['logits' if logits else 'probs'] = prior

        # sample
        sample = torch.distributions.Categorical(**kwargs).sample()
        sample = sample.reshape([batch, 1, *shape])

        return sample


class RandomMixture(Module):
    """Sample from a mixture of distributions."""

    def __init__(self, distributions):
        """

        Parameters
        ----------
        distributions : sequence[distribution]
            Distributions should have attribute `batch_shape` and
            method `sample` with argument `sample_shape`.
            The distribution sample shape should be broadcastable to
            (batch, channel).


        """
        super().__init__()
        self.distributions = distributions
        self._algo = 'where'

    def forward(self, index, **overload):

        distributions = list(overload.get('distributions', self.distributions))

        # call distribution in case it is a random distribution
        for i, dist in enumerate(distributions):
            if isinstance(dist, RandomDistribution):
                distributions[i] = dist()
            elif not isinstance(dist, torch.distributions.Distribution):
                raise TypeError('Distributions should be `Distribution` or '
                                '`RandomDistribution` objects.')

        # get shape
        index = torch.as_tensor(index)
        device = index.device
        nb_dim = index.dim() - 2
        batch = index.shape[0]
        channels = index.shape[1]
        shape = index.shape[2:]

        # check distribution shape
        batches = [batch]
        channels = [] if channels == 1 else [channels]
        for dist in distributions:
            sample_shape = dist.batch_shape + dist.event_shape
            if len(sample_shape) > 0:
                channels.append(sample_shape[0])
            if len(sample_shape) > 1:
                batches.append(sample_shape[1])
            if len(sample_shape) > 2:
                raise ValueError('Samples should have shape (batch, channel). '
                                 'Got {}'.format(sample_shape))
        if len(set(batches)) != 1:
            raise ValueError('Batch shapes not consistent: {}'
                             .format(batches))
        if len(set(channels)) not in (0, 1):
            raise ValueError('Channel shapes not consistent: {}'
                             .format(channels))
        channels = channels[0] if channels else 1

        # sample
        if self._algo == 'scatter':
            raise NotImplementedError
        else:
            assert self._algo == 'where'
            sample = torch.zeros([batch, channels, *shape], device=device)
            for i, dist in enumerate(distributions):
                sample_shape = dist.batch_shape + dist.event_shape
                if len(sample_shape) == 0:
                    sample_shape = [batch, channels, *shape]
                    permute = [*range(len(sample_shape))]
                elif len(sample_shape) == 1:
                    sample_shape = [batch, *shape]
                    permute = [0, -1, *range(1, len(sample_shape))]
                else:
                    permute = [-2, -1, *range(len(sample_shape))]
                sample1 = dist.sample(sample_shape)
                sample1 = sample1.permute(permute)
                sample = torch.where(index == i, sample1, sample)

        return sample



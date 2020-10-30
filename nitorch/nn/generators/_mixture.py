import torch
from ..modules._base import Module
from ...core.utils import expand, unsqueeze, channel2last
from ._distribution import RandomDistribution


class CategoricalSample(Module):
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


class MixtureSample(Module):
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
        channels = [] if channels is 1 else [channels]
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



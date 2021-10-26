import torch
from nitorch import spatial
from nitorch.core.utils import expand, channel2last
from nitorch.core import py, utils, math
from nitorch.nn.base import Module
from .distribution import RandomDistribution, _get_dist
from .field import RandomFieldSpline, HyperRandomFieldSpline


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
        amplitude : float or (channel,) vector_like, default=5
            Amplitude of the random field (per channel).
        fwhm : float or (channel,) vector_like, default=15
            Full-width at half-maximum of the random field (per channel).
        logits : bool, default=False
            Outputs are log-odds instead of probabilities
        implicit : bool, default=False
            Outputs have an implicit K+1-th class.
        device : torch.device, optional
            Output tensor device.
        dtype : torch.dtype, optional
            Output tensor datatype.
        """

        super().__init__()
        self.logits = logits
        self.implicit = implicit
        shape = py.make_list(shape)
        self.field = RandomFieldSpline(shape=shape, channel=nb_classes+1,
                                       amplitude=amplitude, fwhm=fwhm,
                                       device=device, dtype=dtype)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1

        Other Parameters
        ----------------
        shape : sequence[int], optional
        dtype : torch.dtype, optional
        device : torch.device, optional

        Returns
        -------
        prob : (batch, nb_classes[+1], *shape) tensor
            Logits or Probabilities

        """

        # get arguments
        opt = {
            'shape': overload.get('shape', self.field.shape),
            'dtype': overload.get('dtype', self.field.dtype),
            'device': overload.get('device', self.field.device),
        }

        sample = self.field(batch, **opt)
        softmax = math.log_softmax if self.logits else math.softmax
        sample = softmax(sample, 1, implicit=(False, self.implicit))
        return sample


class HyperRandomSmoothSimplexMap(Module):
    """Sample a smooth categorical prior"""

    def __init__(self, shape=None, nb_classes=None,
                 amplitude='lognormal', amplitude_exp=5, amplitude_scale=2,
                 fwhm='lognormal', fwhm_exp=15, fwhm_scale=5,
                 logits=False, implicit=False, device=None, dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int], optional
            Output shape
        nb_classes : int, optional
            Number of classes (excluding background)
        amplitude : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        amplitude_exp : float or (channel,) vector_like, default=5
        amplitude_scale : float or (channel,) vector_like, default=2
        fwhm : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        fwhm_exp : float or (channel,) vector_like, default=15
        fwhm_scale : float or (channel,) vector_like, default=5
        logits : bool, default=False
            Outputs are log-odds instead of probabilities
        implicit : bool, default=False
            Outputs have an implicit K+1-th class.
        device : torch.device, optional
            Output tensor device.
        dtype : torch.dtype, optional
            Output tensor datatype.
        """

        super().__init__()
        self.logits = logits
        self.implicit = implicit
        shape = py.make_list(shape)
        self.field = HyperRandomFieldSpline(
            shape=shape, channel=nb_classes+1, amplitude=amplitude,
            amplitude_exp=amplitude_exp, amplitude_scale=amplitude_scale,
            fwhm=fwhm, fwhm_exp=fwhm_exp, fwhm_scale=fwhm_scale,
            device=device, dtype=dtype)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1

        Other Parameters
        ----------------
        shape : sequence[int], optional
        dtype : torch.dtype, optional
        device : torch.device, optional

        Returns
        -------
        prob : (batch, nb_classes[+1], *shape) tensor
            Logits or Probabilities

        """

        # get arguments
        opt = {
            'shape': overload.get('shape', self.field.shape),
            'dtype': overload.get('dtype', self.field.dtype),
            'device': overload.get('device', self.field.device),
        }

        sample = self.field(batch, **opt)
        softmax = math.log_softmax if self.logits else math.softmax
        sample = softmax(sample, 1, implicit=(False, self.implicit))
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


class RandomGaussianMixture(Module):
    """Sample from a Gaussian mixture with known label/responsibilities.

    A FWHM can be provided to model the effect of a conditional random field.
    """

    def __init__(self,
                 means=tuple(range(10)),
                 scales=1,
                 fwhm=0,
                 dtype=None):
        """

        Parameters
        ----------
        means : ([C], K) tensor_like, default=`0:10`
        scales : float or ([C], K) tensor_like, default=1
        fwhm : float or (C,) vector_like, default=0
        dtype : torch.dtype, optional
        """
        super().__init__()
        self.means = means
        self.scales = scales
        self.fwhm = fwhm
        self.dtype = dtype or torch.get_default_dtype()

    def forward(self, x):
        """

        Parameters
        ----------
        x : (batch, 1 or classes[-1], *shape) tensor
            Labels or probabilities

        Returns
        -------
        x : (batch, channel, *shape) tensor

        """
        batch, _, *shape = x.shape
        device = x.device
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = self.dtype
        backend = dict(dtype=dtype, device=device)

        means = torch.as_tensor(self.means, **backend)
        scales = torch.as_tensor(self.scales, **backend)
        nb_classes = means.shape[-1]
        if means.dim() == 2:
            channel = len(means)
        elif scales.dim() == 2:
            channel = len(scales)
        else:
            channel = 1
        means = means.expand([channel, nb_classes]).clone()
        scales = scales.expand([channel, nb_classes]).clone()
        fwhm = utils.make_vector(self.fwhm, nb_classes, **backend)

        implicit = x.shape[1] < nb_classes
        out = torch.zeros([batch, channel, *shape], **backend)
        for k in range(nb_classes):
            sampler = _get_dist('normal')(means[:, k], scales[:, k])
            if x.dtype.is_floating_point:
                y1 = sampler.sample([batch, *shape])
                y1 = utils.movedim(y1, -1, 1)
                for c, f in enumerate(fwhm):
                    if f > 0:
                        y1[:, c] = spatial.smooth(
                            y1[:, c], fwhm=f, dim=len(shape),
                            padding='same', bound='dct2')
                if not implicit:
                    x1 = x[:, k, None]
                elif k > 0:
                    x1 = x[:, k-1, None]
                else:
                    x1 = x.sum(1, keepdim=True).neg_().add_(1)
                out.addcmul(y1, x1)
            else:
                mask = (x.squeeze(1) == k)
                y1 = sampler.sample([mask.sum().long()])
                out = utils.movedim(out, 1, -1)
                out[mask, :] = y1
                out = utils.movedim(out, -1, 1)

        return out


class HyperRandomGaussianMixture(Module):
    """Sample from a Gaussian mixture with randomized hyper-parameters."""

    def __init__(self,
                 nb_classes,
                 nb_channels=1,
                 means='normal',
                 means_exp=0,
                 means_scale=10,
                 scales='gamma',
                 scales_exp=1,
                 scales_scale=5,
                 fwhm='lognormal',
                 fwhm_exp=3,
                 fwhm_scale=5,
                 background_zero=False,
                 dtype=None):
        """

        Parameters
        ----------
        nb_classes : int
        nb_channels : int, default=1
        means : {'normal', 'lognormal', 'uniform', 'gamma'}, default='normal'
        means_exp : ([[C], K]) tensor_like, default=0
        means_scale : ([[C], K]) tensor_like, default=10
        scales : {'normal', 'lognormal', 'uniform', 'gamma'}, default='gamma'
        scales_exp : float or ([[C], K]) tensor_like, default=1
        scales_scale : float or ([[C], K]) tensor_like, default=5
        dtype : torch.dtype, optional
        """
        super().__init__()
        self.nb_classes = nb_classes
        self.nb_channels = nb_channels
        self.means = _get_dist(means)
        self.means_exp = means_exp
        self.means_scale = means_scale
        self.scales = _get_dist(scales)
        self.scales_exp = scales_exp
        self.scales_scale = scales_scale
        self.fwhm = _get_dist(fwhm)
        self.fwhm_exp = fwhm_exp
        self.fwhm_scale = fwhm_scale
        self.background_zero = background_zero
        self.dtype = dtype or torch.get_default_dtype()

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, 1 or classes[-1], *shape) tensor
            Labels or probabilities

        Returns
        -------
        x : (batch, channel, *shape) tensor

        """
        batch, _, *shape = x.shape

        device = x.device
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = self.dtype
        backend = dict(dtype=dtype, device=device)

        nb_classes = overload.get('nb_classes', self.nb_classes)
        nb_channels = overload.get('nb_channels', self.nb_channels)
        means_exp = torch.as_tensor(self.means_exp, **backend)
        means_scale = torch.as_tensor(self.means_scale, **backend)
        scales_exp = torch.as_tensor(self.scales_exp, **backend)
        scales_scale = torch.as_tensor(self.scales_scale, **backend)
        means_exp = means_exp.expand([nb_channels, nb_classes]).clone()
        means_scale = means_scale.expand([nb_channels, nb_classes]).clone()
        scales_exp = scales_exp.expand([nb_channels, nb_classes]).clone()
        scales_scale = scales_scale.expand([nb_channels, nb_classes]).clone()
        fwhm_exp = utils.make_vector(self.fwhm_exp, nb_channels, **backend)
        fwhm_scale = utils.make_vector(self.fwhm_scale, nb_channels, **backend)

        out = torch.zeros([batch, nb_channels, *shape], **backend)
        for b in range(batch):
            means = self.means(means_exp, means_scale).sample()
            scales = self.scales(scales_exp, scales_scale).sample().clamp_min_(0)
            if self.background_zero:
                means[:, 0] = 0
                scales[:, 0] = 0.1
            fwhm = self.fwhm(fwhm_exp, fwhm_scale).sample().clamp_min_(0)
            sampler = RandomGaussianMixture(means, scales, fwhm=fwhm)
            out[b] = sampler(x[None, b])[0]
        return out

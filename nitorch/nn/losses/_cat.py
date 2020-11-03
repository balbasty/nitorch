"""Losses for categorical data."""

import torch
from ._base import Loss
from ...core.math import nansum


class CategoricalLoss(Loss):
    """(Expected) Negative log-likelihood of a Categorical distribution.

    This loss loosely corresponds to the categorical cross-entropy loss.
    In this loss, we accept one-hot-encoded "ground truth" on top of
    hard labels.
    Note that no normalization is performed here. It is assumed that a
    softmax or log-softmax has already been applied to the prior (and
    eventually to the observation, if one-hot encoded).

    """

    def __init__(self, one_hot_map=None, log=True, implicit=False, *args, **kwargs):
        """

        Parameters
        ----------
        one_hot_map : list[int] or callable, optional
            Mapping from one-hot to hard index.
            By default: identity mapping.
        log : bool, default=True
            If True, priors are log-probabilities.
            Else, they are probabilities and we take their log in the
            forward pass.
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.one_hot_map = one_hot_map
        self.log = log
        self.implicit = implicit

    def forward(self, prior, obs, **overridden):
        """

        Parameters
        ----------
        prior : (nb_batch, nb_class, *spatial) tensor
            (Log)-prior probabilities
        obs : (nb_batch, nb_class|1, *spatial) tensor
            Observed classes (or their expectation).
            If `nb_class == 1` assume that they are hard labels and
            use `one_hot_map` to map one-hot labels to hard labels.
            Hard labels should have an integer data type.
            Else, assume that they are expected labels.
            Expected/soft labels should have a floating point data type.
        overridden : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        # take log if needed
        # TODO: it's a bit ugly, I need to find an elegant yet stable
        #   way to do this.
        log = overridden.get('log', self.log)
        implicit = overridden.get('implicit', self.implicit)
        if log:
            logprior = prior
            if implicit:
                logprior_last = 1 - logprior.exp().sum(dim=1, keepdim=True)
                logprior_last = logprior_last.clamp(min=1e-7, max=1-1e-7).log()
        elif implicit:
            logprior_last = 1 - prior.sum(dim=1, keepdim=True)
            logprior_last = logprior_last.clamp(min=1e-7, max=1-1e-7).log()
            logprior = prior.clamp(min=1e-7, max=1-1e-7).log()

        if obs.dtype in (torch.half, torch.float, torch.double):
            # soft labels
            loss = logprior * obs
            if implicit:
                obs_last = 1-obs.sum(dim=1, keepdim=True)
                loss = torch.cat((loss, logprior_last*obs_last), dim=1)
        else:
            # hard labels
            one_hot_map = overridden.get('one_hot_map', self.one_hot_map)
            if one_hot_map is None:
                one_hot_map = list(range(logprior.shape[1]))
            if len(one_hot_map) != logprior.shape[1]:
                raise ValueError('Number of classes in prior and map '
                                 'do not match: {} and {}.'
                                 .format(logprior.shape[1], len(one_hot_map)))
            if obs.shape[1] != 1:
                raise ValueError('Hard label maps cannot be multi-channel.')
            loss = []
            for soft, hard in enumerate(one_hot_map):
                logprior1 = logprior[:, soft, ...][:, None, ...]
                loss.append(logprior1 * (obs == hard))
            if implicit:
                obs_last = obs not in one_hot_map
                loss.append(logprior_last * obs_last)
            loss = torch.cat(loss, dim=1)

        # negate
        loss = -loss

        # reduction
        return super().forward(loss, **overridden)


class DiceLoss(Loss):
    """Negative Dice/F1 score."""

    def forward(self, predicted, reference, **overridden):
        """

        Parameters
        ----------
        predicted : (batch, channel, *spatial) tensor
            Predicted one-hot encoded classes.
        reference : (batch, channel, *spatial) tensor
            Reference one-hot encoded classes.
        overridden : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        # TODO:
        #   I need to make this a lot more generic:
        #   - inputs can be hard labels or prob
        #   - the background class can be implicit
        #   - we may want to compute the dice w.r.t. a subset of classes

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference,
                                    dtype=predicted.dtype,
                                    device=predicted.device)

        # only compute Dice on foreground (unless the ref has a
        # background class
        nb_class = max(predicted.shape[1], reference.shape[1])
        predicted = predicted[:, :nb_class, ...]
        reference = reference[:, :nb_class, ...]

        nb_dim = predicted.dim() - 2
        dims = list(range(2, nb_dim+2))

        inter = nansum(predicted * reference, dim=dims)
        union = nansum(predicted + reference, dim=dims)
        loss = -2*inter/union

        return super().forward(loss, **overridden)

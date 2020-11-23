"""Losses for categorical data."""

import torch
from ._base import Loss
from nitorch.core.math import nansum
from nitorch.core.utils import isin
from nitorch.core.pyutils import make_list


def _pad_zero(x, implicit=False):
    if not implicit:
        return x
    zero_shape = [x.shape[0], 1, *x.shape[2:]]
    zero = x.new_zeros([1]).expand(zero_shape)
    x = torch.cat((x, zero), dim=1)
    return x


def _pad_norm(x, implicit=False):
    if not implicit:
        return x / x.sum(dim=1, keepdim=True)
    x = torch.cat((x, 1 - x.sum(dim=1, keepdim=True)), dim=1)
    return x


def _softmax(x, implicit=False):
    """Safe softmax (with implicit class)"""
    if implicit:
        x = _pad_zero(x)
    x = torch.softmax(x, dim=1)
    return x


def _logsoftmax(x, implicit=False):
    """Log(softmax(x)) (with implicit class)"""
    x = _pad_zero(x, implicit)
    return torch.log_softmax(x, dim=1)


def _log(x, implicit=False):
    """Log (with implicit class)"""
    x = _pad_norm(x, implicit)
    return x.clamp(min=1e-7, max=1-1e-7).log()


class CategoricalLoss(Loss):
    """(Expected) Negative log-likelihood of a Categorical distribution.

    This loss loosely corresponds to the categorical cross-entropy loss.
    In this loss, we accept one-hot-encoded "ground truth" on top of
    hard labels.

    """

    def __init__(self, one_hot_map=None, log=True, implicit=False,
                 *args, **kwargs):
        """

        Parameters
        ----------
        one_hot_map : list[int or list[int] or None], optional
            Mapping from one-hot to hard index. Default: identity mapping.
            Each index of the list corresponds to a soft label.
            Each soft label can be mapped to a hard label or a list of
            hard labels. Up to one `None` can be used, in which case the
            corresponding soft label will be considered a background class
            and will be mapped to all remaining labels. If `len(one_hot_map)`
            has one less element than the number of soft labels, such a
            background class will be appended to the right.
        log : bool, default=True
            If True, priors are log-probabilities (pre-softmax).
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
        prior : (nb_batch, nb_class[-1], *spatial) tensor
            (Log)-prior probabilities
        obs : (nb_batch, nb_class[-1]|1, *spatial) tensor
            Observed classes (or their expectation).
                * If `obs` has a floating point data type (`half`,
                  `float`, `double`) it is assumed to hold one-hot or
                  soft labels, and its channel dimension should be
                  `nb_class` or `nb_class - 1`.
                * If `obs` has an integer or boolean data type, it is
                  assumed to hold hard labels and its channel dimension
                  should be 1. Eventually, `one_hot_map` is used to map
                  one-hot labels to hard labels.
        overridden : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        log = overridden.get('log', self.log)
        implicit = overridden.get('implicit', self.implicit)

        prior = torch.as_tensor(prior)
        obs = torch.as_tensor(obs, device=prior.device)

        # take log if needed
        if log:
            logprior = _logsoftmax(prior, implicit=implicit)
        else:
            logprior = _log(prior, implicit=implicit)
        nb_classes = logprior.shape[1]

        if obs.dtype in (torch.half, torch.float, torch.double):
            # soft labels
            obs = obs.to(prior.dtype)
            if obs.shape[1] == nb_classes - 1:
                # obs is implicit too
                obs_last = 1 - obs.sum(dim=1, keepdim=True)
                obs = torch.cat((obs, obs_last), dim=1)
            elif obs.shape[1] != nb_classes:
                raise ValueError('Number of classes not consistent. '
                                 'Expected {} or {} but got {}.'.format(
                                 nb_classes, nb_classes-1, obs.shape[1]))
            loss = logprior * obs
        else:
            # hard labels
            one_hot_map = overridden.get('one_hot_map', self.one_hot_map)
            if one_hot_map is None:
                one_hot_map = list(range(logprior.shape[1]))
            if len(one_hot_map) == nb_classes - 1:
                one_hot_map = [*one_hot_map, None]
            if len(one_hot_map) != nb_classes:
                raise ValueError('Number of classes in prior and map '
                                 'do not match: {} and {}.'
                                 .format(nb_classes, len(one_hot_map)))
            one_hot_map = list(map(lambda x: make_list(x) if x is not None else x,
                                   one_hot_map))
            if sum(elem is None for elem in one_hot_map) > 1:
                raise ValueError('Cannot have more than one implicit class')
            if obs.shape[1] != 1:
                raise ValueError('Hard label maps cannot be multi-channel.')
            loss = []
            for soft, hard in enumerate(one_hot_map):
                logprior1 = logprior[:, soft, ...][:, None, ...]
                if hard is None:
                    # implicit class
                    all_labels = [l for l in one_hot_map if l is not None]
                    obs1 = ~isin(obs, all_labels)
                else:
                    obs1 = isin(obs, hard)
                loss.append(logprior1 * obs1)
            loss = torch.cat(loss, dim=1)

        # negate
        loss = -loss

        # reduction
        return super().forward(loss, **overridden)


class DiceLoss(Loss):
    """Negative Dice/F1 score."""

    def __init__(self, one_hot_map=None, log=False, implicit=False,
                 discard_background=False, weighted=False,
                 *args, **kwargs):
        """

        Parameters
        ----------
        one_hot_map : list[int] or list[list[int]], optional
            Mapping from one-hot to hard index.
            By default: identity mapping.
        log : bool, default=False
            If True, priors are log-probabilities.
            Else, they are probabilities and we take their log in the
            forward pass.
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        weighted : bool, default=False
            If True, weight the Dice of each class by its size in the
            reference.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.one_hot_map = one_hot_map
        self.log = log
        self.implicit = implicit
        self.discard_background = discard_background
        self.weighted = weighted

    def forward(self, predicted, reference, **overridden):
        """

        Parameters
        ----------
        predicted : (batch, nb_class[-1], *spatial) tensor
            Predicted classes.
        reference : (batch, nb_class[-1]|1, *spatial) tensor
            Reference classes (or their expectation).
                * If `reference` has a floating point data type (`half`,
                  `float`, `double`) it is assumed to hold one-hot or
                  soft labels, and its channel dimension should be
                  `nb_class` or `nb_class - 1`.
                * If `reference` has an integer or boolean data type, it is
                  assumed to hold hard labels and its channel dimension
                  should be 1. Eventually, `one_hot_map` is used to map
                  one-hot labels to hard labels.
        overridden : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        log = overridden.get('log', self.log)
        implicit = overridden.get('implicit', self.implicit)
        weighted = overridden.get('weighted', self.weighted)
        one_hot_map = overridden.get('one_hot_map', self.one_hot_map)

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)

        # if only one predicted class -> must be implicit
        implicit = implicit or (predicted.shape[1] == 1)

        # take softmax if needed
        if log:
            predicted = _softmax(predicted, implicit=implicit)
        else:
            predicted = _pad_norm(predicted, implicit=implicit)

        nb_classes = predicted.shape[1]
        spatial_dims = list(range(2, predicted.dim()))

        # preprocess reference
        if reference.dtype in (torch.half, torch.float, torch.double):
            # one-hot labels
            reference = reference.to(predicted.dtype)
            if reference.shape[1] == nb_classes - 1:
                reference = _pad_norm(reference, implicit=True)
            elif reference.shape[1] != nb_classes:
                raise ValueError('Number of classes not consistent. '
                                 'Expected {} or {} but got {}.'.format(
                                 nb_classes, nb_classes-1, reference.shape[1]))

            inter = nansum(predicted * reference, dim=spatial_dims)
            union = nansum(predicted + reference, dim=spatial_dims)
            loss = -2 * inter / union
            if weighted:
                weights = nansum(reference, dim=spatial_dims)
                weights = weights / weights.sum(dim=1, keepdim=True)
                loss = loss * weights

        else:
            # hard labels
            if one_hot_map is None:
                one_hot_map = list(range(predicted.shape[1]))
            one_hot_map = list(map(make_list, one_hot_map))

            loss = []
            weights = []
            print(one_hot_map)
            for soft, hard in enumerate(one_hot_map):
                pred1 = predicted[:, soft, ...][:, None, ...]
                ref1 = isin(reference, hard)

                inter = nansum(pred1 * ref1, dim=spatial_dims)
                union = nansum(pred1 + ref1, dim=spatial_dims)
                loss1 = -2 * inter / union
                if weighted:
                    weight1 = ref1.sum()
                    loss1 = loss1 * weight1
                    weights.append(weight1)
                loss.append(loss1)

            loss = torch.cat(loss, dim=1)
            if weighted:
                weights = sum(weights)
                loss = loss / weights

        return super().forward(loss, **overridden)

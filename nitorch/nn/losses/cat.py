"""Losses for categorical data."""

import torch
from .base import Loss
from nitorch.core import math
from nitorch.core.math import nansum, softmax
from nitorch.core.utils import isin, unsqueeze
from nitorch.core.py import make_list, flatten


def _pad_zero(x, implicit=False):
    """Add a zero-channels if the input has an implicit background class"""
    if not implicit:
        return x
    zero_shape = [x.shape[0], 1, *x.shape[2:]]
    zero = x.new_zeros([1]).expand(zero_shape)
    x = torch.cat((x, zero), dim=1)
    return x


def _pad_norm(x, implicit=False):
    """Add a channel that ensures that prob sum to one if the input has
    an implicit background class. Else, ensures that prob sum to one."""
    if not implicit:
        return x / x.sum(dim=1, keepdim=True)
    x = torch.cat((x, 1 - x.sum(dim=1, keepdim=True)), dim=1)
    return x


def _softmax(x, implicit=False):
    """Add a zero-channels if the input has an implicit background class.
    Then, take the softmax."""
    # if implicit:
    #     x = _pad_zero(x, implicit=implicit)
    # x = torch.softmax(x, dim=1)
    x = softmax(x, implicit=(implicit, False))
    return x


def _logsoftmax(x, implicit=False):
    """Add a zero-channels if the input has an implicit background class.
    Then, take the softmax then its log."""
    x = _pad_zero(x, implicit=implicit)
    return torch.log_softmax(x, dim=1)


def _log(x, implicit=False):
    """Add a channel that ensures that prob sum to one if the input has
    an implicit background class. Else, ensures that prob sum to one.
    Then, take the log."""
    x = _pad_norm(x, implicit=implicit)
    return x.clamp(min=1e-7, max=1-1e-7).log()


def get_prob_explicit(x, log=False, implicit=False):
    """Return a tensor of probabilities with all classes explicit"""
    if log:
        return _softmax(x, implicit=implicit)
    else:
        return _pad_norm(x, implicit=implicit)


def get_logprob_explicit(x, log=False, implicit=False):
    """Return a tensor of log-probabilities with all classes explicit"""
    if log:
        return _logsoftmax(x, implicit=implicit)
    else:
        return _log(x, implicit=implicit)


def get_one_hot_map(one_hot_map, nb_classes):
    """Return a well-formed one-hot map"""
    one_hot_map = make_list(one_hot_map or [])
    if not one_hot_map:
        one_hot_map = list(range(nb_classes))
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
    return one_hot_map


def get_log_confusion(confusion, nb_classes_pred, nb_classes_ref, dim, **backend):
    """Return a well formed (log) confusion matrix"""
    if confusion is None:
        confusion = torch.eye(nb_classes_pred, nb_classes_ref,
                              **backend)
    confusion = unsqueeze(confusion, -1, dim)  # spatial shape
    if confusion.dim() < dim + 3:
        confusion = unsqueeze(confusion, 0, dim)  # batch shape
    confusion = confusion / confusion.sum(dim=[-1, -2], keepdim=True)
    confusion = confusion.clamp(min=1e-7, max=1-1e-7).log()
    return confusion


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
        logprior = get_logprob_explicit(prior, log=log, implicit=implicit)
        nb_classes = logprior.shape[1]

        if obs.dtype in (torch.half, torch.float, torch.double):
            # soft labels
            obs = obs.to(prior.dtype)
            obs = get_prob_explicit(obs, implicit=obs.shape[1] == nb_classes-1)
            if obs.shape[1] != nb_classes:
                raise ValueError('Number of classes not consistent. '
                                 'Expected {} or {} but got {}.'.format(
                                 nb_classes, nb_classes-1, obs.shape[1]))
            loss = logprior * obs
        else:
            # hard labels
            if obs.shape[1] != 1:
                raise ValueError('Hard label maps cannot be multi-channel.')
            obs = obs[:, None]
            one_hot_map = overridden.get('one_hot_map', self.one_hot_map)
            one_hot_map = get_one_hot_map(one_hot_map, nb_classes)

            loss = torch.empty_like(logprior)
            for soft, hard in enumerate(one_hot_map):
                if hard is None:
                    # implicit class
                    all_labels = filter(lambda x: x is not None, one_hot_map)
                    obs1 = ~isin(obs, flatten(all_labels))
                else:
                    obs1 = isin(obs, hard)
                loss[:, soft] = logprior[:, soft] * obs1

        # negate
        loss = -loss

        # reduction
        return super().forward(loss, **overridden)


class JointCategoricalLoss(Loss):
    """(Expected) Negative log-likelihood of a joint Categorical distribution.

    L = -trace(a @ log(pi) @ b.T)

    The confusion matrix can be defined at build time or provided at
    call time.

    """

    def __init__(self, confusion=None, one_hot_map=None, log=False,
                 implicit=False, *args, **kwargs):
        """

        Parameters
        ----------
        confusion : (nb_class_pred, nb_class_ref) tensor_like, optional
            Confusion matrix. By default, an identity matrix is used.
        one_hot_map : list[int or list[int] or None], optional
            Mapping from one-hot to hard index. Default: identity mapping.
            Each index of the list corresponds to a soft label.
            Each soft label can be mapped to a hard label or a list of
            hard labels. Up to one `None` can be used, in which case the
            corresponding soft label will be considered a background class
            and will be mapped to all remaining labels. If `len(one_hot_map)`
            has one less element than the number of soft labels, such a
            background class will be appended to the right.
        log : bool, default=False
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
        self.confusion = confusion

    def forward(self, predicted, reference, **overload):
        """

        Parameters
        ----------
        predicted : (nb_batch, nb_class_pred[-1], *spatial) tensor
            (Log)-prior probabilities
        reference : (nb_batch, nb_class_ref[-1]|1, *spatial) tensor
            Observed classes (or their expectation).
                * If `reference` has a floating point data type (`half`,
                  `float`, `double`) it is assumed to hold one-hot or
                  soft labels, and its channel dimension should be
                  `nb_class` or `nb_class - 1`.
                * If `reference` has an integer or boolean data type, it is
                  assumed to hold hard labels and its channel dimension
                  should be 1. Eventually, `one_hot_map` is used to map
                  one-hot labels to hard labels.
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        log = overload.get('log', self.log)
        implicit = overload.get('implicit', self.implicit)
        confusion = overload.get('confusion', self.confusion)
        implicit_pred, implicit_ref = make_list(implicit, 2)

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        backend = dict(dtype=predicted.dtype, device=predicted.device)

        predicted = get_prob_explicit(predicted, log=log, implicit=implicit_pred)
        nb_classes_pred = predicted.shape[1]
        dim = predicted.dim() - 2

        if reference.dtype in (torch.half, torch.float, torch.double):
            # soft labels
            reference = reference.to(predicted.dtype)
            reference = get_prob_explicit(reference, log=log, implicit=implicit_ref)
            nb_classes_ref = reference.shape[1]

            confusion = get_log_confusion(confusion, nb_classes_pred, nb_classes_ref,
                                          dim, **backend)
            loss = (predicted[:, :, None] * confusion).sum(dim=1)
            loss = (loss * reference).sum(dim=1)
        else:
            # hard labels
            if reference.shape[1] != 1:
                raise ValueError('Hard label maps cannot be multi-channel.')
            reference = reference[:, None]
            nb_classes_ref = nb_classes_pred
            one_hot_map = overload.get('one_hot_map', self.one_hot_map)
            one_hot_map = get_one_hot_map(one_hot_map, nb_classes_ref)

            predicted = (predicted[:, :, None] * confusion).sum(dim=1)
            loss = 0
            for soft, hard in enumerate(one_hot_map):
                if hard is None:
                    # implicit class
                    all_labels = filter(lambda x: x is not None, one_hot_map)
                    obs1 = ~isin(reference, flatten(all_labels))
                else:
                    obs1 = isin(reference, hard)
                loss += predicted[:, soft] * obs1

        # negate
        loss = -loss

        # reduction
        return super().forward(loss, **overload)


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
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its size in the
            reference. If a list, use these weights for each class.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.one_hot_map = one_hot_map
        self.log = log
        self.implicit = implicit
        self.discard_background = discard_background
        self.weighted = weighted

    def forward(self, predicted, reference, **overload):
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
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        log = overload.get('log', self.log)
        implicit = overload.get('implicit', self.implicit)
        weighted = overload.get('weighted', self.weighted)
        one_hot_map = overload.get('one_hot_map', self.one_hot_map)

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        backend = dict(dtype=predicted.dtype, device=predicted.device)

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
                if isinstance(weighted, bool):
                    weights = nansum(reference, dim=spatial_dims)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                else:
                    weights = torch.as_tensor(weighted, **backend)[None, :]
                loss = loss * weights

        else:
            # hard labels
            if not one_hot_map:
                one_hot_map = list(range(predicted.shape[1]))
            one_hot_map = list(map(make_list, one_hot_map))

            loss = []
            weights = []
            for soft, hard in enumerate(one_hot_map):
                pred1 = predicted[:, soft, ...][:, None, ...]
                ref1 = isin(reference, hard)

                inter = math.sum(pred1 * ref1, dim=spatial_dims)
                union = math.sum(pred1 + ref1, dim=spatial_dims)
                loss1 = -2 * inter / union
                if weighted:
                    if isinstance(weighted, bool):
                        weight1 = ref1.sum()
                    else:
                        weight1 = float(weighted[soft])
                    loss1 = loss1 * weight1
                    weights.append(weight1)
                loss.append(loss1)

            loss = torch.cat(loss, dim=1)
            if weighted and isinstance(weighted, bool):
                weights = sum(weights)
                loss = loss / weights

        loss += 1
        return super().forward(loss, **overload)

"""Losses for categorical data."""

import torch
from .base import Metric
from nitorch.core.utils import isin
from nitorch.core.py import make_list, flatten
from nitorch.core import py, utils, math


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
    x = math.softmax(x, implicit=(implicit, False))
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
        one_hot_map = list(range(1, nb_classes))
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


def get_hard_labels(x, one_hot_map, implicit=False):
    """Get MAP labels

    Parameters
    ----------
    x : (B, C[-1] | 1, *spatial) tensor
    one_hot_map : list[list[int] or None]
    implicit : bool, default=False

    Returns
    -------
    x : (B, 1, *spatial) tensor[int]

    """
    if x.dtype in (torch.half, torch.float, torch.double):
        x = _pad_norm(x, implicit)
        x = x.argmax(dim=1)
    else:
        new_x = torch.zeros_like(x)
        for soft, hard in enumerate(one_hot_map):
            if hard is None:
                # implicit class
                hard = flatten([l for l in one_hot_map if l is not None])
            new_x[isin(x, hard)] = soft
        x = new_x
    return x


class Accuracy(Metric):
    """Accuracy = Percentage of correctly predicted classes."""

    def __init__(self, one_hot_map=None, implicit=False, *args, **kwargs):
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
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.one_hot_map = one_hot_map
        self.implicit = implicit

    def forward(self, predicted, reference, **overload):
        """

        Notes
        -----
        .. If a tensor has a floating point data type (`half`,
           `float`, `double`) it is assumed to hold one-hot or
           soft labels, and its channel dimension should be
           `nb_class` or `nb_class - 1`.
        .. If a tensor has an integer or boolean data type, it is
           assumed to hold hard labels and its channel dimension
           should be 1. Eventually, `one_hot_map` is used to map
           soft labels to hard labels.

        Parameters
        ----------
        predicted : (nb_batch, nb_class[-1]|1, *spatial) tensor
            Predicted classes (proba, log-proba or hard labels)
        reference : (nb_batch, nb_class[-1]|1, *spatial) tensor
            Observed classes (proba, log-proba or hard labels)
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        accuracy : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        implicit = overload.get('implicit', self.implicit)

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        nb_classes = max(predicted.shape[1], reference.shape[1]) + implicit
        dtype = utils.max_dtype(predicted, reference, force_float=True)

        # hard labels
        one_hot_map = overload.get('one_hot_map', self.one_hot_map)
        one_hot_map = get_one_hot_map(one_hot_map, nb_classes)

        # preprocess
        predicted = get_hard_labels(predicted, one_hot_map, implicit)
        reference = get_hard_labels(reference, one_hot_map, implicit)

        return super().forward((reference == predicted).to(dtype), **overload)


class Dice(Metric):
    """Dice/F1 score."""

    def __init__(self, one_hot_map=None, implicit=False, weighted=False,
                 *args, **kwargs):
        """

        Parameters
        ----------
        one_hot_map : list[int] or list[list[int]], optional
            Mapping from one-hot to hard index.
            By default: identity mapping.
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
        self.implicit = implicit
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
        implicit = overload.get('implicit', self.implicit)
        weighted = overload.get('weighted', self.weighted)

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        nb_classes = max(predicted.shape[1], reference.shape[1]) + implicit
        dtype = utils.max_dtype(predicted, reference, force_float=True)

        # hard labels
        one_hot_map = overload.get('one_hot_map', self.one_hot_map)
        one_hot_map = get_one_hot_map(one_hot_map, nb_classes)

        # preprocess
        predicted = get_hard_labels(predicted, one_hot_map, implicit)
        reference = get_hard_labels(reference, one_hot_map, implicit)

        # dice
        spatial_dims = list(range(2, predicted.dim()))

        loss = []
        weights = []
        for label in range(len(one_hot_map)):
            prd1 = predicted == label
            ref1 = reference == label
            inter = math.sum(prd1 * ref1, dim=spatial_dims, dtype=dtype)
            prd1 = math.sum(prd1, dim=spatial_dims, dtype=dtype)
            ref1 = math.sum(ref1, dim=spatial_dims, dtype=dtype)
            union = prd1 + ref1
            loss1 = 2 * inter / union
            if weighted is not False:
                if weighted is True:
                    weight1 = ref1
                else:
                    weight1 = float(weighted[label])
                loss1 = loss1 * weight1
                weights.append(weight1)
            loss.append(loss1)

        loss = torch.cat(loss, dim=1)
        if weighted is True:
            weights = sum(weights)
            loss = loss / weights

        return super().forward(loss, **overload)

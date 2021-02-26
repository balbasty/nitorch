"""Losses for categorical data."""

import torch
from .base import Metric
from nitorch.core.utils import isin
from nitorch.core.py import make_list, flatten
from nitorch.core import py, utils, math


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
        x = _pad_zero(x, implicit=implicit)
    x = torch.softmax(x, dim=1)
    return x


def _logsoftmax(x, implicit=False):
    """Log(softmax(x)) (with implicit class)"""
    x = _pad_zero(x, implicit=implicit)
    return torch.log_softmax(x, dim=1)


def _log(x, implicit=False):
    """Log (with implicit class)"""
    x = _pad_norm(x, implicit=implicit)
    return x.clamp(min=1e-7, max=1-1e-7).log()


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
        dtype = None

        # hard labels
        one_hot_map = overload.get('one_hot_map', self.one_hot_map)
        if one_hot_map is None:
            one_hot_map = list(range(nb_classes))
        if len(one_hot_map) == nb_classes - 1:
            one_hot_map = [*one_hot_map, None]
        if len(one_hot_map) != nb_classes:
            raise ValueError('Number of classes in tensor and map '
                             'do not match: {} and {}.'
                             .format(nb_classes, len(one_hot_map)))
        one_hot_map = list(map(lambda x: make_list(x) if x is not None else x,
                               one_hot_map))

        # preprocess prediction
        if predicted.dtype in (torch.half, torch.float, torch.double):
            predicted = _pad_norm(predicted, implicit)
            predicted = predicted.argmax(dim=1)
            dtype = predicted.dtype
        else:
            new_pred = torch.zeros_like(predicted)
            for soft, hard in enumerate(one_hot_map):
                if hard is None:
                    # implicit class
                    hard = flatten([l for l in one_hot_map if l is not None])
                new_pred[isin(predicted, hard)] = soft
            predicted = new_pred

        # preprocess observed
        if reference.dtype in (torch.half, torch.float, torch.double):
            reference = _pad_norm(reference, implicit)
            reference = reference.argmax(dim=1)
            dtype = utils.max_dtype(dtype, reference.dtype)
        else:
            new_obs = torch.zeros_like(reference)
            for soft, hard in enumerate(one_hot_map):
                if hard is None:
                    # implicit class
                    hard = flatten([l for l in one_hot_map if l is not None])
                new_obs[isin(reference, hard)] = soft
            reference = new_obs

        # accuracy + reduction
        dtype = utils.max_dtype(dtype, force_float=True)
        return super().forward((reference == predicted).to(dtype), **overload)


class Dice(Metric):
    """Dice/F1 score."""

    def __init__(self, one_hot_map=None, implicit=False,
                 discard_background=False, weighted=False,
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

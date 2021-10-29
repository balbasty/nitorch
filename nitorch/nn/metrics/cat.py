"""Losses for categorical data."""

import torch
from .base import Metric
from nitorch.core.utils import isin
from nitorch.core.py import make_list, flatten
from nitorch.core import py, utils, math


def _pad_norm(x, implicit=False):
    """Add a channel that ensures that prob sum to one if the input has
    an implicit background class. Else, ensures that prob sum to one."""
    if implicit:
        x = torch.cat((1 - x.sum(dim=1, keepdim=True), x), dim=1)
    return x


def get_hard_labels(x, implicit=False):
    """Get MAP labels

    Parameters
    ----------
    x : (B, C[-1] | 1, *spatial) tensor
    implicit : bool, default=False

    Returns
    -------
    x : (B, 1, *spatial) tensor[int]

    """
    if x.dtype.is_floating_point:
        x = _pad_norm(x, implicit)
        x = x.argmax(dim=1, keepdim=True)
    return x


class Accuracy(Metric):
    """Accuracy = Percentage of correctly predicted classes."""

    def __init__(self, implicit=False, logit=False, *args, **kwargs):
        """

        Parameters
        ----------
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.implicit = implicit
        self.logit = logit

    def forward(self, predicted, reference):
        """

        Notes
        -----
        .. If a tensor has a floating point data type (`half`,
           `float`, `double`) it is assumed to hold one-hot or
           soft labels, and its channel dimension should be
           `nb_class` or `nb_class - 1`.
        .. If a tensor has an integer or boolean data type, it is
           assumed to hold hard labels and its channel dimension
           should be 1.

        Parameters
        ----------
        predicted : (nb_batch, nb_class[-1]|1, *spatial) tensor
            Predicted classes (proba, log-proba or hard labels)
        reference : (nb_batch, nb_class[-1]|1, *spatial) tensor
            Observed classes (proba, log-proba or hard labels)

        Returns
        -------
        accuracy : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        implicit = self.implicit

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        nb_classes = max(predicted.shape[1], reference.shape[1]) + implicit
        dtype = utils.max_dtype(predicted, reference, force_float=True)

        # preprocess
        if self.logit:
            predicted = math.softmax(predicted, 1, (implicit, False))
            implicit = False
        predicted = get_hard_labels(predicted, implicit)
        reference = get_hard_labels(reference, reference.shape[1] != nb_classes)

        return super().forward((reference == predicted).to(dtype))


class Dice(Metric):
    """Dice/F1 score."""

    def __init__(self, implicit=False, logit=False, weighted=False,
                 exclude_background=True, *args, **kwargs):
        """

        Parameters
        ----------
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
        self.implicit = implicit
        self.logit = logit
        self.weighted = weighted
        self.exclude_background = exclude_background

    def forward(self, predicted, reference):
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

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        implicit = self.implicit
        weighted = self.weighted
        exclude_background = self.exclude_background

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        nb_classes = max(predicted.shape[1], reference.shape[1]) + implicit
        dtype = utils.max_dtype(predicted, reference, force_float=True)

        # preprocess
        if self.logit:
            predicted = math.softmax(predicted, 1, (implicit, False))
            implicit = False
        predicted = get_hard_labels(predicted, implicit)
        reference = get_hard_labels(reference, reference.shape != nb_classes)

        # dice
        spatial_dims = list(range(2, predicted.dim()))

        loss = []
        weights = []
        first_index = 1 if exclude_background else 0
        for label in range(first_index, nb_classes):
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

        return super().forward(loss)

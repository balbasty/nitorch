"""Losses for categorical data."""

import torch
from .base import Loss
from nitorch.nn.base import Module
from nitorch.core import py, utils, math


def _pad_norm(x, implicit=False):
    """Add a channel that ensures that prob sum to one if the input has
    an implicit background class. Else, ensures that prob sum to one."""
    if not implicit:
        return x / x.sum(dim=1, keepdim=True).clamp_min_(1e-3)
    x = torch.cat((x, 1 - x.sum(dim=1, keepdim=True)), dim=1)
    return x


def _log(x, implicit=False):
    """Add a channel that ensures that prob sum to one if the input has
    an implicit background class. Else, ensures that prob sum to one.
    Then, take the log."""
    x = _pad_norm(x, implicit=implicit)
    return x.clamp(min=1e-7, max=1-1e-7).log()


def get_prob_explicit(x, log=False, implicit=False):
    """Return a tensor of probabilities with all classes explicit"""
    if log:
        return math.softmax(x, dim=1, implicit=[implicit, False])
    else:
        return _pad_norm(x, implicit=implicit)


def get_logprob_explicit(x, log=False, implicit=False):
    """Return a tensor of log-probabilities with all classes explicit"""
    if log:
        return math.log_softmax(x, dim=1, implicit=[implicit, False])
    else:
        return _log(x, implicit=implicit)


def get_score_explicit(x, log=True, implicit=False):
    """Return a tensor of "scores"" with all classes explicit"""
    if not log:
        x = x.log()
    if implicit:
        x = torch.cat((x, x.sum(dim=1, keepdim=True).neg()), dim=1)
    return x


def get_one_hot_map(one_hot_map, nb_classes):
    """Return a well-formed one-hot map"""
    one_hot_map = py.make_list(one_hot_map or [])
    if not one_hot_map:
        one_hot_map = list(range(1, nb_classes))
    if len(one_hot_map) == nb_classes - 1:
        one_hot_map = [*one_hot_map, None]
    if len(one_hot_map) != nb_classes:
        raise ValueError('Number of classes in prior and map '
                         'do not match: {} and {}.'
                         .format(nb_classes, len(one_hot_map)))
    one_hot_map = list(map(lambda x: py.make_list(x) if x is not None else x,
                           one_hot_map))
    if sum(elem is None for elem in one_hot_map) > 1:
        raise ValueError('Cannot have more than one implicit class')
    return one_hot_map


def get_log_confusion(confusion, nb_classes_pred, nb_classes_ref, dim, **backend):
    """Return a well formed (log) confusion matrix"""
    if confusion is None:
        confusion = torch.eye(nb_classes_pred, nb_classes_ref,
                              **backend).exp()
    confusion = utils.unsqueeze(confusion, -1, dim)  # spatial shape
    if confusion.dim() < dim + 3:
        confusion = utils.unsqueeze(confusion, 0, 1)  # batch shape
    confusion = confusion / confusion.sum(dim=[-1, -2], keepdim=True)
    confusion = confusion.clamp(min=1e-7, max=1-1e-7).log()
    return confusion


class CatDotProduct(Module):
    """"Dot product between a vector of (soft or hard) labels and a score.

    Returns <phi(score), truth>

    warning::
        .. the background class (label 0) is assumed to be the last
           (often implicit) "soft" class.

    """

    def __init__(self, weighted=False, one_hot_map=None, *args, **kwargs):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
        one_hot_map : list[int or list[int] or None], optional
            Mapping from one-hot to hard index. Default: [1:n] + [None].
            Each index of the list corresponds to a soft label.
            Each soft label can be mapped to a hard label or a list of
            hard labels. Up to one `None` can be used, in which case the
            corresponding soft label will be considered a background class
            and will be mapped to all remaining labels. If `len(one_hot_map)`
            has one less element than the number of soft labels, such a
            background class will be appended to the right.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.weighted = weighted
        self.one_hot_map = one_hot_map

    def forward(self, score, truth, **overload):
        """

        Parameters
        ----------
        score : (nb_batch, nb_class, *spatial) tensor
            Pre-transformed score vector.
        truth : (nb_batch, nb_class[-1]|1, *spatial) tensor
            Observed classes (or their expectation).
                * If `obs` has a floating point data type (`half`,
                  `float`, `double`) it is assumed to hold one-hot or
                  soft labels, and its channel dimension should be
                  `nb_class` or `nb_class - 1`.
                * If `obs` has an integer or boolean data type, it is
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

        weighted = overload.get('weighted', self.weighted)

        score = torch.as_tensor(score)
        truth = torch.as_tensor(truth, device=score.device)
        nb_classes = score.shape[1]  # (includes background)

        if truth.dtype.is_floating_point:
            # soft labels
            truth = truth.to(score.dtype)
            truth_implicit = truth.shape[1] == nb_classes-1
            truth = get_prob_explicit(truth, implicit=truth_implicit)
            if truth.shape[1] != nb_classes:
                raise ValueError('Number of classes not consistent. '
                                 'Expected {} or {} but got {}.'.format(
                                 nb_classes, nb_classes-1, truth.shape[1]))

            loss = score * truth
            if weighted is True:
                dim = truth.dim() - 2
                weighted = truth.sum(dim=list(range(2, 2 + dim)), keepdim=True)
                weighted = weighted.float().clamp_min_(0.5)
                weighted = weighted / py.prod(truth.shape[2:])
                loss = loss / weighted
            elif weighted not in (None, False):
                dim = truth.dim() - 2
                weighted = utils.make_vector(**utils.backend(loss))
                loss = loss * utils.unsqueeze(weighted, -1, dim)

        else:
            # hard labels
            if truth.shape[1] != 1:
                raise ValueError('Hard label maps cannot be multi-channel.')
            one_hot_map = overload.get('one_hot_map', self.one_hot_map)
            one_hot_map = get_one_hot_map(one_hot_map, nb_classes)

            loss = torch.empty_like(score)
            for soft, hard in enumerate(one_hot_map):
                if hard is None:
                    # implicit class
                    all_labels = filter(lambda x: x is not None, one_hot_map)
                    all_labels = py.flatten(list(all_labels))
                    truth1 = ~utils.isin(truth, all_labels)
                else:
                    truth1 = utils.isin(truth, hard)
                loss[:, soft] = score[:, soft] * truth1.squeeze()
                if weighted is True:
                    dim = truth1.dim() - 2
                    nvox = py.prod(truth1.shape[2:]) or 1
                    w = truth1.sum(dim=list(range(2, 2 + dim)), keepdim=True)
                    w = w.float().clamp_min_(0.5).div_(nvox)
                    loss[:, soft] = loss[:, soft] / w
                elif weighted not in (None, False):
                    w = utils.make_vector(weighted, **utils.backend(loss))
                    loss[:, soft] = loss[:, soft] * w[soft]

        return loss


class CategoricalLoss(Loss):
    """(Expected) Negative log-likelihood of a Categorical distribution.

    This loss loosely corresponds to the categorical cross-entropy loss.
    In this loss, we accept one-hot-encoded "ground truth" on top of
    hard labels.

    """

    def __init__(self, weighted=False, one_hot_map=None,
                 log=True, implicit=False, *args, **kwargs):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
        one_hot_map : list[int or list[int] or None], optional
            Mapping from one-hot to hard index. Default: [1:n] + [None].
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
        self.dot = CatDotProduct(weighted, one_hot_map)
        self.weighted = weighted
        self.one_hot_map = one_hot_map
        self.log = log
        self.implicit = implicit

    def posterior(self, score, **overload):
        implicit = overload.get('implicit', self.implicit)
        return math.softmax(score, dim=1, implicit=implicit)

    def forward(self, prior, obs, **overload):
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
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        log = overload.pop('log', self.log)
        implicit = overload.pop('implicit', self.implicit)

        prior = torch.as_tensor(prior)
        prior = get_logprob_explicit(prior, log=log, implicit=implicit)

        loss = self.dot(prior, obs, **overload)
        loss = loss.sum(dim=1, keepdim=True).neg_()

        return super().forward(loss, **overload)


class DiceLoss(Loss):
    """1 - SoftDice.

    Examples
    --------
    ```python
    >> # Predictions are soft probabilities
    >> loss = Dice()
    >>
    >> # Predictions are soft probabilities, with an implicit background
    >> loss = Dice(implicit=True)
    >>
    >> # Dice are pre-softmax log-probabilities
    >> loss = Dict(log=True)
    >>
    >> # Weight Dice of each class by the reference volume
    >> loss = Dice(weighted=True)
    >>
    >> # Predictions are soft probabilities with 5 classes
    >> # References are hard labels
    >> # We want the background to be in the last soft class
    >> loss = Dice(one_hot_map=[1, 2, 3, 4, 0])
    >>
    >> # Predictions are soft probabilities with 4 classes
    >> # References are hard labels with many more classes
    >> # We want a subset of labels to map to the 3 non-background soft
    >> # classes and all other labels to map to the background class.
    >> loss = Dice(one_hot_map=[(1, 3), 5, (10, 12), None])
    ```
    """

    def __init__(self, one_hot_map=None, log=False, implicit=False,
                 weighted=False, exclude_background=False, *args, **kwargs):
        """

        Parameters
        ----------
        one_hot_map : list[int] or list[list[int]], optional
            Mapping from one-hot to hard index.
            By default: identity mapping.
        log : bool, default=False
            If True, predictions are (pre-softmax) log-probabilities.
        implicit : bool, default=False
            If True, the background class is implicit in the one-hot tensors.
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
        self.weighted = weighted
        self.exclude_background = exclude_background

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
        exclude_background = overload.get('exclude_background', self.exclude_background)

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        backend = dict(dtype=predicted.dtype, device=predicted.device)

        # if only one predicted class -> must be implicit
        implicit = implicit or (predicted.shape[1] == 1)

        # take softmax if needed
        predicted = get_prob_explicit(predicted, log=log, implicit=implicit)

        nb_classes = predicted.shape[1]
        spatial_dims = list(range(2, predicted.dim()))

        # prepare weights
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = utils.make_vector(weighted, nb_classes, **backend)[None]

        # preprocess reference
        if reference.dtype in (torch.half, torch.float, torch.double):
            # one-hot labels
            
            reference = reference.to(predicted.dtype)
            implicit_ref = reference.shape[1] == nb_classes-1
            reference = get_prob_explicit(reference, implicit=implicit_ref)
            if reference.shape[1] != nb_classes:
                raise ValueError('Number of classes not consistent. '
                                 'Expected {} or {} but got {}.'.format(
                                 nb_classes, nb_classes-1, reference.shape[1]))

            if exclude_background:
                predicted = predicted[:, :-1]
                reference = reference[:, :-1]
            inter = math.nansum(predicted * reference, dim=spatial_dims)
            union = math.nansum(predicted + reference, dim=spatial_dims)
            loss = -2 * inter / union
            if weighted is not False:
                if weighted is True:
                    weights = math.nansum(reference, dim=spatial_dims)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                else:
                    weights = weighted
                loss = loss * weights

        else:
            # hard labels
            one_hot_map = overload.get('one_hot_map', self.one_hot_map)
            one_hot_map = get_one_hot_map(one_hot_map, nb_classes)

            loss = []
            weights = []
            for soft, hard in enumerate(one_hot_map):
                if exclude_background and soft == predicted.shape[-1] - 1:
                    continue
                pred1 = predicted[:, None, soft, ...]

                if hard is None:
                    # implicit class
                    all_labels = filter(lambda x: x is not None, one_hot_map)
                    all_labels = py.flatten(list(all_labels))
                    ref1 = ~utils.isin(reference, all_labels)
                else:
                    ref1 = utils.isin(reference, hard)

                inter = math.sum(pred1 * ref1, dim=spatial_dims)
                union = math.sum(pred1 + ref1, dim=spatial_dims)
                loss1 = -2 * inter / union
                if weighted is not False:
                    if weighted is True:
                        weight1 = ref1.sum()
                    else:
                        weight1 = float(weighted[soft])
                    loss1 = loss1 * weight1
                    weights.append(weight1)
                loss.append(loss1)

            loss = torch.cat(loss, dim=1)
            if weighted is True:
                weights = sum(weights)
                loss = loss / weights

        loss += 1
        return super().forward(loss, **overload)


class FocalLoss(Loss):
    """Focal loss.

    References
    ----------
    .. [1] "Focal Loss for Dense Object Detection"
           Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar
           ICCV (2017)
           https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weighted=False, log=False, implicit=False,
                 one_hot_map=None, *args, **kwargs):
        """

        Parameters
        ----------
        gamma : float, default=2
            Focal parameter (0 == CategoricalLoss)
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
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
        self.gamma = gamma
        self.log = log
        self.implicit = implicit
        self.dot = CatDotProduct(weighted, one_hot_map)

    def forward(self, prior, obs, **overload):
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
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        gamma = overload.pop('gamma', self.gamma)
        log = overload.pop('log', self.log)
        implicit = overload.pop('implicit', self.implicit)

        # take log if needed
        prior = torch.as_tensor(prior)
        logprior = get_logprob_explicit(prior, log=log, implicit=implicit)
        prior = get_prob_explicit(prior, log=log, implicit=implicit)
        prior = logprior * (1 - prior).pow(gamma)

        loss = self.dot(prior, obs)
        loss = loss.sum(dim=1, keepdim=True).neg_()
        return super().forward(loss, **overload)


class HingeLoss(Loss):
    """Hinge loss.
    """

    def __init__(self, weighted=False, one_hot_map=None,
                 log=True, implicit=False, mode='cs', *args, **kwargs):
        """

        Parameters
        ----------
        gamma : float, default=2
            Focal parameter (1 == CategoricalLoss)
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
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
            If True, priors are scores (pre-softmax).
            Else, they are probabilities and we take their log in the
            forward pass.
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        mode : {'cs', 'ww', 'llw'}, default='llw'
            Extension to multi-class:
                'cw' : max_j max(0, 1 + fj - fk) (Crammer and Singer)
                'ww' : sum_j max(0, 1 + fj - fk) (Weston and Watkins)
                'llw' : sum_j max(0, 1 + fj) and sum(f) = 0 (Lee, Lin and Wahba)
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.log = log
        self.implicit = implicit
        self.dot = CatDotProduct(weighted, one_hot_map)

    def forward(self, score, obs, **overload):
        """

        Parameters
        ----------
        score : (nb_batch, nb_class[-1], *spatial) tensor
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
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        mode = overload.pop('mode', self.mode)
        log = overload.pop('log', self.log)
        implicit = overload.pop('implicit', self.implicit)

        # product between score and target
        score = get_score_explicit(score, log=log, implicit=implicit)

        if mode == 'llw':
            score = score.add_(1)
            score = score.clamp_min_(0)
            score = score.sum(dim=1, keepdim=True) - score
        else:
            score = score[:, :, None] - score[:, None, :]
            score = score.div_(2).add_(1).clamp_min_(0)
            score.diagonal(dim1=1, dim2=2).zero_()
            if mode == 'cs':
                score = score.max(dim=1).values
            else:
                assert mode == 'ww'
                score = score.sum(dim=1)

        loss = self.dot(score, obs, **overload)
        loss = loss.sum(dim=1, keepdim=True)
        return super().forward(loss, **overload)


class AlphaLoss(Loss):
    """Focal loss.

    References
    ----------
    .. [1] "A Tunable Loss Function for Robust Classification:
            Calibration, Landscape, and Generalization"
           Sypherd, Diaz, Cava, Dasarathy, Kairouz, Sankar
           ISIT (2019)
    """

    def __init__(self, alpha=0.5, weighted=False, log=False, implicit=False,
                 one_hot_map=None, *args, **kwargs):
        """

        Parameters
        ----------
        alpha : float, default=0.5
            Approximation parameter (1 == CategoricalLoss)
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
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
        self.alpha = alpha
        self.log = log
        self.implicit = implicit
        self.dot = CatDotProduct(weighted, one_hot_map)

    def forward(self, prior, obs, **overload):
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
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        alpha = overload.pop('alpha', self.alpha)
        log = overload.pop('log', self.log)
        implicit = overload.pop('implicit', self.implicit)

        ratio = (alpha - 1)/alpha

        prior = torch.as_tensor(prior)
        prior = get_prob_explicit(prior, log=log, implicit=implicit)
        prior = prior.pow(ratio).neg_().add_(1).div_(ratio)

        loss = self.dot(prior, obs)
        loss = loss.sum(dim=1, keepdim=True)
        return super().forward(loss, **overload)

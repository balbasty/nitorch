"""Losses for categorical data."""

import torch
import torch.nn.functional as F
from .base import Loss
from nitorch.core import py, utils, math, linalg


def _pad_norm(x, implicit=False):
    """Add a channel that ensures that prob sum to one if the input has
    an implicit background class. Else, ensures that prob sum to one."""
    if not implicit:
        return x / x.sum(dim=1, keepdim=True).clamp_min_(1e-3)
    x = torch.cat((1 - x.sum(dim=1, keepdim=True), x), dim=1)
    return x


def _log(x, implicit=False):
    """Add a channel that ensures that prob sum to one if the input has
    an implicit background class. Else, ensures that prob sum to one.
    Then, take the log."""
    x = _pad_norm(x, implicit=implicit)
    return x.clamp(min=1e-7, max=1-1e-7).logit()


def get_prob_explicit(x, logit=False, implicit=False):
    """Return a tensor of probabilities with all classes explicit"""
    if logit:
        return math.softmax(x, dim=1, implicit=[implicit, False],
                            implicit_index=0)
    else:
        return _pad_norm(x, implicit=implicit)


def get_logprob_explicit(x, logit=False, implicit=False):
    """Return a tensor of log-probabilities with all classes explicit"""
    if logit:
        return math.log_softmax(x, dim=1, implicit=[implicit, False],
                                implicit_index=0)
    else:
        return _log(x, implicit=implicit)


def get_score_explicit(x, logit=True, implicit=False):
    """Return a tensor of "scores"" with all classes explicit"""
    if not logit:
        x = x.logit()
    if implicit:
        x = torch.cat((x.sum(dim=1, keepdim=True).neg(), x), dim=1)
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
    confusion = confusion.clamp(min=1e-7, max=1-1e-7).logit()
    return confusion


def _process_weights(weighted, dim, nb_classes, **backend):
    weighted_channelwise = False
    if weighted is not False:
        weighted = torch.as_tensor(weighted, **backend)
        if weighted.dim() == 1:
            weighted = utils.unsqueeze(weighted, -1, dim)
        if weighted.numel() == nb_classes:
            weighted_channelwise = True
            weighted = weighted.flatten()
    else:
        weighted = None
    return weighted, weighted_channelwise


def _auto_weighted_soft(truth):
    dim = truth.dim() - 2
    nvox = py.prod(truth.shape[-dim:])
    weighted = truth.sum(dim=list(range(2, 2 + dim)), keepdim=True)
    weighted = weighted.clamp_min_(0.5).div_(nvox).reciprocal_()
    return weighted


def _auto_weighted_hard(truth, nb_classes, **backend):
    dim = truth.dim() - 2
    nvox = py.prod(truth.shape[-dim:])
    weighted = [(truth == i).sum(dim=list(range(-dim, 0)), keepdim=True)
                for i in range(nb_classes)]
    weighted = torch.cat(weighted, dim=1).to(**backend)
    weighted = weighted.clamp_min_(0.5).div_(nvox).reciprocal_()
    return weighted


class CatDotProduct(Loss):
    """"Dot product between a vector of (soft or hard) labels and a score.

    Returns <phi(score), truth>

    warning::
        .. the background class (label 0) is assumed to be the last
           (often implicit) "soft" class.

    """

    def __init__(self, weighted=False, *args, **kwargs):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.weighted = weighted

    def forward(self, score, truth, mask=None):
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
                  should be 1.
        mask : (nb_batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        weighted = self.weighted

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
                weighted = _auto_weighted_soft(truth)
                if mask is not None:
                    weighted = weighted * mask
                loss *= weighted
            elif weighted not in (None, False):
                dim = truth.dim() - 2
                weighted = utils.make_vector(weighted, nb_classes,
                                             **utils.backend(loss))
                weighted = utils.unsqueeze(weighted, -1, dim)
                if mask is not None:
                    weighted = weighted * mask
                loss *= weighted
            elif mask is not None:
                loss *= mask

        else:
            # hard labels
            channelwise = True
            if weighted is True:
                channelwise = False
                weighted = _auto_weighted_hard(truth, nb_classes,
                                               **utils.backend(score))
            elif weighted not in (None, False):
                weighted = utils.make_vector(weighted, **utils.backend(score))
            else:
                weighted = None

            truth = truth.squeeze(1).long()
            # If weights are a list of length C (or none), use nll_loss
            if channelwise and isinstance(self.reduction, str) and mask is None:
                return F.nll_loss(score, truth,
                    weighted, reduction=self.reduction or 'none').neg_()
            # Otherwise, use our own implementation
            else:
                if weighted is not None:
                    score = score * weighted
                loss = score.gather(dim=1, index=truth)
                if mask is not None:
                    mask.squeeze(1)
                    loss *= mask

        if mask is not None and self.reduction == 'mean':
            return loss.sum() / mask.sum()
        return super().forward(loss)


class CategoricalLoss(CatDotProduct):
    """(Expected) Negative log-likelihood of a Categorical distribution.

    This loss loosely corresponds to the categorical cross-entropy loss.
    In this loss, we accept one-hot-encoded "ground truth" on top of
    hard labels.

    """

    def __init__(self, weighted=False, logit=True, implicit=False, **kwargs):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
        logit : bool, default=True
            If True, priors are logits (pre-softmax).
            Else, they are probabilities and we take their log in the
            forward pass.
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        reduction : {'mean', 'sum', 'none'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(weighted, **kwargs)
        self.logit = logit
        self.implicit = implicit

    def posterior(self, score):
        return math.softmax(score, dim=1, implicit=self.implicit,
                            implicit_index=self.implicit_index)

    def forward(self, prior, obs, mask=None):
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
        mask : (nb_batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        implicit = self.implicit or (prior.shape[1] == 1)
        nb_classes = prior.shape[1] + implicit

        # --- preprocess weights ---------------------------------------
        weighted = self.weighted
        if weighted is True:
            weighted = _auto_weighted_hard(obs, nb_classes,
                                           **utils.backend(prior))
        elif weighted not in (None, False):
            weighted = torch.as_tensor(weighted, **utils.backend(prior))
        else:
            weighted = None

        # --- Binary (sigmoid) version ---------------------------------
        if nb_classes == 2:
            if prior.shape == 2:
                prior = prior[:, 1]
            reduction = self.reduction if isinstance(self.reduction, str) else 'none'
            mask_reduction = False
            if mask is not None:
                mask_reduction = reduction
                reduction = 'none'
            if self.logit:
                obs = obs.to(**utils.backend(prior))
                loss = F.binary_cross_entropy_with_logits(
                    prior, obs, weighted, reduction=reduction)
            else:
                obs = obs.to(**utils.backend(prior))
                loss = F.binary_cross_entropy(
                    prior, obs, weighted, reduction=reduction)
            if mask is not None:
                mask = mask.squeeze(1)
                loss *= mask
                if mask_reduction in ('mean', 'sum'):
                    loss = loss.sum()
                if mask_reduction == 'mean':
                    loss /= mask.sum()
            if not isinstance(self.reduction, str):
                loss = self.reduction(loss)
        # --- Multiclass (softmax) version -----------------------------
        else:
            use_native = (self.logit and not obs.dtype.is_floating_point
                          and weighted is not True and not implicit)
            if use_native:
                reduction = self.reduction if isinstance(self.reduction, str) else 'none'
                mask_reduction = False
                if mask is not None:
                    mask_reduction = reduction
                    reduction = 'none'
                obs = obs.squeeze(1)
                obs = obs.long()
                loss = F.cross_entropy(
                    prior, obs, weighted, reduction=reduction)
                if mask is not None:
                    mask = mask.squeeze(1)
                    loss *= mask
                    if mask_reduction in ('mean', 'sum'):
                        loss = loss.sum()
                    if mask_reduction == 'mean':
                        loss /= mask.sum()
                if not isinstance(self.reduction, str):
                    loss = self.reduction(loss)
            else:
                prior = get_logprob_explicit(prior, logit=self.logit,
                                             implicit=self.implicit)
                loss = super().forward(prior, obs, mask=mask).neg_()
        return loss


class DiceLoss(Loss):
    """1 - SoftDice.
    """

    def __init__(self, logit=False, implicit=False,
                 weighted=False, exclude_background=False, *args, **kwargs):
        """

        Parameters
        ----------
        logit : bool, default=False
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
        self.logit = logit
        self.implicit = implicit
        self.weighted = weighted
        self.exclude_background = exclude_background

    def forward(self, predicted, reference, mask=None):
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
        mask : (nb_batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        logit = self.logit
        implicit = self.implicit
        weighted = self.weighted
        exclude_background = self.exclude_background

        predicted = torch.as_tensor(predicted)
        reference = torch.as_tensor(reference, device=predicted.device)
        backend = dict(dtype=predicted.dtype, device=predicted.device)
        dim = predicted.dim() - 2

        # if only one predicted class -> must be implicit
        implicit = implicit or (predicted.shape[1] == 1)

        # take softmax if needed
        predicted = get_prob_explicit(predicted, logit=logit, implicit=implicit)

        nb_classes = predicted.shape[1]
        spatial_dims = list(range(2, predicted.dim()))

        # prepare weights
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = utils.make_vector(weighted, nb_classes, **backend)[None]

        # preprocess reference
        if reference.dtype.is_floating_point:
            # one-hot labels
            reference = reference.to(predicted.dtype)
            implicit_ref = reference.shape[1] == nb_classes-1
            reference = get_prob_explicit(reference, implicit=implicit_ref)
            if reference.shape[1] != nb_classes:
                raise ValueError('Number of classes not consistent. '
                                 'Expected {} or {} but got {}.'.format(
                                 nb_classes, nb_classes-1, reference.shape[1]))

            if exclude_background:
                predicted = predicted[:, 1:]
                reference = reference[:, 1:]
            if mask is not None:
                predicted = predicted * mask
                reference = reference * mask
            predicted = predicted.reshape([*predicted.shape[:-dim], -1])
            reference = reference.reshape([*reference.shape[:-dim], -1])
            inter = linalg.dot(predicted, reference)
            sumpred = predicted.sum(-1)
            sumref = reference.sum(-1)
            union = sumpred + sumref
            # inter = math.nansum(predicted * reference, dim=spatial_dims)
            # union = math.nansum(predicted + reference, dim=spatial_dims)
            loss = -2 * inter / union.clamp_min_(1e-5)
            del inter, union
            if weighted is not False:
                if weighted is True:
                    # weights = math.nansum(reference, dim=spatial_dims)
                    weights = sumref / sumref.sum(dim=1, keepdim=True)
                else:
                    weights = weighted
                loss = loss * weights

        else:
            # hard labels
            loss = []
            weights = []
            first_index = 1 if exclude_background else 0
            for index in range(first_index, nb_classes):
                pred1 = predicted[:, None, index, ...]
                ref1 = reference == index
                if mask is not None:
                    pred1 = pred1 * mask
                    ref1 = ref1 * mask

                inter = math.sum(pred1 * ref1, dim=spatial_dims)
                union = math.sum(pred1 + ref1, dim=spatial_dims)
                loss1 = -2 * inter / union.clamp_min_(1e-5)
                del inter, union
                if weighted is not False:
                    if weighted is True:
                        weight1 = ref1.sum()
                    else:
                        weight1 = float(weighted[index])
                    loss1 = loss1 * weight1
                    weights.append(weight1)
                loss.append(loss1)

            loss = torch.cat(loss, dim=1)
            if weighted is True:
                weights = sum(weights)
                loss = loss / weights

        loss += 1
        return super().forward(loss)


class FocalLoss(CatDotProduct):
    """Focal loss.

    References
    ----------
    .. [1] "Focal Loss for Dense Object Detection"
           Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar
           ICCV (2017)
           https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weighted=False, logit=False, implicit=False,
                 *args, **kwargs):
        """

        Parameters
        ----------
        gamma : float, default=2
            Focal parameter (0 == CategoricalLoss)
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
        logit : bool, default=True
            If True, priors are log-probabilities (pre-softmax).
            Else, they are probabilities and we take their log in the
            forward pass.
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(weighted, *args, **kwargs)
        self.gamma = gamma
        self.logit = logit
        self.implicit = implicit

    def forward(self, prior, obs, mask=None):
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
        mask : (nb_batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        gamma = self.gamma
        log = self.logit
        implicit = self.implicit

        # take log if needed
        prior = torch.as_tensor(prior)
        logprior = get_logprob_explicit(prior, logit=log, implicit=implicit)
        prior = get_prob_explicit(prior, logit=log, implicit=implicit)
        prior = logprior * (1 - prior).pow(gamma)

        return super().forward(prior, obs, mask=mask).neg_()


class HingeLoss(CatDotProduct):
    """Hinge loss.
    """

    def __init__(self, weighted=False, logit=True, implicit=False,
                 mode='cs', *args, **kwargs):
        """

        Parameters
        ----------
        gamma : float, default=2
            Focal parameter (1 == CategoricalLoss)
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
        logit : bool, default=True
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
        super().__init__(weighted, *args, **kwargs)
        self.mode = mode
        self.logit = logit
        self.implicit = implicit
        self.dot = CatDotProduct(weighted)

    def forward(self, score, obs, mask=None):
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
        mask : (nb_batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        mode = self.mode
        logit = self.logit
        implicit = self.implicit

        # product between score and target
        score = get_score_explicit(score, logit=logit, implicit=implicit)

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

        return super().forward(score, obs, mask=mask)


class AlphaLoss(CatDotProduct):
    """Focal loss.

    References
    ----------
    .. [1] "A Tunable Loss Function for Robust Classification:
            Calibration, Landscape, and Generalization"
           Sypherd, Diaz, Cava, Dasarathy, Kairouz, Sankar
           ISIT (2019)
    """

    def __init__(self, alpha=0.5, weighted=False, logit=False, implicit=False,
                 *args, **kwargs):
        """

        Parameters
        ----------
        alpha : float, default=0.5
            Approximation parameter (1 == CategoricalLoss)
        weighted : bool or list[float], default=False
            If True, weight by the inverse class frequency.
            If a list of float, they are user-provided weights.
        logit : bool, default=True
            If True, priors are log-probabilities (pre-softmax).
            Else, they are probabilities and we take their log in the
            forward pass.
        implicit : bool, default=False
            If True, the one-hot tensors only use K-1 channels to encode
            K classes.
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(weighted, *args, **kwargs)
        self.alpha = alpha
        self.logit = logit
        self.implicit = implicit

    def forward(self, prior, obs, mask=None):
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
        mask : (nb_batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """
        alpha = self.alpha
        logit = self.logit
        implicit = self.implicit

        ratio = (alpha - 1)/alpha

        prior = torch.as_tensor(prior)
        prior = get_prob_explicit(prior, logit=logit, implicit=implicit)
        prior = prior.pow(ratio).neg_().add_(1).div_(ratio)

        return super().forward(prior, obs, mask=mask)

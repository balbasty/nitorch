from .base import *
from .cat import *
from .cc import *
from .dice import *
from .gmm import *
from .mi import *
from .emmi import *
from .mse import *
from .robust import *
from .prod import *


def make_loss(loss, dim=None):
    """Instantiate loss object from string.
    Does nothing if `loss` is already a loss object.

    Parameters
    ----------
    loss : {'mse', 'mad', 'tukey', 'cat', 'ncc', 'nmi'} or OptimizationLoss
        'mse' : Means Squared Error (l2 loss)
        'mad' : Median Absolute Deviation (l1 loss, using IRLS)
        'tukey' : Tukey's biweight (~ truncated l2 loss)
        'cat' : Categorical cross-entropy
        'ncc' : Normalized Cross Correlation (zero-normalized version)
        'nmi' : Normalized Mutual Information (studholme's normalization)
    dim : int, optional
        Number of spatial dimensions

    Returns
    -------
    loss : OptimizationLoss

    """
    loss = (MSE(dim=dim) if loss == 'mse' else
            MAD(dim=dim) if loss == 'mad' else
            Tukey(dim=dim) if loss in ('tuk', 'tukey') else
            Cat(dim=dim) if loss == 'cat' else
            Dice(dim=dim) if loss == 'dice' else
            CC(dim=dim) if loss in ('cc', 'ncc') else
            LCC(dim=dim) if loss in ('lcc', 'lncc') else
            MI(dim=dim) if loss in ('mi', 'nmi') else
            EMMI(dim=dim) if loss == 'emi' else
            ProdLoss(dim=dim) if loss == 'prod' else
            NormProdLoss(dim=dim) if loss == 'normprod' else
            SqueezedProdLoss(dim=dim) if loss in ('sqz', 'squeezed') else
            loss)
    if isinstance(loss, str):
        raise ValueError(f'Unknown loss {loss}')
    return loss

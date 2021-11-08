# losses for contrastive pretraining

import torch
from .base import Loss


class InfoNCE(Loss):
    """
    InfoNCE loss used in SimCLR.

    References
    ----------
    ..[1] ""
          https://arxiv.org/abs/2002.05709

    """
    def __init__(self, temp=1, crit=None, nb_views=4):
        super().__init__()
        self.temp = temp
        self.nb_views = nb_views
        # maybe change CE to .cat.CategoricalLoss ?
        self.crit = crit if crit else torch.nn.CrossEntropyLoss()

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        labels = torch.cat([torch.arange(x.shape[0]) for i in range(self.nb_views)])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        x = torch.nn.functional.normalize(x)

        similarity = x @ x.T

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity = similarity[~mask].view(similarity[0], -1)

        pos = similarity[labels.bool()].view(labels.shape[0], -1)
        neg = similarity[~labels.bool()].view(similarity.shape[0], -1)

        logits = torch.cat([pos, neg], 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits /= self.temp

        return self.crit(logits, labels)


class BarlowTwins(Loss):
    """
    Barlow Twins calculation of cross-correlation across different views.

    References
    ----------
    ..[1] ""
          https://arxiv.org/abs/2103.03230
    """
    def __init__(self, lambda_=1, temp=1):
        self.lambda_ = lambda_
        self.temp = temp

    def forward(self, x1, x2):
        B = x1.shape[0]
        dim = x1.shape[1]
        device = x1.device

        x1 = torch.nn.functional.normalize(x1)
        x2 = torch.nn.functional.normalize(x2)

        correlation = x1.T @ x2
        correlation /= B

        correlation -= torch.eye(dim, device=device)
        correlation = correlation ** 2

        correlation[torch.eye(dim)==0] *= self.lambda_
        correlation /= self.temp

        return correlation


# class BYOL(Loss):
#     """
#     Bootstrap your own latent (BYOL) loss.

#     References
#     ----------
#     ..[1] ""
#           https://arxiv.org/abs/2006.07733
#     """
#     def __init__(self, ):


# class DetCon(Loss):
#     """
#     Contrastive detection loss. Will also include pre-processing with uniseg to generate unsupervised masks.

#     References
#     ----------
#     ..[1] ""
#           https://arxiv.org/abs/2103.10957
#     """
#     def __init__(self, ):


# class MoCo(Loss):
#     """
#     Momentum contrast loss (as in MoCo v1).

#     References
#     ----------
#     ..[1] ""
#           https://arxiv.org/abs/1911.05722
#     """
#     def __init__(self, ):

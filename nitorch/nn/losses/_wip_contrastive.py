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
    def __init__(self, temp=1, crit=None, sim=None):
        super().__init__()
        self.temp = temp
        # maybe change CE to .cat.CategoricalLoss ?
        self.crit = crit or torch.nn.CrossEntropyLoss(reduction="sum")
        self.sim = sim or torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, x1, x2):
        B = x1.shape[0]
        N = 2 * B

        mask = self.mask_correlated_samples(B)

        x = torch.cat((x1, x2), dim=0)
        if len(x.shape)==3:
            x = x[...,0]

        sim = self.sim(x.unsqueeze(1), x.unsqueeze(0)) / self.temp

        sim1 = torch.diag(sim, B)
        sim2 = torch.diag(sim, -B)

        pos = torch.cat((sim1, sim2), dim=0).reshape(N, 1)
        neg = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(pos.device).long()
        logits = torch.cat((pos, neg), dim=1)
        loss = self.crit(logits, labels)
        loss /= N
        return loss


class BarlowTwins(Loss):
    """
    Barlow Twins calculation of cross-correlation across different views.

    References
    ----------
    ..[1] ""
          https://arxiv.org/abs/2103.03230
    """
    def __init__(self, lambda_=1, temp=0.005):
        super().__init__()
        self.lambda_ = lambda_
        self.temp = temp
        self.tiny = 1e-8

    def norm(self, x):
        if (x.std(0)==0).sum() != 0:
            x = (x - x.mean(0)) / (x.std(0) + self.tiny)
        else:
            x = (x - x.mean(0)) / x.std(0)
        return x

    def forward(self, x1, x2, reduction=None):
        N = x1.shape[0]
        D = x1.shape[1]
        x1, x2 = x1.reshape(N, D), x2.reshape(N, D)
        device = x1.device

        x1 = self.norm(x1)
        x2 = self.norm(x2)

        correlation = x1.T @ x2
        correlation /= N

        correlation -= torch.eye(D, device=device)
        correlation = correlation ** 2

        correlation[torch.eye(D)==0] *= self.lambda_
        correlation /= self.temp

        if reduction == 'mean':
            correlation = correlation.mean()
        elif reduction == 'sum':
            correlation = correlation.sum()

        return correlation


# class DetCon(Loss):
#     """
#     Contrastive detection loss. Will also include pre-processing with uniseg to generate unsupervised masks.

#     References
#     ----------
#     ..[1] ""
#           https://arxiv.org/abs/2103.10957
#     """
#     def __init__(self, ):

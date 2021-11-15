# losses for adversarial training e.g. GANs
import torch
from .base import Loss
from torch.nn.functional import softplus


def Wasserstein(discriminator, real, fake):
    """
    Wasserstein gradient penalty for W-GAN, based on @eriklindernoren on GitHub
    """
    device = real.device
    fake = fake.to(device)

    dim = len(real.shape) - 2
    shape = [real.shape[0], 1]
    _ = [shape.append(1) for i in range(dim)]
    eps = torch.rand(shape)
    eps = eps.to(device)

    mix = (real * eps + fake * (1 - eps)).requires_grad_(True)

    d_mix = discriminator(mix)

    if isinstance(d_mix, (list, tuple)):
        disc_mix = d_mix[0]

    fake_ = torch.ones(d_mix.shape, requires_grad=False)
    fake_ = fake_.to(device)

    grad = torch.autograd.grad(
        outputs=d_mix,
        inputs=mix,
        grad_outputs=fake_,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    grad = grad[0]
    grad = grad.view(grad.shape[0], -1)

    gp = ((grad.norm(2, dim=1) - 1)**2)
    gp = gp.mean()
    return gp


def R1(discriminator, real, fake=None):
    """
    R1 penalty for discriminator regularisation.
    Based on advice from Guilherme Pombo
    """
    real.requires_grad = True
    d_out = discriminator(real)
    # zero-centered gradient penalty for real images
    batch_size = real.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=real,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == real.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


class DiscriminatorLoss(Loss):
    """
    General class for discriminator loss function in GAN-style training.
    """
    def __init__(self,
                 discriminator,
                 gradient_penalty=None,
                 softplus=False,
                 lambda_gp=1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = discriminator
        if gradient_penalty == 'wasserstein' or 'Wasserstein' or 'wgan' or 'WGAN':
            gradient_penalty = Wasserstein
        elif self.gradient_penalty == 'r1' or 'R1':
            gradient_penalty = R1
        self.gradient_penalty = gradient_penalty
        self.lambda_gp = lambda_gp
        self.softplus = softplus

    def forward(self, real, fake, model=None):
        loss = 0
        if not model:
            model = self.model
        model.eval()

        if self.gradient_penalty is not None:
            loss += self.gradient_penalty(model, real, fake) * self.lambda_gp

        pred_real = model(real)
        pred_fake = model(fake)

        if self.softplus:
            loss += torch.mean(softplus(-pred_real))
            loss += torch.mean(softplus(pred_fake))
        else:
            loss -= torch.mean(pred_real)
            loss += torch.mean(pred_fake)

        return loss
        

class GeneratorLoss(Loss):
    """
    General class for generator loss function in GAN-style training.
    """
    def __init__(self,
                 discriminator,
                 softplus=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = discriminator
        self.softplus = softplus

    def forward(self, fake, model=None):
        loss = 0
        if not model:
            model = self.model
        model.eval()

        pred_fake = model(fake)

        if self.softplus:
            loss += torch.mean(softplus(-pred_fake))
        else:
            loss -= torch.mean(pred_fake)

        return loss

# losses for adversarial training e.g. GANs
import torch

def Wasserstein(discriminator,
                real, fake):
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


def R1(discriminator, x_in):
    """
    R1 penalty for discriminator regularisation.
    Based on advice from Guilherme Pombo
    """
    x_in.requires_grad = True
    d_out = discriminator(x_in)
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

import torch
from nitorch.core import utils, py
from nitorch import spatial
from ..registration import losses, phantoms


def denoise(image=None, lam=1, max_iter=64, sub_iter=32, optim='relax',
            plot=False, jtv=True, dim=None, **prm):
    """Denoise an image using a (joint) total variation prior.

    This implementation uses a reweighted least squares approach.

    Parameters
    ----------
    image : (..., K, *spatial) tensor
        Image to denoise with K channels
    lam : float, default=1
        Regularisartion factor
    max_iter : int, default=64
        Number of RLS iterations
    sub_iter : int, default=32
        Number of relaxation/cg iterations per RLS iteration
    optim : {'cg', 'relax'}, default='relax'
    plot : bool, default=False
    jtv : bool, default=True
        Joint regularisation across channels ($\ell_{1,2}$ norm)
    dim : int, default=image.dim()-1

    Returns
    -------
    denoised : (..., K, *spatial) tensor
        Denoised image

    """

    if image is None:
        torch.random.manual_seed(1234)
        image = phantoms.augment(phantoms.circle(), fwhm=0)[None]

    image = torch.as_tensor(image)
    dim = dim or (image.dim() - 1)

    lr = None
    if isinstance(optim, (tuple, list)):
        optim, lr = optim
    lr = lr or (1e-3 if optim.startswith('gd') else 1)

    weight_shape = [1, *image.shape[-dim:]] if jtv else [image.shape[-dim-1:]]
    weights = image.new_ones(weight_shape)
    denoised = image.clone()

    if jtv:
        lam = lam / image.shape[-dim-1]
    prm['membrane'] = 1
    prm['factor'] = lam

    l_prev = None
    l_max = None
    for n_iter in range(1, max_iter+1):

        ll, g, h = losses.mse(image, denoised, dim=dim)
        gr = spatial.regulariser(denoised, dim=dim, weights=weights, **prm)
        lg = (denoised * gr).sum()
        g += gr

        optim0 = 'cg' if n_iter < 10 else optim  # it seems it helps
        g = spatial.solve_field_sym(h, g, dim=dim, optim=optim0, max_iter=sub_iter,
                                    weights=weights, **prm)
        if lr != 1:
            g.mul_(lr)
        denoised -= g

        weights, lw = spatial.membrane_weights(denoised, dim=dim, return_sum=True,
                                               joint=jtv, factor=lam)

        ll = ll.item() / image.numel()
        lg = lg.item() / image.numel()
        lw = lw.item() / image.numel()
        l = ll + lg + lw

        # print objective
        if l_prev is None:
            l_prev = l
            l_max = l
            print(f'{n_iter:03d} | {ll:12.6g} + {lg:12.6g} + {lw:12.6g} = {l:12.6g}', end='\r')
            print('')
        else:
            gain = (l_prev - l) / max(abs(l_max - l), 1e-8)
            l_prev = l
            print(f'{n_iter:03d} | {ll:12.6g} + {lg:12.6g} + {lw:12.6g} = {l:12.6g} | gain = {gain:12.6g}', end='\r')

        if plot and (n_iter-1) % (max_iter//10+1) == 0:
            import matplotlib.pyplot as plt
            plt.subplot(1, 3, 1)
            plt.imshow(image[0])
            plt.subplot(1, 3, 2)
            plt.imshow(denoised[0])
            plt.subplot(1, 3, 3)
            plt.imshow(weights[0].reciprocal())
            plt.colorbar()
            plt.show()
    print('')

    return denoised

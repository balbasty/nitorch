import torch
from nitorch import spatial
from nitorch.core.py import make_list
from nitorch.core.utils import make_vector
from ..registration import losses, phantoms


def denoise(image=None, lam=1, sigma=1, max_iter=64, sub_iter=32, optim='cg',
            tol=1e-5, sub_tol=1e-5, plot=False, jtv=True, dim=None, **prm):
    """Denoise an image using a (joint) total variation prior.

    This implementation uses a reweighted least squares approach.

    Parameters
    ----------
    image : (..., K, *spatial) tensor, Image to denoise with K channels
    lam : [list of] float, default=1, Regularisation factor
    max_iter : int, default=64, Number of RLS iterations
    sub_iter : int, default=32, Number of relaxation/cg iterations
    optim : {'cg', 'relax', 'fmg+cg', 'fmg+relax'}, default='cg'
    plot : bool, default=False
    jtv : bool, default=True, Joint TV across channels ($\ell_{1,2}$)
    dim : int, default=image.dim()-1

    Returns
    -------
    denoised : (..., K, *spatial) tensor, Denoised image

    """

    if image is None:
        torch.random.manual_seed(1234)
        image = phantoms.augment(phantoms.circle(), fwhm=0)[None]

    image = torch.as_tensor(image)
    dim = dim or (image.dim() - 1)
    nvox = image.shape[-dim:].numel()

    # regularization (prior)
    lam = make_list(lam, image.shape[-dim-1])
    if jtv:
        lam = [l / image.shape[-dim-1] for l in lam]
    prm['membrane'] = 1
    prm['factor'] = lam

    # noise variance (likelihood)
    sigma = make_vector(sigma, image.shape[-dim-1], 
                        dtype=image.dtype, device=image.device)
    isigma2 = 1 / (sigma * sigma)

    # solver
    optim, lr = make_list(optim, 2, default=1)
    do_fmg, optim = ('fmg' in optim), ('relax' if 'relax' in optim else 'cg')
    if do_fmg:
        def solve(h, g, w, optim):
            return spatial.solve_field_fmg(
                h, g, dim=dim, weights=w, **prm, optim=optim)
    else:
        def solve(h, g, w, optim):
            return spatial.solve_field(
                h, g, dim=dim, weights=w, **prm,
                optim=optim, max_iter=sub_iter, tolerance=sub_tol)

    # initialize
    denoised = image.clone()

    l_prev = None
    l_max = None
    for n_iter in range(1, max_iter+1):

        # update weight map / tv loss
        weights, lw = spatial.membrane_weights(denoised, dim=dim, return_sum=True,
                                               joint=jtv, factor=lam)

        # gradient / hessian of least square problem
        ll, g, h = losses.mse(denoised, image, dim=dim, lam=isigma2)
        ll.mul_(nvox), g.mul_(nvox), h.mul_(nvox)
        g += spatial.regulariser(denoised, dim=dim, weights=weights, **prm)

        # solve least square problem
        optim0 = 'cg' if n_iter < 10 else optim  # it seems it helps
        g = solve(h, g, weights, optim0)
        if lr != 1:
            g.mul_(lr)
        denoised -= g

        ll = ll.item() / image.numel()
        lw = lw.item() / image.numel()
        l = ll + lw

        # print objective
        if l_prev is None:
            l_prev = l_max = l
            gain = float('inf')
            print(f'{n_iter:03d} | {ll:12.6g} + {lw:12.6g} = {l:12.6g}', end='\r')
        else:
            gain = (l_prev - l) / max(abs(l_max - l), 1e-8)
            l_prev = l
            print(f'{n_iter:03d} | {ll:12.6g} + {lw:12.6g} = {l:12.6g} '
                  f'| gain = {gain:12.6g}', end='\r')

        if plot and (n_iter-1) % (max_iter//10+1) == 0:
            import matplotlib.pyplot as plt
            img = image[0, image.shape[1]//2] if dim == 3 else image[0]
            den = denoised[0, denoised.shape[1]//2] if dim == 3 else denoised[0]
            wgt = weights[0, weights.shape[1]//2] if dim == 3 else weights[0]
            plt.subplot(1, 3, 1)
            plt.imshow(img.cpu())
            plt.subplot(1, 3, 2)
            plt.imshow(den.cpu())
            plt.subplot(1, 3, 3)
            plt.imshow(wgt.reciprocal().cpu())
            plt.colorbar()
            plt.show()

        if gain < tol:
            break
    print('')

    return denoised

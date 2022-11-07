from nitorch.tools.img_statistics import estimate_noise
from nitorch.core import math
from nitorch import spatial
import torch


def chi_fit(dat, sigma=None, df=None, lam=0, max_iter=50, tol=1e-5):
    """Fit a noncentral Chi noise model to a magnitude image

    Parameters
    ----------
    dat : (*spatial) tensor, Magnitude image
    sigma : float, optional, Noise standard deviation
    df : float, optional, Noise degrees of freedom
    lam : float, default=0, TV regularization
    max_iter : int, default=50, Maximum number of iterations
    tol : float, default=1e-5, Tolerance for early stopping

    Returns
    -------
    fit : (*spatial) tensor, Bias-free image
    sigma : float, Estimated noise standard deviation
    df : float, Estimated noise degrees of freedom

    """
    if lam:
        return _chi_fit_tv(dat, sigma, df, lam, max_iter, tol)
    else:
        return _chi_fit(dat, sigma, df, max_iter, tol)


def _chi_fit(dat, sigma=None, df=None, max_iter=50, tol=1e-5):

    if not sigma or not df:
        noise, _ = estimate_noise(dat, chi=True)
        sigma = sigma or noise['sd']
        df = df or noise['dof']
        print(f'sigma = {sigma}, dof = {df}')
    lam = 1/(sigma*sigma)

    msk = torch.isfinite(dat)
    if msk.any():
        dat = dat.masked_fill(msk.bitwise_not_(), 0)
    fit = chi_bias_correction(dat, sigma, df)[0]
    msk = dat > 0
    n = msk.sum()

    ll_prev = float('inf')
    for n_iter in range(max_iter):

        ll, res = nll_chi(dat, fit, msk, lam, df)
        fit.sub_(res, alpha=1/lam).clamp_min_(1e-8*sigma)

        gain, ll_prev = ll_prev - ll, ll
        print(f'{n_iter:3d} | {ll/n:12.6g} | gain = {gain/n:6.3}')
        if abs(gain) < tol * n:
            break

    return fit, sigma, df


def _chi_fit_tv(dat, sigma=None, df=None, lam=10, max_iter=50, tol=1e-5):

    noise, tissue = estimate_noise(dat, chi=True)
    mu = tissue['mean']
    sigma = sigma or noise['sd']
    df = df or noise['dof']
    print(f'sigma = {sigma}, dof = {df}, mu = {mu}')
    isigma2 = 1/(sigma*sigma)
    lam = lam / mu

    msk = torch.isfinite(dat)
    if msk.any():
        dat = dat.masked_fill(msk.bitwise_not_(), 0)
    fit = chi_bias_correction(dat, sigma, df)[0]
    msk = dat > 0
    n = msk.sum()

    w = torch.ones_like(dat)
    h = w.new_full([1] * dat.dim(), isigma2)[None]

    ll_prev = float('inf')
    for n_iter in range(max_iter):

        w, tv = spatial.membrane_weights(fit[None], factor=lam, return_sum=True)

        l, delta = nll_chi(dat, fit, msk, isigma2, df)
        delta += spatial.regulariser(fit[None], membrane=lam, weights=w)[0]
        delta = spatial.solve_field(h, delta[None], membrane=lam, weights=w)[0]
        fit.sub_(delta).clamp_min_(1e-8*sigma)

        ll = l + tv
        gain, ll_prev = ll_prev - ll, ll
        print(f'{n_iter:3d} | {l/n:12.6g} + {tv/n:12.6g} = {ll/n:12.6g} '
              f'| gain = {gain/n:6.3}')
        if abs(gain) < tol * n:
            break

    return fit, sigma, df


def chi_bias_correction_(dat, sigma=None, df=None):
    """Apply the Chi bias correction, inplace

    Parameters
    ----------
    dat : (*spatial) tensor, Magnitude image
    sigma : float, optional, Noise standard deviation
    df : float, optional, Noise degrees of freedom

    Returns
    -------
    dat, sigma, df

    """

    if not sigma or not df:
        noise, _ = estimate_noise(dat, chi=True)
        sigma = sigma or noise['sd']
        df = df or noise['dof']
        print(f'sigma = {sigma}, dof = {df}')

    return dat.square_().sub_(2*sigma/df).abs_().sqrt_(), sigma, df


def chi_bias_correction(dat, sigma=None, df=None):
    """Apply the Chi bias correction

    Parameters
    ----------
    dat : (*spatial) tensor, Magnitude image
    sigma : float, optional, Noise standard deviation
    df : float, optional, Noise degrees of freedom

    Returns
    -------
    dat, sigma, df

    """
    return chi_bias_correction_(dat.clone(), sigma, df)


def rice_bias_correction_(dat, sigma=None):
    """Apply the Rice bias correction, inplace

    Parameters
    ----------
    dat : (*spatial) tensor, Magnitude image
    sigma : float, optional, Noise standard deviation

    Returns
    -------
    dat, sigma

    """
    if not sigma:
        noise, _ = estimate_noise(dat, chi=False)
        sigma = sigma or noise['sd']
        print(f'sigma = {sigma}')
    return chi_bias_correction_(dat, sigma, 2)[:2]


def rice_bias_correction(dat, sigma=None):
    """Apply the Rice bias correction

    Parameters
    ----------
    dat : (*spatial) tensor, Magnitude image
    sigma : float, optional, Noise standard deviation

    Returns
    -------
    dat, sigma

    """
    return rice_bias_correction_(dat.clone(), sigma)


def ssq(x):
    return x.flatten().dot(x.flatten())


def dot(x, y):
    return x.flatten().dot(y.flatten())


def nll_chi(dat, fit, msk, lam, df, return_grad=True, out=None):
    """Negative log-likelihood of the noncentral Chi distribution

    Parameters
    ----------
    dat : tensor
        Observed data
    fit : tensor
        Signal fit
    msk : tensor
        Mask of observed values
    lam : float
        Noise precision
    df : float
        Degrees of freedom
    return_grad : bool
        Return gradient on top of nll

    Returns
    -------
    nll : () tensor
        Negative log-likelihood
    grad : tensor, if `return_grad`
        Gradient

    """
    fitm = fit[msk]
    datm = dat[msk]

    # components of the log-likelihood
    sumlogfit = fitm.clamp_min(1e-32).log_().sum(dtype=torch.double)
    sumfit2 = ssq(fitm)
    sumlogdat = datm.clamp_min(1e-32).log_().sum(dtype=torch.double)
    sumdat2 = ssq(datm)

    # reweighting
    z = (fitm * datm).mul_(lam).clamp_min_(1e-32)
    xi = math.besseli_ratio(df / 2 - 1, z)
    logbes = math.besseli(df / 2 - 1, z, 'log')
    logbes = logbes.sum(dtype=torch.double)

    # sum parts
    crit = (df / 2 - 1) * sumlogfit - (df / 2) * sumlogdat - logbes
    crit += 0.5 * lam * (sumfit2 + sumdat2)
    if not return_grad:
        return crit

    # compute residuals
    grad = out.zero_() if out is not None else torch.zeros_like(dat)
    grad[msk] = datm.mul_(xi).neg_().add_(fitm).mul_(lam)
    return crit, grad

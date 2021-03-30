import torch
import math
from nitorch.core import utils, constants, py
from .noise import add_noise


def compute_exponentials(r1, tx, tp, tw, td, tr):
    # type: (Tensor, float, float, float, float, float) -> List[Tensor]
    ex = r1.mul(-tx).exp()
    ep = r1.mul(-tp).exp()
    ew = r1.mul(-tw).exp()
    ed = r1.mul(-td).exp()
    e1 = r1.mul(-tr).exp()
    return ex, ep, ew, ed, e1


def compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float) -> Tensor
    mss = (1 - ep) * (c1*ex).pow(n)
    mss = mss + (1 - ex) * (1 - (c1*ex).pow(n)) / (1 - c1*ex)
    mss = mss * ew + (1 - ew)
    mss = mss * (c2*ex).pow(n)
    mss = mss + (1 - ex) * (1 - (c2*ex).pow(n)) / (1 - c2*ex)
    mss = mss * ed + (1 - ed)
    mss = mss * pd / (1 + eff * (c1*c2).pow(n) * e1)
    return mss


def compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, m, eff):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float) -> List[Tensor]
    mi1 = -eff * mss * ep / pd + (1 - ep)
    mi1 = mi1 * (c1*ex).pow(m-1)
    mi1 = mi1 + (1 - ex) * (1 - (c1*ex).pow(m-1)) / (1 - c1*ex)
    mi1 = mi1 * pd * s1
    mi1 = mi1.abs()

    mi2 = (mss / pd - (1 - ed)) / (ed * (c2*ex).pow(m))
    mi2 = mi2 + (1 - ex) * (1 - (c2*ex).pow(-m)) / (1 - c2*ex)
    mi2 = mi2 * pd * s2
    mi2 = mi2.abs()
    return mi1, mi2


def compute_mp2rage(mi1, mi2):
    # type: (Tensor, Tensor) -> Tensor
    m = (mi1 * mi2) / (mi1.pow(2) + mi2.pow(2))
    return m


@torch.jit.script
def mp2rage_from_ir(mi1, mi2):
    # type: (Tensor, Tensor) -> Tensor
    return compute_mp2rage(mi1, mi2)


@torch.jit.script
def mp2rage_nonoise_nob1(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff):
    # type: (Tensor, Tensor, float, float, float, float, float, float, float, int, float) -> Tensor
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = torch.as_tensor(fa1, dtype=pd.dtype, device=pd.device)
    fa2 = torch.as_tensor(fa2, dtype=pd.dtype, device=pd.device)
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    m = compute_mp2rage(mi1, mi2)
    return m


@torch.jit.script
def mp2rage_nonoise_b1p(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1):
    # type: (Tensor, Tensor, float, float, float, float, float, float, float, int, float, Tensor) -> Tensor
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = fa1 * b1
    fa2 = fa2 * b1
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    m = compute_mp2rage(mi1, mi2)
    return m


@torch.jit.script
def mp2rage_uncombined_none(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff):
    # type: (Tensor, Tensor, float, float, float, float, float, float, float, int, float) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = torch.as_tensor(fa1, dtype=pd.dtype, device=pd.device)
    fa2 = torch.as_tensor(fa2, dtype=pd.dtype, device=pd.device)
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    return mi1, mi2


@torch.jit.script
def mp2rage_uncombined_b1p(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1p):
    # type: (Tensor, Tensor, float, float, float, float, float, float, float, int, float, Tensor) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = fa1 * b1p
    fa2 = fa2 * b1p
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    return mi1, mi2


@torch.jit.script
def mp2rage_uncombined_b1m(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1m):
    # type: (Tensor, Tensor, float, float, float, float, float, float, float, int, float, Tensor) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = torch.as_tensor(fa1, dtype=pd.dtype, device=pd.device)
    fa2 = torch.as_tensor(fa2, dtype=pd.dtype, device=pd.device)
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    mi1 = mi1 * b1m
    mi2 = mi2 * b1m
    return mi1, mi2


@torch.jit.script
def mp2rage_uncombined_b1p_b1m(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1p, b1m):
    # type: (Tensor, Tensor, float, float, float, float, float, float, float, int, float, Tensor, Tensor) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = fa1 * b1p
    fa2 = fa2 * b1p
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    mi1 = mi1 * b1m
    mi2 = mi2 * b1m
    return mi1, mi2


@torch.jit.script
def mp2rage_uncombined_r2s(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, int, float) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = torch.as_tensor(fa1, dtype=pd.dtype, device=pd.device)
    fa2 = torch.as_tensor(fa2, dtype=pd.dtype, device=pd.device)
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    e2 = r2s.mul(-te).exp()
    mi1 = mi1 * e2
    mi2 = mi2 * e2
    return mi1, mi2


@torch.jit.script
def mp2rage_uncombined_r2s_b1m(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff, b1m):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, int, float, Tensor) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = torch.as_tensor(fa1, dtype=pd.dtype, device=pd.device)
    fa2 = torch.as_tensor(fa2, dtype=pd.dtype, device=pd.device)
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    mi1 = mi1 * b1m
    mi2 = mi2 * b1m
    e2 = r2s.mul(-te).exp()
    mi1 = mi1 * e2
    mi2 = mi2 * e2
    return mi1, mi2


@torch.jit.script
def mp2rage_uncombined_r2s_b1p(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff, b1p):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, int, float, Tensor) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = fa1 * b1p
    fa2 = fa2 * b1p
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    e2 = r2s.mul(-te).exp()
    mi1 = mi1 * e2
    mi2 = mi2 * e2
    return mi1, mi2


@torch.jit.script
def mp2rage_uncombined_r2s_b1p_b1m(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff, b1p, b1m):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, int, float, Tensor, Tensor) -> List[Tensor]
    ex, ep, ew, ed, e1 = compute_exponentials(r1, tx, tp, tw, td, tr)
    fa1 = fa1 * b1p
    fa2 = fa2 * b1p
    c1, c2, s1, s2 = fa1.cos(), fa2.cos(), fa1.sin(), fa2.sin()
    mss = compute_mss(pd, ex, ep, ew, ed, e1, c1, c2, n, eff)
    mi1, mi2 = compute_ir(mss, pd, ex, ep, ed, c1, c2, s1, s2, n // 2, eff)
    e2 = r2s.mul(-te).exp()
    mi1 = mi1 * e2 * b1m
    mi2 = mi2 * e2 * b1m
    return mi1, mi2


def mp2rage_nonoise(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1=None):
    if b1 is None:
        return mp2rage_nonoise_nob1(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff)
    else:
        return mp2rage_nonoise_b1p(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1)


def mp2rage_uncombined(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff, b1p=None, b1m=None):
    if r2s is None and b1p is None and b1m is None:
        return mp2rage_uncombined_none(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff)
    if r2s is not None and b1p is None and b1m is None:
        return mp2rage_uncombined_r2s(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff)
    if r2s is not None and b1p is not None and b1m is None:
        return mp2rage_uncombined_r2s_b1p(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff, b1p)
    if r2s is not None and b1p is None and b1m is not None:
        return mp2rage_uncombined_r2s_b1m(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff, b1p)
    if r2s is not None and b1p is not None and b1m is not None:
        return mp2rage_uncombined_r2s_b1p_b1m(pd, r1, r2s, tx, tp, tw, td, tr, te, fa1, fa2, n, eff, b1p, b1m)
    if r2s is None and b1p is not None and b1m is None:
        return mp2rage_uncombined_b1p(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1p)
    if r2s is None and b1p is None and b1m is not None:
        return mp2rage_uncombined_b1m(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1m)
    if r2s is None and b1p is not None and b1m is not None:
        return mp2rage_uncombined_b1p_b1m(pd, r1, tx, tp, tw, td, tr, fa1, fa2, n, eff, b1p, b1m)


def mp2rage(pd, r1, r2s=None, transmit=None, receive=None, gfactor=None,
            tr=6.25, ti1=0.8, ti2=2.2, tx=None, te=None, fa=(4, 5), n=160, eff=0.96,
            sigma=None, device=None, return_combined=True):
    """Simulate data generated by a (simplified) MP2RAGE sequence.

    The defaults are parameters used at 3T in the original MP2RAGE paper.
    However, I don't get a nice image with these parameters when applied
    to maps obtained at 3T with the hmri toolbox.
    Here are (unrealistic) parameters that seem to give a decent contrast:
    tr=6.25, ti1=1.4, ti2=4.5, tx=5.8e-3, fa=(4, 5), n=160, eff=0.96

    Tissue parameters
    -----------------
    pd : tensor_like
        Proton density
    r1 : tensor_like
        Longitudinal relaxation rate, in 1/sec
    r2s : tensor_like, optional
        Transverse relaxation rate, in 1/sec.
        If not provided, T2*-bias is not included.

    Fields
    ------
    transmit : tensor_like, optional
        Transmit B1 field
    receive : tensor_like, optional
        Receive B1 field
    gfactor : tensor_like, optional
        G-factor map.
        If provided and `sigma` is not `None`, the g-factor map is used
        to sample non-stationary noise.

    Sequence parameters
    -------------------
    tr : float default=6.25
        Full Repetition time, in sec.
        (Time between two inversion pulses)
    ti1 : float, default=0.8
        First inversion time, in sec.
        (Time between inversion pulse and middle of the first echo train)
    ti2 : float, default=2.2
        Second inversion time, in sec.
        (Time between inversion pulse and middle of the second echo train)
    tx : float, default=te*2 or 5.8e-3
        Excitation repetition time, in sec.
        (Time between two excitation pulses within the echo train)
    te : float, default=tx/2
        Echo time, in sec.
    fa : float or (float, float), default=(4, 5)
        Flip angle of the first and second acquisition block, in deg
        If only one value, it is shared between the blocks.
    n : int, default=160
        Number of excitation pulses (= phase encoding steps) per train.
    eff : float, default=0.96
        Efficiency of the inversion pulse.

    Noise
    -----
    sigma : float, optional
        Standard-deviation of the sampled Rician noise (no sampling if `None`)

    Returns
    -------
    mp2rage : tensor, if return_combined is True
        Simulated MP2RAGE image

    image1 : tensor, if return_combined is False
        Image at first inversion time
    image2 : tensor, if return_combined is False
        Image at second inversion time

    References
    ----------
    ..[1] "MP2RAGE, a self bias-field corrected sequence for improved
        segmentation and T1-mapping at high field."
        Marques JP, Kober T, Krueger G, van der Zwaag W, Van de Moortele PF, Gruetter R.
        Neuroimage. 2010 Jan 15;49(2):1271-81.
        doi: 10.1016/j.neuroimage.2009.10.002

    """

    pd, r1, r2s, transmit, receive, gfactor \
        = utils.to_max_backend(pd, r1, r2s, transmit, receive, gfactor)
    pd, r1, r2s, transmit, receive, gfactor \
        = utils.to(pd, r1, r2s, transmit, receive, gfactor, device=device)

    if tx is None and te is None:
        tx = 5.8e-3
    tx = tx or 2*te                 # Time between excitation pulses
    te = te or tx/2                 # Echo time
    fa1, fa2 = py.make_list(fa, 2)
    fa1 = fa1 * constants.pi / 180  # Flip angle of first GRE block
    fa2 = fa2 * constants.pi / 180  # Flip angle of second GRE block
    n = n or min(pd.shape)          # Number of readouts (PE steps) per loop
    tr1 = n*tx                      # First GRE block
    tr2 = n*tx                      # Second GRE block
    tp = ti1 - tr1/2                # Preparation time
    tw = ti2 - tr2/2 - ti1 - tr1/2  # Wait time between GRE blocks
    td = tr - ti2 - tr2/2           # Recovery time

    if return_combined and not sigma:
        m = mp2rage_nonoise(pd, r1, tx, tp, tw, td, tr,
                            fa1, fa2, n, eff, transmit)

        m = torch.where(~torch.isfinite(m), m.new_zeros([1]), m)
        return m

    mi1, mi2 = mp2rage_uncombined(pd, r1, r2s, tx, tp, tw, td, tr,
                                  te, fa1, fa2, n, eff, transmit, receive)

    # noise
    mi1 = add_noise(mi1, std=sigma, gfactor=gfactor)
    mi2 = add_noise(mi2, std=sigma, gfactor=gfactor)

    if return_combined:
        m = mp2rage_from_ir(mi1, mi2)
        m = torch.where(~torch.isfinite(m), m.new_zeros([1]), m)
        return m
    else:
        mi1 = torch.where(~torch.isfinite(mi1), mi1.new_zeros([]), mi1)
        mi2 = torch.where(~torch.isfinite(mi2), mi2.new_zeros([]), mi2)
        return mi1, mi2


# ---
# Non compiled version of mp2rage
# ---
def mp2rage_old(pd, r1, r2s=None, transmit=None, receive=None, gfactor=None,
               tr=6.25, ti1=0.8, ti2=2.2, tx=None, te=None, fa=(4, 5), n=160, eff=0.96,
               sigma=None, device=None, return_combined=True):
    """Simulate data generated by a (simplified) MP2RAGE sequence.

    The defaults are parameters used at 3T in the original MP2RAGE paper.
    However, I don't get a nice image with these parameters when applied
    to maps obtained at 3T with the hmri toolbox.
    Here are (unrealistic) parameters that seem to give a decent contrast:
    tr=6.25, ti1=1.4, ti2=4.5, tx=5.8e-3, fa=(4, 5), n=160, eff=0.96

    Tissue parameters
    -----------------
    pd : tensor_like
        Proton density
    r1 : tensor_like
        Longitudinal relaxation rate, in 1/sec
    r2s : tensor_like, optional
        Transverse relaxation rate, in 1/sec.
        If not provided, T2*-bias is not included.

    Fields
    ------
    transmit : tensor_like, optional
        Transmit B1 field
    receive : tensor_like, optional
        Receive B1 field
    gfactor : tensor_like, optional
        G-factor map.
        If provided and `sigma` is not `None`, the g-factor map is used
        to sample non-stationary noise.

    Sequence parameters
    -------------------
    tr : float default=6.25
        Full Repetition time, in sec.
        (Time between two inversion pulses)
    ti1 : float, default=0.8
        First inversion time, in sec.
        (Time between inversion pulse and middle of the first echo train)
    ti2 : float, default=2.2
        Second inversion time, in sec.
        (Time between inversion pulse and middle of the second echo train)
    tx : float, default=te*2 or 5.8e-3
        Excitation repetition time, in sec.
        (Time between two excitation pulses within the echo train)
    te : float, default=minitr/2
        Echo time, in sec.
    fa : float or (float, float), default=(4, 5)
        Flip angle of the first and second acquisition block, in deg
        If only one value, it is shared between the blocks.
    n : int, default=160
        Number of excitation pulses (= phase encoding steps) per train.
    eff : float, default=0.96
        Efficiency of the inversion pulse.

    Noise
    -----
    sigma : float, optional
        Standard-deviation of the sampled Rician noise (no sampling if `None`)

    Returns
    -------
    mp2rage : tensor, if return_combined is True
        Simulated MP2RAGE image

    image1 : tensor, if return_combined is False
        Image at first inversion time
    image2 : tensor, if return_combined is False
        Image at second inversion time

    References
    ----------
    ..[1] "MP2RAGE, a self bias-field corrected sequence for improved
        segmentation and T1-mapping at high field."
        Marques JP, Kober T, Krueger G, van der Zwaag W, Van de Moortele PF, Gruetter R.
        Neuroimage. 2010 Jan 15;49(2):1271-81.
        doi: 10.1016/j.neuroimage.2009.10.002

    """

    pd, r1, r2s, transmit, receive, gfactor \
        = utils.to_max_backend(pd, r1, r2s, transmit, receive, gfactor)
    pd, r1, r2s, transmit, receive, gfactor \
        = utils.to(pd, r1, r2s, transmit, receive, gfactor, device=device)
    backend = utils.backend(pd)

    if tx is None and te is None:
        tx = 5.8e-3
    tx = tx or 2*te                 # Time between excitation pulses
    te = te or tx/2                 # Echo time
    fa1, fa2 = py.make_list(fa, 2)
    fa1 = fa1 * constants.pi / 180  # Flip angle of first GRE block
    fa2 = fa2 * constants.pi / 180  # Flip angle of second GRE block
    n = n or min(pd.shape)          # Number of readouts (PE steps) per loop
    tr1 = n*tx                      # First GRE block
    tr2 = n*tx                      # Second GRE block
    tp = ti1 - tr1/2                # Preparation time
    tw = ti2 - tr2/2 - ti1 - tr1/2  # Wait time between GRE blocks
    td = tr - ti2 - tr2/2           # Recovery time
    m = n // 2                      # Middle of echo train

    if transmit is not None:
        fa1 = transmit * fa1
        fa2 = transmit * fa2
    del transmit
    fa1 = torch.as_tensor(fa1, **backend)
    fa2 = torch.as_tensor(fa2, **backend)

    # precompute exponential terms
    ex = r1.mul(-tx).exp()
    ep = r1.mul(-tp).exp()
    ew = r1.mul(-tw).exp()
    ed = r1.mul(-td).exp()
    e1 = r1.mul(-tr).exp()
    c1 = fa1.cos()
    c2 = fa2.cos()

    # steady state
    mss = (1 - ep) * (c1*ex).pow(n)
    mss = mss + (1 - ex) * (1 - (c1*ex).pow(n)) / (1 - c1*ex)
    mss = mss * ew + (1 - ew)
    mss = mss * (c2*ex).pow(n)
    mss = mss + (1 - ex) * (1 - (c2*ex).pow(n)) / (1 - c2*ex)
    mss = mss * ed + (1 - ed)
    mss = mss * pd / (1 + eff * (c1*c2).pow(n) * e1)

    # IR components
    mi1 = -eff * mss * ep / pd + (1 - ep)
    mi1 = mi1 * (c1*ex).pow(m-1)
    mi1 = mi1 + (1 - ex) * (1 - (c1*ex).pow(m-1)) / (1 - c1*ex)
    mi1 = mi1 * fa1.sin()
    mi1 = mi1.abs()

    mi2 = (mss / pd - (1 - ed)) / (ed * (c2*ex).pow(m))
    mi2 = mi2 + (1 - ex) * (1 - (c2*ex).pow(-m)) / (1 - c2*ex)
    mi2 = mi2 * fa2.sin()
    mi2 = mi2.abs()

    if return_combined and not sigma:
        m = (mi1*mi2) / (mi1.square() + mi2.square())
        m = torch.where(~torch.isfinite(m), m.new_zeros([]), m)
        return m

    # Common component (pd, B1-, R2*)
    if receive is not None:
        pd = pd * receive
    del receive

    mi1 = mi1 * pd
    mi2 = mi2 * pd

    if r2s is not None:
        e2 = r2s.mul(-te).exp_()
        mi1 = mi1 * e2
        mi2 = mi2 * e2
    del r2s

    # noise
    mi1 = add_noise(mi1, std=sigma, gfactor=gfactor)
    mi2 = add_noise(mi2, std=sigma, gfactor=gfactor)

    if return_combined:
        m = (mi1*mi2) / (mi1.square() + mi2.square())
        m = torch.where(~torch.isfinite(m), m.new_zeros([]), m)
        return m
    else:
        mi1 = torch.where(~torch.isfinite(mi1), mi1.new_zeros([]), mi1)
        mi2 = torch.where(~torch.isfinite(mi2), mi2.new_zeros([]), mi2)
        return mi1, mi2


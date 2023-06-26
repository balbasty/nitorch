"""
References
----------
    "Model-based multi-parameter mapping"
    Balbastre et al., Med Image Anal (2021)
    https://arxiv.org/abs/2102.01604
"""
import torch
from nitorch import core, spatial
# from nitorch.core.math import besseli_ratio, besseli
from ._options import GREEQOptions
from ._preproc import preproc, postproc
from ..utils import (hessian_sym_loaddiag,  smart_grid, smart_pull, smart_push)
from ..utils import (nll_gauss, nll_chi, get_mask_missing,
                     mask_nan_, check_nans_, dot)
from nitorch.core.linalg import sym_inv, sym_solve
from nitorch.tools.qmri.param import ParameterMap
from typing import Optional, Tuple
import math

# import time
Tensor = torch.Tensor

# NOTE:
#   In our model, we first deform the parameter maps and then apply the
#   FLASH signal equation. The objective function is therefore
#                   L = l2(flash(phi @ y) - x) / 2
#   where phi is the deformation encoded in a large matrix.
#   The forward operation phi @ y is implemented by `grid_pull`.
#   By applying the chain rule, we get
#                   dL/dy = phi.T @ (flash'(phi @ y) * res)
#   where res = (flash(phi @ y) - x) is the residual image in
#   acquisition space. The adjoint operation phi.T @ y is implemented
#   by `grid_push`. In practice, this means that we pull the parameter
#   maps in observed space, compute the derivative of the flash equation,
#   multiply it with the residuals and push back the whole thing to
#   parameter space.
#
#   Alternately, we could have first applied the FLASH equation and then
#   deformed the resulting signal
#                       L = l2(phi @ flash(y) - x) / 2
#   which, after chain rule, yields
#                   dL/dy = flash'(y) * (phi.T @ res)
#   where this time res = (phi @ flash(y) - x)
#   In that case, we pull the precomputed signal only to compute the
#   residuals that are then pushed back to parameter space.
#   The derivative of the flash equation is computed in parameter space
#   and multiplied with the pushed residuals.
#
#   It is not too complicated to implement the alternate, and it might
#   even save some computation (although many of the components of the
#   flash function depend on te/tr/fa which are at least contrast-specific,
#   and even echo-time specific for the r2* component, so it may not save
#   that much; and we still need to pull from param to observed to compute
#   the residuals).


def greeq(data, transmit=None, receive=None, opt=None, **kwopt):
    """Fit a non-linear relaxometry model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
        Observed GRE data.
    transmit : sequence[PrecomputedFieldMap], optional
        Map(s) of the transmit field (b1+). If a single map is provided,
        it is used to correct all contrasts. If multiple maps are
        provided, there should be one for each contrast.
    receive : sequence[PrecomputedFieldMap], optional
        Map(s) of the receive field (b1-). If a single map is provided,
        it is used to correct all contrasts. If multiple maps are
        provided, there should be one for each contrast.
        If no receive map is provided, the output `pd` map will have
        a remaining b1- bias field.
    opt : GREEQOptions or dict, optional
        Algorithm options.
        {'preproc': {'register':      True},     # Co-register contrasts
         'optim':   {'nb_levels':     1,         # Number of pyramid levels
                     'max_iter_rls':  10,        # Max reweighting iterations
                     'max_iter_gn':   5,         # Max Gauss-Newton iterations
                     'max_iter_cg':   32,        # Max Conjugate-Gradient iterations
                     'tolerance': 1e-05,     # Tolerance for early stopping (RLS)
                     'tolerance':  1e-05,         ""
                     'tolerance_cg':  1e-03},        ""
         'backend': {'dtype':  torch.float32,    # Data type
                     'device': 'cpu'},           # Device
         'penalty': {'norm':    'jtv',           # Type of penalty: {'tkh', 'tv', 'jtv', None}
                     'factor':  {'r1':  10,      # Penalty factor per (log) map
                                 'pd':  10,
                                 'r2s': 2,
                                 'mt':  2}},
         'verbose': 1}

    Returns
    -------
    pd : ParameterMap
        Proton density
    r1 : ParameterMap
        Longitudinal relaxation rate
    r2s : ParameterMap
        Apparent transverse relaxation rate
    mt : ParameterMap, optional
        Magnetisation transfer saturation
        Only returned is MT-weighted data is provided.

    """
    opt = GREEQOptions().update(opt or {}, **kwopt)
    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    # --- estimate noise / register / initialize maps ---
    data, transmit, receive, maps = preproc(data, transmit, receive, opt)
    has_mt = hasattr(maps, 'mt')

    # --- prepare penalty factor ---
    lam = opt.penalty.factor
    if isinstance(lam, dict):
        lam = [lam.get('pd', 0), lam.get('r1', 0),
               lam.get('r2s', 0), lam.get('mt', 0)]
        if not has_mt:
            lam = lam[:3]
    lam = core.utils.make_vector(lam, 3 + has_mt, **backend)  # PD, R1, R2*, [MT]

    # --- initialize weights (RLS) ---
    if str(opt.penalty.norm).lower().startswith('no') or all(lam == 0):
        opt.penalty.norm = ''
    opt.penalty.norm = opt.penalty.norm.lower()
    mean_shape = maps[0].shape
    rls = None
    sumrls = 0
    if opt.penalty.norm in ('tv', 'jtv'):
        rls_shape = mean_shape
        if opt.penalty.norm == 'tv':
            rls_shape = (len(maps),) + rls_shape
        rls = torch.ones(rls_shape, **backend)
        sumrls = 0.5 * core.py.prod(rls_shape)

    if opt.penalty.norm:
        print(f'With {opt.penalty.norm.upper()} penalty:')
        print(f'    - PD:  {lam[0]:.3g}')
        print(f'    - R1:  {lam[1]:.3g}')
        print(f'    - R2*: {lam[2]:.3g}')
        if has_mt:
            print(f'    - MT:  {lam[3]:.3g}')
    else:
        print('Without penalty')

    if opt.penalty.norm not in ('tv', 'jtv'):
        # no reweighting -> do more gauss-newton updates instead
        opt.optim.max_iter_gn *= opt.optim.max_iter_rls
        opt.optim.max_iter_rls = 1
    print('Optimization:')
    print(f'    - Tolerance:        {opt.optim.tolerance}')
    if opt.penalty.norm.endswith('tv'):
        print(f'    - IRLS iterations:  {opt.optim.max_iter_rls}')
    print(f'    - GN iterations:    {opt.optim.max_iter_gn}')
    if opt.optim.solver == 'fmg':
        print(f'    - FMG cycles:       2')
        print(f'    - CG iterations:    2')
    else:
        print(f'    - CG iterations:    {opt.optim.max_iter_cg}'
              f' (tolerance: {opt.optim.tolerance_cg})')
    if opt.optim.nb_levels > 1:
        print(f'    - Levels:           {opt.optim.nb_levels}')

    printer = CritPrinter(max_levels=opt.optim.nb_levels,
                          max_rls=opt.optim.max_iter_rls,
                          max_gn=opt.optim.max_iter_gn,
                          penalty=opt.penalty.norm,
                          verbose=opt.verbose)
    printer.print_head()

    shape0 = shape = maps.shape[1:]
    aff0 = aff = maps.affine
    vx0 = vx = spatial.voxel_size(aff0)
    vol0 = vx0.prod()
    vol = vx.prod() / vol0
    ll_scl = sum(core.py.prod(dat.shape) for dat in data)

    for level in range(opt.optim.nb_levels, 0, -1):
        printer.level = level

        if opt.optim.nb_levels > 1:
            aff, shape = _get_level(level, aff0, shape0)
            vx = spatial.voxel_size(aff)
            vol = vx.prod() / vol0
            maps, rls = resize(maps, rls, aff, shape)
            if opt.penalty.norm in ('tv', 'jtv'):
                sumrls = 0.5 * vol * rls.reciprocal().sum(dtype=torch.double)
        
        # --- compute derivatives ---
        nb_prm = len(maps)
        nb_hes = nb_prm * (nb_prm + 1) // 2
        grad = torch.empty((nb_prm,) + shape, **backend)
        hess = torch.empty((nb_hes,) + shape, **backend)
    
        ll_rls = []
        ll_gn = []
        ll_max = float('inf')

        max_iter_rls = max(opt.optim.max_iter_rls // level, 1)
        for n_iter_rls in range(max_iter_rls):
            # --- Reweighted least-squares loop ---
            printer.rls = n_iter_rls

            # --- Gauss Newton loop ---
            for n_iter_gn in range(opt.optim.max_iter_gn):
                # start = time.time()
                printer.gn = n_iter_gn
                crit = 0
                grad.zero_()
                hess.zero_()
                # --- loop over contrasts ---
                for contrast, b1m, b1p in zip(data, receive, transmit):
                    crit1, g1, h1 = derivatives_parameters(contrast, maps, b1m, b1p, opt)

                    # increment
                    if hasattr(maps, 'mt') and not contrast.mt:
                        # we optimize for mt but this particular contrast
                        # has no information about mt so g1/h1 are smaller
                        # than grad/hess.
                        grad[:-1] += g1
                        hind = list(range(nb_prm-1))
                        cnt = nb_prm
                        for i in range(nb_prm):
                            for j in range(i+1, nb_prm):
                                if i != nb_prm-1 and j != nb_prm-1:
                                    hind.append(cnt)
                                cnt += 1
                        hess[hind] += h1
                        crit += crit1
                    else:
                        grad += g1
                        hess += h1
                        crit += crit1
                    
                    del g1, h1, crit1
                # duration = time.time() - start
                # print('grad', duration)

                # start = time.time()
                reg = 0.
                if opt.penalty.norm:
                    g = spatial.regulariser(maps.volume, weights=rls, dim=3,
                                            voxel_size=vx, membrane=1,
                                            factor=lam * vol)
                    grad += g
                    reg = 0.5 * dot(maps.volume, g)
                    del g
                # duration = time.time() - start
                # print('reg', duration)

                # --- gauss-newton ---
                # start = time.time()
                grad = check_nans_(grad, warn='gradient')
                hess = check_nans_(hess, warn='hessian')
                if opt.penalty.norm:
                    # hess = hessian_sym_loaddiag(hess, 1e-5, 1e-8)  # 1e-5 1e-8
                    if opt.optim.solver == 'fmg':
                        deltas = spatial.solve_field_fmg(
                                hess, grad, rls, factor=lam * vol,
                                membrane=1, voxel_size=vx, nb_iter=2)
                    else:
                        deltas = spatial.solve_field(
                            hess, grad, rls, factor=lam * vol, membrane=1,
                            voxel_size=vx, verbose=max(0, opt.verbose - 1),
                            optim='cg', max_iter=opt.optim.max_iter_cg,
                            tolerance=opt.optim.tolerance_cg, stop='diff')
                else:
                    # hess = hessian_sym_loaddiag(hess, 1e-3, 1e-4)
                    deltas = spatial.solve_field_closedform(hess, grad)
                deltas = check_nans_(deltas, warn='deltas')
                # duration = time.time() - start
                # print('solve', duration)

                for map, delta in zip(maps, deltas):
                    map.volume -= delta
                    map.volume.clamp_(-64, 64)  # avoid exp overflow
                    del delta
                del deltas

                # --- Compute gain ---
                ll = crit + reg + sumrls
                ll_max = max(ll_max, ll)
                ll_prev = ll_gn[-1] if ll_gn else ll_max
                gain = ll_prev - ll
                ll_gn.append(ll)
                printer.print_crit(crit, reg, sumrls, gain / ll_scl)
                if gain < opt.optim.tolerance * ll_scl:
                    # print('GN converged: ', ll_prev.item(), '->', ll.item())
                    break

            # --- Update RLS weights ---
            if opt.penalty.norm in ('tv', 'jtv'):
                rls, sumrls = spatial.membrane_weights(
                    maps.volume, lam, vx, return_sum=True, dim=3,
                    joint=opt.penalty.norm == 'jtv',
                    eps=core.constants.eps(rls.dtype))
                sumrls *= 0.5 * vol

                # --- Compute gain ---
                # (we are late by one full RLS iteration when computing the 
                #  gain but we save some computations)
                ll = ll_gn[-1]
                ll_prev = ll_rls[-1] if ll_rls else ll_max
                ll_rls.append(ll)
                gain = ll_prev - ll
                if abs(gain) < opt.optim.tolerance * ll_scl:
                    # print(f'RLS converged ({gain:7.2g})')
                    break

    del grad
    if opt.uncertainty:
        uncertainty = compute_uncertainty(hess, rls, lam * vol, vx, opt)
        maps.pd.uncertainty = uncertainty[0]
        maps.r1.uncertainty = uncertainty[1]
        maps.r2s.uncertainty = uncertainty[2]
        if hasattr(maps, 'mt'):
            maps.mt.uncertainty = uncertainty[3]
    
    # --- Prepare output ---
    return postproc(maps)


def compute_uncertainty(hess, rls, lam, vx, opt):
    """Diagonal posterior uncertainty"""
    uncertainty = hess.clone()
    if opt.penalty:
        m = len(hess)
        n = int(round((math.sqrt(1 + 8*m) - 1)/2))
        diag = uncertainty[:n]
        if rls is None:
            vx = core.utils.make_vector(vx, 3)
            smo = vx.square().reciprocal().sum().item()
            diag.transpose(0, -1).add_(lam * (4 * smo))
        else:
            vx = core.utils.make_vector(vx, 3, **core.utils.backend(hess))
            reg = spatial.membrane_diag(vx, dim=3, weights=rls)
            if reg.dim() == 3:
                reg = reg[None]
            diag.transpose(0, -1).addcmul_(reg.transpose(0, -1), lam)

    return sym_inv(uncertainty.transpose(0, -1), diag=True).transpose(0, -1)


class CritPrinter:
    """Utility to print info about convergence"""

    def __init__(self, penalty='', max_levels=1, max_rls=1, max_gn=1,
                 verbose=True):
        self.penalty = penalty
        self.max_levels = max_levels
        self.max_rls = max_rls
        self.max_gn = max_gn
        self.verbose = verbose
        self.level = max_levels
        self.rls = 1
        self.gn = 1

    def print_head(self):
        if not self.verbose:
            return
        def cond_append(s, x):
            if s:
                s += x
            return s
        pstr = ''
        if self.max_levels > 1:
            pstr = f'{"lvl":^3s}'
        if self.max_rls > 1:
            pstr = cond_append(pstr, ' | ')
            pstr += f'{"rls":^3s}'
        if self.max_gn > 1:
            pstr = cond_append(pstr, ' | ')
            pstr += f'{"gn":^3s}'
        pstr = cond_append(pstr, ' | ')
        pstr += f'{"fit":^12s}'
        if self.penalty:
            pstr += f' + {"reg":^12s}'
            if self.penalty in ('tv', 'jtv'):
                pstr += f' + {"rls":^12s}'
            pstr += f' = {"crit":^12s}'
        pstr += f' | {"gain":^7s}'
        print(pstr)
        print('-' * len(pstr))

    def print_crit(self, fit, reg=None, rls=None, gain=None):
        if not self.verbose:
            return
        def cond_append(s, x):
            if s:
                s += x
            return s
        pstr = ''
        if self.max_levels > 1:
            pstr = f'{self.level:3d}'
        if self.max_rls > 1:
            pstr = cond_append(pstr, ' | ')
            pstr += f'{self.rls:3d}'
        if self.max_gn > 1:
            pstr = cond_append(pstr, ' | ')
            pstr += f'{self.gn:3d}'
        pstr = cond_append(pstr, ' | ')
        pstr += f'{fit:12.6g}'
        if self.penalty:
            crit = fit
            pstr += f' + {reg:12.6g}'
            crit = crit + reg
            if self.penalty in ('tv', 'jtv'):
                pstr += f' + {rls:12.6}'
                crit = crit + rls
            pstr += f' = {crit:12.6g}'
        pstr += f' | {gain:7.2g}'
        print(pstr)


def _get_level(level, aff0, shape0):
    return spatial.affine_resize(aff0, shape0, 1/(2 ** (level-1)))


def resize(maps, rls, aff, shape):
    maps.volume = spatial.resize(maps.volume, shape=shape)
    maps.affine = aff
    if rls is not None:
        rls = spatial.resize(rls, shape=shape)
    return maps, rls


def pull_parameters(maps, transmit, receive,
                    target_aff, target_shape, **backend):
    """Reslice all parameters to the observed grid"""
    aff = core.linalg.lmdiv(maps.affine, target_aff)
    identity = torch.eye(aff.shape[-1], **core.utils.backend(aff))
    if torch.allclose(aff, identity) and target_shape == maps.shape[1:]:
        pd, r1, r2s, *mt = maps.fdata(**backend, copy=True)
    else:
        aff = aff.to(**backend)
        grid = smart_grid(aff, target_shape, maps[0].shape)
        pd, r1, r2s, *mt = smart_pull(maps.fdata(**backend, copy=False), grid)
    mt = mt[0] if mt else None

    # pull field maps to observed space
    b1m = None
    if receive is not None:
        aff1 = core.linalg.lmdiv(receive.affine, target_aff)
        aff1 = aff1.to(**backend)
        grid1 = smart_grid(aff1, target_shape, receive.shape)
        b1m = smart_pull(receive.fdata(**backend)[None], grid1)[0]
        if receive.unit in ('%', 'pct', 'p.u.'):
            b1m = b1m.div(100.) if grid1 is None else b1m.div_(100.)
        del grid1, aff1

    b1p = None
    if transmit is not None:
        aff1 = core.linalg.lmdiv(transmit.affine, target_aff)
        aff1 = aff1.to(**backend)
        grid1 = smart_grid(aff1, target_shape, transmit.shape)
        b1p = smart_pull(transmit.fdata(**backend)[None], grid1)[0]
        if grid1 is None:
            b1p = b1p.clone()
        if transmit.unit in ('%', 'pct', 'p.u.'):
            b1p /= 100.
        del grid1, aff1

    return pd, r1, r2s, mt, b1m, b1p


@torch.jit.script
def recon_intercept(pd, r1, mt: Optional[Tensor],
                    b1p: Optional[Tensor], b1m: Optional[Tensor],
                    fa: float, tr: float, has_mt: float):
    """Reconstruct signal at TE=0

    Parameters
    ----------
    pd : Log-PD map
    r1 : Log-R1 map
    mt : Logit-MTsat map, optional
    b1p : Transmit map, optional
    b1m : Receive map, optional
    fa : Nominal flip angle
    tr : Repetition time
    has_mt : True if an MT pulse was used

    Returns
    -------
    fit0 : Fitted signal at TE=0

    """
    # exponentiate parameters
    pd = pd.exp_()
    r1 = r1.exp_()
    if has_mt and mt is not None:
        mt = mt.neg_().exp_().add_(1).reciprocal_()
        mt = mt.neg_().add_(1)  # inverse probability

    # modulate flip angle
    fa_tensor: Optional[Tensor] = None
    if b1p is not None:
        fa_tensor = b1p.mul_(fa)

    if b1m is not None:
        pd = pd.mul_(b1m)

    # precompute intercept
    #                             (1 - mt) * (1 - exp(-r1*tr))
    # fit0 = pd * sin(fa) * ----------------------------------------
    #                        (1 - cos(fa) * (1 - mt) * exp(-r1*tr))

    e1 = (-tr * r1).exp()
    num = pd * (1 - e1)
    den = e1
    if fa_tensor is not None:
        num = num * fa_tensor.sin()
        den = den * fa_tensor.cos()
    else:
        num = num * math.sin(fa)
        den = den * math.cos(fa)
    if has_mt and mt is not None:
        num = num * mt
        den = den * mt
    den = 1 - den
    return num / den


@torch.jit.script
def recon_intercept_plus(pd, r1, mt: Optional[Tensor],
                    b1p: Optional[Tensor], b1m: Optional[Tensor],
                    fa: float, tr: float, has_mt: bool)\
        -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Reconstruct signal at TE=0 + return values we need

    Parameters
    ----------
    pd : Log-PD map
    r1 : Log-R1 map
    mt : Logit-MTsat map, optional
    b1p : Transmit map, optional
    b1m : Receive map, optional
    fa : Nominal flip angle
    tr : Repetition time
    has_mt : True if an MT pulse was used

    Returns
    -------
    fit0 : Fitted signal at TE=0
    r1 : R1 map
    e1 : exp(-TR * R1)
    omt : (1-MTsat)
    cosfa : cos(fa)

    """
    # exponentiate parameters
    pd = pd.exp_()
    r1 = r1.exp_()
    if has_mt and mt is not None:
        mt = mt.neg_().exp_().add(1).reciprocal_()
        mt = mt.neg_().add_(1)  # inverse probability

    # modulate flip angle
    fa_tensor: Optional[Tensor] = None
    if b1p is not None:
        fa_tensor = b1p.mul_(fa)

    if b1m is not None:
        pd = pd.mul_(b1m)

    # precompute intercept
    #                             (1 - mt) * (1 - exp(-r1*tr))
    # fit0 = pd * sin(fa) * ----------------------------------------
    #                        (1 - cos(fa) * (1 - mt) * exp(-r1*tr))

    e1 = (-tr * r1).exp()
    num = pd * (1 - e1)
    den = e1
    if fa_tensor is not None:
        num = num * fa_tensor.sin()
        den = den * fa_tensor.cos()
    else:
        num = num * math.sin(fa)
        den = den * math.cos(fa)
    if has_mt and mt is not None:
        num = num * mt
        den = den * mt
    den = 1 - den

    if fa_tensor is not None:
        fa_tensor = fa_tensor.cos()
    return num / den, r1, e1, mt, fa_tensor


@torch.jit.script
def gradhess_signal(fit, r1, r2s, e1,
                    omt: Optional[Tensor], cosfa: Optional[Tensor],
                    tr: float, te: float, fa: float):
    """Compute the gradient of the signal term"""
    # Returns a full-sized hessian (with offdiag terms) even though they
    # are not filled. This saves an allocation when applying the chaine rule.

    nb_prm = 3 + int(omt is not None)
    nb_hess = nb_prm * (nb_prm + 1) // 2

    if omt is not None and cosfa is not None:
        omt_x_cosfa = omt * cosfa
    elif cosfa is not None:
        omt_x_cosfa = cosfa
    elif omt is not None:
        omt_x_cosfa = omt * math.cos(fa)
    else:
        omt_x_cosfa = torch.as_tensor(math.cos(fa),
                                      dtype=fit.dtype, device=fit.device)

    grad = fit[None].expand([nb_prm] + fit.shape).clone()
    grad[1] *= -tr * r1 * e1 * (omt_x_cosfa - 1) / ((1 - e1) * (1 - omt_x_cosfa * e1))
    grad[2] *= -te * r2s
    if omt is not None:
        grad[3] *= (omt - 1) / (1 - omt_x_cosfa * e1)

    hess = grad.new_empty([nb_hess] + fit.shape)
    hess[:nb_prm].copy_(grad)
    hess[1] *= 1 - tr * r1 * (1 + omt_x_cosfa * e1) / (1 - omt_x_cosfa * e1)
    hess[2] *= (1 - te * r2s)
    if omt is not None:
        hess_mt = (omt - 1) * omt_x_cosfa * e1 / (1 - omt_x_cosfa * e1)
        hess[3] *= 2 * (hess_mt + omt - 0.5)

    return grad, hess


def gradhess_chain(grad, hess, res):
    """Apply the chain rule"""
    # Performs computation in-place.
    # Assumes that the input hessian is full-sized (with offdiag terms).

    # grad_crit = grad_signal * residuals
    # hess_crit = grad_signal^2 + abs(hess_signal * residuals)

    nb_prm = len(grad)

    hess[:nb_prm].mul_(res).abs_().addcmul_(grad, grad)
    cnt = nb_prm
    for i in range(nb_prm):
        for j in range(i+1, nb_prm):
            torch.mul(grad[i], grad[j], out=hess[cnt])
            cnt += 1

    grad.mul_(res)
    return grad, hess


def derivatives_parameters(contrast, maps, receive, transmit, opt, do_grad=True):
    """Compute the gradient and Hessian of the parameter maps with
    respect to one contrast.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    maps : (*recon_shape) ParameterMaps
        Current parameter maps, with fields pd, r1, r2s and mt.
    receive :
    transmit :
    opt : Options
    do_grad : bool, default=True
        If False, only compute the lnegative og-likelihood.

    Returns
    -------
    crit : () tensor
        Negative log-likelihood
    grad : (3|4, *recon_shape) tensor
        Gradient with respect to (the log of):
            [0] pd
            [1] r1
            [2] r2s
            [3] mt (optional)
    hess : (6|10, *recon_shape) tensor
        Hessian with respect to:
            [0] pd ** 2
            [1] r1 ** 2
            [2] r2s ** 2
            [ |3] mt ** 2 (optional)
            [3|4] pd * r1
            [4|5] pd * r2s
            [ |6] pd * mt (optional)
            [5|7] r1 * r2s
            [ |8] r1 * mt (optional)
            [ |9] r2s * mt (optional)

    """
    dtype = opt.backend.dtype
    device = opt.backend.device
    chi = opt.likelihood[0].lower() == 'c'
    backend = dict(dtype=dtype, device=device)

    # sequence parameters
    lam = 1 / contrast.noise
    tr = contrast.tr                                # TR is stored in sec
    fa = contrast.fa / 180. * core.constants.pi     # FA is stored in deg
    dof = contrast.dof
    
    obs_shape = contrast.shape[1:]
    recon_shape = maps.shape[1:]

    # pull parameter maps to observed space
    has_mt = bool(hasattr(maps, 'mt') and getattr(contrast, 'mt', False))
    if not has_mt:
        maps = maps.drop_mt()
    pd, r1, r2s, mt, b1m, b1p \
        = pull_parameters(maps, transmit, receive, contrast.affine, obs_shape,
                          **backend)

    # precompute intercepts + useful values
    r2s = r2s.exp_()
    fit0, r1, e1, omt, cosfa \
        = recon_intercept_plus(pd, r1, mt, b1p, b1m, fa, tr, has_mt)
    del pd, mt, b1p, b1m

    nb_prm, nb_hes = (4, 10) if has_mt else (3, 6)
    grad = torch.zeros((nb_prm,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((nb_hes,) + obs_shape, **backend) if do_grad else None
    crit = 0

    for echo in contrast:

        # compute residuals
        dat = echo.fdata(**backend, rand=True, missing=0)    # observed
        fit = r2s.mul(-echo.te).exp_().mul_(fit0)            # fitted

        msk = get_mask_missing(dat, fit)
        dat.masked_fill_(msk, 0)
        fit.masked_fill_(msk, 0)
        msk = msk.bitwise_not_()

        if chi:
            crit1, res = nll_chi(dat, fit, msk, lam, dof)
        else:
            crit1, res = nll_gauss(dat, fit, msk, lam)
        del dat
        crit += crit1

        if do_grad:

            grad1, hess1 = gradhess_signal(fit, r1, r2s, e1, omt, cosfa,
                                           tr, echo.te, fa)
            grad1, hess1 = gradhess_chain(grad1, hess1, res)

            mask_nan_(hess1)
            hess1[:(3+has_mt)].masked_fill_(msk.bitwise_not(), 1e-3)
            hess1[(3+has_mt):].masked_fill_(msk.bitwise_not(), 0)
            hess.add_(hess1, alpha=lam)
            del hess1

            mask_nan_(grad1)
            grad.add_(grad1, alpha=lam)
            del grad1
        del res
    del r1, r2s, omt, e1, fit0, cosfa

    if not do_grad:
        return crit

    # push gradient and Hessian to recon space
    aff = core.linalg.lmdiv(maps.affine, contrast.affine).to(**backend)
    grid = smart_grid(aff, obs_shape, recon_shape)
    grad = smart_push(grad, grid, recon_shape)
    hess = smart_push(hess, grid, recon_shape)
    return crit, grad, hess


# def _nonlin_gradient(contrast, maps, receive, transmit, opt, do_grad=True):
#     """Compute the gradient and Hessian of the parameter maps with
#     respect to one contrast.
#
#     Parameters
#     ----------
#     contrast : (nb_echo, *obs_shape) GradientEchoMulti
#         A single echo series (with the same weighting)
#     maps : (*recon_shape) ParameterMaps
#         Current parameter maps, with fields pd, r1, r2s and mt.
#     receive :
#     transmit :
#     opt : Options
#     do_grad : bool, default=True
#         If False, only compute the lnegative og-likelihood.
#
#     Returns
#     -------
#     crit : () tensor
#         Negative log-likelihood
#     grad : (3|4, *recon_shape) tensor
#         Gradient with respect to (the log of):
#             [0] pd
#             [1] r1
#             [2] r2s
#             [3] mt (optional)
#     hess : (6|10, *recon_shape) tensor
#         Hessian with respect to:
#             [0] pd ** 2
#             [1] r1 ** 2
#             [2] r2s ** 2
#             [ |3] mt ** 2 (optional)
#             [3|4] pd * r1
#             [4|5] pd * r2s
#             [ |6] pd * mt (optional)
#             [5|7] r1 * r2s
#             [ |8] r1 * mt (optional)
#             [ |9] r2s * mt (optional)
#
#     """
#
#     dtype = opt.backend.dtype
#     device = opt.backend.device
#     chi = opt.likelihood[0].lower() == 'c'
#     backend = dict(dtype=dtype, device=device)
#
#     # sequence parameters
#     lam = 1 / contrast.noise
#     tr = contrast.tr  # TR is stored in sec
#     fa = contrast.fa / 180. * core.constants.pi  # FA is stored in deg
#     dof = contrast.dof
#
#     obs_shape = contrast.shape[1:]
#     recon_shape = maps.shape[1:]
#
#     # pull parameter maps to observed space
#     aff = core.linalg.lmdiv(maps.affine, contrast.affine)
#     aff = aff.to(**backend)
#     grid = smart_grid(aff, obs_shape, recon_shape)
#     drop_mt = hasattr(maps, 'mt') and not getattr(contrast, 'mt', False)
#     if drop_mt:
#         maps = maps[:-1]
#     dmaps = torch.stack([map.fdata(**backend) for map in maps])
#     dmaps = smart_pull(dmaps, grid)
#     grid = None if grid is None else True
#     pd, r1, r2s, *mt = dmaps
#     mt = mt[0] if mt else None
#     has_mt = mt is not None
#     if has_mt:
#         nb_prm = 4
#         nb_hes = 10
#     else:
#         nb_prm = 3
#         nb_hes = 6
#
#     # pull field maps to observed space
#     if receive is not None:
#         aff1 = core.linalg.lmdiv(receive.affine, contrast.affine)
#         aff1 = aff1.to(**backend)
#         grid1 = smart_grid(aff1, obs_shape, receive.shape)
#         b1m = smart_pull(receive.fdata(**backend)[None], grid1)[0]
#         if receive.unit in ('%', 'pct', 'p.u.'):
#             b1m = b1m.div(100.) if grid1 is None else b1m.div_(100.)
#         del grid1, aff1
#
#     if transmit is not None:
#         aff1 = core.linalg.lmdiv(transmit.affine, contrast.affine)
#         aff1 = aff1.to(**backend)
#         grid1 = smart_grid(aff1, obs_shape, transmit.shape)
#         b1p = smart_pull(transmit.fdata(**backend)[None], grid1)[0]
#         if grid1 is None:
#             b1p = b1p.clone()
#         if transmit.unit in ('%', 'pct', 'p.u.'):
#             b1p /= 100.
#         del grid1, aff1
#
#     # exponentiate
#     pd = pd.exp() if grid is None else pd.exp_()
#     r1 = r1.exp() if grid is None else r1.exp_()
#     r2s = r2s.exp() if grid is None else r2s.exp_()
#     if has_mt:
#         # mt is encoded by a sigmoid:
#         # > mt = 1 / (1 + exp(-prm))
#         mt = mt.neg() if grid is None else mt.neg_()
#         mt = mt.exp_().add_(1).reciprocal_()
#
#     # precompute intercept
#     #                             (1 - mt) * (1 - exp(-r1*tr))
#     # fit0 = pd * sin(fa) * ----------------------------------------
#     #                        (1 - cos(fa) * (1 - mt) * exp(-r1*tr))
#     fit0 = pd
#     if receive is not None:
#         fit0 *= b1m
#         del b1m
#     if transmit:
#         b1p *= fa
#         fa = b1p
#         del b1p
#     else:
#         fa = torch.as_tensor(fa, **backend)
#     fit0 *= fa.sin()
#     cosfa = fa.cos_()
#     e1 = r1
#     e1 *= -tr
#     e1 = e1.exp_()
#     fit0 *= (1 - e1)
#     if has_mt:
#         omt = mt.neg_().add_(1)
#         omt_x_cosfa = cosfa.mul(omt)
#         fit0 *= omt
#         del mt
#     else:
#         omt = None
#         omt_x_cosfa = cosfa
#     fit0 /= (1 - omt_x_cosfa * e1)
#     del cosfa
#
#     crit = 0
#     grad = torch.zeros((nb_prm,) + obs_shape, **backend) if do_grad else None
#     hess = torch.zeros((nb_hes,) + obs_shape, **backend) if do_grad else None
#
#     for echo in contrast:
#
#         # compute residuals
#         dat = echo.fdata(**backend)  # observed
#         fit = fit0 * (-echo.te * r2s).exp_()  # fitted
#         if chi:
#             msk = torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0) & (
#                         fit > 0)
#             tiny = torch.tensor(1e-32, dtype=dtype, device=device)
#             dat[~msk] = 0
#             fit[~msk] = 0
#         else:
#             msk = torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0)
#             dat[~msk] = 0
#             fit[~msk] = 0
#
#         if chi:
#             z = (dat[msk] * fit[msk] * lam).clamp_min_(tiny)
#             critn = ((dof / 2. - 1.) * fit[msk].log()
#                      - (dof / 2.) * dat[msk].log()
#                      + 0.5 * lam * (fit[msk].square() + dat[msk].square())
#                      - besseli(dof / 2. - 1., z, 'log'))
#             critn = torch.sum(critn, dtype=torch.double)
#             crit = crit + critn
#
#             z = besseli_ratio(dof / 2. - 1., z, N=2, K=4)
#             res = torch.zeros(obs_shape, **backend)  # unsure
#             res[msk] = z.mul_(dat[msk]).neg_().add_(fit[msk])
#             del z
#         else:
#             # gaussian log-likelihood
#             res = dat.neg_().add_(fit)
#             crit = crit + 0.5 * lam * res.square().sum(dtype=torch.double)
#
#         if do_grad:
#
#             # PD / R1 / R2* / MT
#             grad1 = torch.empty_like(grad)
#
#             # grad_crit = grad_signal * residuals
#             # hess_crit = grad_signal^2 + hess_signal * residuals
#
#             # grad_signal: compute gradient of the signal term
#             #   all gradients are multiplied by fit at some point
#             #   so we initialize them with fit
#             grad1[...] = fit[None].expand((3 + has_mt, *fit.shape))
#             del fit
#             grad1[1] *= -tr * r1 * (omt_x_cosfa - 1) * e1
#             grad1[1] /= (1 - e1) * (1 - omt_x_cosfa * e1)
#             grad1[2] *= -echo.te * r2s
#             if has_mt:
#                 grad1[3] *= (omt - 1) / (1 - omt_x_cosfa * e1)
#
#             # hess_signal: compute diagonal of the hessian of the signal term
#             hess0 = torch.empty_like(grad)
#
#             hess0[0] = grad1[0]
#             hess0[1] = - tr * r1
#             hess0[1] *= (1 + omt_x_cosfa * e1)
#             hess0[1] /= (1 - omt_x_cosfa * e1)
#             hess0[1] += 1
#             hess0[1] *= grad1[1]
#             hess0[2] = r2s
#             hess0[2] *= -echo.te
#             hess0[2] += 1
#             hess0[2] *= grad1[2]
#             if has_mt:
#                 hess0[3] = (omt - 1) * omt_x_cosfa * e1
#                 hess0[3] /= 1 - omt_x_cosfa * e1
#                 hess0[3] += omt
#                 hess0[3] -= 0.5
#                 hess0[3] *= 2
#                 hess0[3] *= grad1[3]
#
#             # hess_signal: multiply by residuals
#             hess0 *= res
#             hess0 = hess0.abs_()
#
#             # hess_crit: add grad_signal^2 to hess_signal
#             hess1 = torch.empty_like(hess)
#             hess1[0] = grad1[0].square().add_(hess0[0])
#             hess1[1] = grad1[1].square().add_(hess0[1])
#             hess1[2] = grad1[2].square().add_(hess0[2])
#             if has_mt:
#                 hess1[3] = grad1[3].square().add_(hess0[3])
#                 del hess0
#                 hess1[4] = grad1[0] * grad1[1]
#                 hess1[5] = grad1[0] * grad1[2]
#                 hess1[6] = grad1[0] * grad1[3]
#                 hess1[7] = grad1[1] * grad1[2]
#                 hess1[8] = grad1[1] * grad1[3]
#                 hess1[9] = grad1[2] * grad1[3]
#             else:
#                 del hess0
#                 hess1[3] = grad1[0] * grad1[1]
#                 hess1[4] = grad1[0] * grad1[2]
#                 hess1[5] = grad1[1] * grad1[2]
#             hess1 *= lam
#             hess1[:, ~msk] = 0
#             diag = hess1[:(3 + has_mt)]
#             diag.masked_fill_(torch.isfinite(diag).bitwise_not_(), 1e-3)
#             # diag[~torch.isfinite(diag)] = 1e-3
#             offdiag = hess1[(3 + has_mt):]
#             # offdiag[~torch.isfinite(offdiag)] = 0
#             offdiag.masked_fill_(torch.isfinite(offdiag).bitwise_not_(), 0)
#             hess += hess1
#             del hess1
#
#             # grad_crit: multiply by residuals
#             grad1 *= res
#             grad1 *= lam
#             # grad1[~torch.isfinite(grad1)] = 0
#             grad1.masked_fill_(torch.isfinite(grad1).bitwise_not_(), 0)
#             grad += grad1
#             del grad1
#         del res
#     del r1, r2s, omt, e1, fit0, omt_x_cosfa
#     if do_grad:
#         # push gradient and Hessian to recon space
#         grid = smart_grid(aff, obs_shape, recon_shape)
#         grad = smart_push(grad, grid, recon_shape)
#         hess = smart_push(hess, grid, recon_shape)
#         return crit, grad, hess
#     return crit
#
#
# def _nonlin_reg(map, vx=1., rls=None, lam=1., do_grad=True):
#     """Compute the gradient of the regularisation term.
#
#     The regularisation term has the form:
#     `0.5 * lam * sum(w[i] * (g+[i]**2 + g-[i]**2) / 2)`
#     where `i` indexes a voxel, `lam` is the regularisation factor,
#     `w[i]` is the RLS weight, `g+` and `g-` are the forward and
#     backward spatial gradients of the parameter map.
#
#     Parameters
#     ----------
#     map : (*shape) tensor
#         Parameter map
#     vx : float or sequence[float]
#         Voxel size
#     rls : (*shape) tensor, optional
#         Weights from the reweighted least squares scheme
#     lam : float, default=1
#         Regularisation factor
#     do_grad : bool, default=True
#         Return both the criterion and gradient
#
#     Returns
#     -------
#     reg : () tensor
#         Regularisation term
#     grad : (*shape) tensor
#         Gradient with respect to the parameter map
#
#     """
#
#     grad_fwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='f')
#     grad_bwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='b')
#     if rls is not None:
#         grad_fwd *= rls[..., None]
#         grad_bwd *= rls[..., None]
#     grad_fwd = spatial.div(grad_fwd, dim=[0, 1, 2], voxel_size=vx, side='f')
#     grad_bwd = spatial.div(grad_bwd, dim=[0, 1, 2], voxel_size=vx, side='b')
#
#     grad = grad_fwd
#     grad += grad_bwd
#     grad *= lam / 2.   # average across directions (3) and side (2)
#
#     if do_grad:
#         reg = (map * grad).sum(dtype=torch.double)
#         return 0.5 * reg, grad
#     else:
#         grad *= map
#         return 0.5 * grad.sum(dtype=torch.double)
#
#
# def _nonlin_solve(hess, grad, rls, lam, vx, opt):
#     """Solve the regularized linear system
#
#     Parameters
#     ----------
#     hess : (P*(P+1)//2, *shape) tensor
#     grad : (P, *shape) tensor
#     rls : (P, *shape) tensor_like
#     lam : (P,) sequence[float]
#     vx : (D,) sequence[float]
#     opt : Options
#
#     Returns
#     -------
#     delta : (P, *shape) tensor
#
#     """
#
#     def hess_fn(x):
#         result = hessian_sym_matmul(hess, x)
#         if not opt.penalty.norm:
#             return result
#         for i, (map, weight, l) in enumerate(zip(x, rls, lam)):
#             if not l:
#                 continue
#             _, res1 = _nonlin_reg(map, vx, weight, l)
#             result[i] += res1
#         return result
#
#     # The Hessian is A = H + L, where H corresponds to the data term
#     # and L to the regularizer. Note that, L = D'WD where D is the
#     # gradient operator, D' the divergence and W a diagonal matrix
#     # that contains the RLS weights.
#     # We use (H + diag(|D'D|w)) as a preconditioner because it is
#     # easy to invert and majorises the true Hessian.
#     hessp = hess.clone()
#     smo = torch.as_tensor(vx).square().reciprocal().sum().item()
#     for i, (weight, l) in enumerate(zip(rls, lam)):
#         hessp[i] += l * (rls_maj(weight, vx) if weight is not None else 4*smo)
#
#     def precond(x):
#         return hessian_sym_solve(hessp, x)
#
#     result = core.optim.cg(hess_fn, grad, precond=precond,
#                            max_iter=opt.optim.max_iter_cg,
#                            tolerance=opt.optim.tolerance_cg,
#                            verbose=(opt.verbose > 1))
#     return result
#
#
# def update_rls(maps, lam=1., norm='jtv'):
#     """Update the (L1) weights.
#
#     Parameters
#     ----------
#     map : (P, *shape) ParameterMaps
#         Parameter map
#     lam : float or (P,) sequence[float], default=1
#         Regularisation factor
#     norm : {'tv', 'jtv'}, default='jtv'
#
#     Returns
#     -------
#     rls : ([P], *shape) tensor
#         Weights from the reweighted least squares scheme
#     """
#
#     if norm not in ('tv', 'jtv', '__internal__'):
#         return None
#
#     if isinstance(maps, ParameterMap):
#         # single map
#         # this should only be an internal call
#         # -> we return the squared gradient map
#         assert norm == '__internal__'
#         vx = spatial.voxel_size(maps.affine)
#         grad_fwd = spatial.diff(maps.fdata(), dim=[0, 1, 2], voxel_size=vx, side='f')
#         grad_bwd = spatial.diff(maps.fdata(), dim=[0, 1, 2], voxel_size=vx, side='b')
#
#         grad = grad_fwd.square_().sum(-1)
#         grad += grad_bwd.square_().sum(-1)
#         grad *= lam / 2.   # average across sides (2)
#         return grad
#
#     # multiple maps
#
#     if norm == 'tv':
#         rls = []
#         for map, l in zip(maps, lam):
#             rls1 = update_rls(map, l, '__internal__')
#             rls1 = rls1.sqrt_()
#             rls.append(rls1)
#         return torch.stack(rls, dim=0)
#     else:
#         assert norm == 'jtv'
#         rls = 0
#         for map, l in zip(maps, lam):
#             rls += update_rls(map, l, '__internal__')
#         rls = rls.sqrt_()
#
#     return rls

import torch
from nitorch import core, spatial
from ._options import GREEQOptions
from ._preproc import preproc, postproc
from ..utils import (hessian_sym_loaddiag, hessian_sym_matmul,
                     hessian_sym_solve, hessian_sym_inv, rls_maj,
                     smart_grid, smart_pull, smart_push)
from scipy.special import jve
from nitorch.tools.qmri.param import ParameterMap




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
#   In that case, we pull the param the signal only to compute the
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
                     'tolerance_rls': 1e-05,     # Tolerance for early stopping (RLS)
                     'tolerance_gn':  1e-05,         ""
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
    opt = GREEQOptions().update(opt, **kwopt)
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
    lam = core.utils.make_vector(lam, 4, **backend)  # PD, R1, R2*, MT

    # --- initialize weights (RLS) ---
    if str(opt.penalty.norm).lower() == 'none' or all(lam == 0):
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
    for level in range(opt.optim.nb_levels, 0, -1):
        printer.level = level

        if opt.optim.nb_levels > 1:
            aff, shape = _get_level(level, aff0, shape0)
            vx = spatial.voxel_size(aff)
            vol = vx.prod() / vol0
            maps, rls = _resize(maps, rls, aff, shape)
            if opt.penalty.norm in ('tv', 'jtv'):
                sumrls = 0.5 * vol * rls.reciprocal().sum(dtype=torch.double)
        
        # --- compute derivatives ---
        nb_prm = len(maps)
        nb_hes = nb_prm * (nb_prm + 1) // 2
        grad = torch.empty((nb_prm,) + shape, **backend)
        hess = torch.empty((nb_hes,) + shape, **backend)
    
        ll_rls = []
        ll_max = core.constants.ninf

        max_iter_rls = max(opt.optim.max_iter_rls // level, 1)
        for n_iter_rls in range(max_iter_rls):
            # --- Reweighted least-squares loop ---
            printer.rls = n_iter_rls
            multi_rls = rls if opt.penalty.norm == 'tv' else [rls] * len(maps)

            # --- Gauss Newton loop ---
            ll_gn = []
            for n_iter_gn in range(opt.optim.max_iter_gn):
                printer.gn = n_iter_gn
                crit = 0
                grad.zero_()
                hess.zero_()
                # --- loop over contrasts ---
                for contrast, b1m, b1p in zip(data, receive, transmit):
                    # compute gradient
                    crit1, g1, h1 = _nonlin_gradient(contrast, maps, b1m, b1p, opt)

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
                    torch.cuda.empty_cache()
                # --- penalty ---
                reg = 0.
                if opt.penalty.norm:
                    for i, (map, weight, l) in enumerate(zip(maps, multi_rls, lam)):
                        if not l:
                            continue
                        reg1, g1 = _nonlin_reg(map.fdata(**backend), vx, weight, l * vol)
                        reg += reg1
                        grad[i] += g1
                        del g1, reg1

                # --- gauss-newton ---
                if not torch.isfinite(hess).all():
                    print('WARNING: NaNs in hess')
                if not torch.isfinite(grad).all():
                    print('WARNING: NaNs in grad')
                if opt.penalty.norm:
                    hess = hessian_sym_loaddiag(hess, 1e-5, 1e-8)
                    deltas = _nonlin_solve(hess, grad, multi_rls, lam * vol, vx, opt)
                else:
                    hess = hessian_sym_loaddiag(hess, 1e-3, 1e-4)
                    deltas = hessian_sym_solve(hess, grad)
                if not torch.isfinite(deltas).all():
                    print('WARNING: NaNs in delta')

                for map, delta in zip(maps, deltas):
                    map.volume -= delta
                    if map.min is not None or map.max is not None:
                        map.volume.clamp_(map.min, map.max)
                    del delta
                del deltas

                # --- Compute gain ---
                ll = crit + reg + sumrls
                ll_max = max(ll_max, ll)
                ll_prev = ll_gn[-1] if ll_gn else ll_max
                gain = (ll_prev - ll) / (ll_max - ll_prev)
                ll_gn.append(ll)
                printer.print_crit(crit, reg, sumrls, gain)
                if gain < opt.optim.tolerance_gn:
                    print('GN converged: ', ll_prev.item(), '->', ll.item())
                    break
            # --- Update RLS weights ---
            if opt.penalty.norm in ('tv', 'jtv'):
                del multi_rls
                rls = _nonlin_rls(maps, lam, opt.penalty.norm)
                sumrls = (0.5 * vol) * rls.sum(dtype=torch.double)
                eps = core.constants.eps(rls.dtype)
                rls = rls.clamp_min_(eps).reciprocal_()

                # --- Compute gain ---
                # (we are late by one full RLS iteration when computing the 
                #  gain but we save some computations)
                ll = ll_gn[-1]
                ll_prev = ll_rls[-1][-1] if ll_rls else ll_max
                ll_rls.append(ll_gn)
                gain = (ll_prev - ll) / (ll_max - ll_prev)
                if abs(gain) < opt.optim.tolerance_rls:
                    print(f'RLS converged ({gain:7.2g})')
                    break

    del grad
    if opt.uncertainty:
        multi_rls = rls if opt.penalty.norm == 'tv' else [rls] * len(maps)
        uncertainty = _nonlin_uncertainty(hess, multi_rls, lam * vol, vx, opt)
        maps.pd.uncertainty = uncertainty[0]
        maps.r1.uncertainty = uncertainty[1]
        maps.r2s.uncertainty = uncertainty[2]
        if hasattr(maps, 'mt'):
            maps.mt.uncertainty = uncertainty[3]
    
    # --- Prepare output ---
    return postproc(maps)


def _nonlin_uncertainty(hess, rls, lam, vx, opt):
    """Diagonal posterior uncertainty"""
    uncertainty = hess.clone()
    if opt.penalty:
        smo = torch.as_tensor(vx).square().reciprocal().sum().item()
        for i, (weight, l) in enumerate(zip(rls, lam)):
            uncertainty[i] += l * (rls_maj(weight, vx) if weight is not None
                                   else 4 * smo)
    return hessian_sym_inv(uncertainty, diag=True)


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
        pattern = ''
        args = []
        if self.max_levels > 1:
            pattern = '{:^3s}'
            args.append('lvl')
        if self.max_rls > 1:
            if pattern:
                pattern += ' | '
            pattern += '{:^3s}'
            args.append('rls')
        if self.max_gn > 1:
            if pattern:
                pattern += ' | '
            pattern += '{:^3s}'
            args.append('gn')
        if pattern:
            pattern += ' | '
        pattern += '{:^12s}'
        args.append('fit')
        if self.penalty:
            pattern += ' + {:^12s}'
            args.append('reg')
            if self.penalty in ('tv', 'jtv'):
                pattern += ' + {:^12s}'
                args.append('rls')
            pattern += ' = {:^12s}'
            args.append('crit')
        pattern += ' | {:^7s}'
        args.append('gain')
        pattern = pattern.format(*args)
        print(pattern)
        print('-' * len(pattern))

    def print_crit(self, fit, reg=None, rls=None, gain=None):
        if not self.verbose:
            return
        pattern = ''
        args = []
        if self.max_levels > 1:
            pattern = '{:3d}'
            args.append(self.level)
        if self.max_rls > 1:
            if pattern:
                pattern += ' | '
            pattern += '{:3d}'
            args.append(self.rls)
        if self.max_gn > 1:
            if pattern:
                pattern += ' | '
            pattern += '{:3d}'
            args.append(self.gn)
        if pattern:
            pattern += ' | '
        pattern += '{:12.6g}'
        args.append(fit)
        if self.penalty:
            crit = fit
            pattern += ' + {:12.6g}'
            args.append(reg)
            crit = crit + reg
            if self.penalty in ('tv', 'jtv'):
                pattern += ' + {:12.6g}'
                args.append(rls)
                crit = crit + rls
            pattern += ' = {:12.6g}'
            args.append(crit)
        pattern += ' | {:7.2g}'
        args.append(gain)
        pattern = pattern.format(*args)
        print(pattern)


def _get_level(level, aff0, shape0):
    return spatial.affine_resize(aff0, shape0, 1/(2 ** (level-1)))


def _resize(maps, rls, aff, shape):
    for map in maps:
        map.volume = spatial.resize(map.volume[None, None, ...], 
                                    shape=shape)[0, 0]
        map.affine = aff
    maps.affine = aff
    if rls is not None:
        if rls.dim() == len(shape):
            rls = spatial.resize(rls[None, None], shape=shape)[0, 0]
        else:
            rls = spatial.resize(rls[None], shape=shape)[0]
    return maps, rls


def _nonlin_gradient(contrast, maps, receive, transmit, opt, do_grad=True):
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
    backend = dict(dtype=dtype, device=device)

    # sequence parameters
    lam = 1 / contrast.noise
    tr = contrast.tr                                # TR is stored in sec
    fa = contrast.fa / 180. * core.constants.pi     # FA is stored in deg
    
    obs_shape = contrast.shape[1:]
    recon_shape = maps.shape[1:]

    # pull parameter maps to observed space
    aff = core.linalg.lmdiv(maps.affine, contrast.affine)
    aff = aff.to(**backend)
    grid = smart_grid(aff, obs_shape, recon_shape)
    drop_mt = hasattr(maps, 'mt') and not getattr(contrast, 'mt', False)
    if drop_mt:
        maps = maps[:-1]
    dmaps = torch.stack([map.fdata(**backend) for map in maps])
    dmaps = smart_pull(dmaps, grid)
    grid = None if grid is None else True
    pd, r1, r2s, *mt = dmaps
    mt = mt[0] if mt else None
    has_mt = mt is not None
    if has_mt:
        nb_prm = 4
        nb_hes = 10
    else:
        nb_prm = 3
        nb_hes = 6

    # pull field maps to observed space
    if receive is not None:
        aff1 = core.linalg.lmdiv(receive.affine, contrast.affine)
        aff1 = aff1.to(**backend)
        grid1 = smart_grid(aff1, obs_shape, receive.shape)
        b1m = smart_pull(receive.fdata(**backend)[None], grid1)[0]
        if receive.unit in ('%', 'pct', 'p.u.'):
            b1m = b1m.div(100.) if grid1 is None else b1m.div_(100.)
        del grid1, aff1

    if transmit is not None:
        aff1 = core.linalg.lmdiv(transmit.affine, contrast.affine)
        aff1 = aff1.to(**backend)
        grid1 = smart_grid(aff1, obs_shape, transmit.shape)
        b1p = smart_pull(transmit.fdata(**backend)[None], grid1)[0]
        if grid1 is None:
            b1p = b1p.clone()
        if transmit.unit in ('%', 'pct', 'p.u.'):
            b1p /= 100.
        del grid1, aff1

    # exponentiate
    pd = pd.exp() if grid is None else pd.exp_()
    r1 = r1.exp() if grid is None else r1.exp_()
    r2s = r2s.exp() if grid is None else r2s.exp_()
    if has_mt:
        # mt is encoded by a sigmoid:
        # > mt = 1 / (1 + exp(-prm))
        mt = mt.neg() if grid is None else mt.neg_()
        mt = mt.exp_().add_(1).reciprocal_()

    # precompute intercept
    #                             (1 - mt) * (1 - exp(-r1*tr))
    # fit0 = pd * sin(fa) * ----------------------------------------
    #                        (1 - cos(fa) * (1 - mt) * exp(-r1*tr))
    fit0 = pd
    if receive is not None:
        fit0 *= b1m
        del b1m
    if transmit:
        b1p *= fa
        fa = b1p
        del b1p
    else:
        fa = torch.as_tensor(fa, **backend)
    fit0 *= fa.sin()
    cosfa = fa.cos_()
    e1 = r1
    e1 *= -tr
    e1 = e1.exp_()
    fit0 *= (1 - e1)
    if has_mt:
        omt = mt.neg_().add_(1)
        omt_x_cosfa = cosfa.mul(omt)
        fit0 *= omt
        del mt
    else:
        omt = None
        omt_x_cosfa = cosfa
    fit0 /= (1 - omt_x_cosfa * e1)
    del cosfa

    crit = 0
    grad = torch.zeros((nb_prm,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((nb_hes,) + obs_shape, **backend) if do_grad else None

    for echo in contrast:

        # compute residuals
        dat = echo.fdata(**backend)                          # observed
        fit = fit0 * (-echo.te * r2s).exp_()                 # fitted
        msk = torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0)    # mask of observed
        dat[~msk] = 0
        fit[~msk] = 0
        res = dat.neg_()
        res += fit
        del dat

        # compute log-likelihood
        #crit = crit + 0.5 * lam * res.square().sum(dtype=torch.double)

        # chi log likelihood
        dof= 14.1450
        sig = 3
        besin = dat*fit/(sig**2)
        bes = jve(dof/2-1, besin)
        crit = crit + (dof/2-1)*torch.log(fit)-dof/2*torch.log(dat)+(fit.square()+dat.square())/(2*sig**2)\
            - torch.log(bes)



        if do_grad:

            # PD / R1 / R2* / MT
            grad1 = torch.empty_like(grad)

            # grad_crit = grad_signal * residuals
            # hess_crit = grad_signal^2 + hess_signal * residuals
            
            # grad_signal: compute gradient of the signal term
            #   all gradients are multiplied by fit at some point
            #   so we initialize them with fit
            grad1[...] = fit[None].expand((3+has_mt, *fit.shape))

            grad1[1] *= -tr * r1 * (omt_x_cosfa - 1) * e1
            grad1[1] /= (1 - e1) * (1 - omt_x_cosfa * e1)
            grad1[2] *= -echo.te * r2s
            if has_mt:
                grad1[3] *= (omt - 1) / (1 - omt_x_cosfa * e1)
            
            # hess_signal: compute diagonal of the hessian of the signal term 
            hess0 = torch.empty_like(grad)
            
            hess0[0] = grad1[0]
            hess0[1] = - tr * r1 
            hess0[1] *= (1 + omt_x_cosfa * e1) 
            hess0[1] /= (1 - omt_x_cosfa * e1)
            hess0[1] += 1
            hess0[1] *= grad1[1]
            hess0[2] = r2s
            hess0[2] *= -echo.te
            hess0[2] += 1
            hess0[2] *= grad1[2]
            if has_mt:
                hess0[3] = (omt - 1) * omt_x_cosfa * e1 
                hess0[3] /= 1 - omt_x_cosfa * e1
                hess0[3] += omt 
                hess0[3] -= 0.5
                hess0[3] *= 2
                hess0[3] *= grad1[3]
            
            # hess_signal: multiply by residuals
            hess0 *= res
            hess0 = hess0.abs_()
            
            # hess_crit: add grad_signal^2 to hess_signal
            hess1 = torch.empty_like(hess)
            hess1[0] = grad1[0].square().add_(hess0[0])
            hess1[1] = grad1[1].square().add_(hess0[1])
            hess1[2] = grad1[2].square().add_(hess0[2])
            if has_mt:
                hess1[3] = grad1[3].square().add_(hess0[3])
                del hess0
                hess1[4] = grad1[0] * grad1[1]
                hess1[5] = grad1[0] * grad1[2]
                hess1[6] = grad1[0] * grad1[3]
                hess1[7] = grad1[1] * grad1[2]
                hess1[8] = grad1[1] * grad1[3]
                hess1[9] = grad1[2] * grad1[3]
            else:
                del hess0
                hess1[3] = grad1[0] * grad1[1]
                hess1[4] = grad1[0] * grad1[2]
                hess1[5] = grad1[1] * grad1[2]
            hess1 *= lam
            hess1[:, ~msk] = 0
            diag = hess1[:(3+has_mt)]
            diag[~torch.isfinite(diag)] = 1e-3
            offdiag = hess1[(3+has_mt):]
            offdiag[~torch.isfinite(offdiag)] = 0
            hess += hess1
            del hess1

            # grad_crit: multiply by residuals
            grad1 *= res
            grad1 *= lam
            grad1[~torch.isfinite(grad1)] = 0
            grad += grad1
            del grad1
        del res
        torch.cuda.empty_cache()
    del r1, r2s, omt, e1, fit0, omt_x_cosfa
    if do_grad:
        # push gradient and Hessian to recon space
        grid = smart_grid(aff, obs_shape, recon_shape)
        grad = smart_push(grad, grid, recon_shape)
        hess = smart_push(hess, grid, recon_shape)
        return crit, grad, hess
    torch.cuda.empty_cache()
    return crit


def _nonlin_reg(map, vx=1., rls=None, lam=1., do_grad=True):
    """Compute the gradient of the regularisation term.

    The regularisation term has the form:
    `0.5 * lam * sum(w[i] * (g+[i]**2 + g-[i]**2) / 2)`
    where `i` indexes a voxel, `lam` is the regularisation factor,
    `w[i]` is the RLS weight, `g+` and `g-` are the forward and
    backward spatial gradients of the parameter map.

    Parameters
    ----------
    map : (*shape) tensor
        Parameter map
    vx : float or sequence[float]
        Voxel size
    rls : (*shape) tensor, optional
        Weights from the reweighted least squares scheme
    lam : float, default=1
        Regularisation factor
    do_grad : bool, default=True
        Return both the criterion and gradient

    Returns
    -------
    reg : () tensor
        Regularisation term
    grad : (*shape) tensor
        Gradient with respect to the parameter map

    """

    grad_fwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='b')
    if rls is not None:
        grad_fwd *= rls[..., None]
        grad_bwd *= rls[..., None]
    grad_fwd = spatial.div(grad_fwd, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.div(grad_bwd, dim=[0, 1, 2], voxel_size=vx, side='b')

    grad = grad_fwd
    grad += grad_bwd
    grad *= lam / 2.   # average across directions (3) and side (2)

    if do_grad:
        reg = (map * grad).sum(dtype=torch.double)
        return 0.5 * reg, grad
    else:
        grad *= map
        return 0.5 * grad.sum(dtype=torch.double)


def _nonlin_solve(hess, grad, rls, lam, vx, opt):
    """Solve the regularized linear system

    Parameters
    ----------
    hess : (P*(P+1)//2, *shape) tensor
    grad : (P, *shape) tensor
    rls : (P, *shape) tensor_like
    lam : (P,) sequence[float]
    vx : (D,) sequence[float]
    opt : Options

    Returns
    -------
    delta : (P, *shape) tensor

    """

    def hess_fn(x):
        result = hessian_sym_matmul(hess, x)
        if not opt.penalty.norm:
            return result
        for i, (map, weight, l) in enumerate(zip(x, rls, lam)):
            if not l:
                continue
            _, res1 = _nonlin_reg(map, vx, weight, l)
            result[i] += res1
        return result

    # The Hessian is A = H + L, where H corresponds to the data term
    # and L to the regularizer. Note that, L = D'WD where D is the
    # gradient operator, D' the divergence and W a diagonal matrix
    # that contains the RLS weights.
    # We use (H + diag(|D'D|w)) as a preconditioner because it is 
    # easy to invert and majorises the true Hessian.
    hessp = hess.clone()
    smo = torch.as_tensor(vx).square().reciprocal().sum().item()
    for i, (weight, l) in enumerate(zip(rls, lam)):
        hessp[i] += l * (rls_maj(weight, vx) if weight is not None else 4*smo)

    def precond(x):
        return hessian_sym_solve(hessp, x)

    result = core.optim.cg(hess_fn, grad, precond=precond,
                           verbose=(opt.verbose > 1), stop='norm',
                           max_iter=opt.optim.max_iter_cg,
                           tolerance=opt.optim.tolerance_cg)
    return result


def _nonlin_rls(maps, lam=1., norm='jtv'):
    """Update the (L1) weights.

    Parameters
    ----------
    map : (P, *shape) ParameterMaps
        Parameter map
    lam : float or (P,) sequence[float], default=1
        Regularisation factor
    norm : {'tv', 'jtv'}, default='jtv'

    Returns
    -------
    rls : ([P], *shape) tensor
        Weights from the reweighted least squares scheme
    """

    if norm not in ('tv', 'jtv', '__internal__'):
        return None

    if isinstance(maps, ParameterMap):
        # single map
        # this should only be an internal call
        # -> we return the squared gradient map
        assert norm == '__internal__'
        vx = spatial.voxel_size(maps.affine)
        grad_fwd = spatial.diff(maps.fdata(), dim=[0, 1, 2], voxel_size=vx, side='f')
        grad_bwd = spatial.diff(maps.fdata(), dim=[0, 1, 2], voxel_size=vx, side='b')

        grad = grad_fwd.square_().sum(-1)
        grad += grad_bwd.square_().sum(-1)
        grad *= lam / 2.   # average across sides (2)
        return grad

    # multiple maps

    if norm == 'tv':
        rls = []
        for map, l in zip(maps, lam):
            rls1 = _nonlin_rls(map, l, '__internal__')
            rls1 = rls1.sqrt_()
            rls.append(rls1)
        return torch.stack(rls, dim=0)
    else:
        assert norm == 'jtv'
        rls = 0
        for map, l in zip(maps, lam):
            rls += _nonlin_rls(map, l, '__internal__')
        rls = rls.sqrt_()

    return rls

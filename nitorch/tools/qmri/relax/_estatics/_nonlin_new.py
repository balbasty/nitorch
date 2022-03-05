import torch
from nitorch import core, spatial
from ._options import ESTATICSOptions
from ._preproc import preproc, postproc
from ._utils import (hessian_loaddiag_, hessian_matmul, hessian_solve,
                     smart_grid, smart_pull, smart_push, smart_grad, )
from nitorch.core.math import besseli, besseli_ratio
from nitorch.tools.qmri.param import ParameterMap, SVFDeformation, DenseDeformation
from typing import Optional


def nonlin(data, opt=None):
    """Fit the ESTATICS model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
        Observed GRE data.
    opt : Options, optional
        Algorithm options.

    Returns
    -------
    intecepts : sequence[GradientEcho]
        Echo series extrapolated to TE=0
    decay : estatics.ParameterMap
        R2* decay map
    distortions : sequence[ParameterizedDeformation], if opt.distortion.enable
        B0-induced distortion fields

    """
    model = _ESTATICS_nonlin(opt)
    return model.fit(data)


class _ESTATICS_nonlin:

    IMAGE_BOUND = 'dft'
    DIST_BOUND = 'dct2'

    def __init__(self, opt):
        self.opt = ESTATICSOptions().update(opt).cleanup_()
        self.data = None
        self.maps = None
        self.dist = None
        self.rls = None
        self.lam = None
        self.shape0 = None
        self.affine0 = None
        self.lam_scale = 1
        self.last_step = ''

    @property
    def numel(self):
        return sum(core.py.prod(dat.shape) for dat in self.data)

    @property
    def shape(self):
        return self.maps.decay.shape

    @property
    def affine(self):
        return self.maps.affine

    @property
    def voxel_size(self):
        return spatial.voxel_size(self.affine)

    @property
    def nb_contrasts(self):
        return len(self.maps) - 1

    @property
    def lam_dist(self):
        return dict(factor=self.opt.distortion.factor,
                    absolute=self.opt.distortion.absolute,
                    membrane=self.opt.distortion.membrane,
                    bending=self.opt.distortion.bending)

    @property
    def backend(self):
        return dict(dtype=self.opt.backend.dtype,
                    device=self.opt.backend.device)

    def iter_rls(self):
        if self.rls is None:
            for _ in range(self.nb_contrasts + 1):
                yield None
        elif self.rls.dim() == 3:
            for _ in range(self.nb_contrasts + 1):
                yield self.rls
        else:
            assert self.rls.dim() == 4
            for rls1 in self.rls:
                yield rls1

    def fit(self, data):

        # --- be polite ------------------------------------------------
        if self.opt.verbose > 0:
            print(f'Fitting a (multi) exponential decay model with '
                  f'{len(data)} contrasts. Echo times:')
            for i, contrast in enumerate(data):
                print(f'    - contrast {i:2d}: [' + ', '.join(
                    [f'{te * 1e3:.1f}' for te in contrast.te]) + '] ms')

        # --- estimate noise / register / initialize maps --------------
        self.data, self.maps, self.dist = preproc(data, self.opt)
        self.affine0 = self.maps.affine
        self.shape0 = self.maps.decay.shape

        # --- prepare regularization factor ----------------------------
        # -> we want lam = [*lam_intercepts, lam_decay]
        *lam, lam_decay = self.opt.regularization.factor
        lam = core.py.ensure_list(lam, self.nb_contrasts)
        lam.append(lam_decay)
        if not any(lam):
            self.opt.regularization.norm = ''
        self.lam = lam

        # --- initialize weights (RLS) ---------------------------------
        self.rls = None
        if self.opt.regularization.norm.endswith('tv'):
            rls_shape = self.shape
            if self.opt.regularization.norm == 'tv':
                rls_shape = (len(self.maps), *rls_shape)
            self.rls = ParameterMap(rls_shape, fill=1, **self.backend).volume

        # --- initialize nb of iterations ------------------------------
        if not self.opt.regularization.norm.endswith('tv'):
            # no reweighting -> do more gauss-newton updates instead
            self.opt.optim.max_iter_gn *= self.opt.optim.max_iter_rls
            self.opt.optim.max_iter_rls = 1

        # --- be polite (bis) ------------------------------------------
        self.print_opt()
        self.print_header()

        # --- main loop ------------------------------------------------
        self.loop()

        # --- prepare output -----------------------------------------------
        out = postproc(self.maps, self.data)
        if self.opt.distortion.enable:
            out = (*out, self.dist)
        return out

    def print_opt(self):
        if self.opt.verbose <= 1:
            return
        if self.opt.regularization.norm:
            print('Regularization:')
            print(f'    - type:           {self.opt.regularization.norm.upper()}')
            print(f'    - log intercepts: [' + ', '.join(
                [f'{i:.3g}' for i in self.lam[:-1]]) + ']')
            print(f'    - decay:          {self.lam[-1]:.3g}')
        else:
            print('Without regularization')

        if self.opt.distortion.enable:
            print('Distortion correction:')
            print(f'    - model:          {self.opt.distortion.model.lower()}')
            print(
                f'    - absolute:       {self.opt.distortion.absolute * self.opt.distortion.factor}')
            print(
                f'    - membrane:       {self.opt.distortion.membrane * self.opt.distortion.factor}')
            print(
                f'    - bending:        {self.opt.distortion.bending * self.opt.distortion.factor}')
            print(f'    - te_scaling:     {self.opt.distortion.te_scaling or "no"}')

        else:
            print('Without distortion correction')

        print('Optimization:')
        if self.opt.regularization.norm.endswith('tv'):
            print(f'    - IRLS iterations: {self.opt.optim.max_iter_rls}'
                  f' (tolerance: {self.opt.optim.tolerance_rls})')
        print(f'    - GN iterations:   {self.opt.optim.max_iter_gn}'
              f' (tolerance: {self.opt.optim.tolerance_gn})')
        print(f'    - FMG cycles:      2')
        print(f'    - CG iterations:   {self.opt.optim.max_iter_cg}'
              f' (tolerance: {self.opt.optim.tolerance_cg})')

    def loop(self):
        """Nested optimization loops"""

        rls = self.rls.reciprocal().sum() if self.rls is not None else 0
        self.nll = dict(obs=float('inf'), obs_prev=float('inf'),
                        reg=0, reg_prev=0, vreg=0, vreg_prev=0,
                        rls=rls, rls_prev=rls, all=[])
        nll = float('inf')

        nll_scl = self.numel * len(self.data)

        # --- Multi-Resolution loop ------------------------------------
        for self.level in range(self.opt.optim.nb_levels, 0, -1):
            if self.opt.optim.nb_levels > 1:
                self.resize()

            # --- RLS loop ---------------------------------------------
            #   > max_iter_rls == 1 if regularization is not (J)TV
            for self.n_iter_rls in range(1, self.opt.optim.max_iter_rls + 1):

                # --- Gauss-Newton (prm + dist) ------------------------
                for self.n_iter_gn in range(1, self.opt.optim.max_iter_gn + 1):
                    nll0_gn = nll
                    self.n_iter_dist = 0
                    self.n_iter_prm = 0

                    # --- Gauss-Newton (prm) ---------------------------
                    max_iter_prm = self.opt.optim.max_iter_prm
                    if self.n_iter_gn == 1 and self.n_iter_rls == 1:
                        max_iter_prm = max_iter_prm * 2
                    for self.n_iter_prm in range(1, max_iter_prm + 1):
                        nll0_prm = nll
                        nll = self.update_prm()
                        if (self.n_iter_gn > 1 and
                                (nll0_prm - nll) < self.opt.optim.tolerance * nll_scl):
                            break

                        # ----------------------------------------------
                        # this is where we should check for RLS (== global) gain
                        if (self.n_iter_prm == 0 and self.n_iter_gn == 0 and
                                self.n_iter_rls > 1 and
                                (nll0_rls - nll) < self.opt.optim.tolerance * nll_scl * 2):
                            return
                        # ----------------------------------------------

                    # --- Gauss-Newton (dist) --------------------------
                    for self.n_iter_dist in range(1, self.opt.optim.max_iter_dist + 1):
                        nll0_dist = nll
                        nll = self.update_dist()
                        if (self.n_iter_dist > 1 and
                                (nll0_dist - nll) < self.opt.optim.tolerance * nll_scl):
                            break

                    if (self.n_iter_gn > 1 and
                            (nll0_gn - nll) < self.opt.optim.tolerance * nll_scl):
                        break

                nll0_rls = nll
                nll = self.update_rls()

    # ------------------------------------------------------------------
    #                       UPDATE PARAMETERS MAPS
    # ------------------------------------------------------------------

    def momentum_prm(self, dat):
        """Momentum of the parameter maps"""
        return spatial.regulariser(dat, weights=self.rls, dim=3,
                                   **self.lam_prm, voxel_size=self.voxel_size)

    def solve_prm(self, hess, grad):
        """Solve Newton step"""
        hess = check_nans_(hess, warn='hessian')
        hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
        delta = spatial.solve_field_fmg(hess, grad, self.rls, **self.lam_prm,
                                        voxel_size=self.voxel_size)
        delta = check_nans_(delta, warn='delta')
        return delta

    def update_prm(self):
        """Update parameter maps (log-intercept and decay)"""

        nmaps = len(self.data)
        grad = torch.zeros((nmaps + 1, *self.shape), **self.backend)
        hess = torch.zeros((nmaps * 2 + 1, *self.shape), **self.backend)

        # --- loop over contrasts --------------------------------------
        iterator = zip(self.data, self.maps.intercepts, self.dist)
        nll = 0
        for i, (contrast, intercept, distortion) in enumerate(iterator):
            # compute gradient
            nll1, g1, h1 = self.derivatives_prm(
                contrast, distortion, intercept, self.maps.decay)
            # increment
            gind = [i, -1]
            grad[gind] += g1
            hind = [i, len(grad)-1, len(grad)+i]
            hess[hind] += h1
            nll += nll1
            del g1, h1

        # --- regularization ------------------------------------------
        reg = 0
        if self.opt.regularization.norm:
            g1 = self.momentum_prm(self.data.volume)
            reg = 0.5 * dot(g1, self.data.volume)
            grad += g1
            del g1

        # --- gauss-newton ---------------------------------------------
        # Computing the GN step involves solving H\g
        deltas = self.solve_prm(hess, grad)

        # No need for a line search (hopefully)
        for map, delta in zip(self.maps, deltas):
            map.volume -= delta
            if map.min is not None or map.max is not None:
                map.volume.clamp_(map.min, map.max)

        # --- track general improvement --------------------------------
        self.nll['obs_prev'] = self.nll['obs']
        self.nll['obs'] = nll
        self.nll['reg_prev'] = self.nll['reg']
        self.nll['reg'] = reg
        nll = self.print_nll()
        self.last_step = 'prm'
        return nll

    # ------------------------------------------------------------------
    #                       UPDATE DISTORTION MAPS
    # ------------------------------------------------------------------

    def momentum_dist(self, dat, vx, readout):
        """Momentum of the distortion maps"""
        lam = dict(self.lam_dist)
        lam['factor'] = lam['factor'] * (vx[readout] ** 2)
        return spatial.regulariser(dat[None], **self.lam_dist, dim=3,
                                   bound=self.DIST_BOUND, voxel_size=vx)[0]

    def solve_dist(self, hess, grad, vx, readout):
        """Solve Newton step"""
        hess = check_nans_(hess, warn='hessian')
        hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
        lam = dict(self.lam_dist)
        lam['factor'] = lam['factor'] * (vx[readout] ** 2)
        delta = spatial.solve_field_fmg(hess, grad, **self.lam_dist, dim=3,
                                        bound=self.DIST_BOUND, voxel_size=vx)
        delta = check_nans_(delta, warn='delta')
        return delta

    def update_dist(self):
        """Update distortions"""

        nll = 0
        reg = 0
        # --- loop over contrasts --------------------------------------
        iterator = zip(self.data, self.maps.intercepts, self.dist)
        for i, (contrast, intercept, distortion) in enumerate(iterator):

            momentum = lambda dat: self.momentum_dist(
                dat, distortion.voxel_size, contrast.readout)
            solve = lambda h, g: self.solve_dist(
                h, g, distortion.voxel_size, contrast.readout)
            vol = distortion.volume

            # --- likelihood -------------------------------------------
            nll1, g, h = self.derivatives_dist(
                contrast, distortion, intercept, self.maps.decay)

            # --- regularization ---------------------------------------
            g1 = momentum(vol)
            reg1 = 0.5 * dot(g1, vol)
            g += g1
            del g1

            # --- gauss-newton -----------------------------------------
            delta = solve(h, g)
            del g, h

            # --- line search ------------------------------------------
            armijo, armijo_prev = 1, 0
            dd = momentum(delta)
            dv = dot(dd, vol)
            dd = dot(dd, delta)
            success = False
            for n_ls in range(12):
                vol.sub_(delta, alpha=(armijo - armijo_prev))
                armijo_prev = armijo
                delta_reg1 = 0.5 * armijo * (armijo * dd - 2 * dv)
                new_nll1 = self.derivatives_dist(
                    contrast, distortion, intercept, self.maps.decay,
                    do_grad=False)
                if new_nll1 + delta_reg1 <= nll1:
                    success = True
                    break
                armijo /= 2
            if not success:
                vol.add_(delta, alpha=armijo_prev)
                new_nll1 = nll1
                delta_reg1 = 0
            nll += new_nll1
            reg += reg1 + delta_reg1

            del delta

        # --- track general improvement --------------------------------
        self.nll['obs_prev'] = self.nll['obs']
        self.nll['obs'] = nll
        self.nll['vreg_prev'] = self.nll['vreg']
        self.nll['vreg'] = reg
        nll = self.print_nll()
        self.last_step = 'dist'
        return nll

    # ------------------------------------------------------------------
    #                        UPDATE WEIGHT MAP
    # ------------------------------------------------------------------

    def update_rls(self):
        if self.opt.regularization.norm not in ('tv', 'jtv'):
            return sum(self.nll[k] for k in ['obs', 'reg', 'vreg', 'rls'])
        rls, sumrls = update_rls(self.maps, self.lam, self.opt.regularization.norm)
        self.nll['rls_prev'] = self.nll['rls']
        self.nll['rls'] = 0.5 * sumrls
        self.last_step = 'rls'
        return sum(self.nll[k] for k in ['obs', 'reg', 'vreg', 'rls'])

    def print_header(self):
        if self.opt.verbose <= 0:
            return
        pstr = ''
        if self.opt.optim.max_iter_rls > 1:
            pstr += f'{"rls":^3s} | '
        if self.opt.optim.max_iter_gn > 1:
            pstr += f'{"gn":^3s} | '
        if self.opt.optim.max_iter_prm > 1 or self.opt.optim.max_iter_dist > 1:
            pstr += f'{"sub":^3s} | '
        pstr += f'{"step":^4s} | '
        pstr += f'{"fit":^12s} '
        if self.opt.regularization.norm:
            pstr += f'+ {"reg":^12s} '
        if self.opt.regularization.norm.endswith('tv'):
            pstr += f'+ {"rls":^12s} '
        if self.opt.distortion.enable:
            pstr += f'+ {"dist":^12s} '
        pstr += f'= {"crit":^12s}'
        print('\n' + pstr)
        print('-' * len(pstr))

    def print_nll(self):
        nll = sum(self.nll[k] for k in ['obs', 'reg', 'vreg', 'rls'])
        self.nll['all'].append(nll)
        if self.opt.verbose <= 0:
            return nll
        obs = self.nll['obs']
        reg = self.nll['reg']
        vreg = self.nll['vreg']
        rls = self.nll['rls']
        obs0 = self.nll['obs_prev']
        reg0 = self.nll['reg_prev']
        vreg0 = self.nll['vreg_prev']
        rls0 = self.nll['rls_prev']

        if self.last_step == 'rls':
            nll = obs0 + reg + rls + vreg
            if reg + rls <= reg0 + rls0:
                evol = '<='
            else:
                evol = '>'
        elif self.last_step == 'prm':
            nll = obs0 + reg + rls + vreg
            if obs + reg <= obs0 + reg0:
                evol = '<='
            else:
                evol = '>'
        elif self.last_step == 'dist':
            if obs + vreg <= obs0 + vreg0:
                evol = '<='
            else:
                evol = '>'
        else:
            evol = ''

        nll = obs + reg + vreg + rls
        pstr = ''
        if self.opt.optim.max_iter_rls > 1:
            pstr += f'{self.n_iter_rls:3d} | '
        if self.opt.optim.max_iter_gn > 1:
            pstr += f'{self.n_iter_gn:3d} | '
        if self.opt.optim.max_iter_prm > 1 or self.opt.optim.max_iter_dist > 1:
            if self.last_step == 'rls':
                pstr += f'{"-"*3} | '
            else:
                pstr += f'{self.n_iter_prm+self.n_iter_dist:3d} | '
        pstr += f'{self.last_step:4s} | '
        pstr += f'{obs:12.6g} '
        if self.opt.regularization.norm:
            pstr += f'+ {reg:12.6g} '
        if self.opt.regularization.norm.endswith('tv'):
            pstr += f'+ {rls:12.6g} '
        if self.opt.distortion.enable:
            pstr += f'+ {vreg:12.6g} '
        pstr += f'= {nll:12.6g} | '
        pstr += f'{evol}'
        print(pstr)
        self.show_maps()
        return nll

    def resize(self):
        affine, shape = spatial.affine_resize(
            self.affine0, self.shape0, 1 / (2 ** (self.level - 1)))

        scl0 = spatial.voxel_size(self.affine0).prod()
        scl = spatial.voxel_size(affine).prod() / scl0
        self.lam_scale = scl

        for map in self.maps:
            map.volume = spatial.resize(map.volume[None, None, ...],
                                        shape=shape)[0, 0]
            map.affine = affine
        self.maps.affine = affine
        if self.rls is not None:
            if self.rls.dim() == len(shape):
                self.rls = spatial.resize(self.rls[None, None], hape=shape)[0, 0]
            else:
                self.rls = spatial.resize(self.rls[None], shape=shape)[0]
            self.nll['rls'] = self.rls.reciprocal().sum(dtype=torch.double)

    def show_maps(self):
        if not self.opt.plot:
            return
        import matplotlib.pyplot as plt
        has_dist = any([d is not None for d in self.dist])
        ncol = max(len(self.maps), len(self.dist))
        for i, map in enumerate(self.maps):
            plt.subplot(2 + has_dist, ncol, i+1)
            vol = map.volume[:, :, map.shape[-1]//2]
            if i < len(self.maps) - 1:
                vol = vol.exp()
            plt.imshow(vol.cpu())
            plt.axis('off')
            if i < len(self.maps) - 1:
                plt.title('TE=0')
            else:
                plt.title('R2*')
            plt.colorbar()
        if has_dist:
            for i, (dat, dst) in enumerate(zip(self.data, self.dist)):
                if dst is None:
                    continue
                readout = dat.readout
                vol = dst.volume
                plt.subplot(2 + has_dist, ncol, i+1+ncol)
                vol = vol[:, :, dst.shape[-2]//2]
                if readout is not None:
                    vol = vol[..., readout]
                else:
                    vol = vol.square().sum(-1).sqrt()
                plt.imshow(vol.cpu())
                plt.axis('off')
                plt.colorbar()
        plt.subplot(2+has_dist, 1, 2+has_dist)
        plt.plot(self.nll['all'])
        plt.show()

@torch.jit.script
def recon_fit(inter, slope, te: float):
    """Reconstruct a single echo"""
    return inter.add(slope, alpha=-te).exp()


@torch.jit.script
def ssq(x):
    """Sum of squares"""
    return (x * x).sum(dtype=torch.double)


@torch.jit.script
def dot(x, y):
    """Dot product"""
    return (x * y).sum(dtype=torch.double)


def get_mask_missing(dat, fit):
    """Mask of voxels excluded from the objective"""
    return ~(torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0))


def mask_nan_(x, value: float = 0.):
    """Mask out all non-finite values"""
    return x.masked_fill_(torch.isfinite(x).bitwise_not(), value)


def check_nans_(x, warn: Optional[str] = None, value: float = 0):
    """Mask out all non-finite values + warn if `warn is not None`"""
    msk = torch.isfinite(x)
    if warn is not None:
        if ~(msk.all()):
            print(f'WARNING: NaNs in {warn}')
    x.masked_fill_(msk.bitwise_not(), value)
    return x


def nll_chi(dat, fit, msk, lam, df, return_residuals=True):
    """Negative log-likelihood of the noncentral Chi distribution

    Parameters
    ----------
    dat : tensor
        Observed data (should be zero where not observed)
    fit : tensor
        Signal fit (should be zero where not observed)
    msk : tensor
        Mask of observed values
    lam : float
        Noise precision
    df : float
        Degrees of freedom
    return_residuals : bool
        Return residuals (gradient) on top of nll

    Returns
    -------
    nll : () tensor
        Negative log-likelihood
    res : tensor, if `return_residuals`
        Residuals

    """
    z = (dat * fit * lam).clamp_min_(1e-32)
    xi = besseli_ratio(df / 2 - 1, z)
    logbes = besseli(df / 2 - 1, z, 'log')
    logbes = logbes[msk].sum(dtype=torch.double)

    # chi log-likelihood
    fitm = fit[msk]
    sumlogfit = fitm.clamp_min(1e-32).log_().sum(dtype=torch.double)
    sumfit2 = fitm.flatten().dot(fitm.flatten())
    del fitm
    datm = dat[msk]
    sumlogdat = datm.clamp_min(1e-32).log_().sum(dtype=torch.double)
    sumdat2 = datm.flatten().dot(datm.flatten())
    del datm

    crit = (df / 2 - 1) * sumlogfit - (df / 2) * sumlogdat - logbes
    crit += 0.5 * lam * (sumfit2 + sumdat2)
    if not return_residuals:
        return crit
    res = dat.mul_(xi).neg_().add_(fit)
    return crit, res


def nll_gauss(dat, fit, msk, lam, return_residuals=True):
    """Negative log-likelihood of the noncentral Chi distribution

    Parameters
    ----------
    dat : tensor
        Observed data (should be zero where not observed)
    fit : tensor
        Signal fit (should be zero where not observed)
    msk : tensor
        Mask of observed values
    lam : float
        Noise precision
    nu : float
        Degrees of freedom
    return_residuals : bool
        Return residuals (gradient) on top of nll

    Returns
    -------
    nll : () tensor
        Negative log-likelihood
    res : tensor, if `return_residuals`
        Residuals

    """
    res = dat.neg_().add_(fit)
    crit = 0.5 * lam * ssq(res[msk])
    return (crit, res) if return_residuals else crit


# if core.utils.torch_version('>', (1, 4)):
# For some reason, the output of torch.isfinite is not understood
# as a tensor by TS. I am disabling TS for these functions until
# I find a better solution.
# get_mask_missing = torch.jit.script(get_mask_missing)
# mask_nan_ = torch.jit.script(mask_nan_)
# check_nans_ = torch.jit.script(check_nans_)


def derivatives_parameters(contrast, distortion, intercept, decay, opt,
                           do_grad=True):
    """Compute the gradient and Hessian of the parameter maps with
    respect to one contrast.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    distortion : ParameterizedDeformation
        A model of distortions caused by B0 inhomogeneities.
    intercept : (*recon_shape) ParameterMap
        Log-intercept of the contrast
    decay : (*recon_shape) ParameterMap
        Exponential decay
    opt : Options
    do_grad : bool, default=True

    Returns
    -------
    crit : () tensor
        Log-likelihood
    grad : (2, *recon_shape) tensor, if `do_grad`
        Gradient with respect to:
            [0] intercept
            [1] decay
    hess : (3, *recon_shape) tensor, if `do_grad`
        Hessian with respect to:
            [0] intercept ** 2
            [1] decay ** 2
            [2] intercept * decay

    """

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    obs_shape = contrast.volume.shape[1:]
    recon_shape = intercept.volume.shape
    aff = core.linalg.lmdiv(intercept.affine, contrast.affine)
    aff = aff.to(**backend)
    lam = 1 / contrast.noise
    df = contrast.dof
    chi = opt.likelihood[0].lower() == 'c'

    # pull parameter maps to observed space
    grid = smart_grid(aff, obs_shape, recon_shape)
    inter = smart_pull(intercept.fdata(**backend), grid)
    slope = smart_pull(decay.fdata(**backend), grid)
    if distortion and opt.distortion.te_scaling != 'pre':
        grid_up, grid_down = distortion.exp2(
            add_identity=not opt.distortion.te_scaling)
    else:
        grid_up = grid_down = None

    crit = 0
    grad = torch.zeros((2,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((3,) + obs_shape, **backend) if do_grad else None

    te0 = 0
    for e, echo in enumerate(contrast):

        te = echo.te
        te0 = te0 or te
        blip = echo.blip or (2 * (e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down
        if distortion:
            vscl = te / te0
            if opt.distortion.te_scaling == 'pre':
                vexp = distortion.iexp if blip < 0 else distortion.exp
                grid_blip = vexp(add_identity=True, alpha=vscl)
            elif opt.distortion.te_scaling == 'post':
                grid_blip = spatial.add_identity_grid_(vscl * grid_blip)

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)
        fit = recon_fit(inter, slope, te)
        pull_fit = smart_pull(fit, grid_blip, bound='dft', extrapolate=True)
        msk = get_mask_missing(dat, pull_fit)
        dat.masked_fill_(msk, 0)
        pull_fit.masked_fill_(msk, 0)
        msk = msk.bitwise_not_()

        if chi:
            crit1, res = nll_chi(dat, pull_fit, msk, lam, df)
        else:
            crit1, res = nll_gauss(dat, pull_fit, msk, lam)
        del dat, pull_fit
        crit += crit1

        if do_grad:
            msk = msk.to(fit.dtype)
            if grid_blip is not None:
                res0 = res
                res = smart_push(res0, grid_blip, bound='dft',
                                 extrapolate=True)
                abs_res = smart_push(res0.abs_(), grid_blip, bound='dft',
                                     extrapolate=True)
                abs_res.mul_(fit)
                msk = smart_push(msk, grid_blip, bound='dft', extrapolate=True)
                del res0

            # ----------------------------------------------------------
            # compute gradient and (majorised) Hessian in observed space
            #
            #   grad[inter]       =           lam * fit * res
            #   grad[decay]       =     -te * lam * fit * res
            #   hess[inter**2]    =           lam * fit * (fit + abs(res))
            #   hess[decay**2]    = (te*te) * lam * fit * (fit + abs(res))
            #   hess[inter*decay] =     -te * lam * fit * fit
            #
            # I tried to put that into an "accumulation" function but it
            # does super weird stuff, so I keep it in the main loop. I am
            # saving a few allocations here so I think it's faster than
            # torchscript.
            # ----------------------------------------------------------

            res.mul_(fit)
            grad[0].add_(res, alpha=lam)
            grad[1].add_(res, alpha=-te * lam)
            if grid_blip is None:
                abs_res = res.abs_()
            fit2 = fit.mul_(fit).mul_(msk)
            del msk
            hess[2].add_(fit2, alpha=-te * lam)
            fit2.add_(abs_res)
            hess[0].add_(fit2, alpha=lam)
            hess[1].add_(fit2, alpha=lam * (te * te))

            del res, fit, abs_res, fit2

    if not do_grad:
        return crit

    mask_nan_(grad)
    mask_nan_(hess[:-1], 1e-3)  # diagonal
    mask_nan_(hess[-1])  # off-diagonal

    # push gradient and Hessian to recon space
    grad = smart_push(grad, grid, recon_shape)
    hess = smart_push(hess, grid, recon_shape)
    return crit, grad, hess


def derivatives_distortion(contrast, distortion, intercept, decay, opt,
                           do_grad=True):
    """Compute the gradient and Hessian of the distortion field.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    distortion : ParameterizedDeformation
        A model of distortions caused by B0 inhomogeneities.
    intercept : (*recon_shape) ParameterMap
        Log-intercept of the contrast
    decay : (*recon_shape) ParameterMap
        Exponential decay
    opt : Options

    Returns
    -------
    crit : () tensor
        Log-likelihood
    grad : (*shape, 3) tensor
    hess : (*shape, 6) tensor

    """

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    obs_shape = contrast.volume.shape[1:]
    recon_shape = intercept.volume.shape
    aff = core.linalg.lmdiv(intercept.affine, contrast.affine)
    aff = aff.to(**backend)
    lam = 1 / contrast.noise
    df = contrast.dof
    chi = opt.likelihood[0].lower() == 'c'

    # pull parameter maps to observed space
    grid = smart_grid(aff, obs_shape, recon_shape)
    inter = smart_pull(intercept.fdata(**backend), grid)
    slope = smart_pull(decay.fdata(**backend), grid)
    readout = contrast.readout
    if opt.distortion.te_scaling != 'pre':
        grid_up, grid_down = distortion.exp2(
            add_identity=not opt.distortion.te_scaling)
    else:
        grid_up = grid_down = None

    crit = 0
    grad = torch.zeros(obs_shape + (3,), **backend) if do_grad else None
    hess = torch.zeros(obs_shape + (6,), **backend) if do_grad else None

    te0 = 0
    for e, echo in enumerate(contrast):

        te = echo.te
        te0 = te0 or te
        blip = echo.blip or (2 * (e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down
        vscl = te / te0
        if opt.distortion.te_scaling == 'pre':
            vexp = distortion.iexp if blip < 0 else distortion.exp
            grid_blip = vexp(add_identity=True, alpha=vscl)
        elif opt.distortion.te_scaling:
            grid_blip = spatial.add_identity_grid_(vscl * grid_blip)

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)  # observed
        fit = recon_fit(inter, slope, te)  # fitted
        if do_grad and isinstance(distortion, DenseDeformation):
            # D(fit) o phi
            gfit = smart_grad(fit, grid_blip, bound='dft', extrapolate=True)
        fit = smart_pull(fit, grid_blip, bound='dft', extrapolate=True)
        msk = get_mask_missing(dat, fit)  # mask of missing values
        if do_grad and isinstance(distortion, SVFDeformation):
            # D(fit o phi)
            gfit = spatial.diff(fit, bound='dft', dim=[-3, -2, -1])
            gfit.masked_fill_(msk.unsqueeze(-1), 0)
        dat.masked_fill_(msk, 0)
        fit.masked_fill_(msk, 0)
        msk = msk.bitwise_not_()

        if chi:
            crit1, res = nll_chi(dat, fit, msk, lam, df)
        else:
            crit1, res = nll_gauss(dat, fit, msk, lam)
        del dat, fit, msk
        crit += crit1

        if do_grad:
            g1 = res.unsqueeze(-1).mul(gfit)
            h1 = torch.zeros_like(hess)
            if readout is None:
                h1[..., :3] = gfit.square()
                h1[..., 3] = gfit[..., 0] * gfit[..., 1]
                h1[..., 4] = gfit[..., 0] * gfit[..., 2]
                h1[..., 5] = gfit[..., 1] * gfit[..., 2]
            else:
                h1[..., readout] = gfit[..., readout].square()

            # propagate backward
            if isinstance(distortion, SVFDeformation):
                vel = distortion.volume
                if opt.distortion.te_scaling == 'pre':
                    vel = ((-vscl) * vel) if blip < 0 else (vscl * vel)
                elif blip < 0:
                    vel = -vel
                g1, h1 = spatial.exp_backward(vel, g1, h1,
                                              steps=distortion.steps)

            alpha_g = alpha_h = lam
            alpha_g = alpha_g * blip
            if opt.distortion.te_scaling == 'pre':
                alpha_g = alpha_g * vscl
                alpha_h = alpha_h * (vscl * vscl)
            grad.add_(g1, alpha=alpha_g)
            hess.add_(h1, alpha=alpha_h)

    if not do_grad:
        return crit

    if readout is None:
        mask_nan_(grad)
        mask_nan_(hess[:-3], 1e-3)  # diagonal
        mask_nan_(hess[-3:])  # off-diagonal
    else:
        grad = grad[..., readout]
        hess = hess[..., readout]
        mask_nan_(grad)
        mask_nan_(hess)

    return crit, grad, hess


def solve_parameters(hess, grad, rls, lam, vx, opt):
    """Solve the regularized linear system

    Parameters
    ----------
    hess : (2*P+1, *shape) tensor
    grad : (P+1, *shape) tensor
    rls : ([P+1], *shape) tensor or None
    lam : (P,) sequence[float]
    vx : (D,) sequence[float]
    opt : Options

    Returns
    -------
    delta : (P+1, *shape) tensor

    """
    # The ESTATICS Hessian has a very particular form (intercepts do not
    # have cross elements). We therefore need to tell the solver how to operate
    # on it.

    def matvec(m, x):
        m = m.transpose(-1, -4)
        x = x.transpose(-1, -4)
        return hessian_matmul(m, x).transpose(-4, -1)

    def matsolve(m, x):
        m = m.transpose(-1, -4)
        x = x.transpose(-1, -4)
        return hessian_solve(m, x).transpose(-4, -1)

    def matdiag(m, d):
        return m[..., ::2]

    return spatial.solve_field_fmg(hess, grad, rls, factor=lam, membrane=1,
                                   voxel_size=vx, verbose=opt.verbose - 1,
                                   nb_iter=opt.optim.max_iter_cg,
                                   tolerance=opt.optim.tolerance_cg,
                                   matvec=matvec, matsolve=matsolve, matdiag=matdiag)

def update_rls(maps, lam=1., norm='jtv'):
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
        (Inverted) Weights from the reweighted least squares scheme
    sumrls : () tensor
        Sum of the (non-inverted) weights
    """
    # vx = spatial.voxel_size(maps.affine)
    # return spatial.membrane_weights(maps.volume, dim=3, factor=lam,
    #                                 joint=(norm == 'jtv'), voxel_size=vx,
    #                                 return_sum=True)

    # ----------------------------------------------------------------
    # This is the old version of the code, before it got refactored
    # and generalized in the `spatial` module.
    # ----------------------------------------------------------------

    if norm not in ('tv', 'jtv', '__internal__'):
        return None

    if isinstance(maps, ParameterMap):
        # single map
        # this should only be an internal call
        # -> we return the squared gradient map
        assert norm == '__internal__'
        vx = spatial.voxel_size(maps.affine)
        grad_fwd = spatial.diff(maps.volume, dim=[0, 1, 2], voxel_size=vx, side='f')
        grad_bwd = spatial.diff(maps.volume, dim=[0, 1, 2], voxel_size=vx, side='b')

        grad = grad_fwd.square_().sum(-1)
        grad += grad_bwd.square_().sum(-1)
        grad *= lam / 2   # average across side (2)
        return grad

    # multiple maps

    if norm == 'tv':
        rls = []
        for map, l in zip(maps, lam):
            rls1 = update_rls(map, l, '__internal__')
            rls1 = rls1.sqrt_()
            rls.append(rls1)
    else:
        assert norm == 'jtv'
        rls = 0
        for map, l in zip(maps, lam):
            rls += update_rls(map, l, '__internal__')
        rls = rls.sqrt_()

    sumrls = rls.sum(dtype=torch.double)
    eps = core.constants.eps(rls.dtype)
    rls = rls.clamp_min_(eps).reciprocal_()

    return rls, sumrls
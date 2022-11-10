from nitorch.core.struct import Structure
from nitorch._C.solve import c_precond
from torch import Tensor
import math
import torch
from nitorch.core.linalg import sym_solve
from nitorch.core.utils import min_intensity_step
from .relax.utils import nll_chi, nll_gauss
from ..img_statistics import estimate_noise

# ----------------------------------------------------------------------
#       SEQUENCES
# ----------------------------------------------------------------------


class Encoding(Structure):
    bandwidth: float = float('inf')    # Readout bandwidth (Hz/pixel)


class Encoding3D(Encoding):
    readout: int = -1       # Readout direction


class EncodingPlanar(Encoding):
    slice: int = -1        # Slice direction
    readout: int = -2      # Readout direction
    thickness: float = 1   # Thickness, in percentage of the voxel size


class MRI(Structure):
    dat: Tensor = None
    TR: float = 0
    SNR: float = 1
    encoding: Encoding = None
    PDindex: int = 0


class SPGR(MRI):
    FA: float = 0
    TE: float = 0
    Spoiling: float = 0


class bSSFP(MRI):
    FA: float = 0
    Phi: float = 0
    B0index: int = 0


class Tissue(Structure):
    PD = None
    R1 = None
    _R2star = None
    R2 = None
    R2prime = None
    DeltaPhi = None
    B1 = None
    SSE = None

    @property
    def R2star(self):
        if self._R2star is not None:
            return self._R2star
        elif self.R2 is not None and self.R2prime is not None:
            return self.R2 + self.R2prime
        else:
            return None

    @R2star.setter
    def R2star(self, value):
        self._R2star = value

    @property
    def T1(self):
        return 1/self.R1

    @property
    def T2(self):
        return 1/self.R2

    @property
    def T2prime(self):
        return 1/self.R2prime

    @property
    def T2star(self):
        return 1/self.R2star

    def log(self):
        kwargs = dict(
            PD=log(self.PD),
            R1=log(self.R1),
            # DeltaPhi=exp1(self.DeltaPhi),
            DeltaPhi=clone(self.DeltaPhi),
            B1=log(self.B1))
        if self._R2star is not None:
            kwargs['R2star'] = log(self.R2star)
        else:
            kwargs['R2'] = log(self.R2)
            kwargs['R2prime'] = log(self.R2prime)
        return LogTissue(**kwargs)

    def to(self, *args, **kwargs):
        for field in ('PD', 'R1', '_R2star', 'R2', 'R2prime', 'DeltaPhi', 'B1', 'SSE'):
            if torch.is_tensor(getattr(self, field)):
                setattr(self, field, getattr(self, field).to(*args, **kwargs))
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')


class LogTissue(Structure):
    PD = None
    R1 = None
    R2 = None
    _R2star = None
    R2prime = None
    DeltaPhi = None
    B1 = None

    @property
    def R2star(self):
        if self._R2star is not None:
            return self._R2star
        elif self.R2 is not None and self.R2prime is not None:
            return log(exp(self.R2) + exp(self.R2prime))
        else:
            return None

    @R2star.setter
    def R2star(self, value):
        self._R2star = value

    def exp(self):
        kwargs = dict(
            PD=exp(self.PD),
            R1=exp(self.R1),
            # DeltaPhi=exp1(self.DeltaPhi),
            DeltaPhi=clone(self.DeltaPhi),
            B1=exp(self.B1))
        if self._R2star is not None:
            kwargs['R2star'] = exp(self.R2star)
        else:
            kwargs['R2'] = exp(self.R2)
            kwargs['R2prime'] = exp(self.R2prime)
        return Tissue(**kwargs)


class Optim(Structure):
    parameters = ('PD', 'R1', 'R2', 'R2prime', 'DeltaPhi', 'B1')
    lr: float = 1
    newton: bool = True
    robust: float = 1
    levenberg: float = 1e-3
    marquardt: float = 0
    lm_update: bool = False
    max_iter: int = 100
    jtv: float = 0
    device: str = 'cpu'


def multifit(dat, opt, init=None):

    shape = dat[0].dat.shape[-3:]
    backend = dict(dtype=dat[0].dat.dtype, device=opt.device)

    for dat1 in dat:
        dat1.dat = dat1.dat.cuda()
        dat1.step = min_intensity_step(dat1.dat)
        dat1.dat = torch.rand_like(dat1.dat).mul_(dat1.step).add_(dat1.dat)
        noise, _ = estimate_noise(dat1.dat.cuda(), chi=True)
        dat1.SNR = 1/noise['sd']
        dat1.df = noise['dof']
        dat1.dat = dat1.dat.cpu().pin_memory()

    mask = 0
    for dat1 in dat:
        mask += dat1.dat.cuda().square()
    mask /= len(dat)
    mask = mask.sqrt_()
    pd0 = mask[mask > 100].mean()
    mask = mask > 50

    nb_phi = max([dat1.B0index for dat1 in dat if isinstance(dat1, bSSFP)]) + 1
    nb_pd = max([dat1.PDindex for dat1 in dat]) + 1

    fit = init or Tissue()
    fit.PD = torch.full([nb_pd, *shape], fit.PD or pd0, **backend) if not torch.is_tensor(fit.PD) else fit.PD.to(**backend)
    fit.R1 = torch.full(shape, fit.R1 or 1, **backend) if not torch.is_tensor(fit.R1) else fit.R1.to(**backend)
    fit.R2 = torch.full(shape, fit.R2 or 1/70e-3, **backend) if not torch.is_tensor(fit.R2) else fit.R2.to(**backend)
    fit.R2prime = torch.full(shape, fit.R2prime or 1/30e-3, **backend) if not torch.is_tensor(fit.R2prime) else fit.R2prime.to(**backend)
    fit.DeltaPhi = torch.full([nb_phi, *shape], fit.DeltaPhi or 0, **backend) if not torch.is_tensor(fit.DeltaPhi) else fit.DeltaPhi.to(**backend)
    fit.B1 = torch.full(shape, fit.B1 or 1, **backend) if not torch.is_tensor(fit.B1) else fit.B1.to(**backend)
    logfit = fit.log()

    K = len(opt.parameters)
    if 'PD' in opt.parameters:
        i = opt.parameters.index('PD')
        opt.parameters = ('PD',) + opt.parameters[:i] + opt.parameters[i+1:]
        K += nb_pd - 1
    if 'DeltaPhi' in opt.parameters:
        i = opt.parameters.index('DeltaPhi')
        opt.parameters = opt.parameters[:i] + opt.parameters[i+1:] + ('DeltaPhi',)
        K += nb_phi - 1
    print(K, opt.parameters)
    KK = (K * (K + 1)) // 2
    grad = torch.zeros([K, *shape], **backend)
    if opt.newton:
        hess = torch.zeros([KK, *shape], **backend)

    # utility to index into the symmetric hessian
    hind = torch.zeros([K, K], dtype=torch.long)
    cnt = K
    for k in range(K):
        hind[k, k] = k
        for kk in range(k+1, K):
            hind[k, kk] = hind[kk, k] = cnt
            cnt += 1

    loss = float('inf')
    losses = []
    for n_iter in range(opt.max_iter):
        loss_prev = loss

        # Compute derivatives
        grad.zero_()
        if opt.newton:
            hess.zero_()
        loss = 0
        for dat1 in dat:
            dat0 = dat1.dat
            dat1.dat = dat1.dat.cuda()
            lam, dof = dat1.SNR**2, dat1.df
            f, g, h = forward(dat1, fit)
            # d = f - dat1.dat
            loss1, d = nll_chi(dat1.dat, f, mask, lam, dof)
            loss += loss1
            mask1 = torch.isfinite(d)
            d.masked_fill_(~(mask & mask1), 0)
            for k, prm in enumerate(opt.parameters):
                gk = getattr(g, prm)
                if gk is None:
                    continue
                if prm == 'PD':
                    k += dat1.PDindex
                elif 'PD' in opt.parameters:
                    k += nb_pd - 1
                if prm == 'DeltaPhi':
                    k += dat1.B0index
                grad[k].addcmul_(gk, d, value=lam)
                if opt.newton:
                    hess[k].addcmul_(gk, gk, value=lam)
                    if opt.robust:
                        hk = getattr(h, prm)
                        hess[k].add_(hk.mul(d).abs_(), alpha=lam)
                    for kk, prm2 in enumerate(opt.parameters[k+1:]):
                        kk += k+1
                        if getattr(g, prm2) is not None:
                            gkk = getattr(g, prm2)
                            if prm2 == 'PD':
                                kk += dat1.PDindex
                            elif 'PD' in opt.parameters:
                                kk += nb_pd - 1
                            if prm2 == 'DeltaPhi':
                                kk += dat1.B0index
                            c = hind[k, kk]
                            hess[c].addcmul_(gk, gkk, value=lam)
            dat1.dat = dat0
        loss /= len(dat)
        losses.append(loss)

        if opt.jtv:
            raise NotImplemented

        if opt.newton:
            if opt.marquardt:
                hess[:K].add_(hess[:K].abs(), alpha=opt.marquardt)
            if opt.levenberg:
                hess[:K].add_(hess.abs().max(0).values, alpha=opt.levenberg)
            hess[:K] += 1e-5
            if opt.jtv:
                raise NotImplemented
            else:
                delta = c_precond(grad[None], None, hess[None])[0]
                # delta = sym_solve(hess.movedim(0, -1), grad.movedim(0, -1))
                # delta = delta.movedim(-1, 0)
        else:
            delta = grad
        mask1 = torch.isfinite(delta).all(0)
        delta.masked_fill(~(mask & mask1), 0)

        for k, prm in enumerate(opt.parameters):
            ref = getattr(logfit, prm)
            if prm == 'PD':
                delta1 = delta[k:k+nb_pd]
            elif 'PD' in opt.parameters:
                k += nb_pd - 1
            if prm == 'DeltaPhi':
                delta1 = delta[k:k+nb_phi]
            elif prm != 'PD':
                delta1 = delta[k]
            ref.add_(delta1, alpha=-opt.lr)
            ref.clamp_(-64, 64)
        fit = logfit.exp()

        print(f'{n_iter:03d} | {loss:g} | {loss_prev - loss:g}')

    sse = 0
    for dat1 in dat:
        dat0 = dat1.dat
        dat1.dat = dat1.dat.cuda()
        lam = dat1.SNR**2
        dof = dat1.df
        f, g, h = forward(dat1, fit)
        sse1 = nll_chi(dat1.dat, f, mask, lam, dof,
                       return_residuals=False, sum_crit=False)
        sse += sse1
        dat1.dat = dat0
    fit.SSE = sse

    return fit


def spgrfit(dat, opt, init=None):

    shape = dat[0].dat.shape[-3:]
    backend = dict(dtype=dat[0].dat.dtype, device=opt.device)

    for dat1 in dat:
        dat1.dat = dat1.dat.pin_memory()

    for dat1 in dat:
        dat1.dat = dat1.dat.cuda()
        dat1.step = min_intensity_step(dat1.dat)
        dat1.dat = torch.rand_like(dat1.dat).mul_(dat1.step).add_(dat1.dat)
        noise, _ = estimate_noise(dat1.dat.cuda(), chi=True)
        dat1.SNR = 1/noise['sd']
        dat1.df = noise['dof']
        dat1.dat = dat1.dat.cpu().pin_memory()

    mask = 0
    for dat1 in dat:
        mask += dat1.dat.cuda().square()
    mask /= len(dat)
    mask = mask.sqrt_()
    pd0 = mask[mask > 100].mean()
    mask = mask > 50

    fit = init or Tissue()
    fit.PD = torch.full(shape, pd0, **backend) if not torch.is_tensor(fit.PD) else fit.PD.to(**backend)
    fit.R1 = torch.ones(shape, **backend) if not torch.is_tensor(fit.R1) else fit.R1.to(**backend)
    fit.R2star = torch.full(shape, 1/20e-3, **backend) if not torch.is_tensor(fit.R2star) else fit.R2star.to(**backend)
    fit.B1 = torch.ones(shape, **backend) if not torch.is_tensor(fit.B1) else fit.B1.to(**backend)
    logfit = fit.log()

    K = len(opt.parameters)
    KK = (K * (K + 1)) // 2
    grad = torch.zeros([K, *shape], **backend)
    if opt.newton:
        hess = torch.zeros([KK, *shape], **backend)

    # utility to index into the symmetric hessian
    hind = torch.zeros([K, K], dtype=torch.long)
    cnt = K
    for k in range(K):
        hind[k, k] = k
        for kk in range(k+1, K):
            hind[k, kk] = hind[kk, k] = cnt
            cnt += 1

    loss = float('inf')
    losses = []
    for n_iter in range(opt.max_iter):
        loss_prev = loss

        # Compute derivatives
        grad.zero_()
        if opt.newton:
            hess.zero_()
        loss = 0
        for dat1 in dat:
            dat0 = dat1.dat
            dat1.dat = dat1.dat.cuda()
            lam = dat1.SNR**2
            dof = dat1.df
            f, g, h = forward_spgr(dat1, fit, star=True)
            loss1, d = nll_chi(dat1.dat, f, mask, lam, dof)
            loss += loss1
            mask1 = torch.isfinite(d)
            d.masked_fill_(~(mask & mask1), 0)
            for k, prm in enumerate(opt.parameters):
                if getattr(g, prm) is None:
                    continue
                grad[k].addcmul_(getattr(g, prm), d, value=lam)
                if opt.newton:
                    hess[k].addcmul_(getattr(g, prm), getattr(g, prm), value=lam)
                    if opt.robust:
                        hess[k].add_(getattr(h, prm).mul(d).abs_(), alpha=lam)
                    for kk in range(k+1, K):
                        prm2 = opt.parameters[kk]
                        if getattr(g, prm2) is not None:
                            c = hind[k, kk]
                            hess[c].addcmul_(getattr(g, prm), getattr(g, prm2), value=lam)
            dat1.dat = dat0
        loss /= len(dat)
        losses.append(loss)

        if opt.jtv:
            raise NotImplemented

        if opt.newton:
            if opt.marquardt:
                hess[:K].add_(hess[:K].abs(), alpha=opt.marquardt)
            if opt.levenberg:
                hess[:K].add_(hess.abs().max(0).values, alpha=opt.levenberg)
            hess[:K] += 1e-5
            if opt.jtv:
                raise NotImplemented
            else:
                delta = c_precond(grad[None], None, hess[None])[0]
                # delta = sym_solve(hess.movedim(0, -1), grad.movedim(0, -1))
                # delta = delta.movedim(-1, 0)
        else:
            delta = grad
        mask1 = torch.isfinite(delta).all(0)
        delta.masked_fill(~(mask & mask1), 0)

        for k, prm in enumerate(opt.parameters):
            ref = getattr(logfit, prm)
            ref.add_(delta[k], alpha=-opt.lr)
            ref.clamp_(-64, 64)
        fit = logfit.exp()

        print(f'{n_iter:03d} | {loss:g} | {loss_prev - loss:g}')

    sse = 0
    for dat1 in dat:
        dat0 = dat1.dat
        dat1.dat = dat1.dat.cuda()
        lam = dat1.SNR**2
        dof = dat1.df
        f, g, h = forward_spgr(dat1, fit, star=True)
        sse1 = nll_chi(dat1.dat, f, mask, lam, dof,
                       return_residuals=False, sum_crit=False)
        sse += sse1
        dat1.dat = dat0
    fit.SSE = sse

    return fit


def forward(dat, fit):
    if isinstance(dat, SPGR):
        return forward_spgr(dat, fit)
    if isinstance(dat, bSSFP):
        return forward_bssfp(dat, fit)
    raise NotImplementedError(type(dat))


def forward_spgr(dat, fit, star=False):
    PD = fit.PD
    if fit.PD.dim() > fit.R1.dim():
        PD = PD[dat.PDindex]
    R1 = fit.R1
    R2star = fit.R2star
    Alpha = fit.B1 * dat.FA
    cosAlpha = cos(Alpha)
    sinAlpha = sin(Alpha)
    E1 = exp(-dat.TR * R1)
    E2 = exp(-dat.TE * R2star)

    f = PD * sinAlpha * (1 - E1) * E2 / (1 - cosAlpha * E1)
    g = LogTissue()
    h = LogTissue()

    g.DeltaPhi = None
    h.DeltaPhi = None

    # gradient
    g.PD = f

    num = (R1*dat.TR) * E1 * (1 - cosAlpha)
    den = (1 - E1*cosAlpha) * (1 - E1)
    g.R1 = num / den
    g.R1 = g.R1 * f

    if star:
        g.R2star = (-dat.TE) * (R2star * f)
    else:
        g.R2 = (-dat.TE) * (fit.R2 * f)
        g.R2prime = (-dat.TE) * (fit.R2prime * f)

    num = Alpha * (E1 - cosAlpha)
    den = sinAlpha * (E1*cosAlpha - 1)
    g.B1 = num / den
    g.B1 = g.B1 * f

    # hessian
    h.PD = f

    num = 2 * dat.TR * R1 * cosAlpha * E1
    den = E1 * cosAlpha - 1
    add = 1 - dat.TR * R1
    h.R1 = num/den + add
    h.R1 = h.R1 * g.R1

    if star:
        h.R2star = 1 - dat.TE * R2star
        h.R2star = h.R2star * g.R2star
    else:
        h.R2 = 1 - dat.TE * fit.R2
        h.R2 = h.R2 * g.R2

        h.R2prime = 1 - dat.TE * fit.R2prime
        h.R2prime = h.R2prime * g.R2prime

    num = 2*square(E1) - E1 * cosAlpha - 1
    num = Alpha * sinAlpha * num
    den = (E1 * cosAlpha - 1) * (E1 - cosAlpha)
    h.B1 = num/den + 1
    h.B1 = h.B1 * g.B1

    return f, g, h


def forward_bssfp(dat, fit):
    PD = fit.PD[dat.PDindex]
    R1 = fit.R1
    R2 = fit.R2
    Alpha = fit.B1 * dat.FA
    cosAlpha = cos(Alpha)
    sinAlpha = sin(Alpha)
    Phi = dat.Phi + fit.DeltaPhi[dat.B0index]
    cosPhi = cos(Phi)
    sinPhi = sin(Phi)
    E1 = exp(-dat.TR * R1)
    E2 = exp(-dat.TR * R2)

    g = LogTissue()
    h = LogTissue()

    g.R2prime = None
    h.R2prime = None

    # signal
    Denom = (E1*cosAlpha - 1) * (E2*cosPhi - 1) - E2*(E1 - cosAlpha)*(E2 - cosPhi)
    Complex = sqrt(1 + square(E2) - 2*cosPhi*E2)
    f = PD * (1 - E1) * sinAlpha * Complex / Denom
    f = f * sqrt(E2)

    # gradient
    g.PD = f

    denom = (1 - E1) * Denom
    num = E1*R1*dat.TR*(square(E2) - 1)*(cosAlpha - 1)
    g.R1 = num / denom
    g.R1 = g.R1 * f

    num1 = 1 - E1*cosAlpha - (cosAlpha - E1)*square(E2)
    denom1 = Denom
    num2 = square(E2) - 1
    denom2 = 2*(square(E2) - 2*cosPhi*E2 + 1)
    g.R2 = num1 / denom1 + num2 / denom2
    g.R2 = -dat.TR * R2 * g.R2
    g.R2 = g.R2 * f

    if False:  # LogDeltaPhi
        num1 = 1
        denom1 = E2 * (E2 - 2*cosPhi) + 1
        num2 = -(1 - E1)*(cosAlpha + 1)
        denom2 = Denom
        g.DeltaPhi = num1 / denom1 + num2 / denom2
        g.DeltaPhi = g.DeltaPhi * fit.DeltaPhi * E2 * sinPhi
        g.DeltaPhi = g.DeltaPhi * f
    else:  # DeltaPhi
        num1 = 1
        den1 = square(E2) - 2*cosPhi*E2 + 1
        num2 = -(1 - E1)*(cosAlpha + 1)
        den2 = Denom
        g.DeltaPhi = num1 / den1 + num2 / den2
        g.DeltaPhi = g.DeltaPhi * E2*sinPhi
        g.DeltaPhi = g.DeltaPhi * f

    num1 = cosAlpha
    denom1 = sinAlpha
    num2 = - sinAlpha*(E1 + E2*cosPhi - square(E2) - E1*E2*cosPhi)
    denom2 = Denom
    g.B1 = num1 / denom1 + num2 / denom2
    g.B1 = g.B1 * Alpha
    g.B1 = g.B1 * f

    # hessian
    h.PD = f

    num1 = E1 + cosAlpha
    den1 = E1 - cosAlpha
    num2 = 2*E1*square(sinAlpha)*(1 - E2*cosPhi)
    den2 = (cosAlpha - E1)*Denom
    h.R1 = num1/den1 + num2/den2
    h.R1 = h.R1 * R1*dat.TR
    h.R1 = h.R1 + 1
    h.R1 = h.R1 * g.R1

    num = Complex*(E1 - 1)/(4*Denom) \
             - ((-   E2**2 +   cosPhi*E2)**2 * (E1 - 1))/(Complex**3 * Denom) \
             - ((- 3*E2**2 + 2*cosPhi*E2)    * (E1 - 1))/(Complex    * Denom) \
             + (  Complex*(E1 - 1)*E2   *((6*E2 - 2*cosPhi)*(E1 - cosAlpha) - 2*cosPhi*(E1*cosAlpha - 1))   )/Denom**2 \
             + (2*Complex*(E1 - 1)*E2**2 * ((2*E2 -   cosPhi)*(E1 - cosAlpha) -   cosPhi*(E1*cosAlpha - 1))**2)/Denom**3 \
             - ((- 2*E2**2 + 2*cosPhi*E2)*(E1 - 1)*(E2*(2*E2 - cosPhi)*(E1 - cosAlpha) - E2*cosPhi*(E1*cosAlpha - 1)))/(Complex * Denom**2)
    den = (Complex*(E1 - 1))/(2*Denom) \
             + (Complex*(E1 - 1)*E2*((2*E2 - cosPhi)*(E1 - cosAlpha) - cosPhi*(E1*cosAlpha - 1)))/Denom**2 \
             - (E2*(cosPhi - E2)*(E1 - 1))/(Complex*Denom)
    h.R2 = num / den
    h.R2 = -R2*dat.TR * h.R2
    h.R2 = h.R2 + 1
    h.R2 = h.R2 * g.R2

    if False:  # LogDeltaPhi
        num = \
              Denom*cosPhi*fit.DeltaPhi/sinPhi  \
            - Denom*E2*sinPhi*fit.DeltaPhi/Complex**2 \
            + Denom \
            + Complex**2 * (1 + cosPhi*fit.DeltaPhi/sinPhi)*(E1 - 1)*(cosAlpha + 1) \
            + 2*fit.DeltaPhi*E2*sinPhi*(E1 - 1)*(cosAlpha + 1) \
            + 2*Complex**2 * fit.DeltaPhi*E2*sinPhi*(E1 - 1)**2 * (cosAlpha + 1)**2 / Denom
        den = Denom + Complex**2 * (E1 - cosAlpha - 1 + E1*cosAlpha)
        h.DeltaPhi = num / den
        h.DeltaPhi = h.DeltaPhi * g.DeltaPhi
    else:  # DeltaPhi
        num1 = 2 * (E1 - 1) * (cosAlpha + 1)
        den1 = Denom
        num2 = Denom
        den2 = square(Complex) * (Denom + square(Complex) * (cosAlpha + 1) * (E1 - 1))
        h.DeltaPhi = num1 / den1 - num2 / den2
        h.DeltaPhi = h.DeltaPhi * E2 * sinPhi
        h.DeltaPhi = h.DeltaPhi + cosPhi / sinPhi
        h.DeltaPhi = h.DeltaPhi * g.DeltaPhi

    if torch.is_tensor(sinPhi):
        h.DeltaPhi.masked_fill_(sinPhi == 0, 0)
    elif sinPhi == 0:
        h.DeltaPhi.zero_()

    foo = E1*(E2*cosPhi - 1) + E2*(E2 - cosPhi)
    num1 = 2*foo
    den1 = Denom
    num2 = cosAlpha*foo - Denom
    den2 = cosAlpha*Denom + (sinAlpha**2*foo)
    h.B1 = num1/den1 + num2/den2
    h.B1 = h.B1 * sinAlpha*Alpha
    h.B1 = h.B1 + 1
    h.B1 = h.B1 * g.B1

    return f, g, h


def exp(x):
    return (torch.exp(x) if torch.is_tensor(x) else
            math.exp(x) if x is not None else None)


def log(x):
    return (torch.log(x) if torch.is_tensor(x) else
            math.log(x) if x is not None else None)


def cos(x):
    return (torch.cos(x) if torch.is_tensor(x) else
            math.cos(x) if x is not None else None)


def sin(x):
    return (torch.sin(x) if torch.is_tensor(x) else
            math.sin(x) if x is not None else None)


def square(x):
    return (torch.square(x) if torch.is_tensor(x) else
            (x*x) if x is not None else None)


def sqrt(x):
    return (torch.sqrt(x) if torch.is_tensor(x) else
            math.sqrt(x) if x is not None else None)


def pow(x, y):
    return (torch.pow(x, y) if torch.is_tensor(x) else
            math.pow(x, y) if x is not None else None)


def clone(x):
    return (torch.clone(x) if torch.is_tensor(x) else x)
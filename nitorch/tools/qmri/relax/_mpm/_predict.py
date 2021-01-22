from nitorch import core
from nitorch.core.pyutils import make_list
from nitorch.tools.qmri.param import ParameterMap
from nitorch.tools.qmri.io import PrecomputedFieldMap, GradientEchoMulti
from ..utils import smart_grid, smart_pull
import torch


def gre(pd, r1, r2s=None, mt=None, transmit=None, receive=None, gfactor=None,
        te=0, tr=25e-3, fa=20, mtpulse=False, sigma=None, noise='rician',
        affine=None, shape=None, device=None):
    """Simulate data generated by a Gradient-Echo (FLASH) sequence.

    Tissue parameters
    -----------------
    pd : ParameterMap or tensor_like
        Proton density
    r1 : ParameterMap or tensor_like
        Longitudinal relaxation rate, in 1/sec
    r2s : ParameterMap, optional
        Transverse relaxation rate, in 1/sec. Mandatory if any `te > 0`.
    mt : ParameterMap, optional
        MTsat. Mandatory if any `mtpulse == True`.

    Fields
    ------
    transmit : (N-sequence of) PrecomputedFieldMap or tensor_like, optional
        Transmit B1 field
    receive : (N-sequence of) PrecomputedFieldMap or tensor_like, optional
        Receive B1 field
    gfactor : (N-sequence of) PrecomputedFieldMap or tensor_like, optional
        G-factor map.
        If provided and `sigma` is not `None`, the g-factor map is used
        to sample non-stationary noise.

    Sequence parameters
    -------------------
    te : ((N-sequence of) M-sequence of) float, default=0
        Echo time, in sec
    tr : (N-sequence of) float default=2.5e-3
        Repetition time, in sec
    fa : (N-sequence of) float, default=20
        Flip angle, in deg
    mtpulse : (N-sequence of) bool, default=False
        Presence of an off-resonance pulse

    Noise
    -----
    sigma : (N-sequence of) float, optional
        Standard-deviation of the sampled noise (no sampling if `None`)
    noise : {'rician', 'gaussian'}, default='rician'
        Noise distribution

    Space
    -----
    affine : ([N], 4, 4) tensor, optional
        Orientation matrix of the simulation space
    shape : (N-sequence of) sequence[int], default=pd.shape
        Shape of the simulation space

    Returns
    -------
    sim : (N-sequence of) GradientEchoMulti
        Simulated series of multi-echo GRE images

    """

    # 1) Find out the number of contrasts requested
    te = make_list(te)
    if any(map(lambda x: isinstance(x, (list, tuple)), te)):
        te = [make_list(t) for t in te]
    else:
        te = [te]
    tr = make_list(tr)
    fa = make_list(fa)
    mtpulse = make_list(mtpulse)
    mtpulse = [bool(p) for p in mtpulse]
    sigma = make_list(sigma)
    transmit = make_list(transmit or [])
    receive = make_list(receive or [])
    gfactor = make_list(gfactor or [])
    shape = make_list(shape)
    if any(map(lambda x: isinstance(x, (list, tuple)), shape)):
        shape = [make_list(s) for s in shape]
    else:
        shape = [shape]
    if torch.is_tensor(affine):
        affine = [affine] if affine.dim() == 2 else affine.unbind(0)
    else:
        affine = make_list(affine)
    nb_contrasts = max(len(te), len(tr), len(fa), len(mtpulse), len(sigma),
                      len(transmit), len(receive), len(gfactor),
                      len(shape), len(affine))

    # 2) Pad all lists up to `nb_contrasts`
    te = make_list(te, nb_contrasts)
    tr = make_list(tr, nb_contrasts)
    fa = make_list(fa, nb_contrasts)
    mtpulse = make_list(mtpulse, nb_contrasts)
    sigma = make_list(sigma, nb_contrasts)
    transmit = make_list(transmit or [None], nb_contrasts)
    receive = make_list(receive or [None], nb_contrasts)
    gfactor = make_list(gfactor or [None], nb_contrasts)
    shape = make_list(shape, nb_contrasts)
    affine = make_list(affine, nb_contrasts)

    # 3) ensure parameters are `ParameterMap`s
    has_r2s = r2s is not None
    has_mt = mt is not None
    if not isinstance(pd, ParameterMap):
        pd = ParameterMap(pd)
    if not isinstance(r1, ParameterMap):
        r1 = ParameterMap(r1)
    if has_r2s and not isinstance(r2s, ParameterMap):
        r2s = ParameterMap(r2s)
    if has_mt and not isinstance(mt, ParameterMap):
        mt = ParameterMap(mt)

    # 4) ensure all fields are `PrecomputedFieldMap`s
    for n in range(nb_contrasts):
        if (transmit[n] is not None and
                not isinstance(transmit[n], PrecomputedFieldMap)):
            transmit[n] = PrecomputedFieldMap(transmit[n])
        if (receive[n] is not None and
                not isinstance(receive[n], PrecomputedFieldMap)):
            receive[n] = PrecomputedFieldMap(receive[n])
        if (gfactor[n] is not None and
                not isinstance(gfactor[n], PrecomputedFieldMap)):
            gfactor[n] = PrecomputedFieldMap(gfactor[n])

    # 5) choose backend
    all_var = [te, tr, fa, mtpulse, sigma, affine]
    all_var += [f.volume for f in transmit
                if f is not None and torch.is_tensor(f.volume)]
    all_var += [f.volume for f in receive
                if f is not None and torch.is_tensor(f.volume)]
    all_var += [f.volume for f in gfactor
                if f is not None and torch.is_tensor(f.volume)]
    all_var += [pd.volume] if torch.is_tensor(pd.volume) else []
    all_var += [r1.volume] if torch.is_tensor(r1.volume) else []
    all_var += [r2s.volume] if r2s is not None and torch.is_tensor(r2s.volume) else []
    all_var += [mt.volume] if mt is not None and torch.is_tensor(mt.volume) else []
    backend = core.utils.max_backend(*all_var)
    if device:
        backend['device'] = device

    # 6) prepare parameter maps
    prm = stack_maps(pd.fdata(**backend), r1.fdata(**backend),
                     r2s.fdata(**backend) if r2s is not None else None,
                     mt.fdata(**backend) if mt is not None else None)
    fpd, fr1, fr2s, fmt = unstack_maps(prm, has_r2s, has_mt)
    if has_mt and mt.unit in ('%', 'pct', 'p.u.'):
        fmt /= 100.
    if any(aff is not None for aff in affine):
        logprm = stack_maps(safelog(fpd), safelog(fr1),
                            safelog(fr2s) if r2s is not None else None,
                            safelog(fmt) + safelog(1 - fmt)
                            if mt is not None else None)

    # 7) generate noise-free signal
    contrasts = []
    for n in range(nb_contrasts):

        shape1 = shape[n]
        if shape1 is None:
            shape1 = pd.shape
        aff1 = affine[n]
        if aff1 is not None:
            aff1 = aff1.to(**backend)
        te1 = torch.as_tensor(te[n], **backend)
        tr1 = torch.as_tensor(tr[n], **backend)
        fa1 = torch.as_tensor(fa[n], **backend) / 180. * core.constants.pi
        sigma1 = torch.as_tensor(sigma[n], **backend) if sigma[n] else None
        mtpulse1 = mtpulse[n]
        transmit1 = transmit[n]
        receive1 = receive[n]
        gfactor1 = gfactor[n]

        if aff1 is not None:
            mat = core.linalg.lmdiv(pd.affine.to(**backend), aff1)
            grid = smart_grid(mat, shape1, pd.shape)
            prm1 = smart_pull(logprm, grid)
            del grid
            f1pd, f1r1, f1r2s, f1mt = unstack_maps(prm1, has_r2s, has_mt)
            f1pd, f1r1, f1r2s, f1mt = exp_maps_(f1pd, f1r1, f1r2s, f1mt)
        else:
            f1pd, f1r1, f1r2s, f1mt = (fpd, fr1, fr2s, fmt)
            f1pd = f1pd.clone()  # clone it so that we can work in-place later

        if transmit1 is not None:
            unit = transmit1.unit
            taff1 = transmit1.affine.to(**backend)
            transmit1 = transmit1.fdata(**backend)
            if unit in ('%', 'pct', 'p.u'):
                transmit1 = transmit1 / 100.
            if aff1 is not None:
                mat = core.linalg.lmdiv(taff1, aff1)
                grid = smart_grid(mat, shape1, transmit1.shape)
                transmit1 = smart_pull(transmit1[None], grid)[0]
                del grid
            fa1 = fa1 * transmit1
            del transmit1

        if receive1 is not None:
            unit = receive1.unit
            raff1 = receive1.affine.to(**backend)
            receive1 = receive1.fdata(**backend)
            if unit in ('%', 'pct', 'p.u'):
                receive1 = receive1 / 100.
            if aff1 is not None:
                mat = core.linalg.lmdiv(raff1, aff1)
                grid = smart_grid(mat, shape1, receive1.shape)
                receive1 = smart_pull(receive1[None], grid)[0]
                del grid
            f1pd = f1pd * receive1
            del receive1

        # generate signal
        flash = f1pd
        flash *= fa1.sin()
        cosfa = fa1.cos_()
        e1 = f1r1
        e1 *= -tr1
        e1 = e1.exp_()
        flash *= (1 - e1)
        if mtpulse1:
            if not has_mt:
                raise ValueError('Cannot simulate an MT pulse: '
                                 'an MT must be provided.')
            omt = f1mt.neg_()
            omt += 1
            flash *= omt
            flash /= (1 - cosfa * omt * e1)
            del omt
        else:
            flash /= (1 - cosfa * e1)
        del e1, cosfa, fa1, f1r1, f1mt

        # multiply with r2*
        if any(t > 0 for t in te1):
            if not has_r2s:
                raise ValueError('Cannot simulate an R2* decay: '
                                 'an R2* must be provided.')
            te1 = te1.reshape([-1] + [1] * f1r2s.dim())
            flash = flash * (-te1 * f1r2s).exp_()
            del f1r2s

        # sample noise
        if sigma1:
            if gfactor1 is not None:
                gfactor1 = gfactor1.fdata(**backend)
                if gfactor1.unit in ('%', 'pct', 'p.u'):
                    gfactor1 = gfactor1 / 100.
                if aff1 is not None:
                    mat = core.linalg.lmdiv(gfactor1.affine.to(**backend), aff1)
                    grid = smart_grid(mat, shape1, gfactor1.shape)
                    gfactor1 = smart_pull(gfactor1[None], grid)[0]
                    del grid
                sigma1 = sigma1 * gfactor1
                del gfactor1

            noise_shape = flash.shape
            if noise == 'rician':
                noise_shape = (2,) + noise_shape
            sample = torch.randn(noise_shape, **backend)
            sample *= sigma1
            del sigma1
            if noise == 'rician':
                sample = sample.square_().sum(dim=0)
                flash = flash.square_().add_(sample).sqrt_()
            else:
                flash += sample
            del sample

        te1 = te1.tolist()
        tr1 = tr1.item()
        fa1 = torch.as_tensor(fa[n]).item()
        mtpulse1 = torch.as_tensor(mtpulse1).item()
        flash = GradientEchoMulti(flash, affine=aff1,
                                  tr=tr1, fa=fa1, te=te1, mt=mtpulse1)
        contrasts.append(flash)

    return contrasts[0] if len(contrasts) == 1 else contrasts


def stack_maps(*prm):
    prm = [p for p in prm if p is not None]
    return torch.stack(prm)


def unstack_maps(prm, has_r2s, has_mt):
    pd = prm[0]
    r1 = prm[1]
    r2s = prm[2] if has_r2s else None
    mt = prm[-1] if has_mt else None
    return pd, r1, r2s, mt


def exp_maps(pd, r1, r2s, mt, inplace=False):
    if not inplace:
        pd = pd.clone()
        r1 = r1.clone()
        if r2s is not None:
            r2s = r2s.clone()
        if mt is not None:
            mt = mt.clone()
    pd = pd.exp_()
    r1 = r1.exp_()
    if r2s is not None:
        r2s = r2s.exp_()
    if mt is not None:
        mt = mt.neg_().exp_().add_(1).reciprocal_()
    return pd, r1, r2s, mt


def exp_maps_(pd, r1, r2s, mt):
    return exp_maps(pd, r1, r2s, mt, inplace=True)


def safelog(x, eps=1e-42):
    return x.clamp_min(eps).log_()
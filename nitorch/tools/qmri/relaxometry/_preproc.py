import torch
from nitorch import core, spatial
from nitorch.tools.img_statistics import estimate_noise
from nitorch.tools.preproc import affine_align
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot', _as=True)
from ._options import Options
from ..param import ParameterMap
from ._param import ParameterMaps


def postproc(maps):
    """Generate PD, R1, R2* (and MTsat) volumes from log-parameters

    Parameters
    ----------
    maps : ParameterMaps

    Returns
    -------
    pd : ParameterMap
    r1 : ParameterMap
    r2s : ParameterMap
    mt : ParameterMap, optional

    """
    maps.r1.volume = maps.r1.volume.exp_()
    maps.r1.name = 'R1'
    maps.r1.unit = '1/s'
    maps.r2s.volume = maps.r2s.volume.exp_()
    maps.r2s.name = 'R2*'
    maps.r2s.unit = '1/s'
    maps.pd.volume = maps.pd.volume.exp_()
    maps.r2s.name = 'PD'
    maps.r2s.unit = 'a.u.'
    if hasattr(maps, 'mt'):
        maps.mt.volume = maps.mt.volume.neg_().exp_()
        maps.mt.volume += 1
        maps.mt.volume = maps.mt.volume.reciprocal_()
        maps.mt.volume *= 100
        maps.mt.name = 'MTsat'
        maps.mt.unit = 'p.u.'
        return maps.pd, maps.r1, maps.r2s, maps.mt
    return maps.pd, maps.r1, maps.r2s


def preproc(data, transmit=[], receive=[], opt=None):
    """Estimate noise variance + register + compute recon space + init maps

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
    transmit : sequence[PrecomputedFieldMap], optional
    receive : sequence[PrecomputedFieldMap], optional
    opt : Options, optional

    Returns
    -------
    data : sequence[GradientEchoMulti]
    maps : ParametersMaps

    """

    if opt is None:
        opt = Options()

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    # --- estimate hyper parameters ---
    logmeans = []
    te = []
    tr = []
    fa = []
    mt = []
    for c, contrast in enumerate(data):
        means = []
        vars = []
        for e, echo in enumerate(contrast):
            if opt.verbose:
                print(f'Estimate noise: contrast {c+1:d} - echo {e+1:2d}', end='\r')
            dat = echo.fdata(**backend, rand=True, cache=False)
            sd0, sd1, mu0, mu1 = estimate_noise(dat)
            echo.mean = mu1.item()
            echo.sd = sd0.item()
            means.append(mu1)
            vars.append(sd0.square())
        means = torch.stack(means)
        vars = torch.stack(vars)
        var = (means*vars).sum() / means.sum()
        contrast.noise = var.item()

        te.append(contrast.te)
        tr.append(contrast.tr)
        fa.append(contrast.fa / 180 * core.constants.pi)
        mt.append(contrast.mt)
        logmeans.append(means.log())
    if opt.verbose:
        print('')

    # --- initial minifit ---
    print('Compute initial parameters')
    inter, r2s = _loglin_minifit(logmeans, te)
    pd, r1, mt = _rational_minifit(inter, tr, fa, mt)
    print(f'    - PD:    {pd.tolist():.3g}')
    print(f'    - R1:    {r1.tolist():.3g} s')
    print(f'    - R2*:   {r2s.tolist():.3g} s')
    pd = pd.log()
    r1 = r1.log()
    r2s = r2s.log()
    if mt is not None:
        print(f'    - MT:    {100*mt.tolist():.3g} %')
        mt = mt.log() - (1 - mt).log()

    # --- initial align ---
    if opt.preproc.register and len(data) > 1:
        print('Register volumes')
        data_reg = [(contrast.echo(0).fdata(rand=True, cache=False, **backend),
                     contrast.affine) for contrast in data]
        dats, affines, _ = affine_align(data_reg, device=device)
        if opt.verbose > 1 and plt:
            plt.figure()
            for i in range(len(dats)):
                plt.subplot(1, len(dats), i+1)
                plt.imshow(dats[i, :, dats.shape[2]//2, :].cpu())
            plt.show()
        for contrast, aff in zip(data, affines):
            aff, contrast.affine = core.utils.to_common(aff, contrast.affine)
            contrast.affine = torch.matmul(aff.inverse(), contrast.affine)

    # --- compute recon space ---
    affines = [contrast.affine for contrast in data]
    shapes = [dat.volume.shape[1:] for dat in data]
    if opt.recon.space == 'mean':
        print('Estimate recon space')
        if isinstance(opt.recon.space, int):
            mean_affine = affines[opt.recon.space]
            mean_shape = shapes[opt.recon.space]
        elif isinstance(opt.recon.space, str) and opt.recon.space.lower() == 'mean':
            mean_affine, mean_shape = spatial.mean_space(affines, shapes)
        else:
            raise NotImplementedError()
    else:
        mean_affine = affines[opt.recon.space]
        mean_shape = shapes[opt.recon.space]

    # --- allocate maps ---
    maps = ParameterMaps()
    maps.pd = ParameterMap(mean_shape, fill=pd, affine=mean_affine, **backend)
    maps.r1 = ParameterMap(mean_shape, fill=r1, affine=mean_affine, **backend)
    maps.r2s = ParameterMap(mean_shape, fill=r2s, affine=mean_affine, **backend)
    if mt is not None:
        maps.mt = ParameterMap(mean_shape, fill=mt, affine=mean_affine, **backend)
    maps.affine = mean_affine
    maps.shape = mean_shape

    return data, transmit, receive, maps


def _loglin_minifit(dat, te):
    """Log-linear fit on a single voxel

    Parameters
    ----------
    dat: (C,) sequence of (E,) sequence of float
        Observed log data.
        - Outter sequence: contrasts
        - Inner sequence: echoes
    te: (C,) sequence of (E,) sequence of float
        Echo times
        - Outter sequence: contrasts
        - Inner sequence: echoes

    Returns
    -------
    inter : (C,) tensor
        Intercepts (extrapolated at TE = 0)
    decay : () tensor
        Decay

    """

    nb_contrasts = len(dat)
    nb_images = sum(len(contrast) for contrast in dat)

    # build data and contrast matrix
    contrast_matrix = torch.zeros([nb_images, nb_contrasts + 1])
    observed = torch.zeros([nb_images])
    i = 0
    for c, contrast in enumerate(dat):
        for e, echo in enumerate(contrast):
            contrast_matrix[i, c] = 1
            contrast_matrix[i, -1] = -te[c][e]
            observed[i] = echo
            i += 1

    contrast_matrix = torch.pinverse(contrast_matrix)
    param = core.linalg.matvec(contrast_matrix, observed)

    return param[:-1].exp(), param[-1]


def _rational_minifit(inter, tr, fa, mt):
    """Rational approximation on a single voxel

    Parameters
    ----------
    inter : (C,) sequence[float]
        Intercepts (=data extrapolated at TE = 0)
    tr : (C,) sequence[float]
        Repetition times, in sec.
    fa : (C,) sequence[float]
        Flip angles, in rad.
    mt : (C,) sequence[float or bool or None]
        MT pulse (frequency or boolean)

    Returns
    -------
    pd : () tensor
    r1 : () tensor
    mt : () tensor or None

    """

    data_for_pdr1 = [(inter1, tr1, fa1)
                     for inter1, tr1, fa1, mt1 in zip(inter, tr, fa, mt)
                     if not mt]
    data_for_pdr1 = data_for_pdr1[:2]
    (pdw, pdw_tr, pdw_fa), (t1w, t1w_tr, t1w_fa) = data_for_pdr1

    pdw = torch.as_tensor(pdw)
    t1w = torch.as_tensor(t1w)

    r1 = 0.5 * (t1w * (t1w_fa / t1w_tr) - pdw * (pdw_fa / pdw_tr))
    r1 /= ((pdw / pdw_fa) - (t1w / t1w_fa))

    pd = (t1w * pdw) * (t1w_tr * (pdw_fa / t1w_fa) - pdw_tr * (t1w_fa / pdw_fa))
    pd /= (pdw * (pdw_tr * pdw_fa) - t1w * (t1w_tr * t1w_fa))

    data_for_mt = [(inter1, tr1, fa1)
                   for inter1, tr1, fa1, mt1 in zip(inter, tr, fa, mt)
                   if mt]
    if not data_for_mt:
        return pd, r1, None
    data_for_mt = data_for_mt[0]
    mtw, mtw_tr, mtw_fa = data_for_mt
    mtw = torch.as_tensor(mtw)
    mt = (mtw_fa * pd / mtw - 1) * r1 * mtw_tr - 0.5 * (mtw_fa ** 2)
    return pd, r1, mt

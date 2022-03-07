import torch
from nitorch import core, spatial
from nitorch.tools.img_statistics import estimate_noise
from nitorch.tools.preproc import affine_align
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot', _as=True)
from ._options import GREEQOptions
from nitorch.tools.qmri.param import ParameterMap
from ._param import GREEQParameterMaps


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
    # TODO:
    #   I could also convert the log-space uncertainty into
    #   "exponentiated" uncertainty. PD/R1/R2* are log-normal distirbuted
    #   under the Laplace approximation, so that's easy. But MTsat
    #   is logit-normal distributed, and that's harder (there's no
    #   analytical solution for its mean and variance).
    #   https://en.wikipedia.org/wiki/Log-normal_distribution
    #   https://en.wikipedia.org/wiki/Logit-normal_distribution

    pd = maps.pd
    r1 = maps.r1
    r2s = maps.r2s
    r1.volume = r1.fdata().exp_()
    r1.name = 'R1'
    r1.unit = '1/s'
    r2s.volume = r2s.fdata().exp_()
    r2s.name = 'R2*'
    r2s.unit = '1/s'
    pd.volume = maps.pd.fdata().exp_()
    pd.name = 'PD'
    pd.unit = 'a.u.'
    if hasattr(maps, 'mt'):
        mt = maps.mt
        mt.volume = mt.fdata().neg_().exp_().add_(1).reciprocal_().mul_(100)
        mt.name = 'MTsat'
        mt.unit = 'p.u.'
        return pd, r1, r2s, mt
    return pd, r1, r2s


def preproc(data, transmit=None, receive=None, opt=None):
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

    opt = GREEQOptions().update(opt)
    dtype = opt.backend.dtype
    device = opt.backend.device

    backend = dict(dtype=dtype, device=device)
    chi = opt.likelihood[0].lower() == 'c'
    
    # --- estimate hyper parameters ---
    logmeans = []
    te = []
    tr = []
    fa = []
    mt = []
    for c, contrast in enumerate(data):
        means = []
        vars = []
        dofs = []
        for e, echo in enumerate(contrast):
            if opt.verbose:
                print(f'Estimate noise: contrast {c+1:d} - echo {e+1:2d}', end='\r')
            dat = echo.fdata(**backend, rand=True, cache=False)

            prm_noise, prm_not_noise = estimate_noise(dat, chi=chi)
            sd0 = prm_noise['sd']
            mu1 = prm_not_noise['mean']
            dof0 = prm_noise.get('dof', 0)

            echo.mean = mu1.item()
            means.append(mu1)
            vars.append(sd0.square())
            dofs.append(dof0)
        means = torch.stack(means)
        vars = torch.stack(vars)
        var = (means*vars).sum() / means.sum()
        if chi:
            dofs = torch.stack(dofs)
            # degrees of freedom weighted by the mean of each echo
            # later echoes contribute less to the average dof
            # could be changed to weigthing by the noise standard deviation
            dofs = (dofs*means).sum() / means.sum()

        if not getattr(contrast, 'noise', 0):
            contrast.noise = var.item()
        if not getattr(contrast, 'dof', 0):
            contrast.dof = dofs.item() if chi else 2

        te.append(contrast.te)
        tr.append(contrast.tr)
        fa.append(contrast.fa / 180 * core.constants.pi)
        mt.append(contrast.mt)
        logmeans.append(means.log())
    if opt.verbose:
        print('')
        sds = [c.noise ** 0.5 for c in data]
        print('    - standard deviation:  [' + ', '.join([f'{s:.2f}' for s in sds]) + ']')
        if chi:
            dofs = [c.dof for c in data]
            print('    - degrees of freedom:  [' + ', '.join([f'{s:.2f}' for s in dofs]) + ']')

    print('Estimating maps from volumes:')
    for i in range(len(data)):
        print(f'    - Contrast {i:d}: ', end='')
        print(f'FA = {fa[i]*180/core.constants.pi:2.0f} deg  / ', end='')
        print(f'TR = {tr[i]*1e3:4.1f} ms / ', end='')
        print('TE = [' + ', '.join([f'{t*1e3:.1f}' for t in te[i]]) + '] ms', end='')
        if mt[i]:
            print(f' / MT = True', end='')
        print()
        
    # --- initial minifit ---
    print('Compute initial parameters')
    inter, r2s = _loglin_minifit(logmeans, te)
    pd, r1, mt = _rational_minifit(inter, tr, fa, mt)
    print(f'    - PD:    {pd.tolist():9.3g} a.u.')
    print(f'    - R1:    {r1.tolist():9.3g} 1/s')
    print(f'    - R2*:   {r2s.tolist():9.3g} 1/s')
    pd = pd.log()
    r1 = r1.log()
    r2s = r2s.log()
    if mt is not None:
        print(f'    - MT:    {100*mt.tolist():9.3g} %')
        mt = mt.log() - (1 - mt).log()

    # --- initial align ---
    transmit = core.py.make_list(transmit or [])
    receive = core.py.make_list(receive or [])
    if opt.preproc.register and len(data) > 1:
        print('Register volumes')
        data_reg = [(contrast.echo(0).fdata(rand=True, cache=False, **backend),
                     contrast.affine) for contrast in data]
        data_reg += [(map.magnitude.fdata(rand=True, cache=False, **backend),
                      map.magnitude.affine) for map in transmit
                     if map and map.magnitude is not None]
        data_reg += [(map.magnitude.fdata(rand=True, cache=False, **backend),
                      map.magnitude.affine) for map in receive
                     if map and map.magnitude is not None]
        dats, affines, _ = affine_align(data_reg, device=device)
        # if opt.verbose > 1 and plt:
        #     plt.figure()
        #     for i in range(len(dats)):
        #         plt.subplot(1, len(dats), i+1)
        #         plt.imshow(dats[i, :, dats.shape[2]//2, :].cpu())
        #         plt.axis('off')
        #     plt.show()
        for contrast, aff in zip(data + transmit + receive, affines):
            if contrast is not None:
                aff, contrast.affine = core.utils.to_max_device(aff, contrast.affine)
                contrast.affine = torch.matmul(aff.inverse(), contrast.affine)

    # --- compute recon space ---
    affines = [contrast.affine for contrast in data]
    shapes = [dat.volume.shape[1:] for dat in data]
    if opt.recon.affine is None:
        opt.recon.affine = opt.recon.space
    if opt.recon.fov is None:
        opt.recon.fov = opt.recon.space
    if isinstance(opt.recon.affine, str):
        assert opt.recon.affine == 'mean'
        mean_affine, _ = spatial.mean_space([dat.affine for dat in data],
                                            [dat.shape[1:] for dat in data])
    elif isinstance(opt.recon.affine, int):
        mean_affine = affines[opt.recon.affine]
    else:
        mean_affine = torch.as_tensor(opt.recon.affine)
    if isinstance(opt.recon.affine, str):
        assert opt.recon.affine in ('mean', 'bb')
        mean_affine, mean_shape = spatial.fov_max(mean_affine,
                                                  [dat.affine for dat in data],
                                                  [dat.shape[1:] for dat in data])
    elif isinstance(opt.recon.fov, int):
        mean_shape = shapes[opt.recon.fov]
    else:
        mean_shape = tuple(opt.recon.fov)

    # --- allocate maps ---
    maps = GREEQParameterMaps([3+(mt is not None), *mean_shape],
                              affine=mean_affine, **backend)
    maps.pd.volume.fill_(pd)
    maps.r1.volume.fill_(r1)
    maps.r2s.volume.fill_(r2s)
    if mt is not None:
        maps.mt.volume.fill_(mt)

    # --- repeat fields if not enough ---
    if transmit:
        transmit = core.py.make_list(transmit, len(data))
    else:
        transmit = [None] * len(data)
    if receive:
        receive = core.py.make_list(receive, len(data))
    else:
        receive = [None] * len(data)
    
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
                     if not mt1]
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
                   if mt1]
    if not data_for_mt:
        return pd, r1, None
    data_for_mt = data_for_mt[0]
    mtw, mtw_tr, mtw_fa = data_for_mt
    mtw = torch.as_tensor(mtw)
    mt = (mtw_fa * pd / mtw - 1) * r1 * mtw_tr - 0.5 * (mtw_fa ** 2)
    return pd, r1, mt

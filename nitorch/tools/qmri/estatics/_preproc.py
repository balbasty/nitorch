import torch
from nitorch import core, spatial
from nitorch.tools.img_statistics import estimate_noise
from nitorch.tools.preproc import affine_align
from nitorch.tools.qmri import io as qio
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot', _as=True)
from ._options import Options
from ._param import ParameterMap, ParameterMaps


def postproc(maps, contrasts):
    """Generate TE=0 and R2* volumes from log-parameters

    Parameters
    ----------
    maps : ParameterMaps
    contrasts : sequence[GradientEchoMulti]

    Returns
    -------
    intercepts : sequence[GradientEchoSingle]
    decay : ParameterMap

    """
    intercepts = []
    for loginter, contrast in zip(maps.intercepts, contrasts):
        volume = loginter.volume.exp_()
        attributes = {key: getattr(contrast, key)
                      for key in contrast.attributes()}
        attributes['affine'] = loginter.affine.clone()
        attributes['te'] = 0.
        inter = qio.GradientEchoSingle(volume, **attributes)
        intercepts.append(inter)
    decay = maps.decay
    decay.name = 'R2*'
    decay.unit = '1/s'

    return intercepts, decay


def preproc(data, opt):
    """Estimate noise variance + register + compute recon space + init maps

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
    opt : Options

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
        logmeans.append(means.log())
    if opt.verbose:
        print('')

    # --- initial minifit ---
    print('Compute initial parameters')
    inter, decay = _loglin_minifit(logmeans, te)
    print('    - log intercepts: [' + ', '.join([f'{i:.1f}' for i in inter.tolist()]) + ']')
    print(f'    - decay:          {decay.tolist():.3g}')

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
    maps.intercepts = [ParameterMap(mean_shape, fill=inter[c], affine=mean_affine, **backend)
                       for c in range(len(data))]
    maps.decay = ParameterMap(mean_shape, fill=decay, affine=mean_affine, min=0, **backend)
    maps.affine = mean_affine
    maps.shape = mean_shape

    return data, maps


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
        Log-intercepts
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

    return param[:-1], param[-1]

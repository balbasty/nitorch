import torch
from nitorch import core, spatial
from nitorch.tools.img_statistics import estimate_noise
from nitorch.tools.preproc import affine_align
from ._options import Options
from ._param import ParameterMap, ParameterMaps


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
    info = dict(dtype=dtype, device=device)

    # --- estimate hyper parameters ---
    logmeans = []
    te = []
    for contrast in data:
        means = []
        vars = []
        for echo in contrast:
            dat = echo.fdata(**info, rand=True, cache=False)
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

    # --- initial minifit ---
    inter, decay = _loglin_minifit(logmeans, te)

    # --- initial align ---
    if opt.preproc.register and len(data) > 1:
        data_reg = [(contrast.echo(0).fdata(rand=True, cache=False),
                     contrast.affine) for contrast in data]
        _, affines, _ = affine_align(data_reg)
        for contrast, aff in zip(data, affines):
            contrast.affine = torch.matmul(aff.inverse(), contrast.affine)

    # --- compute recon space ---
    affines = [contrast.affine for contrast in data]
    shapes = [dat.volume.shape[1:] for dat in data]
    if opt.recon.space == 'mean':
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
    maps.intercepts = [ParameterMap(mean_shape, fill=inter[c], affine=mean_affine, **info)
                       for c in range(len(data))]
    maps.decay = ParameterMap(mean_shape, fill=decay, affine=mean_affine, min=0, **info)
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

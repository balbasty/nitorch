import torch
from nitorch import core, spatial
from nitorch.tools.img_statistics import estimate_noise
from nitorch.tools.preproc import affine_align
from nitorch.tools.qmri import io as qio
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot', _as=True)
from ._options import ESTATICSOptions
from nitorch.tools.qmri.param import ParameterMap, SVFDistortion, DenseDistortion
from ._param import ESTATICSParameterMaps


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


def preproc(data, dist, opt):
    """Estimate noise variance + register + compute recon space + init maps

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
    dist : sequence[Optional[ParameterizedDistortion]]
    opt : Options

    Returns
    -------
    data : sequence[GradientEchoMulti]
    maps : ParametersMaps
    dist : sequence[ParameterizedDistortion]

    """

    if opt is None:
        opt = ESTATICSOptions()

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)
    chi = opt.likelihood[0].lower() == 'c'

    # --- guess readout/blip if not provided ---
    for contrast in data:
        shape = contrast.spatial_shape
        if contrast.readout is None:
            contrast.readout = core.py.argmax(shape)
        if contrast.readout >= 0:
            contrast.readout = contrast.readout - (len(shape) + 1)

    # --- estimate hyper parameters ---
    logmeans = []
    te = []
    for c, contrast in enumerate(data):
        means = []
        vars = []
        dofs = []
        for e, echo in enumerate(contrast):
            if opt.verbose:
                print(f'Estimate noise: contrast {c+1:d} - echo {e+1:2d}', end='\r')
            dat = echo.fdata(**backend, rand=True, cache=False, missing=0)

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
            dofs = (dofs*means).sum() / means.sum()

        if not getattr(contrast, 'noise', 0):
            contrast.noise = var.item()
        if not getattr(contrast, 'dof', 0):
            contrast.dof = dofs.item() if chi else 2

        te.append(contrast.te)
        logmeans.append(means.log())

    if opt.verbose:
        print('')
        sds = [c.noise ** 0.5 for c in data]
        print('    - standard deviation:  [' + ', '.join([f'{s:.2f}' for s in sds]) + ']')
        if chi:
            dofs = [c.dof for c in data]
            print('    - degrees of freedom:  [' + ', '.join([f'{s:.2f}' for s in dofs]) + ']')

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
    maps = ESTATICSParameterMaps([len(data)+1, *mean_shape], **backend, affine=mean_affine)
    for c in range(len(data)):
        maps.intercepts[c].volume.fill_(inter[c])
    maps.decay.volume.fill_(decay)

    # --- allocate distortions ---
    dist = dist or [None]
    dist = core.py.make_list(dist, len(data))
    if opt.distortion.enable:
        for c, contrast in enumerate(data):
            if dist[c] is None:
                if opt.distortion.model == 'smalldef':
                    dist[c] = DenseDistortion(contrast.spatial_shape,
                                              dim=contrast.readout,
                                              affine=contrast.affine,
                                              **backend)
                elif opt.distortion.model == 'svf':
                    dist[c] = SVFDistortion(contrast.spatial_shape,
                                            dim=contrast.readout,
                                            affine=contrast.affine,
                                            steps=opt.distortion.steps,
                                            **backend)
                else:
                    raise ValueError('Unknown distortion model',
                                     opt.distortion.model)
            else:
                dist[c].displacement_dim = contrast.readout
                dist[c].affine = contrast.affine
                if opt.distortion.model == 'svf':
                    dist[c].steps = opt.distortion.steps

    return data, maps, dist


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

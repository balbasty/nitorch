import warnings
import torch
from nitorch import core, spatial
from nitorch.tools.preproc import affine_align
from nitorch.core.optionals import try_import
from nitorch.tools.qmri.param import ParameterMap
from ._options import VFAOptions
plt = try_import('matplotlib.pyplot', _as=True)


def vfa(data, transmit=None, receive=None, opt=None, **kwopt):
    """Compute PD, R1 (and MTsat) from two GRE at two flip angles using
    rational approximations of the Ernst equations.

    Parameters
    ----------
    data : sequence[GradientEcho]
        Volumes with different contrasts (flip angle or MT pulse) but with
        the same echo time. Note that they do not need to be real echoes;
        they often are images extrapolated to TE = 0.
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
    opt : VFAOptions
        Algorithm options.
        {'preproc': {'register':      True},     # Co-register contrasts
         'backend': {'dtype':  torch.float32,    # Data type
                     'device': 'cpu'},           # Device
         'verbose': 1}                           # Verbosity: 1=print, 2=plot

    Returns
    -------
    pd : ParameterMap
        Proton density (potentially, with R2* bias)
    r1 : ParameterMap
        Longitudinal relaxation rate
    mt : ParameterMap, optional
        Magnetization transfer

    References
    ----------
    ..[1] Tabelow et al., "hMRI - A toolbox for quantitative MRI in
          neuroscience and clinical research", NeuroImage (2019).
          https://www.sciencedirect.com/science/article/pii/S1053811919300291
    ..[2] Helms et al., "Quantitative FLASH MRI at 3T using a rational
          approximation of the Ernst equation", MRM (2008).
          https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21732
    ..[3] Helms et al., "High-resolution maps of magnetization transfer
          with inherent correction for RF inhomogeneity and T1 relaxation
          obtained from 3D FLASH MRI", MRM (2008).
          https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21732

    """
    opt = VFAOptions().update(opt, **kwopt)
    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    data = core.pyutils.make_list(data)
    if len(data) < 2:
        raise ValueError('Expected at least two input images')
    transmit = core.pyutils.make_list(transmit or [])
    receive = core.pyutils.make_list(receive or [])

    # --- Copy instances to avoid modifying the inputs ---
    data = list(map(lambda x: x.copy(), data))
    transmit = list(map(lambda x: x.copy(), transmit))
    receive = list(map(lambda x: x.copy(), receive))

    # --- check TEs ---
    if len(set([contrast.te for contrast in data])) > 1:
        raise ValueError('Echo times not consistent across contrasts')

    # --- register ---
    if opt.preproc.register:
        print('Register volumes')
        data_reg = [(contrast.fdata(rand=True, cache=False, **backend),
                     contrast.affine) for contrast in data]
        data_reg += [(fmap.magnitude.fdata(rand=True, cache=False, **backend),
                      fmap.magnitude.affine) for fmap in transmit]
        data_reg += [(fmap.magnitude.fdata(rand=True, cache=False, **backend),
                      fmap.magnitude.affine) for fmap in receive]
        dats, affines, _ = affine_align(data_reg, device=device, fwhm=3)
        
        if opt.verbose > 1 and plt:
            plt.figure()
            for i in range(len(dats)):
                plt.subplot(1, len(dats), i+1)
                plt.imshow(dats[i, :, dats.shape[2]//2, :].cpu())
                plt.axis('off')
            plt.suptitle('Registered magnitude images')
            plt.show()

        for vol, aff in zip(data + transmit + receive, affines):
            aff, vol.affine = core.utils.to_max_device(aff, vol.affine)
            vol.affine = torch.matmul(aff.inverse(), vol.affine)

    # --- repeat fields if not enough ---
    transmit = core.pyutils.make_list(transmit or [None], len(data))
    receive = core.pyutils.make_list(receive or [None], len(data))

    # --- compute recon space ---
    affines = [contrast.affine for contrast in data]
    shapes = [dat.volume.shape for dat in data]
    if opt.recon.space == 'mean':
        print('Estimate recon space')
        if isinstance(opt.recon.space, int):
            mean_affine = affines[opt.recon.space]
            mean_shape = shapes[opt.recon.space]
        elif isinstance(opt.recon.space, str) and opt.recon.space == 'mean':
            mean_affine, mean_shape = spatial.mean_space(affines, shapes)
        else:
            raise NotImplementedError()
    else:
        mean_affine = affines[opt.recon.space]
        mean_shape = shapes[opt.recon.space]

    # --- compute PD/R1 ---
    pdt1 = [(id, contrast) for id, contrast in enumerate(data) if not contrast.mt]
    if len(pdt1) > 2:
        warnings.warn('More than two volumes could be used to compute PD+R1')
    pdt1 = pdt1[:2]
    if len(pdt1) < 2:
        raise ValueError('Not enough volumes to compute PD+R1')
    (pdw_idx, pdw_struct), (t1w_idx, t1w_struct) = pdt1
    
    if t1w_struct.te != pdw_struct.te:
        warnings.warn('Echo times not consistent across volumes')
    
    print('Computing PD and R1 from volumes:')
    print(f'  - '
          f'FA = {pdw_struct.fa:2.0f} deg  /  '
          f'TR = {pdw_struct.tr*1e3:4.1f} ms /  '
          f'TE = {pdw_struct.te*1e3:4.1f} ms')
    
    pdw = load_and_pull(pdw_struct, mean_affine, mean_shape)
    pdw_fa = pdw_struct.fa / 180. * core.constants.pi
    pdw_tr = pdw_struct.tr
    if receive[pdw_idx]:
        b1m = load_and_pull(receive[pdw_idx], mean_affine, mean_shape)
        unit = receive[pdw_idx].unit
        minval = b1m[b1m > 0].min()
        maxval = b1m[b1m > 0].max()
        meanval = b1m[b1m > 0].mean()
        print(f'    with B1- map ('
              f'min= {minval:.2f}, '
              f'max = {maxval:.2f}, '
              f'mean = {meanval:.2f} {unit})')
        pdw /= b1m
        if unit in ('%', 'pct', 'p.u.'):
            pdw *= 100
        del b1m 
    if transmit[pdw_idx]:
        b1p = load_and_pull(transmit[pdw_idx], mean_affine, mean_shape)
        unit = transmit[pdw_idx].unit
        minval = b1p[b1p > 0].min()
        maxval = b1p[b1p > 0].max()
        meanval = b1p[b1p > 0].mean()
        print(f'    with B1+ map ('
              f'min= {minval:.2f}, '
              f'max = {maxval:.2f}, '
              f'mean = {meanval:.2f} {unit})')
        pdw_fa = b1p * pdw_fa
        if unit in ('%', 'pct', 'p.u.'):
            pdw_fa /= 100
        del b1p

    print(f'  - '
          f'FA = {t1w_struct.fa:2.0f} deg  /  '
          f'TR = {t1w_struct.tr*1e3:4.1f} ms /  '
          f'TE = {t1w_struct.te*1e3:4.1f} ms')
    
    t1w = load_and_pull(t1w_struct, mean_affine, mean_shape)
    t1w_fa = t1w_struct.fa / 180. * core.constants.pi
    t1w_tr = t1w_struct.tr
    if receive[t1w_idx]:
        b1m = load_and_pull(receive[t1w_idx], mean_affine, mean_shape)
        unit = receive[t1w_idx].unit
        minval = b1m[b1m > 0].min()
        maxval = b1m[b1m > 0].max()
        meanval = b1m[b1m > 0].mean()
        print(f'    with B1- map ('
              f'min= {minval:.2f}, '
              f'max = {maxval:.2f}, '
              f'mean = {meanval:.2f} {unit})')
        t1w /= b1m
        if unit in ('%', 'pct', 'p.u.'):
            t1w *= 100
        del b1m
    if transmit[pdw_idx]:
        b1p = load_and_pull(transmit[t1w_idx], mean_affine, mean_shape)
        unit = transmit[t1w_idx].unit
        minval = b1p[b1p > 0].min()
        maxval = b1p[b1p > 0].max()
        meanval = b1p[b1p > 0].mean()
        print(f'    with B1+ map ('
              f'min= {minval:.2f}, '
              f'max = {maxval:.2f}, '
              f'mean = {meanval:.2f} {unit})')
        t1w_fa = b1p * t1w_fa
        if unit in ('%', 'pct', 'p.u.'):
            t1w_fa /= 100
        del b1p
    
    r1 = 0.5 * (t1w * (t1w_fa / t1w_tr) - pdw * (pdw_fa / pdw_tr))
    r1 /= ((pdw / pdw_fa) - (t1w / t1w_fa))
    r1[~torch.isfinite(r1)] = 0

    pd = (t1w * pdw) * (t1w_tr * (pdw_fa / t1w_fa) - pdw_tr * (t1w_fa / pdw_fa))
    pd /= (pdw * (pdw_tr * pdw_fa) - t1w * (t1w_tr * t1w_fa))
    pd[~torch.isfinite(pd)] = 0

    # --- compute MTsat ---
    mtw_struct = [(id, contrast) for id, contrast in enumerate(data)
                  if contrast.mt]
    if len(mtw_struct) == 0:
        return (ParameterMap(pd, affine=mean_affine, unit=None),
                ParameterMap(r1, affine=mean_affine, unit='1/s'))

    if len(mtw_struct) > 1:
        warnings.warn('More than one volume could be used to compute MTsat')
    mtw_idx, mtw_struct = mtw_struct[0]

    print('Computing MTsat from PD/R1 maps and volume:')
    print(f'  - '
          f'FA = {mtw_struct.fa:2.0f} deg  /  '
          f'TR = {mtw_struct.tr*1e3:4.1f} ms /  '
          f'TE = {mtw_struct.te*1e3:4.1f} ms')
    
    mtw = load_and_pull(mtw_struct, mean_affine, mean_shape)
    mtw_fa = mtw_struct.fa / 180. * core.constants.pi
    mtw_tr = mtw_struct.tr
    if receive[mtw_idx]:
        b1m = load_and_pull(receive[mtw_idx], mean_affine, mean_shape)
        unit = receive[mtw_idx].unit
        minval = b1m[b1m > 0].min()
        maxval = b1m[b1m > 0].max()
        meanval = b1m[b1m > 0].mean()
        print(f'    with B1- map ('
              f'min= {minval:.2f}, '
              f'max = {maxval:.2f}, '
              f'mean = {meanval:.2f} {unit})')
        mtw /= b1m
        if unit in ('%', 'pct', 'p.u.'):
            mtw *= 100
        del b1m
    if transmit[mtw_idx]:
        b1p = load_and_pull(transmit[mtw_idx], mean_affine, mean_shape)
        unit = transmit[mtw_idx].unit
        minval = b1p[b1p > 0].min()
        maxval = b1p[b1p > 0].max()
        meanval = b1p[b1p > 0].mean()
        print(f'    with B1+ map ('
              f'min= {minval:.2f}, '
              f'max = {maxval:.2f}, '
              f'mean = {meanval:.2f} {unit})')
        mtw_fa = b1p * mtw_fa
        if unit in ('%', 'pct', 'p.u.'):
            mtw_fa /= 100
        del b1p
        
    mtsat = (mtw_fa * pd / mtw - 1) * r1 * mtw_tr - 0.5 * (mtw_fa ** 2)
    mtsat *= 100
    mtsat[~torch.isfinite(mtsat)] = 0

    return (ParameterMap(pd, affine=mean_affine, unit=None),
            ParameterMap(r1, affine=mean_affine, unit='1/s'),
            ParameterMap(mtsat, affine=mean_affine, unit='%'))


def load_and_pull(volume, aff, shape):
    """

    Parameters
    ----------
    volume : Volume3D
    aff : (D+1,D+1) tensor
    shape : (D,) tuple

    Returns
    -------
    dat : tensor

    """

    backend = dict(dtype=aff.dtype, device=aff.device)
    identity = torch.eye(aff.shape[-1], **backend)
    fdata = volume.fdata(cache=False, **backend)
    inshape = fdata.shape
    inaff = volume.affine.to(**backend)
    aff = core.linalg.lmdiv(inaff, aff)
    if torch.allclose(aff, identity) and tuple(shape) == tuple(inshape):
        return fdata
    else:
        grid = spatial.affine_grid(aff, shape)
        return spatial.grid_pull(fdata[None, None, ...], grid[None, ...])[0, 0]

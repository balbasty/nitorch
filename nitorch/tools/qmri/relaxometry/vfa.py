import warnings
import torch
from nitorch import core, spatial
from nitorch.tools.preproc import affine_align
from ..estatics._param import ParameterMap
from ..estatics._options import Options


def rational(data, transmit=None, receive=None, opt=None):
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
    opt : Options

    Returns
    -------
    pd : ParameterMap
    r1 : ParameterMap
    mt : ParameterMap, optional

    """
    if opt is None:
        opt = Options()

    data = core.pyutils.make_list(data)
    if len(data) != 2:
        raise ValueError('Expected two input images')
    transmit = core.pyutils.make_list(transmit or [])
    receive = core.pyutils.make_list(receive or [])

    # --- check TEs ---
    if len(set([contrast.te for contrast in data])) > 1:
        raise ValueError('Echo times not consistent across contrasts')

    # --- register ---
    if opt.preproc.register:
        data_reg = [(contrast.fdata(rand=True, cache=False),
                     contrast.affine) for contrast in data]
        data_reg += [(map.magnitude.fdata(rand=True, cache=False),
                      map.magnitude.affine) for map in transmit]
        data_reg += [(map.magnitude.fdata(rand=True, cache=False),
                      map.magnitude.affine) for map in receive]
        _, affines, _ = affine_align(data_reg)
        for map, aff in zip(data + transmit + receive, affines):
            map.affine = torch.matmul(aff.inverse(), map.affine)

    # --- repeat fields if not enough ---
    if transmit:
        transmit = core.pyutils.make_list(transmit, len(data))
    else:
        transmit = [None] * len(data)
    if receive:
        receive = core.pyutils.make_list(receive, len(data))
    else:
        receive = [None] * len(data)

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

    # --- compute PD/R1 ---
    pdt1 = [(id, contrast) for id, contrast in enumerate(data) if not contrast.mt]
    if len(pdt1) > 2:
        warnings.warn('More than two volumes could be used to compute PD+R1')
    pdt1 = pdt1[:2]
    if len(pdt1) < 2:
        raise ValueError('Not enough volumes to compute PD+R1')
    (pdw_idx, pdw_struct), (t1w_idx, t1w_struct) = pdt1

    pdw = load_and_pull(pdw_struct, mean_affine, mean_shape)
    pdw_fa = pdw_struct.fa
    pdw_tr = pdw_struct.tr
    if receive[pdw_idx]:
        b1m = load_and_pull(receive[pdw_idx], mean_affine, mean_shape)
        pdw /= b1m
        if receive[pdw_idx].unit in ('%', 'pct', 'p.u.'):
            pdw *= 100
        del b1m
    if transmit[pdw_idx]:
        b1p = load_and_pull(transmit[pdw_idx], mean_affine, mean_shape)
        pdw_fa = b1p * pdw_fa
        del b1p

    t1w = load_and_pull(t1w_struct, mean_affine, mean_shape)
    t1w_fa = t1w_struct.fa
    t1w_tr = t1w_struct.tr
    if receive[t1w_idx]:
        b1m = load_and_pull(receive[t1w_idx], mean_affine, mean_shape)
        t1w /= b1m
        if receive[t1w_idx].unit in ('%', 'pct', 'p.u.'):
            t1w *= 100
        del b1m
    if transmit[pdw_idx]:
        b1p = load_and_pull(transmit[t1w_idx], mean_affine, mean_shape)
        t1w_fa = b1p * t1w_fa
        del b1p

    eps = core.constants.eps(t1w.dtype)

    r1 = 0.5 * (t1w * (t1w_fa / t1w_tr) - pdw * (pdw_fa / pdw_tr))
    r1 /= ((pdw / pdw_fa) - (t1w / t1w_fa)).clamp_min_(eps)

    pd = (t1w * pdw) * (t1w_tr * (pdw_fa / t1w_fa) - pdw_tr * (t1w_fa / pdw_fa))
    pd /= (pdw * (pdw_tr * pdw_fa) - t1w * (t1w_tr * t1w_fa)).clamp_min_(eps)

    # --- compute MTsat ---
    mtw_struct = [(id, contrast) for id, contrast in enumerate(data)
                  if contrast.mt]
    if len(mtw_struct) == 0:
        return (ParameterMap(pd, affine=mean_affine, unit=None),
                ParameterMap(r1, affine=mean_affine, unit='1/s'))

    if len(mtw_struct) > 1:
        warnings.warn('More than one volume could be used to compute MTsat')
    mtw_idx, mtw_struct = mtw_struct[0]

    mtw = load_and_pull(mtw_struct, mean_affine, mean_shape)
    mtw_fa = mtw_struct.fa
    mtw_tr = mtw_struct.tr
    if receive[mtw_idx]:
        b1m = load_and_pull(receive[mtw_idx], mean_affine, mean_shape)
        mtw /= b1m
        if receive[mtw_idx].unit in ('%', 'pct', 'p.u.'):
            mtw *= 100
        del b1m
    if transmit[mtw_idx]:
        b1p = load_and_pull(transmit[mtw_idx], mean_affine, mean_shape)
        mtw_fa = b1p * mtw_fa
        del b1p

    mtsat = (mtw_fa * pd / mtw - 1) * r1 * mtw_tr - 0.5 * (mtw_fa ** 2)
    mtsat *= 100

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

    inshape = volume.shape
    backend = dict(dtype=aff.dtype, device=aff.device)
    identity = torch.eye(aff.shape[-1], **backend)
    fdata = volume.fdata(cache=False)
    if torch.allclose(aff, identity) and shape == inshape:
        return fdata
    else:
        grid = spatial.affine_grid(aff, shape)
        return spatial.grid_pull(fdata[None, None, ...], grid[None, ...])[0]

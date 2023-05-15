"""Functions for data pre-processing.

"""


from .affine_reg._align import (_affine_align, _atlas_align)
from ._preproc_fov import (_atlas_crop, _reset_origin, _subvol)
from ._preproc_img import (_world_reslice, _reslice2ref)
from ._preproc_utils import (_format_input, _process_reg, _write_output)


def atlas_crop(img, write=False, nam=None, odir=None, prefix='ac_',
               device='cpu', do_align=True, fov='head', mat_a=None):
    """Crop an image to the NITorch T1w brain atlas' field-of-view.

    Parameters
    ----------
    img : str or [tensor_like, tensor_like]
        Either path to an image file, or a list of two tensors
        containing the image data (X, Y, Z) and the image affine matrix
        (4, 4).
    write : bool, default=False
        Write preprocessed data to disk.
    nam = str, optional
        Output filename, if None, uses same as input.
    odir = str, optional
        Output directory, if None, uses same as input.
    prefix = str, default='ac_'
        Output prefix.
    device : torch.device, default='cpu'
        Output device, only used if input are paths. If input are tensors,
        then the device of those tensors will be used.
    do_align : bool, default=True
        Do alignment to MNI space.
    fov : str, default='head'
        * 'head' : Head FOV.
        * 'brain' : Brain FOV.
    mat_a : (4, 4) tensor_like, dtype=float64, optional
        Pre-computed atlas alignment affine matrix.

    Returns
    -------
    dat : (X1, Y1, Z1) tensor_like, dtype=float32
        Preprocessed image data.
    mat : (4, 4) tensor_like, dtype=float64
        New affine matrix.
    pth : str
        Paths to preprocessed data (only if write=True).

    """
    # Sanity check
    if fov not in ['brain', 'head']:
        raise ValueError('Option fov should be one of: brain, head')
    # Get properly formatted function input
    dat, mat, file = _format_input(img, device=device)
    if len(dat) != 1:
        raise ValueError('Only one input image should be given!')
    # Do preprocessing
    dat[0], mat[0] = _atlas_crop(dat[0], mat[0], do_align=do_align,
                                 fov=fov, mat_a=mat_a)
    # Possibly write output to disk
    pth = None
    if write:
        pth = _write_output(dat, mat, file=file, nam=nam, odir=odir,
                            prefix=prefix)

    return dat[0], mat[0], pth


def affine_align(img, write=None, nam=None, odir=None, prefix='aa_',
                 device='cpu', cost_fun='nmi', samp=(3, 1.5),
                 mean_space=False, group='SE', fix=0, fwhm=7.0,
                 verbose=False):
    """Affinely align images.

    This function aligns N images affinely, either pairwise or groupwise,
    by non-gradient based optimisation. An affine transformation has maximum
    12 parameters that control: translation, rotation, scaling and shearing.
    It is a linear registration.

    The transformation model is:
        mat_mov\mat_a*mat_fix,
    where mat_mov is the affine matrix of the source/moving image, mat_fix is the affine matrix
    of the fixed/target image, and mat_a is the affine matrix that we
    optimise to align voxels
    in the source and target image. The affine matrix is represented by its Lie algebra (q),
    which a vector with as many parameters as the affine transformation.

    When running the algorithm in a pair-wise setting (i.e., the cost-function takes as input two images),
    one image is set as fixed, and all other images are registered to this fixed image. When running the
    algorithm in a groupwese setting (i.e., the cost-function takes as input all images) two options are
    available:
        1. One of the input images are used as the fixed image and all others are aligned to this image.
        2. A mean-space is defined, containing all of the input images, and all images are optimised
           to aling in this mean-space.

    At the end the affine transformation is returned, together with the defintion of the fixed space, and
    the affine group used. For getting the data resliced to the fixed space, look at the function reslice2fix().
    To adjust the affine in the image header, look at apply2affine().

    The registration methods used here are described in:
    A Collignon, F Maes, D Delaere, D Vandermeulen, P Suetens & G Marchal (1995)
    "Automated Multi-modality Image Registration Based On Information Theory", IPMI
    and:
    M Brudfors, Y Balbastre, J Ashburner (2020) "Groupwise Multimodal Image
    Registration using Joint Total Variation", MIUA

    Parameters
    ----------
    img : [N, ]
        Either: list of paths to N image files; or list of N lists,
        where each element contains two tensors: the image data (X, Y, Z)
        and the image affine matrix (4, 4).
    write : str, default=None
        Write preprocessed data to disk.
        * 'reslice' : writes image data resliced to fixed image.
        * 'affine' : writes original image data with affine modified.
    nam = str, optional
        Output filename, if None, uses same as input.
    odir = str, optional
        Output directory, if None, uses same as input.
    prefix = str, default='aa_'
        Output prefix.
    device : torch.device, default='cpu'
        Output device, only used if input are paths. If input are tensors,
        then the device of those tensors will be used.
    cost_fun : str, default='nmi'
        * 'nmi' : Normalised Mutual Information (pairwise method)
        * 'mi' : Mutual Information (pairwise method)
        * 'ncc' : Normalised Cross Correlation (pairwise method)
        * 'ecc' : Entropy Correlation Coefficient (pairwise method)
        * 'njtv' : Normalised Joint Total variation (groupwise method)
        * 'jtv' : Joint Total variation (groupwise method)
    group : str, default='SE'
        * 'T'   : Translations
        * 'SO'  : Special Orthogonal (rotations)
        * 'SE'  : Special Euclidean (translations + rotations)
        * 'D'   : Dilations (translations + isotropic scalings)
        * 'CSO' : Conformal Special Orthogonal
                  (translations + rotations + isotropic scalings)
        * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
        * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
        * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
    mean_space : bool, default=False
        Optimise a mean-space fit, only available if cost_fun='njtv'.
    samp : (float, ), default=(3, 1.5)
    fix : int, default=0
        Index of image to used as fixed image, not used if mean_space=True.
    verbose : bool, default=False
        Show registration results.

    Returns
    -------
    rdat : (N, X1, Y1, Z1) tensor_like, dtype=float32
        Preprocessed image data.
    mat_a : (N, 4, 4) tensor_like, dtype=float64
        Affine matrices aligning as mat_mov\mat_a*mat_fix.
    pth : [str, ...]
        Paths to preprocessed data (only if write=True).

    """
    # Get properly formatted function input
    dat, mat, file = _format_input(img, device=device, rand=True,
                                   cutoff=(0.0005, 0.9995))
    # Do preprocessing
    mat_a, mat_fix, dim_fix, _ = _affine_align(dat, mat,
        samp=samp, cost_fun=cost_fun, mean_space=mean_space,
        group=group, fix=fix, verbose=verbose, fwhm=fwhm)
    # Get original data
    dat = _format_input(img, device=device)[0]
    # Process registration results
    dat, mat, write, rdat = _process_reg(
        dat, mat, mat_a, mat_fix, dim_fix, write)
    # Possibly write output to disk
    pth = None
    if write:
        pth = _write_output(dat, mat, file=file, nam=nam, odir=odir,
                            prefix=prefix)

    return rdat, mat_a, pth


def atlas_align(img, rigid=True, write=None, nam=None, odir=None, prefix='aa_',
                device='cpu', pth_atlas=None, default_atlas='atlas_t1'):
    """Affinely align an image to some atlas space.

    Parameters
    ----------
    img : [N, ]
        Either: list of paths to N image files; or list of N lists,
        where each element contains two tensors: the image data (X, Y, Z)
        and the image affine matrix (4, 4).
    rigid = bool, default=True
        Do rigid alignment, else does rigid+isotropic scaling.
    write : str, default=None
        Write preprocessed data to disk.
        * 'reslice' : writes image data resliced to fixed image.
        * 'affine' : writes original image data with affine modified.
    nam = str, optional
        Output filename, if None, uses same as input.
    odir = str, optional
        Output directory, if None, uses same as input.
    prefix = str, default='aa_'
        Output prefix.
    device : torch.device, default='cpu'
        Output device, only used if input are paths. If input are tensors,
        then the device of those tensors will be used.
    pth_atlas : str, optional
        Path to atlas image to match to. Uses Brain T1w atlas by default.
    default_atlas : str, optional
        Name of an atlas available in the collection of default nitorch atlases.

    Returns
    -------
    rdat : (N, X1, Y1, Z1) tensor_like, dtype=float32
        Preprocessed image data.
    mat_a : (4, 4) tensor_like, dtype=float64
        Affine matrix aligning as mat_mov\mat_a*mat_fix.
    pth : str
        Paths to preprocessed data (only if write=True).
    mat_cso : (4, 4) tensor_like
        CSO transformation.

    """
    # Get properly formatted function input
    dat, mat, file = _format_input(img, device=device, rand=True,
                                   cutoff=(0.0005, 0.9995))
    if len(dat) != 1:
        raise ValueError('Only one input image should be given!')
    # Do preprocessing
    mat_a, mat_fix, dim_fix, mat_cso = _atlas_align(dat, mat, rigid=rigid,
                                                    pth_atlas=pth_atlas, default_atlas=default_atlas)
    # Get original data
    dat = _format_input(img, device=device)[0]
    # Process registration results
    dat, mat, write, rdat = _process_reg(
        dat, mat, mat_a, mat_fix, dim_fix, write)
    # Possibly write output to disk
    pth = None
    if write:
        pth = _write_output(dat, mat, file=file, nam=nam, odir=odir,
                            prefix=prefix)

    return rdat[0, ...], mat_a[0, ...], pth, mat_cso[0, ...]


def reset_origin(img, write=False, nam=None, odir=None, prefix='ro_',
                 device='cpu', interpolation=1):
    """Reset affine matrix.

    Parameters
    ----------
    img : str or [tensor_like, tensor_like]
        Either path to an image file, or a list of two tensors
        containing the image data (X, Y, Z) and the image affine matrix
        (4, 4).
    write : bool, default=False
        Write preprocessed data to disk.
    nam = str, optional
        Output filename, if None, uses same as input.
    odir = str, optional
        Output directory, if None, uses same as input.
    prefix = str, default='ro_'
        Output prefix.
    device : torch.device, default='cpu'
        Output device, only used if input are paths. If input are tensors,
        then the device of those tensors will be used.
    interpolation : int, default=1 (linear)
        Interpolation order.

    Returns
    -------
    dat : (X1, Y1, Z1) tensor_like, dtype=float32
        Preprocessed image data.
    mat : (4, 4) tensor_like, dtype=float64
        New affine matrix.
    pth : str
        Paths to preprocessed data (only if write=True).

    """
    # Get properly formatted function input
    dat, mat, file = _format_input(img, device=device)
    if len(dat) != 1:
        raise ValueError('Only one input image should be given!')
    # Do preprocessing
    dat[0], mat[0] = _reset_origin(dat[0], mat[0], interpolation=interpolation)
    # Possibly write output to disk
    pth = None
    if write:
        pth = _write_output(dat, mat, file=file, nam=nam, odir=odir,
                            prefix=prefix)

    return dat[0], mat[0], pth


def subvol(img, bb=None, write=False, nam=None, odir=None, prefix='sv_',
           device='cpu'):
    """Extract a sub-volume.

    Parameters
    ----------
    img : str or [tensor_like, tensor_like]
        Either path to an image file, or a list of two tensors
        containing the image data (X, Y, Z) and the image affine matrix
        (4, 4).
    bb : (2, 3) sequence, optional
        Bounding box, where [[x0, y0, z0], [x1, y1, z1]] modifies an
        image's dimensions as:
            [[1 + x0, 1 + y0, 1 + z0],
             [dm[0] - x1, dm[1] - y1, dm[2] - z1]]
         By default, the bounding box is the same as the input image's
         dimensions.
    write : bool, default=False
        Write preprocessed data to disk.
    nam = str, optional
        Output filename, if None, uses same as input.
    odir = str, optional
        Output directory, if None, uses same as input.
    prefix = str, default='sv_'
        Output prefix.
    device : torch.device, default='cpu'
        Output device, only used if input are paths. If input are tensors,
        then the device of those tensors will be used.

    Returns
    -------
    dat : (X1, Y1, Z1) tensor_like, dtype=float32
        Preprocessed image data.
    mat : (4, 4) tensor_like, dtype=float64
        New affine matrix.
    pth : str
        Paths to preprocessed data (only if write=True).

    """
    # Get properly formatted function input
    dat, mat, file = _format_input(img, device=device)
    if len(dat) != 1:
        raise ValueError('Only one input image should be given!')
    # Do preprocessing
    dat[0], mat[0] = _subvol(dat[0], mat[0], bb=bb)
    # Possibly write output to disk
    pth = None
    if write:
        pth = _write_output(dat, mat, file=file, nam=nam, odir=odir,
                            prefix=prefix)

    return dat[0], mat[0], pth


def reslice2ref(img, write=False, nam=None, odir=None, prefix='r_',
                device='cpu', interpolation=1, ref=0):
    """Reslice images to reference.

    Parameters
    ----------
    img : str or [tensor_like, tensor_like]
        Either paths to image files, or a list of lists with two tensors
        containing the image data (X, Y, Z) and the image affine matrix
        (4, 4).
    write : bool, default=False
        Write preprocessed data to disk.
    nam = str, optional
        Output filename, if None, uses same as input.
    odir = str, optional
        Output directory, if None, uses same as input.
    prefix = str, default='r_'
        Output prefix.
    device : torch.device, default='cpu'
        Output device, only used if input are paths. If input are tensors,
        then the device of those tensors will be used.
    interpolation : int, default=1 (linear)
        Interpolation order.
    ref : int, default=0
        Index of reference image.

    Returns
    -------
    dat : (C, X1, Y1, Z1) tensor_like, dtype=float32
        Resliced image data.
    mat : (4, 4) tensor_like, dtype=float64
        Reference affine matrix.
    pth : str
        Paths to resliced images (only if write=True).

    """
    # Get properly formatted function input
    dat, mat, file = _format_input(img, device=device)
    if len(dat) < 2:
        raise ValueError('More than one image should be given!')
    # Do preprocessing
    dat, mat = _reslice2ref(dat, mat, interpolation=interpolation, ref=ref)
    # Possibly write output to disk
    pth = None
    if write:
        pth = _write_output(dat, mat, file=file, nam=nam, odir=odir,
                            prefix=prefix)

    return dat, mat, pth


def world_reslice(img, write=False, nam=None, odir=None, prefix='wr_',
                  device='cpu', interpolation=1, vx=None):
    """Reslice image data to world space.

    Parameters
    ----------
    img : str or [tensor_like, tensor_like]
        Either path to an image file, or a list of two tensors
        containing the image data (X, Y, Z) and the image affine matrix
        (4, 4).
    write : bool, default=False
        Write preprocessed data to disk.
    nam = str, optional
        Output filename, if None, uses same as input.
    odir = str, optional
        Output directory, if None, uses same as input.
    prefix = str, default='wr_'
        Output prefix.
    device : torch.device, default='cpu'
        Output device, only used if input are paths. If input are tensors,
        then the device of those tensors will be used.
    interpolation : int, default=1 (linear)
        Interpolation order.
    vx : float | [float,] *3, optional
        Output voxel size.

    Returns
    -------
    dat : (X1, Y1, Z1) tensor_like, dtype=float32
        Preprocessed image data.
    mat : (4, 4) tensor_like, dtype=float64
        New affine matrix.
    pth : str
        Paths to preprocessed data (only if write=True).

    """
    # Get properly formatted function input
    dat, mat, file = _format_input(img, device=device)
    if len(dat) != 1:
        raise ValueError('Only one input image should be given!')
    # Do preprocessing
    dat[0], mat[0] = _world_reslice(dat[0], mat[0], interpolation=interpolation, vx=vx)
    # Possibly write output to disk
    pth = None
    if write:
        pth = _write_output(dat, mat, file=file, nam=nam, odir=odir,
                            prefix=prefix)

    return dat[0], mat[0], pth

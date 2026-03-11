"""Self-contained registration pipeline returning in-memory tensors."""

import torch
from nitorch import spatial


def register(fixed_dat, fixed_affine, moving_dat, moving_affine, loss_name,
             is_label=False, affine_basis='similitude', reg_lambda=5,
             penalty_absolute=0.0001, penalty_membrane=0.001,
             penalty_bending=0.2, penalty_lame=(0.05, 0.2),
             aff_max_iter=128, aff_tolerance=1e-4,
             nl_max_iter=64, nl_tolerance=1e-3,
             outer_max_iter=64, outer_tolerance=1e-3,
             pyramid_levels=None, device='cpu', verbose=False):
    """Run affine + nonlinear (SVF) registration.

    All inputs are in-memory tensors — no file I/O.

    Parameters
    ----------
    fixed_dat : torch.Tensor
        (C, *spatial) or (*spatial) fixed image data.
    fixed_affine : torch.Tensor
        (4, 4) voxel-to-world affine of the fixed image.
    moving_dat : torch.Tensor
        (C, *spatial) or (*spatial) moving image data.
    moving_affine : torch.Tensor
        (4, 4) voxel-to-world affine of the moving image.
    loss_name : str
        Loss function name ('lcc', 'mse', 'nmi', 'dice').
    is_label : bool
        Whether inputs are label maps.
    affine_basis : str
        Affine basis: 'translation', 'rotation', 'rigid', 'similitude', 'affine'.
    reg_lambda : float
        Regularization weight for nonlinear registration.
    aff_max_iter : int
        Max iterations for affine optimizer (per pyramid level).
    aff_tolerance : float
        Convergence tolerance for affine optimizer.
    nl_max_iter : int
        Max iterations for nonlinear optimizer (per pyramid level).
    nl_tolerance : float
        Convergence tolerance for nonlinear optimizer.
    outer_max_iter : int
        Max outer iterations for interleaved affine/nonlinear optimization.
    outer_tolerance : float
        Convergence tolerance for outer interleaved optimization.
    pyramid_levels : list of int or None
        Pyramid levels (default: [0, 1, 2]).
    device : str
        Torch device ('cpu' or 'cuda:N').
    verbose : bool
        Print progress information.

    Returns
    -------
    affine_sqrt : torch.Tensor
        (4, 4) square root of the affine transform.
    displacement : torch.Tensor
        (*spatial, 3) displacement field (exponentiated SVF).
    disp_affine : torch.Tensor
        (4, 4) voxel-to-world affine of the displacement field.
    """
    from nitorch.tools.registration.pairwise_preproc import preproc_image
    from nitorch.tools.registration.pairwise_makeobj import (
        make_image, make_loss, make_affine, make_nonlin,
        make_affine_optim, make_nonlin_optim,
    )
    from nitorch.tools.registration.pairwise_pyramid import sequential_pyramid
    from nitorch.tools.registration.pairwise_run import run
    from nitorch.tools.registration import objects

    if pyramid_levels is None:
        pyramid_levels = [0, 1, 2]

    # Ensure channel dimension: (C, *spatial) and move to device
    if fixed_dat.dim() == 3:
        fixed_dat = fixed_dat.unsqueeze(0)
    if moving_dat.dim() == 3:
        moving_dat = moving_dat.unsqueeze(0)
    fixed_dat = fixed_dat.to(device=device, dtype=torch.float32)
    moving_dat = moving_dat.to(device=device, dtype=torch.float32)
    fixed_affine = fixed_affine.to(device=device)
    moving_affine = moving_affine.to(device=device)

    # Rescale parameters
    if loss_name in ('cat', 'dice'):
        rescale = (0, 0)
    else:
        rescale = (0, 95)

    # Preprocess — pass tensors directly, world= overrides affine
    fix_dat, fix_mask, fix_aff = preproc_image(
        fixed_dat, label=is_label, rescale=rescale,
        world=fixed_affine,
        fwhm=1, bound='dct2', dim=3, device=device,
    )
    mov_dat, mov_mask, mov_aff = preproc_image(
        moving_dat, label=is_label, rescale=rescale,
        world=moving_affine,
        fwhm=1, bound='dct2', dim=3, device=device,
    )

    # Create image pyramids
    fix_img = make_image(
        fix_dat, mask=fix_mask, affine=fix_aff,
        pyramid=pyramid_levels, pyramid_method='gaussian',
        bound='dct2', extrapolate=False,
    )
    mov_img = make_image(
        mov_dat, mask=mov_mask, affine=mov_aff,
        pyramid=pyramid_levels, pyramid_method='gaussian',
        bound='dct2', extrapolate=False,
    )

    # Loss
    loss_kwargs = {}
    if loss_name in ('lcc', 'lncc'):
        loss_kwargs = {'patch': 10, 'stride': 1, 'kernel': 'g'}
    elif loss_name in ('nmi', 'mi'):
        loss_kwargs = {'bins': 32, 'fwhm': 0, 'norm': 'studholme'}
    loss_obj = make_loss(loss_name, **loss_kwargs)

    # Similarity and pyramid
    similarity = objects.Similarity(loss_obj, mov_img, fix_img)
    loss_list = sequential_pyramid(similarity)

    # Affine model
    affine_model = make_affine(basis=affine_basis, position='symmetric')

    # Nonlinear model (SVF)
    images = list(loss_list[-1].images())
    penalty = {
        'absolute': penalty_absolute,
        'membrane': penalty_membrane,
        'bending': penalty_bending,
        'lame': penalty_lame,
    }
    nonlin_model = make_nonlin(
        images, 'svf', factor=reg_lambda, penalty=penalty,
        voxel_size=[100, '%'], device=device,
    )

    # Optimizers
    order = loss_obj.order
    optim_name = 'gn' if order >= 2 else 'lbfgs'
    affine_optim = make_affine_optim(
        optim_name, order, max_iter=aff_max_iter, tolerance=aff_tolerance,
    )
    nonlin_optim = make_nonlin_optim(
        optim_name, order, max_iter=nl_max_iter, tolerance=nl_tolerance,
        nonlin=nonlin_model,
    )

    # Run registration
    verbose_level = 1 if verbose else 0
    affine_model, nonlin_model = run(
        loss_list, affine=affine_model, nonlin=nonlin_model,
        affine_optim=affine_optim, nonlin_optim=nonlin_optim,
        pyramid=True, interleaved=True, progressive=False,
        max_iter=outer_max_iter, tolerance=outer_tolerance,
        verbose=verbose_level, framerate=0,
    )

    # Extract affine square root
    affine_sqrt = affine_model.exp(cache_result=True, recompute=True)

    # Extract and exponentiate SVF
    svf_dat = nonlin_model.dat.dat
    if svf_dat.dim() == 4:
        svf_dat = svf_dat.unsqueeze(0)
    displacement = spatial.exp(svf_dat, displacement=True).squeeze(0)
    disp_affine = nonlin_model.affine

    return affine_sqrt, displacement, disp_affine

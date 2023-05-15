"""Plotting utilities for multi-dimensional tensors.

TODO
* Real-time plotting slows down with number of calls to show_slices!

"""

import torch
from nitorch import spatial
from nitorch.core import utils, py, math, linalg
from nitorch.core.optionals import try_import
import math as pymath
# Try import matplotlib
plt = try_import('matplotlib.pyplot', _as=True)
gridspec = try_import('matplotlib.gridspec', _as=True)
make_axes_locatable = try_import(
    'mpl_toolkits.axes_grid1', keys='make_axes_locatable', _as=False)
from . import colormaps as cmap


def get_slice(image, dim=-1, index=None):
    """Extract a 2d slice from a 3d volume

    Parameters
    ----------
    image : (..., *shape3) tensor
        A (batch of) 3d volume
    dim : int, default=-1
        Index of the spatial dimension to slice
    index : int, default=shape//2
        Coordinate (in voxel) of the slice to extract

    Returns
    -------
    slice : (..., *shape2) tensor
        A (batch of) 2d slice

    """
    image = torch.as_tensor(image)
    if index is None:
        index = image.shape[dim] // 2
    return utils.slice_tensor(image, index, dim=dim)


def get_orthogonal_slices(image, index=None):
    """Extract orthogonal 2d slices from a 3d volume.

    Parameters
    ----------
    image : (..., *shape3) tensor
        A (batch of) 3d volume
    index : [sequence of] int, default=shape//2
        Coordinate (in voxel) of the slice to extract

    Returns
    -------
    slices : tuple or (..., *shape2) tensor
        Three (batch of) 2d slices

    """
    image = torch.as_tensor(image)
    dim = [-1, -2, -3]
    index = py.make_list(index, 3)
    index = [idx if idx is not None else image.shape[d] // 2
             for d, idx in zip(dim, index)]

    return tuple(get_slice(image, d, idx)
                 for d, idx in zip(dim, index))


def _get_default_native(affines, shapes):
    """Get default native space

    Parameters
    ----------
    affines : [sequence of] (D+1, D+1) tensor_like or None
    shapes : [sequence of] (D,) tensor_like

    Returns
    -------
    affines : (N, D+1, D+1) tensor
    shapes : (N, D) tensor

    """
    shapes = utils.as_tensor(shapes)
    ndim = shapes.shape[-1]
    shapes = shapes.reshape([-1, ndim])
    shapes = shapes.unbind(dim=0)
    if torch.is_tensor(affines):
        affines = affines.reshape([-1, ndim+1, ndim+1])
        affines = affines.unbind(dim=0)
    shapes = py.make_list(shapes, max(len(shapes), len(affines)))
    affines = py.make_list(affines, max(len(shapes), len(affines)))

    # default affines
    affines = [spatial.affine_default(shape) if affine is None else affine
               for shape, affine in zip(shapes, affines)]

    affines = utils.as_tensor(affines)
    shapes = utils.as_tensor(shapes)
    affines, shapes = utils.to_max_device(affines, shapes)
    return affines, shapes


def _get_default_space(affines, shapes, space=None, bbox=None):
    """Get default visualisation space

    Parameters
    ----------
    affines : [sequence of] (D+1, D+1) tensor_like
        Orientation matrices of all input images
    shapes : [sequence of] (D,) tensor_like
        Shapes of all input images
    space : (D+1, D+1) tensor_like, optional
    bbox : (2, D) tensor_like, optional
        Bounding box: min and max coordinates (in millimetric
        visualisation space). Default: bounding box of the input image.

    Returns
    -------
    space : (D+1, D+1) tensor
    mn : (D,) tensor
    mx : (D,) tensor

    """
    affines, shapes = _get_default_native(affines, shapes)

    if space is None:
        voxel_size = spatial.voxel_size(affines)
        voxel_size = voxel_size.min()
        ndim = shapes.shape[-1]
        space = torch.eye(ndim+1)
        space[:-1, :-1] *= voxel_size

    if bbox is None:
        shapes = torch.as_tensor(shapes)
        mn, mx = spatial.compute_fov(space, affines, shapes)
    else:
        voxel_size = spatial.voxel_size(space)
        mn, mx = utils.as_tensor(bbox)
        mn /= voxel_size
        mx /= voxel_size

    return space, mn, mx


def get_oriented_slice(image, dim=-1, index=None, affine=None,
                       space=None, bbox=None, interpolation=1,
                       transpose_sagittal=False, return_index=False,
                       return_mat=False):
    """Sample a slice in a RAS system

    Parameters
    ----------
    image : (..., *spatial) tensor
    dim : int, default=-1
        Index of spatial dimension to sample in the visualization space
        If RAS: -1 = axial / -2 = coronal / -3 = sagittal
    index : int, default=shape//2
        Coordinate (in voxel) of the slice to extract
    affine : (D+1, D+1) tensor, optional
        Orientation matrix of the image
    space : (D+1, D+1) tensor, optional
        Orientation matrix of the visualisation space.
        Default: RAS with minimum voxel size of all inputs.
    bbox : (2, D) tensor_like, optional
        Bounding box: min and max coordinates (in millimetric
        visualisation space). Default: bounding box of the input image.
    interpolation : {0, 1}, default=1
        Interpolation order.

    Returns
    -------
    slice : (..., *shape2) tensor
        Slice in the visualisation space.

    """
    # preproc dim
    if isinstance(dim, str):
        dim = dim.lower()[0]
        if dim == 'a':
            dim = -1
        if dim == 'c':
            dim = -2
        if dim == 's':
            dim = -3

    backend = utils.backend(image)

    # guess number of spatial dimensions
    if affine is not None:
        ndim = affine.shape[-1] - 1
    elif space is not None:
        ndim = space.shape[-1] - 1
    elif bbox is not None:
        bbox = torch.as_tensor(bbox)
        ndim = bbox.shape[-1]
    else:
        ndim = 3

    # compute default space (mn/mx are in voxels)
    affine, shape = _get_default_native(affine, image.shape[-ndim:])
    space, mn, mx = _get_default_space(affine, [shape], space, bbox)
    affine, shape = (affine[0], shape[0])

    # compute default cursor (in voxels)
    if index is None:
        index = (mx + mn) / 2
    else:
        index = torch.as_tensor(index)
        index = spatial.affine_matvec(spatial.affine_inv(space), index)

    # include slice to volume matrix
    shape = tuple(((mx-mn) + 1).round().int().tolist())
    if ndim == 2:
        shift = [[1, 0, - mn[0] + 1],
                 [0, 1, - mn[1] + 1],
                 [0, 0, 1]]
        shift = utils.as_tensor(shift, **backend)
        index = (index[0] - mn[0] + 1, index[1] - mn[1] + 1)
    elif dim == -1:
        # axial
        shift = [[1, 0, 0, - mn[0] + 1],
                 [0, 1, 0, - mn[1] + 1],
                 [0, 0, 1, - index[2]],
                 [0, 0, 0, 1]]
        shift = utils.as_tensor(shift, **backend)
        shape = shape[:-1]
        index = (index[0] - mn[0] + 1, index[1] - mn[1] + 1)
    elif dim == -2:
        # coronal
        shift = [[1, 0, 0, - mn[0] + 1],
                 [0, 0, 1, - mn[2] + 1],
                 [0, 1, 0, - index[1]],
                 [0, 0, 0, 1]]
        shift = utils.as_tensor(shift, **backend)
        shape = (shape[0], shape[2])
        index = (index[0] - mn[0] + 1, index[2] - mn[2] + 1)
    elif dim == -3:
        # sagittal
        if not transpose_sagittal:
            shift = [[0, 0, 1, - mn[2] + 1],
                     [0, 1, 0, - mn[1] + 1],
                     [1, 0, 0, - index[0]],
                     [0, 0, 0, 1]]
            shift = utils.as_tensor(shift, **backend)
            shape = (shape[2], shape[1])
            index = (index[2] - mn[2] + 1, index[1] - mn[1] + 1)
        else:
            shift = [[0, -1, 0,   mx[1] + 1],
                     [0,  0, 1, - mn[2] + 1],
                     [1,  0, 0, - index[0]],
                     [0,  0, 0, 1]]
            shift = utils.as_tensor(shift, **backend)
            shape = (shape[1], shape[2])
            index = (mx[1] + 1 - index[1], index[2] - mn[2] + 1)
    else:
        raise ValueError(f'Unknown dimension {dim}')

    # sample
    space = spatial.affine_rmdiv(space, shift)
    affine = spatial.affine_lmdiv(affine, space)
    affine = affine.to(**backend)
    grid = spatial.affine_grid(affine, [*shape] + ([] if ndim == 2 else [1]))
    imshape = image.shape[-ndim:]
    channel = image.shape[:-ndim]
    image = image.reshape([1, -1, *imshape])
    image = spatial.grid_pull(image, grid[None], interpolation=interpolation,
                              bound='dct2', extrapolate=False)
    image = image.reshape([*channel, *shape])
    return ((image, index, space) if return_index and return_mat else
            (image, index) if return_index else
            (image, space) if return_mat else image)


def get_orthogonal_oriented_slices(image, index=None, affine=None,
                                   space=None, bbox=None, interpolation=1,
                                   transpose_sagittal=False, return_index=False,
                                   return_mat=False):
    """Sample orthogonal slices in a RAS system

    Parameters
    ----------
    image : (..., *shape3)
        Input image
    index : (3,) sequence or tensor, default=shape//2
        Coordinate (in voxel) of the slice to extract
    affine : (4, 4) tensor, optional
        Orientation matrix of the image
    space : (4, 4) tensor, optional
        Orientation matrix of the visualisation space.
        Default: RAS with minimum voxel size of all inputs.
    bbox : (2, D) tensor_like, optional
        Bounding box: min and max coordinates (in millimetric
        visualisation space). Default: bounding box of the input image.
    interpolation : {0, 1}, default=1
        Interpolation order.

    Returns
    -------
    slice : tuple of (..., *shape2) tensor
        Slices in the visualisation space.

    """
    # guess number of spatial dimensions
    if affine is not None:
        affine = torch.as_tensor(affine)
        ndim = affine.shape[-1] - 1
    elif space is not None:
        space = torch.as_tensor(space)
        ndim = space.shape[-1] - 1
    elif bbox is not None:
        bbox = torch.as_tensor(bbox)
        ndim = bbox.shape[-1]
    else:
        ndim = 3

    shape = image.shape[-ndim:]
    if affine is None:
        affine = spatial.affine_default(shape)
    affine = torch.as_tensor(affine)

    # compute default space (mn/mx are in voxels)
    affines = affine.reshape([-1, ndim+1, ndim+1])
    shapes = [shape] * len(affines)
    space, mn, mx = _get_default_space(affines, shapes, space, bbox)
    voxel_size = spatial.voxel_size(space)
    mn *= voxel_size
    mx *= voxel_size

    prm = {
        'index': index,
        'affine': affine,
        'space': space,
        'bbox': [mn, mx],
        'interpolation': interpolation,
        'transpose_sagittal': transpose_sagittal,
        'return_index': return_index,
        'return_mat': return_mat
    }
    if ndim == 2:
        slices = (get_oriented_slice(image, **prm),)
    else:
        slices = tuple(get_oriented_slice(image, dim=d, **prm)
                       for d in (-1, -2, -3))
    if return_index and return_mat:
        return (tuple(sl[0] for sl in slices),
                tuple(sl[1] for sl in slices),
                tuple(sl[2] for sl in slices))
    elif return_index or return_mat:
        return (tuple(sl[0] for sl in slices),
                tuple(sl[1] for sl in slices))
    else:
        return slices


def show_orthogonal_slices(image, index=None, affine=None, fig=None,
                           colormap=None, layout='row', mode='intensity',
                           interpolation=1, eq=None, clim=None, show_cursor=False,
                           return_mat=False, blit=False, show=True, **kwargs):
    """Show three orthogonal slices

    Parameters
    ----------
    image : (..., [K], X, Y, Z) tensor
        Input image.
    index : (float, float, float), optional
        Cursor index, in world millimetric space
    affine : (4, 4) tensor, optional
        Input orientation matrix.
    fig, [grid] : (int or Figure, [SubplotSpec]) or list[plt.Axes], optional
        Figure in which to draw. Default: create a new figure.
        If a list of axes, each view is drawn in its provided axis.
    colormap : str or (N, 3) tensor, optional
        Colormap. Default depends on `mode`.
    layout : {'row', 'col', 'orth'}
        'row' : views in a row
        'col' : views in a columns
        'orth' : orthogonal views
    mode : {'intensity', 'categorical', 'displacement'}, default='intensity'
        Type of data to plot.
    interpolation : {0, 1}, default=1
        Interpolation order
    eq : {'linear', 'quadratic', 'log', None} or float, default=None
        Histogram equalization method.
        If float, the signal is taken to that power before being equalized.
    show_cursor : bool or float, default=False
        Show the cursor. If float, line size.

    Returns
    -------
    fig : plt.Figure
    ax : list[plt.Axes]
    mats : list[tensor], optional

    """

    def transpose(x):
        return utils.movedim(x, [-2, -3], [-3, -2])

    def reset_extent(ax):
        # https://stackoverflow.com/questions/7433585
        im = ax.images[0]
        shape = im.get_array().shape

        if im.origin == 'upper':
            im.set_extent((-0.5, shape[0] - .5, shape[1] - .5, -.5))
            ax.set_xlim((-0.5, shape[0] - .5))
            ax.set_ylim((shape[1] - .5, -.5))
        else:
            im.set_extent((-0.5, shape[0] - .5, -.5, shape[1] - .5))
            ax.set_xlim((-0.5, shape[0] - .5))
            ax.set_ylim((-.5, shape[1] - .5))

    def show_curs(ax, index, width, blit=False):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x, y = index
        x = (xlim[1] + xlim[0])/2 if x is None else x
        y = (ylim[1] + ylim[0])/2 if y is None else y
        if blit:
            ax.lines[0].set_data((xlim, [y, y]))
            ax.lines[1].set_data(([x, x], ylim,))
        else:
            linex = ax.add_line(plt.Line2D(xlim, [y, y], color='r'))
            liney = ax.add_line(plt.Line2D([x, x], ylim, color='r'))
            if not isinstance(width, bool):
                linex.set_linewidth(width)
                liney.set_linewidth(width)

    if not plt:
        return

    if affine is not None:
        affine = affine.cpu().float()

    # get slices
    image = image.detach()
    slices, index, mat = get_orthogonal_oriented_slices(
        image, index=index, affine=affine, interpolation=interpolation,
        return_index=True, return_mat=True,
        transpose_sagittal=not layout.startswith('orth'), **kwargs)
    ndim = mat[0].shape[-1] - 1

    print('index', index)
    print('shape', [s.shape for s in slices])

    # process intensities
    if mode == 'intensity':
        cmin, cmax = py.make_list(clim, 2)
        slices = cmap.intensity_preproc(*slices, eq=eq, min=cmin, max=cmax)
        if ndim == 2:
            slices = [slices]
        slices = [cmap.intensity_to_rgb(slice, min=0, max=1, colormap=colormap)
                  for slice in slices]
    elif mode.startswith('cat'):
        implicit = 'implicit' in mode
        slices = [cmap.prob_to_rgb(slice, colormap=colormap, implicit=implicit)
                  for slice in slices]
    elif mode.startswith('disp'):
        slices = [cmap.disp_to_rgb(slice) for slice in slices]
    else:
        raise ValueError(f'Unknown mode {mode}')

    if affine is not None:
        vx = spatial.voxel_size(affine)
    else:
        vx = torch.ones(ndim)

    # compute the size, in pseudo-mm, of each image
    if ndim == 2:
        size = [slices[0].shape[1]*vx[1]/(slices[0].shape[0]*vx[0])]
    else:
        size = [slices[0].shape[1]*vx[1]/(slices[0].shape[0]*vx[0]),
                slices[1].shape[1]*vx[2]/(slices[1].shape[0]*vx[0]),
                slices[2].shape[0]*vx[1]/(slices[2].shape[1]*vx[2])]
    size = [s.tolist() for s in size]

    # prepare grid
    # ensure that the h/w ratio is correctly set such that
    # axes have the same size in all views
    ns = (ndim*(ndim-1))//2
    ns2 = int(pymath.ceil(pymath.sqrt(ns)))
    grid = ([ns, 1] if layout.startswith('col') else
            [1, ns] if layout.startswith('row') else
            [ns2, ns2])
    grid_prm = {}
    if ndim == 2:
        grid_prm['width_ratios'] = [1/size[0]]
        grid_prm['height_ratios'] = [size[0]]
    elif layout.startswith('row'):
        grid_prm['width_ratios'] = [1/size[0], 1/size[1], size[2]]
    elif layout.startswith('col'):
        grid_prm['height_ratios'] = [size[0], size[1], 1/size[2]]
    elif layout.startswith('orth'):
        grid_prm['width_ratios'] = [1/size[0], size[2]]
        grid_prm['height_ratios'] = [size[0], size[1]]

    if (isinstance(fig, (list, tuple)) and
            isinstance(fig[0], (plt.Figure, int, type(None)))):
        fig, gs = fig
    else:
        gs = None
    if fig is None or isinstance(fig, int):
        fig = plt.figure(fig)
    if isinstance(fig, plt.Figure):
        if gs is None:
            gs = gridspec.GridSpec(*grid, **grid_prm)
        else:
            gs = gridspec.GridSpecFromSubplotSpec(*grid, **grid_prm,
                                                  subplot_spec=gs)
        ax = [fig.add_subplot(gs[n]) for n in range(ns)]
        if layout.startswith('ort'):
            ax = [ax[0], ax[2], ax[1]]
        blit = False
    else:
        ax = fig
        fig = ax[0].get_figure()
    if len(ax) < ns:
        raise ValueError('Too few axes provided')

    index = py.make_list(index, ndim)

    if ndim == 2:
        if blit:
            ax[0].images[0].set_data(transpose(slices[0]).cpu())
            reset_extent(ax[0])
        else:
            ax[0].imshow(transpose(slices[0]).cpu(), aspect=vx[1]/vx[0],
                         origin='lower', interpolation='nearest')
            ax[0].set_axis_off()
        if show_cursor:
            show_curs(ax[0], index[0], show_cursor, blit)
    elif layout.startswith('ort'):
        # orthogonal views
        if blit:
            ax[0].images[0].set_data(transpose(slices[0]).cpu())
            reset_extent(ax[0])
            ax[1].images[0].set_data(transpose(slices[1]).cpu())
            reset_extent(ax[1])
            ax[2].images[0].set_data(transpose(slices[2]).cpu())
            reset_extent(ax[2])
        else:
            ax[0].imshow(transpose(slices[0]).cpu(), aspect=vx[1]/vx[0],
                         origin='lower', interpolation='nearest')
            ax[0].set_axis_off()
            ax[1].imshow(transpose(slices[1]).cpu(), aspect=vx[2]/vx[0],
                         origin='lower', interpolation='nearest')
            ax[1].set_axis_off()
            ax[2].imshow(transpose(slices[2]).cpu(), aspect=vx[2]/vx[1],
                         origin='lower', interpolation='nearest')
            ax[2].set_axis_off()
        if show_cursor:
            show_curs(ax[0], index[0], show_cursor, blit)
            show_curs(ax[1], index[1], show_cursor, blit)
            show_curs(ax[2], index[2], show_cursor, blit)
    else:
        # aligned row or col views
        if blit:
            ax[0].images[0].set_data(transpose(slices[0]).cpu())
            reset_extent(ax[0])
            ax[1].images[0].set_data(transpose(slices[1]).cpu())
            reset_extent(ax[1])
            ax[2].images[0].set_data(transpose(slices[2]).cpu())
            reset_extent(ax[2])
        else:
            ax[0].imshow(transpose(slices[0]).cpu(), aspect=vx[1]/vx[0], origin='lower')
            ax[0].set_axis_off()
            ax[1].imshow(transpose(slices[1]).cpu(), aspect=vx[2]/vx[0], origin='lower')
            ax[1].set_axis_off()
            ax[2].imshow(transpose(slices[2]).cpu(), aspect=vx[1]/vx[2], origin='lower')
            ax[2].set_axis_off()
        if show_cursor:
            show_curs(ax[0], index[0], show_cursor, blit)
            show_curs(ax[1], index[1], show_cursor, blit)
            show_curs(ax[2], index[2], show_cursor, blit)

    if blit:
        for ax1 in ax:
            for image in ax1.images:
                ax1.draw_artist(image)
            for line in ax1.lines:
                ax1.draw_artist(line)
            try:
                ax1.get_figure().canvas.blit(ax1.bbox)
            except ValueError:
                # Probably out-of-bound, will reupdate later
                pass
    else:
        fig.tight_layout()
    if show:
        fig.canvas.flush_events()
        if not blit:
            fig.show()

    return (fig, ax, mat) if return_mat else (fig, ax)


def show_slices(img, fig_ax=None, title='', cmap='gray', flip=True,
                fig_num=1, colorbar=False, figsize=None, aspect='auto',
                title_img=None, ijk=None, channel='last'):
    """ Display a multi-channel 2D or 3D image.

    Allows for real-time plotting if giving returned fig_ax objects as input.

    Args:
        img (torch.Tensor): Input image (X, Y, C) | (X, Y, Z, C).
        fig_ax ([matplotlib.figure, matplotlib.axes])
        title (string, optional): Figure title, defaults to ''.
        cmap (str, optional): Name of matplotlib color map, defaults to 'gray'.
        flip (bool, optional): Flip channels and anatomical axis, defaults to False.
        fig_num (int, optional): matplotlib figure number, defaults to 1.
        colorbar (bool, optional): Show colorbar, defaults to False.
        figsize (tuple, optional): Figure size, given as (width, height)
        aspect (str, optional): matplotlib imshow aspect, default='auto'.
        title_img (str, optional): title for each individual image.
        ijk ([int,] * 3, optional): slice indices to visualise, if not given uses midline in all axes.
        channel (str, optional): position of channel dimension in image. default='last'

    Returns:
        fig_ax ([matplotlib.figure, matplotlib.axes])

    """
    if plt is None:
        return

    # Work out dimensions/channels
    issequence = False
    if isinstance(img, (list, tuple)):
        issequence = True        
        ix = []        
        num_chan = len(img)  # Number of channels
        for i in img:            
            i = i[..., None, None]
            dm = i.shape
            dm = torch.tensor(dm)
            is_3d = dm[2] > 1
            if ijk is None:
                ix.append(torch.floor(0.5 * dm).int().tolist())        
            else:
                ix.append(ijk)
    else:
        img = img[..., None, None]
        is_3d = img.shape[2] > 1
        if channel == 'first':
            if is_3d:
                img = img.permute(1,2,3,0,4,5)
            else:
                img = img.permute(1,2,0,3,4)
        dm = img.shape
        num_chan = dm[3]  # Number of channels
        dm = torch.tensor(dm)
        if ijk is None:
            ix = torch.floor(0.5 * dm).int().tolist()

    if not isinstance(title_img, (list, tuple)):
        title_img = [title_img]
    if len(title_img) != num_chan:
        title_img = [title_img[0],] * num_chan

    if fig_ax is None:
        # Make figure object
        if is_3d:  # 3D
            if flip:
                fig, ax = plt.subplots(num_chan, 3, num=fig_num, figsize=figsize)
            else:
                fig, ax = plt.subplots(3, num_chan, num=fig_num, figsize=figsize)
        else:  # 2D
            if flip:
                fig, ax = plt.subplots(num_chan, 1, num=fig_num, figsize=figsize)
            else:
                fig, ax = plt.subplots(1, num_chan, num=fig_num, figsize=figsize)
        fig_ax = [fig, ax]
        plt.ion()
        #fig.show()

    # Get figure and axis objects
    fig = fig_ax[0]
    ax = fig_ax[1]

    # Show images
    img_list = []
    for c in range(num_chan):  # loop over image channels
        if issequence:
            im_c = torch.squeeze(img[c][:, :, ix[c][2]]).detach().cpu()
        else:
            im_c = torch.squeeze(img[:, :, ix[2], c]).detach().cpu()
        if is_3d:
            # 3D slice 1
            ax_c = ax[0] if num_chan == 1 else ax[0, c] if not flip else ax[c, 0]
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect=aspect)
            img_list.append(im_c)
            # 3D slice 2
            ax_c = ax[1] if num_chan == 1 else ax[1, c] if not flip else ax[c, 1]
            if issequence:
                im_c = torch.squeeze(img[c][:, ix[c][1], :]).detach().cpu()
            else:
                im_c = torch.squeeze(img[:, ix[1], :, c]).detach().cpu()
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect=aspect)
            img_list.append(im_c)
            # 3D slice 3
            ax_c = ax[2] if num_chan == 1 else ax[2, c] if not flip else ax[c, 2]
            if issequence:
                im_c = torch.squeeze(img[c][ix[c][0], :, :]).detach().cpu()
            else:
                im_c = torch.squeeze(img[ix[0], :, :, c]).detach().cpu()
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect=aspect)
            img_list.append(im_c)
        else:
            # 2D
            ax_c = ax if num_chan == 1 else ax[c]
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect=aspect)
            img_list.append(im_c)

    # Modify axes
    cnt = 0
    fontsize = 9
    for c in range(num_chan):  # loop over image channels
        if is_3d:
            for r in range(3):
                ax_c = ax[r] if num_chan == 1 else ax[r, c] if not flip else ax[c, r]
                ax_c.axis('off')
                # ax_c.clear()
                if colorbar:
                    divider = make_axes_locatable(ax_c)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(img_list[cnt], cax=cax, orientation='vertical')
                if r == 1:
                    ax_c.title.set_text(title_img[c])
                    ax_c.title.set_fontsize(fontsize)
                cnt += 1
        else:
            ax_c = ax if num_chan == 1 else ax[c]
            ax_c.axis('off')
            if colorbar:
                divider = make_axes_locatable(ax_c)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(img_list[cnt], cax=cax, orientation='vertical')
            ax_c.title.set_text(title_img[c])
            ax_c.title.set_fontsize(fontsize)
            cnt += 1

    fig.suptitle(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()
    
    return fig_ax

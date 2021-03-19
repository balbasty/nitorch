"""Plotting utilities for multi-dimensional tensors.

TODO
* Real-time plotting slows down with number of calls to show_slices!

"""


import torch
from ..core.optionals import try_import
# Try import matplotlib
plt = try_import('matplotlib.pyplot', _as=True)
make_axes_locatable = try_import(
    'mpl_toolkits.axes_grid1', keys='make_axes_locatable', _as=False)


def show_slices(img, fig_ax=None, title='', cmap='gray', flip=True,
                fig_num=1, colorbar=False, figsize=None):
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

    Returns:
        fig_ax ([matplotlib.figure, matplotlib.axes])

    """
    if plt is None:
        return

    # Work out dimensions/channels
    img = img[..., None, None]
    dm = img.shape
    num_chan = dm[3]  # Number of channels
    dm = torch.tensor(dm)
    is_3d = dm[2] > 1
    ix = torch.floor(0.5 * dm).int().tolist()

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
        im_c = torch.squeeze(img[:, :, ix[2], c]).detach().cpu()
        if is_3d:
            ax_c = ax[0] if num_chan == 1 else ax[0, c] if not flip else ax[c, 0]
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect='auto')
            img_list.append(im_c)
            ax_c = ax[1] if num_chan == 1 else ax[1, c] if not flip else ax[c, 1]
            im_c = torch.squeeze(img[:, ix[1], :, c]).detach().cpu()
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect='auto')
            img_list.append(im_c)
            ax_c = ax[2] if num_chan == 1 else ax[2, c] if not flip else ax[c, 2]
            im_c = torch.squeeze(img[ix[0], :, :, c]).detach().cpu()
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect='auto')
            img_list.append(im_c)
        else:
            ax_c = ax if num_chan == 1 else ax[c]
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect='auto')
            img_list.append(im_c)

    # Modify axes
    cnt = 0
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
                cnt += 1
        else:
            ax_c = ax if num_chan == 1 else ax[c]
            ax_c.axis('off')
            if colorbar:
                divider = make_axes_locatable(ax_c)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(img_list[cnt], cax=cax, orientation='vertical')
            cnt += 1

    fig.suptitle(title)
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig_ax

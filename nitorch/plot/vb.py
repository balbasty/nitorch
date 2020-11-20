"""Plotting utilities for Variational Bayes (VB) stuff.

"""


import torch
from ..core.optionals import matplotlib


def plot_mixture_fit(X, log_pdf, mu, var, mp, fig_num=1, W=None, title=''):
    """ Plot mixture fit.

    Args:
        X (torch.tensor): (N, C).
        fig_num (int, optional): Defaults to 1.
        W (torch.tensor, optional): Defaults to no weights.
        title (string, optional): Defaults to ''.

    """
    if matplotlib is None:
        return

    import matplotlib.pyplot as plt

    # Number of classes
    dtype = X.dtype
    K = len(mp)
    # To CPU
    X = X.cpu()
    mu = mu.cpu()
    var = var.cpu()
    mp = mp.cpu()
    # Parameters
    num_sd = torch.tensor(5)
    steps = 100
    nN = 128
    inf = torch.tensor(float('inf'))
    if len(X.shape) == 1:
        X = X[:, None]
    C = X.shape[1]
    N = X.shape[0]
    # Max X
    mn_x = torch.min(X, dim=0)[0]
    mx_x = torch.max(X, dim=0)[0]

    if W is not None:
        # To CPU
        W = W.cpu()
        # Weights and observation range given
        W = torch.reshape(W, (N, 1))
        W = W / torch.sum(W)
        H = [1]
    else:
        # Make weights and observation range from data
        nX = torch.zeros(nN, C)
        W = torch.zeros(nN, C)
        H = torch.zeros(C)
        for c in range(C):
            # Bar height
            W[:, c] = torch.histc(X[:, c], bins=nN)
            # Bar start edge
            nX[:, c] = torch.linspace(start=mn_x[c], end=mx_x[c], steps=nN + 1, dtype=dtype)[:-1]
            # Bar width
            H[c] = nX[1, c] - nX[0, c]
            # Normalise height
            W[:, c] = W[:, c] / (torch.sum(W[:, c]) * H[c])
        X = nX
    # Max x and y
    mn_x = torch.min(X, dim=0)[0]
    mx_x = torch.max(X, dim=0)[0]
    mx_y = torch.max(W, dim=0)[0]
    # Plotting
    num_plt = C + 1
    num_row = math.floor(math.sqrt(num_plt))
    num_col = math.ceil(num_plt / num_row)
    plt.figure(fig_num).clear()  # Clear figure
    fig, ax = plt.subplots(num_row, num_col, num=fig_num)  # Make figure and axes
    fig.show()
    # For each channel, plot the data and the marginal density
    c = 0  # channel counter
    for row in range(num_row):  # Loop over subplot rows
        for col in range(num_col):  # Loop over subplot rows
            if c == C:
                continue
            # Get axis
            ax_rc = ax[c] if (num_row == 1 or num_col == 1) else ax[row, col]
            # Data in bar plot
            ax_rc.bar(x=X[:, c], height=W[:, c], width=H[c], alpha=0.25)
            # Marginal density
            plot_list = []  # Store plot handles (used to set colors in bar plot for mix prop)
            for k in range(K):  # Loop over mixture components
                width = num_sd * torch.sqrt(var[c, c, k])
                x0 = mu[c, k] - width
                x1 = mu[c, k] + width
                x = torch.linspace(x0, x1, steps=steps, dtype=dtype)
                y = mp[k] * torch.exp(log_pdf(x.reshape(steps, 1), k, c))
                plot = ax_rc.plot(x, y)
                plot_list.append(plot)

            ax_rc.set_xlim([mn_x[c], mx_x[c]])
            ax_rc.set_ylim([0, 0.5 * mx_y[c]])

            ax_rc.axes.get_yaxis().set_visible(False)
            ax_rc.set_title('Marginal density, C={}'.format(c + 1))
            c += 1

    # Bar plot the mixing proportions
    ax_rc = ax[c] if (num_row == 1 or num_col == 1) else ax[num_row - 1, num_col - 1]
    bp = ax_rc.bar([str(n) for n in range(1, K + 1)], mp)
    for k in range(K):
        bp[k].set_color(plot_list[k][0].get_color())
    ax_rc.axes.get_yaxis().set_visible(False)
    ax_rc.set_title('Mixing proportions')

    plt.suptitle(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
def show_slices(img, fig_ax=None, title='', cmap='gray', flip=True,
                fig_num=1, colorbar=False):
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

    Returns:
        fig_ax ([matplotlib.figure, matplotlib.axes])

    """
    if matplotlib is None:
        return

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                fig, ax = plt.subplots(num_chan, 3, num=fig_num)
            else:
                fig, ax = plt.subplots(3, num_chan, num=fig_num)
        else:  # 2D
            if flip:
                fig, ax = plt.subplots(num_chan, 1, num=fig_num)
            else:
                fig, ax = plt.subplots(1, num_chan, num=fig_num)
        fig_ax = [fig, ax]
        plt.ion()
        fig.show()

    # Get figure and axis objects
    fig = fig_ax[0]
    ax = fig_ax[1]

    # Show images
    img_list = []
    for c in range(num_chan):  # loop over image channels
        im_c = torch.squeeze(img[:, :, ix[2], c]).cpu()
        if is_3d:
            ax_c = ax[0] if num_chan == 1 else ax[0, c] if not flip else ax[c, 0]
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect='auto')
            img_list.append(im_c)
            ax_c = ax[1] if num_chan == 1 else ax[1, c] if not flip else ax[c, 1]
            im_c = torch.squeeze(img[:, ix[1], :, c]).cpu()
            im_c = ax_c.imshow(im_c, interpolation='None', cmap=cmap,  aspect='auto')
            img_list.append(im_c)
            ax_c = ax[2] if num_chan == 1 else ax[2, c] if not flip else ax[c, 2]
            im_c = torch.squeeze(img[ix[0], :, :, c]).cpu()
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

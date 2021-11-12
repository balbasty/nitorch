"""Plotting utilities for Variational Bayes (VB) stuff.

"""


import math
import torch
from ..core.optionals import try_import
# Try import matplotlib
plt = try_import('matplotlib.pyplot', _as=True)


def plot_mixture_fit(X, log_pdf, mu, var, mp, fig_num=1, W=None, title=''):
    """ Plot mixture fit.

    Args:
        X (torch.tensor): (N, C).
        fig_num (int, optional): Defaults to 1.
        W (torch.tensor, optional): Defaults to no weights.
        title (string, optional): Defaults to ''.

    """
    if plt is None:
        return

    # Number of classes
    dtype = X.dtype
    K = len(mp)
    # To CPU
    X = X.detach().cpu()
    mu = mu.detach().cpu()
    var = var.detach().cpu()
    mp = mp.detach().cpu()
    # Parameters
    num_sd = 5
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
        W = W.detach().cpu()
        # Weights and observation range given
        W = torch.reshape(W, (N, 1))
        W = W / torch.sum(W)
        H = X[1:] - X[:-1]
        H = 0.5*(H[1:] + H[:-1])
        H = torch.cat([H[:1], H, H[-1:]])
    else:
        # Make weights and observation range from data
        nX = torch.zeros(nN, C)
        W = torch.zeros(nN, C)
        H = torch.zeros(C)
        for c in range(C):
            # Bar height
            W[:, c] = torch.histc(X[:, c], bins=nN)
            # Bar start edge
            nX[:, c] = torch.linspace(start=mn_x[c], end=mx_x[c],
                                      steps=nN + 1, dtype=dtype)[:-1]
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
    fig, ax = plt.subplots(num_row, num_col,
                           num=fig_num)  # Make figure and axes
    #fig.show()
    # For each channel, plot the data and the marginal density
    c = 0  # channel counter
    for row in range(num_row):  # Loop over subplot rows
        for col in range(num_col):  # Loop over subplot rows
            if c == C:
                continue
            # Get axis
            ax_rc = ax[c] if (num_row == 1 or num_col == 1) else ax[
                row, col]
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
    ax_rc = ax[c] if (num_row == 1 or num_col == 1) else ax[
        num_row - 1, num_col - 1]
    bp = ax_rc.bar([str(n) for n in range(1, K + 1)], mp)
    for k in range(K):
        bp[k].set_color(plot_list[k][0].get_color())
    ax_rc.axes.get_yaxis().set_visible(False)
    ax_rc.set_title('Mixing proportions')

    plt.suptitle(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)


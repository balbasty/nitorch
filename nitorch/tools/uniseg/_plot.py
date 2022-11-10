import torch
from nitorch.core import utils, math
from nitorch.plot.colormaps import prob_to_rgb, disp_to_rgb
from nitorch.core.optionals import try_import_as
plt = try_import_as('matplotlib.pyplot')


def modulate_prior(M, G):
    if G is None:
        return M
    M = utils.movedim(M, 0, -1)
    M = M * G
    M /= M.sum(-1, keepdim=True)
    M = utils.movedim(M, -1, 0)
    return M


def softmax_prior(M, G):
    if G is not None:
        M = utils.movedim(M, 0, -1)
        M = M + G
        M = utils.movedim(M, -1, 0)
    return math.softmax(M, 0, implicit=(True, False))


def plot_lb(lb, fig=None, saved_elem=None):
    """

    Parameters
    ----------
    lb : list or tensor
        Lower bound
    fig : int or plt.Figure, optional
        Figure to update, or index to use when creating it
    saved_elem : list
        Elements of the figure saved for live updating

    Returns
    -------
    fig, saved_elem

    """
    if not plt:
        return None, None

    first = not saved_elem
    if not isinstance(fig, plt.Figure):
        fig = plt.figure(fig)

    if isinstance(lb, (list, tuple)):
        lb = torch.stack(lb)

    if first:
        fig.clf()
        plt.plot([])
        plt.suptitle('# iter. = {}'.format(0))
        fig.canvas.draw()
        saved_elem = [fig.canvas.copy_from_bbox(ax.bbox)
                      for ax in fig.axes]
        plt.show(block=False)

    # --- Restore figure elements --------------------------------------
    for elem in saved_elem:
        fig.canvas.restore_region(elem)

    # --- Update lower bound -------------------------------------------
    ax = fig.axes[-1]
    ax.lines[0].set_data(range(1, len(lb) + 1), lb.cpu())
    ax.relim()
    ax.autoscale_view()
    fig._suptitle.set_text('# iter. = {}'.format(len(lb)))

    # --- Draw figure --------------------------------------------------
    for ax in fig.axes[:-1]:
        ax.draw_artist(ax.images[0])
        fig.canvas.blit(ax.bbox)
    ax = fig.axes[-1]
    ax.draw_artist(ax.lines[0])
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

    return fig, saved_elem


def plot_images_and_lb(lb, X, Z, B=None, M=None, V=None, G=None,
                       mode=None, fig=None, saved_elem=None):
    """

    Parameters
    ----------
    lb : list or tensor
        Lower bound
    X : (C, *spatial) tensor
        Input image
    Z : (K, *spatial) tensor
        Responsibilities
    B : (C, *spatial) tensor
        Log bias field
    M : (K, *spatial) tensor
        Warped template (unmodulated)
    V : (*spatial, D) tensor
        Displacement field
    G : (K,) tensor
        Mixing proportions
    mode : {'bias', 'warp', 'gmm'}, optional
        Name of the step that has just been performed
    fig : int or plt.Figure, optional
        Figure to update, or index to use when creating it
    saved_elem : list
        Elements of the figure saved for live updating

    Returns
    -------
    fig, saved_elem

    """
    if not plt:
        return None, None

    first = False
    if not isinstance(fig, plt.Figure):
        fig = plt.figure(fig)
    if not saved_elem:
        first = True
    if first:
        mode = ''

    if isinstance(lb, (list, tuple)):
        lb = torch.stack(lb)

    # --- Restore figure elements --------------------------------------
    dim = X.dim() - 1
    ncol = len(X) + 1 + 2 * (M is not None) + 2 * len(X) * (B is not None)
    i = 0
    if first:
        fig.clf()
    else:
        for elem in saved_elem:
            fig.canvas.restore_region(elem)

    # ==================================================================
    #                            3D VERSION
    # ==================================================================
    if dim == 3:
        # --- Update original image ------------------------------------
        if not mode or mode == 'bias':
            X1 = X[:, :, :, X.shape[-1] // 2].cpu()
            X2 = X[:, :, X.shape[-2] // 2, :].cpu()
            X3 = X[:, X.shape[-3] // 2, :, :].cpu()
        if not mode:
            for c in range(len(X)):
                mn = X[c].min()
                mx = X[c].max()
                if first:
                    plt.subplot(4, ncol, c + 1)
                    plt.imshow(X1[c], vmin=mn, vmax=0.8 * mx)
                    plt.axis('off')
                    plt.title('Original')
                    plt.subplot(4, ncol, ncol + c + 1)
                    plt.imshow(X2[c], vmin=mn, vmax=0.8 * mx)
                    plt.axis('off')
                    plt.subplot(4, ncol, 2 * ncol + c + 1)
                    plt.imshow(X3[c], vmin=mn, vmax=0.8 * mx)
                    plt.axis('off')
                else:
                    fig.axes[i].images[0].set_data(X1[c])
                    fig.axes[i + 1].images[0].set_data(X2[c])
                    fig.axes[i + 2].images[0].set_data(X3[c])
                    i += 3
        else:
            i += 3 * len(X)

        # --- Update bias and corrected --------------------------------
        offset = 0
        if B is not None:
            offset = 2 * len(B)
            if not mode or mode == 'bias':
                B1 = B[:, :, :, B.shape[-1] // 2].exp().cpu()
                B2 = B[:, :, B.shape[-2] // 2, :].exp().cpu()
                B3 = B[:, B.shape[-3] // 2, :, :].exp().cpu()
                X1 = X1 * B1
                X2 = X2 * B2
                X3 = X3 * B3
                B1 = B1.reciprocal_()
                B2 = B2.reciprocal_()
                B3 = B3.reciprocal_()
                for c in range(len(B)):
                    if first:
                        plt.subplot(4, ncol, len(X) + c + 1)
                        plt.imshow(X1[c], vmin=mn, vmax=0.8 * mx)
                        plt.axis('off')
                        plt.title('Corrected')
                        plt.subplot(4, ncol, ncol + len(X) + c + 1)
                        plt.imshow(X2[c], vmin=mn, vmax=0.8 * mx)
                        plt.axis('off')
                        plt.subplot(4, ncol, 2 * ncol + len(X) + c + 1)
                        plt.imshow(X3[c], vmin=mn, vmax=0.8 * mx)
                        plt.axis('off')

                        plt.subplot(4, ncol, 2*len(X) + c + 1)
                        plt.imshow(B1[c])
                        plt.axis('off')
                        plt.title('Bias')
                        plt.subplot(4, ncol, ncol + 2*len(X) + c + 1)
                        plt.imshow(B2[c])
                        plt.axis('off')
                        plt.subplot(4, ncol, 2 * ncol + 2*len(X) + c + 1)
                        plt.imshow(B3[c])
                        plt.axis('off')
                        # plt.colorbar()
                    else:
                        fig.axes[i].images[0].set_data(X1[c])
                        fig.axes[i + 1].images[0].set_data(X2[c])
                        fig.axes[i + 2].images[0].set_data(X3[c])
                        i += 3

                        fig.axes[i].images[0].set_data(B1[c])
                        fig.axes[i].images[0].autoscale()
                        fig.axes[i + 1].images[0].set_data(B2[c])
                        fig.axes[i + 1].images[0].autoscale()
                        fig.axes[i + 2].images[0].set_data(B3[c])
                        fig.axes[i + 2].images[0].autoscale()
                        i += 3
            else:
                i += 6 * len(B)

        # --- Update responsibilities ----------------------------------
        Z1 = prob_to_rgb(Z[:, :, :, Z.shape[-1] // 2]).cpu()
        Z2 = prob_to_rgb(Z[:, :, Z.shape[-2] // 2, :]).cpu()
        Z3 = prob_to_rgb(Z[:, Z.shape[-3] // 2, :, :]).cpu()
        if first:
            plt.subplot(4, ncol, len(X) + offset + 1)
            plt.imshow(Z1)
            plt.axis('off')
            plt.title('Resp.')
            plt.subplot(4, ncol, ncol + len(X) + offset + 1)
            plt.imshow(Z2)
            plt.axis('off')
            plt.subplot(4, ncol, 2 * ncol + len(X) + offset + 1)
            plt.imshow(Z3)
            plt.axis('off')
        else:
            fig.axes[i].images[0].set_data(Z1)
            fig.axes[i + 1].images[0].set_data(Z2)
            fig.axes[i + 2].images[0].set_data(Z3)
            i += 3

        # --- Update warped and warp -----------------------------------
        if M is not None:
            if not mode or mode in ('gmm', 'warp'):
                M1 = softmax_prior(M[:, :, :, M.shape[-1] // 2], G)
                M2 = softmax_prior(M[:, :, M.shape[-2] // 2, :], G)
                M3 = softmax_prior(M[:, M.shape[-3] // 2, :, :], G)
                M1 = prob_to_rgb(M1).cpu()
                M2 = prob_to_rgb(M2).cpu()
                M3 = prob_to_rgb(M3).cpu()
                if V is not None:
                    V = utils.movedim(V, -1, 0)
                    V1 = disp_to_rgb(V[:, :, :, V.shape[-1] // 2],
                                     amplitude='saturation').cpu()
                    V2 = disp_to_rgb(V[:, :, V.shape[-2] // 2, :],
                                     amplitude='saturation').cpu()
                    V3 = disp_to_rgb(V[:, V.shape[-3] // 2, :, :],
                                     amplitude='saturation').cpu()
                if first:
                    plt.subplot(4, ncol, len(X) + 2 + offset)
                    plt.imshow(M1)
                    plt.axis('off')
                    plt.title('Prior')
                    plt.subplot(4, ncol, ncol + len(X) + 2 + offset)
                    plt.imshow(M2)
                    plt.axis('off')
                    plt.subplot(4, ncol, 2 * ncol + len(X) + 2 + offset)
                    plt.imshow(M3)
                    plt.axis('off')

                    if V is not None:
                        plt.subplot(4, ncol, len(X) + 3 + offset)
                        plt.imshow(V1)
                        plt.axis('off')
                        plt.title('Disp.')
                        plt.subplot(4, ncol, ncol + len(X) + 3 + offset)
                        plt.imshow(V2)
                        plt.axis('off')
                        plt.subplot(4, ncol, 2 * ncol + len(X) + 3 + offset)
                        plt.imshow(V3)
                        plt.axis('off')
                else:
                    fig.axes[i].images[0].set_data(M1)
                    fig.axes[i + 1].images[0].set_data(M2)
                    fig.axes[i + 2].images[0].set_data(M3)
                    if V is not None:
                        fig.axes[i + 3].images[0].set_data(V1)
                        fig.axes[i + 4].images[0].set_data(V2)
                        fig.axes[i + 5].images[0].set_data(V3)
                    i += 6
            else:
                i += 6

    # ==================================================================
    #                            2D VERSION
    # ==================================================================
    else:
        # --- Update original image ------------------------------------
        if not mode or mode == 'bias':
            X1 = X.cpu()
        if not mode:
            for c in range(len(X)):
                mn = X[c].min()
                mx = X[c].max()
                if first:
                    plt.subplot(4, ncol, c + 1)
                    plt.imshow(X1[c], vmin=mn, vmax=0.8 * mx)
                    plt.axis('off')
                    plt.title('Original')
                else:
                    fig.axes[i].images[0].set_data(X1[c])
                    i += 1
        else:
            i += len(X)

        # --- Update bias and corrected --------------------------------
        offset = 0
        if B is not None:
            offset = 2 * len(B)
            if not mode or mode == 'bias':
                B1 = B.exp().cpu()
                X1 = X1 * B1
                B1 = B1.reciprocal_()
                for c in range(len(B)):
                    if first:
                        plt.subplot(4, ncol, len(X) + c + 1)
                        plt.imshow(X1[c], vmin=mn, vmax=0.8 * mx)
                        plt.axis('off')
                        plt.title('Corrected')

                        plt.subplot(4, ncol, len(X) + c + 2)
                        plt.imshow(B1[c])
                        plt.axis('off')
                        plt.title('Bias')
                    else:
                        fig.axes[i].images[0].set_data(X1[c])
                        i += 1

                        fig.axes[i].images[0].set_data(B1[c])
                        fig.axes[i].images[0].autoscale()
                        i += 1
            else:
                i += 2 * len(B)

        # --- Update responsibilities ----------------------------------
        Z1 = prob_to_rgb(Z).cpu()
        if first:
            plt.subplot(4, ncol, len(X) + offset + 1)
            plt.imshow(Z1)
            plt.axis('off')
            plt.title('Resp.')
        else:
            fig.axes[i].images[0].set_data(Z1)
            i += 1

        # --- Update warped and warp -----------------------------------
        if M is not None:
            if not mode or mode in ('gmm', 'warp'):
                M1 = softmax_prior(M, G)
                M1 = prob_to_rgb(M1).cpu()
                V = utils.movedim(V, -1, 0)
                V1 = disp_to_rgb(V).cpu()
                if first:
                    plt.subplot(4, ncol, len(X) + 2 + offset)
                    plt.imshow(M1)
                    plt.axis('off')
                    plt.title('Prior')

                    plt.subplot(4, ncol, len(X) + 3 + offset)
                    plt.imshow(V1)
                    plt.axis('off')
                    plt.title('Prior')
                else:
                    fig.axes[i].images[0].set_data(M1)
                    fig.axes[i + 3].images[0].set_data(V1)
                    i += 2
            else:
                i += 2

    # --- Update lower bound -------------------------------------------
    if first:
        plt.subplot(1+dim*(dim-1)//2, 1, 1+dim*(dim-1)//2)
        plt.plot([])
        plt.suptitle('# iter. = {}'.format(0))

        fig.canvas.draw()
        saved_elem = [fig.canvas.copy_from_bbox(ax.bbox)
                      for ax in fig.axes]
        plt.show(block=False)

    ax = fig.axes[-1]
    ax.lines[0].set_data(list(range(1, len(lb) + 1)), lb.cpu())
    ax.relim()
    ax.autoscale_view()
    fig._suptitle.set_text('# iter. = {}'.format(len(lb)))

    # --- Draw figure --------------------------------------------------
    for ax in fig.axes[:-1]:
        ax.draw_artist(ax.images[0])
    ax = fig.axes[-1]
    ax.draw_artist(ax.lines[0])
    for ax in fig.axes:
        fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

    return fig, saved_elem

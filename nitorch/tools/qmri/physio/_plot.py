def plot_volumes(volume, fig_number=None, colorbar=False, coord=None,
                 title=None):
    import matplotlib.pyplot as plt

    # Get dimensions
    if not isinstance(volume, (list, tuple)):
        volume = [volume]
    nb_volumes = len(volume)
    shape = volume[0].shape
    dim = len(shape)

    if title is not None and not isinstance(title, (list,tuple)):
        title = [title]

    # Get figure
    if fig_number is not None:
        fig = plt.figure(fig_number)
    else:
        fig = plt.figure()

    fig.clf()
    ax = fig.subplots(nb_volumes, dim*(dim-1)//2)
    for n in range(nb_volumes):
        if dim == 2:
            p = ax[n].imshow(volume[n])
            ax[n].axis('off')
            if colorbar:
                plt.colorbar(p, ax=ax[n])
            if title is not None:
                ax[n].title(title[min(n, len(title)-1)])
        elif dim == 3:
            if coord is None:
                c = [s//2 for s in volume[n].shape[:3]]
            else:
                c = coord
            p = ax[n][0].imshow(volume[n][:, :, c[2]])
            ax[n][0].axis('off')
            if colorbar:
                plt.colorbar(p, ax=ax[n][0])
            p = ax[n][1].imshow(volume[n][:, c[1], :])
            ax[n][1].axis('off')
            if colorbar:
                plt.colorbar(p, ax=ax[n][1])
            if title is not None:
                ax[n][1].title(title[min(n, len(title)-1)])
            p = ax[n][2].imshow(volume[n][c[0], :, :])
            ax[n][2].axis('off')
            if colorbar:
                plt.colorbar(p, ax=ax[n][2])
        else:
            raise NotImplementedError

    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig.number



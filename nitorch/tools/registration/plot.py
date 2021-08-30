from nitorch.core import math, utils
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot')


def mov2fix(fixed, moving, warped, vel=None, cat=False, dim=None, title=None):
    """Plot registration live"""

    if plt is None:
        return

    warped = warped.detach()
    if vel is not None:
        vel = vel.detach()

    dim = dim or (fixed.dim() - 1)
    if fixed.dim() < dim + 2:
        fixed = fixed[None]
    if moving.dim() < dim + 2:
        moving = moving[None]
    if warped.dim() < dim + 2:
        warped = warped[None]
    if vel is not None:
        if vel.dim() < dim + 2:
            vel = vel[None]
    nb_channels = fixed.shape[-dim-1]
    nb_batch = len(fixed)

    if dim == 3:
        fixed = fixed[..., fixed.shape[-1]//2]
        moving = moving[..., moving.shape[-1]//2]
        warped = warped[..., warped.shape[-1]//2]
        if vel is not None:
            vel = vel[..., vel.shape[-2]//2, :]
    if vel is not None:
        vel = vel.square().sum(-1).sqrt()

    if cat:
        moving = math.softmax(moving, dim=1, implicit=True)
        warped = math.softmax(warped, dim=1, implicit=True)

    checker = fixed.clone()
    patch = max([s//8 for s in fixed.shape])
    checker_unfold = utils.unfold(checker, [patch]*2, [2*patch]*2)
    warped_unfold = utils.unfold(warped, [patch]*2, [2*patch]*2)
    checker_unfold.copy_(warped_unfold)

    nb_rows = min(nb_batch, 3)
    nb_cols = 4 + (vel is not None)
    for b in range(nb_rows):
        plt.subplot(nb_rows, nb_cols, b*nb_cols + 1)
        plt.imshow(moving[b, 0].cpu())
        plt.title('moving')
        plt.axis('off')
        plt.subplot(nb_rows, nb_cols, b*nb_cols + 2)
        plt.imshow(warped[b, 0].cpu())
        plt.title('moved')
        plt.axis('off')
        plt.subplot(nb_rows, nb_cols, b*nb_cols + 3)
        plt.imshow(checker[b, 0].cpu())
        plt.title('checker')
        plt.axis('off')
        plt.subplot(nb_rows, nb_cols, b*nb_cols + 4)
        plt.imshow(fixed[b, 0].cpu())
        plt.title('fixed')
        plt.axis('off')
        if vel is not None:
            plt.subplot(nb_rows, nb_cols, b*nb_cols + 5)
            plt.imshow(vel[b].cpu())
            plt.title('velocity')
            plt.axis('off')
            plt.colorbar()
    if title:
        plt.suptitle(title)
    plt.gcf().canvas.flush_events()
    plt.show(block=False)

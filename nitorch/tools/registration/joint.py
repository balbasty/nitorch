"""Joint affine + nonlinear registration.

This model is the most polished one. All simpler models (affine, smalldef,
svf, shoot) can be performed by the joint model as sub-cases.

Given its flexibility, there is no simple function to call it. I used
an object-oriented representation of the different blocks (implemented in
registration.objects) to encode the images, transformations and optimizers.

The command line function `nitorch register` calls this model under the hood.

"""
from nitorch import spatial
from nitorch.core import py, utils, linalg, math
import torch
from .utils import jg, jhj
from . import optim as optm, utils as regutils
from .objects import SVFModel, MeanSpace
import copy


# TODO:
#  [x] fix backward gradients for smalldef/shoot (svf should be ok)
#  [x] implement forward pass for affine
#  [x] implement a forward pass for affine only (when nonlin is None).
#      It deserves its own implementation as it is so much simpler (the
#      two half matrices collapse).
#  [ ] write utility function to set optimizer options based on
#      model options (e.g., reg param in GridCG/GridRelax)
#  [x] syntaxic sugar class to transform a forward loss into an OptimizationLoss
#      object using autodiff (set `warped.requires_grad`, wrap under a
#      `with torch.grad()`, call backward on loss, gather grad)
#  [x] integrate cache mechanism into the Displacement/Image classes
#      (e.g., add a `cache` argument to `exp` to retrieve/cache the
#      exponentiated field).
#  [ ] add the possibility to encode velocity.displacement fields using
#      B-splines. Gradients should be relatively straightforward (compute
#      dense gradient and then propagate to nodes?), but regularizers
#      should probably be stored in matrix form (like what JA did for
#      fields encoded with dft/dct is US). If the number of nodes is not
#      too large, it would allow us to use Gauss-Newton directly.


def _almost_identity(aff):
    return torch.allclose(aff, torch.eye(*aff.shape, **utils.backend(aff)))


class RegisterStep:
    """Forward pass of Diffeo+Affine registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(
            self,
            losses,                 # list[LossComponent]
            affine=None,            # AffineModel
            nonlin=None,            # NonLinModel
            verbose=True,           # verbosity level
            ):
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        self.losses = losses
        self.affine = affine
        self.nonlin = nonlin
        self.verbose = verbose

        # pretty printing
        self.n_iter = 0             # current iteration
        self.ll_prev = None         # previous loss value
        self.ll_max = 0             # max loss value
        self.llv = 0                # last velocity penalty
        self.lla = 0                # last affine penalty
        self.all_ll = []

        self.framerate = 1
        self._last_plot = 0
        if self.verbose > 1:
            import matplotlib.pyplot as plt
            self.figure = plt.figure()

    def mov2fix(self, fixed, moving, warped, vel=None, cat=False, dim=None, title=None):
        """Plot registration live"""

        import time
        tic = self._last_plot
        toc = time.time()
        if toc - tic < 1/self.framerate:
            return
        self._last_plot = toc

        import matplotlib.pyplot as plt

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
        nb_channels = fixed.shape[-dim - 1]
        nb_batch = len(fixed)

        if dim == 3:
            fixed = [fixed[..., fixed.shape[-1] // 2],
                     fixed[..., fixed.shape[-2] // 2, :],
                     fixed[..., fixed.shape[-3] // 2, :, :]]
            moving = [moving[..., moving.shape[-1] // 2],
                      moving[..., moving.shape[-2] // 2, :],
                      moving[..., moving.shape[-3] // 2, :, :]]
            warped = [warped[..., warped.shape[-1] // 2],
                      warped[..., warped.shape[-2] // 2, :],
                      warped[..., warped.shape[-3] // 2, :, :]]
            if vel is not None:
                vel = [vel[..., vel.shape[-2] // 2, :],
                       vel[..., vel.shape[-3] // 2, :, :],
                       vel[..., vel.shape[-4] // 2, :, :, :]]
                vel = [v.square().sum(-1).sqrt() for v in vel]
        else:
            fixed = [fixed]
            moving = [moving]
            warped = [warped]
            vel = [vel.square().sum(-1).sqrt()] if vel is not None else []

        if cat:
            moving = [math.softmax(img, dim=1, implicit=True) for img in moving]
            warped = [math.softmax(img, dim=1, implicit=True) for img in warped]

        checker = []
        for f, w in zip(fixed, warped):
            c = f.clone()
            patch = max([s // 8 for s in f.shape])
            checker_unfold = utils.unfold(c, [patch] * 2, [2 * patch] * 2)
            warped_unfold = utils.unfold(w, [patch] * 2, [2 * patch] * 2)
            checker_unfold.copy_(warped_unfold)
            checker.append(c)

        kdim = 3 if dim == 3 else 1
        bdim = min(nb_batch, 3)
        nb_rows = kdim * bdim + 1
        nb_cols = 4 + (vel is not None)

        if len(self.figure.axes) != nb_rows*nb_cols:
            self.figure.clf()

            for b in range(bdim):
                for k in range(kdim):
                    plt.subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 1)
                    plt.imshow(moving[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        plt.title('moving')
                    plt.axis('off')
                    plt.subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 2)
                    plt.imshow(warped[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        plt.title('moved')
                    plt.axis('off')
                    plt.subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 3)
                    plt.imshow(checker[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        plt.title('checker')
                    plt.axis('off')
                    plt.subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 4)
                    plt.imshow(fixed[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        plt.title('fixed')
                    plt.axis('off')
                    if vel is not None:
                        plt.subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 5)
                        plt.imshow(vel[k][b].cpu())
                        if b == 0 and k == 0:
                            plt.title('velocity')
                        plt.axis('off')
                        plt.colorbar()
            plt.subplot(nb_rows, 1, nb_rows)
            plt.plot(list(range(1, len(self.all_ll)+1)), self.all_ll)
            plt.ylabel('NLL')
            plt.xlabel('iteration')
            if title:
                plt.suptitle(title)

            self.figure.canvas.draw()
            self.plt_saved = [self.figure.canvas.copy_from_bbox(ax.bbox)
                              for ax in self.figure.axes]
            self.figure.canvas.flush_events()
            plt.show(block=False)

        else:
            self.figure.canvas.draw()
            for elem in self.plt_saved:
                self.figure.canvas.restore_region(elem)

            for b in range(bdim):
                for k in range(kdim):
                    j = (b + k*bdim) * nb_cols
                    self.figure.axes[j].images[0].set_data(moving[k][b, 0].cpu())
                    self.figure.axes[j+1].images[0].set_data(warped[k][b, 0].cpu())
                    self.figure.axes[j+2].images[0].set_data(checker[k][b, 0].cpu())
                    self.figure.axes[j+3].images[0].set_data(fixed[k][b, 0].cpu())
                    if vel is not None:
                        self.figure.axes[j+4].images[0].set_data(vel[k][b].cpu())
            lldata = (list(range(1, len(self.all_ll)+1)), self.all_ll)
            self.figure.axes[-1].lines[0].set_data(lldata)
            if title:
                self.figure._suptitle.set_text(title)

            for ax in self.figure.axes:
                ax.draw_artist(ax.images[0])
                self.figure.canvas.blit(ax.bbox)
            self.figure.canvas.flush_events()

    def do_vel(self, vel, grad=False, hess=False, in_line_search=False):
        """Forward pass for updating the nonlinear component"""

        sumloss = None
        sumgrad = None
        sumhess = None

        # build affine and displacement field
        if self.affine:
            aff0, iaff0 = self.affine.exp2(cache_result=True, recompute=False)
            aff_pos = self.affine.position[0].lower()
        else:
            aff_pos = 'x'
            aff0 = iaff0 = torch.eye(self.nonlin.dim + 1, **utils.backend(self.nonlin.dat))
        vel0 = vel
        if any(loss.symmetric for loss in self.losses):
            phi0, iphi0 = self.nonlin.exp2(vel0,
                                           recompute=True,
                                           cache_result=not in_line_search)
            ivel0 = -vel0
        else:
            phi0 = self.nonlin.exp(vel0,
                                   recompute=True,
                                   cache_result=not in_line_search)
            iphi0 = ivel0 = None

        # register temporary "backward" loss for symmetric losses
        losses = []
        for loss in self.losses:
            losses.append(loss)
            if loss.symmetric:
                bwdloss = copy.copy(loss)
                bwdloss.moving, bwdloss.fixed = loss.fixed, loss.moving
                bwdloss.symmetric = 'backward'
                losses.append(bwdloss)

        for loss in losses:

            factor = loss.factor
            if loss.symmetric:
                factor = factor / 2
            if loss.symmetric == 'backward':
                phi00 = iphi0
                aff00 = iaff0
                vel00 = ivel0
            else:
                phi00 = phi0
                aff00 = aff0
                vel00 = vel0

            is_level0 = True
            for moving, fixed in zip(loss.moving, loss.fixed):  # pyramid

                # build left and right affine
                if aff_pos in 'fs':
                    aff_right = spatial.affine_matmul(aff00, fixed.affine)
                else:
                    aff_right = fixed.affine
                aff_right = spatial.affine_lmdiv(self.nonlin.affine, aff_right)
                if aff_pos in 'ms':
                    tmp = spatial.affine_matmul(aff00, self.nonlin.affine)
                    aff_left = spatial.affine_lmdiv(moving.affine, tmp)
                else:
                    aff_left = spatial.affine_lmdiv(moving.affine, self.nonlin.affine)

                # build full transform
                if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                    aff_right = None
                    phi = spatial.identity_grid(fixed.shape, **utils.backend(phi00))
                    phi += phi00
                else:
                    phi = spatial.affine_grid(aff_right, fixed.shape)
                    phi += regutils.smart_pull_grid(phi00, phi)
                if _almost_identity(aff_left) and moving.shape == self.nonlin.shape:
                    aff_left = None
                else:
                    phi = spatial.affine_matvec(aff_left, phi)

                # forward
                warped, mask = moving.pull(phi, mask=True)
                if fixed.masked:
                    if mask is None:
                        mask = fixed.mask
                    else:
                        mask = mask * fixed.mask

                if is_level0 and self.verbose > 1 and not in_line_search \
                        and loss.symmetric != 'backward':
                    is_level0 = False
                    init = spatial.affine_lmdiv(moving.affine, fixed.affine)
                    if _almost_identity(init) and moving.shape == fixed.shape:
                        init = moving.dat
                    else:
                        init = spatial.affine_grid(init, fixed.shape)
                        init = moving.pull(init)
                    self.mov2fix(fixed.dat, init, warped, vel0,
                                 dim=fixed.dim,
                                 title=f'(nonlin) {self.n_iter:03d}')

                # gradient/Hessian of the log-likelihood in observed space
                g = h = None
                if not grad and not hess:
                    llx = loss.loss.loss(warped, fixed.dat, dim=fixed.dim, mask=mask)
                elif not hess:
                    llx, g = loss.loss.loss_grad(warped, fixed.dat, dim=fixed.dim, mask=mask)
                else:
                    llx, g, h = loss.loss.loss_grad_hess(warped, fixed.dat, dim=fixed.dim, mask=mask)

                # compose with spatial gradients
                if grad or hess:

                    g, h, mugrad = self.nonlin.propagate_grad(
                        g, h, moving, phi00, aff_left, aff_right,
                        inv=(loss.symmetric == 'backward'))

                    g = regutils.jg(mugrad, g)
                    h = regutils.jhj(mugrad, h)

                    if isinstance(self.nonlin, SVFModel):
                        # propagate backward by scaling and squaring
                        g, h = spatial.exp_backward(vel00, g, h, steps=self.nonlin.steps)

                    sumgrad = g.mul_(factor) if sumgrad is None else sumgrad.add_(g, alpha=factor)
                    if hess:
                        sumhess = h.mul_(factor) if sumhess is None else sumhess.add_(h, alpha=factor)
                sumloss = llx.mul_(factor) if sumloss is None else sumloss.add_(llx, alpha=factor)

        # add regularization term
        vgrad = self.nonlin.regulariser(vel0)
        llv = 0.5 * vel0.flatten().dot(vgrad.flatten())
        if grad:
            sumgrad += vgrad
        del vgrad

        # print objective
        llx = sumloss.item()
        sumloss += llv
        sumloss += self.lla
        self.loss_value = sumloss.item()
        if self.verbose and not in_line_search:
            llv = llv.item()
            self.llv = llv
            ll = sumloss.item()
            self.all_ll.append(ll)
            lla = self.lla
            self.n_iter += 1
            line = '(nonlin) | '
            line += f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} + {lla:12.6g} = {ll:12.6g}'
            if self.ll_prev is not None:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                line += f' | {gain:12.6g}'
            print(line, end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]

    def do_affine(self, logaff, grad=False, hess=False, in_line_search=False):
        """Forward pass for updating the affine component (nonlin is not None)"""

        sumloss = None
        sumgrad = None
        sumhess = None

        # build affine and displacement field
        logaff0 = logaff
        aff_pos = self.affine.position[0].lower()
        if any(loss.symmetric for loss in self.losses):
            aff0, iaff0, gaff0, igaff0 = self.affine.exp2(logaff0, grad=True,
                                                          cache_result=not in_line_search)
            phi0, iphi0 = self.nonlin.exp2(cache_result=True, recompute=False)
        else:
            iaff0 = None
            aff0, gaff0 = self.affine.exp(logaff0, grad=True,
                                          cache_result=not in_line_search)
            phi0 = self.nonlin.exp(cache_result=True, recompute=False)
            iphi0 = None

        # register temporary "backward" loss for symmetric losses
        losses = []
        for loss in self.losses:
            losses.append(loss)
            if loss.symmetric:
                bwdloss = copy.copy(loss)
                bwdloss.moving, bwdloss.fixed = loss.fixed, loss.moving
                bwdloss.symmetric = 'backward'
                losses.append(bwdloss)

        for loss in losses:

            factor = loss.factor
            if loss.symmetric:
                factor = factor / 2
            if loss.symmetric == 'backward':
                phi00 = iphi0
                aff00 = iaff0
                gaff00 = igaff0
            else:
                phi00 = phi0
                aff00 = aff0
                gaff00 = gaff0

            is_level0 = True
            for moving, fixed in zip(loss.moving, loss.fixed):  # pyramid

                # build complete warp
                if aff_pos in 'fs':
                    aff_right = spatial.affine_matmul(aff00, fixed.affine)
                    aff_right = spatial.affine_lmdiv(self.nonlin.affine, aff_right)
                    gaff_right = torch.matmul(gaff00, fixed.affine)
                    gaff_right = linalg.lmdiv(self.nonlin.affine, gaff_right)
                else:
                    aff_right = spatial.affine_lmdiv(self.nonlin.affine, fixed.affine)
                    gaff_right = None
                if aff_pos in 'ms':
                    aff_left = spatial.affine_matmul(aff00, self.nonlin.affine)
                    aff_left = spatial.affine_lmdiv(moving.affine, aff_left)
                    gaff_left = torch.matmul(gaff00, self.nonlin.affine)
                    gaff_left = linalg.lmdiv(moving.affine, gaff_left)
                else:
                    aff_left = spatial.affine_lmdiv(moving.affine, self.nonlin.affine)
                    gaff_left = None
                if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                    right = None
                    phi = spatial.identity_grid(fixed.shape, **utils.backend(phi00))
                    phi += phi00
                else:
                    right = spatial.affine_grid(aff_right, fixed.shape)
                    phi = regutils.smart_pull_grid(phi00, right)
                    phi += right
                phi_right = phi
                if _almost_identity(aff_left) and moving.shape == self.nonlin.shape:
                    left = None
                else:
                    left = spatial.affine_grid(aff_left, self.nonlin.shape)
                    phi = spatial.affine_matvec(aff_left, phi)

                # forward
                warped, mask = moving.pull(phi, mask=True)
                if fixed.masked:
                    if mask is None:
                        mask = fixed.mask
                    else:
                        mask = mask * fixed.mask

                if is_level0 and self.verbose > 1 and not in_line_search \
                        and loss.symmetric != 'backward':
                    is_level0 = False
                    init = spatial.affine_lmdiv(moving.affine, fixed.affine)
                    if _almost_identity(init) and moving.shape == fixed.shape:
                        init = moving.dat
                    else:
                        init = spatial.affine_grid(init, fixed.shape)
                        init = moving.pull(init)
                    self.mov2fix(fixed.dat, init, warped, dim=fixed.dim,
                                 title=f'(affine) {self.n_iter:03d}')

                # gradient/Hessian of the log-likelihood in observed space
                g = h = None
                if not grad and not hess:
                    llx = loss.loss.loss(warped, fixed.dat, dim=fixed.dim, mask=mask)
                elif not hess:
                    llx, g = loss.loss.loss_grad(warped, fixed.dat, dim=fixed.dim, mask=mask)
                else:
                    llx, g, h = loss.loss.loss_grad_hess(warped, fixed.dat, dim=fixed.dim, mask=mask)

                def compose_grad(g, h, g_mu, g_aff):
                    """
                    g, h : gradient/Hessian of loss wrt moving image
                    g_mu : spatial gradients of moving image
                    g_aff : gradient of affine matrix wrt Lie parameters
                    returns g, h: gradient/Hessian of loss wrt Lie parameters
                    """
                    # Note that `h` can be `None`, but the functions I
                    # use deal with this case correctly.
                    dim = g_mu.shape[-1]
                    g = jg(g_mu, g)
                    h = jhj(g_mu, h)
                    g, h = regutils.affine_grid_backward(g, h)
                    dim2 = dim * (dim + 1)
                    g = g.reshape([*g.shape[:-2], dim2])
                    g_aff = g_aff[..., :-1, :]
                    g_aff = g_aff.reshape([*g_aff.shape[:-2], dim2])
                    g = linalg.matvec(g_aff, g)
                    if h is not None:
                        h = h.reshape([*h.shape[:-4], dim2, dim2])
                        h = g_aff.matmul(h).matmul(g_aff.transpose(-1, -2))
                        h = h.abs().sum(-1).diag_embed()
                    return g, h

                # compose with spatial gradients
                if grad or hess:
                    g0, g = g, None
                    h0, h = h, None
                    if aff_pos in 'ms':
                        g_left = regutils.smart_push(g0, phi_right, shape=self.nonlin.shape)
                        h_left = regutils.smart_push(h0, phi_right, shape=self.nonlin.shape)
                        mugrad = moving.pull_grad(left, rotate=False)
                        g_left, h_left = compose_grad(g_left, h_left, mugrad, gaff_left)
                        g = g_left
                        h = h_left
                    if aff_pos in 'fs':
                        g_right = g0
                        h_right = h0
                        mugrad = moving.pull_grad(phi, rotate=False)
                        jac = spatial.grid_jacobian(phi0, right, type='disp', extrapolate=False)
                        jac = torch.matmul(aff_left[:-1, :-1], jac)
                        mugrad = linalg.matvec(jac.transpose(-1, -2), mugrad)
                        g_right, h_right = compose_grad(g_right, h_right, mugrad, gaff_right)
                        g = g_right if g is None else g.add_(g_right)
                        h = h_right if h is None else h.add_(h_right)

                    if loss.symmetric == 'backward':
                        g = g.neg_()
                    sumgrad = g.mul_(factor) if sumgrad is None else sumgrad.add_(g, alpha=factor)
                    if hess:
                        sumhess = h.mul_(factor) if sumhess is None else sumhess.add_(h, alpha=factor)
                sumloss = llx.mul_(factor) if sumloss is None else sumloss.add_(llx, alpha=factor)

        # TODO add regularization term
        lla = 0

        # print objective
        llx = sumloss.item()
        sumloss += lla
        sumloss += self.llv
        self.loss_value = ll.item()
        if self.verbose and not in_line_search:
            self.n_iter += 1
            ll = sumloss.item()
            self.all_ll.append(ll)
            llv = self.llv
            line = '(affine) | '
            line += f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} + {lla:12.6g} = {ll:12.6g}'
            if self.ll_prev is not None:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                line += f' | {gain:12.6g}'
            print(line, end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]

    def do_affine_only(self, logaff, grad=False, hess=False, in_line_search=False):
        """Forward pass for updating the affine component (nonlin is None)"""

        sumloss = None
        sumgrad = None
        sumhess = None

        # build affine and displacement field
        logaff0 = logaff
        aff0, iaff0, gaff0, igaff0 = self.affine.exp2(logaff0, grad=True)

        # register temporary "backward" loss for symmetric losses
        losses = []
        for loss in self.losses:
            losses.append(loss)
            if loss.symmetric:
                bwdloss = copy.copy(loss)
                bwdloss.moving, bwdloss.fixed = loss.fixed, loss.moving
                bwdloss.symmetric = 'backward'
                losses.append(bwdloss)

        for loss in losses:

            factor = loss.factor
            if loss.symmetric:
                factor = factor / 2
            if loss.symmetric == 'backward':
                aff00 = iaff0
                gaff00 = igaff0
            else:
                aff00 = aff0
                gaff00 = gaff0

            is_level0 = True
            for moving, fixed in zip(loss.moving, loss.fixed):  # pyramid

                # build complete warp
                aff = spatial.affine_matmul(aff00, fixed.affine)
                aff = spatial.affine_lmdiv(moving.affine, aff)
                gaff = torch.matmul(gaff00, fixed.affine)
                gaff = linalg.lmdiv(moving.affine, gaff)
                phi = spatial.affine_grid(aff, fixed.shape)

                # forward
                warped, mask = moving.pull(phi, mask=True)
                if fixed.masked:
                    if mask is None:
                        mask = fixed.mask
                    else:
                        mask = mask * fixed.mask

                if is_level0 and self.verbose > 1 and not in_line_search \
                        and loss.symmetric != 'backward':
                    is_level0 = False
                    init = spatial.affine_lmdiv(moving.affine, fixed.affine)
                    if _almost_identity(init) and moving.shape == fixed.shape:
                        init = moving.dat
                    else:
                        init = spatial.affine_grid(init, fixed.shape)
                        init = moving.pull(init)
                    self.mov2fix(fixed.dat, init, warped, dim=fixed.dim,
                                 title=f'(affine) {self.n_iter:03d}')

                # gradient/Hessian of the log-likelihood in observed space
                g = h = None
                if not grad and not hess:
                    llx = loss.loss.loss(warped, fixed.dat, dim=fixed.dim, mask=mask)
                elif not hess:
                    llx, g = loss.loss.loss_grad(warped, fixed.dat, dim=fixed.dim, mask=mask)
                else:
                    llx, g, h = loss.loss.loss_grad_hess(warped, fixed.dat, dim=fixed.dim, mask=mask)

                def compose_grad(g, h, g_mu, g_aff):
                    """
                    g, h : gradient/Hessian of loss wrt moving image
                    g_mu : spatial gradients of moving image
                    g_aff : gradient of affine matrix wrt Lie parameters
                    returns g, h: gradient/Hessian of loss wrt Lie parameters
                    """
                    # Note that `h` can be `None`, but the functions I
                    # use deal with this case correctly.
                    dim = g_mu.shape[-1]
                    g = jg(g_mu, g)
                    h = jhj(g_mu, h)
                    g, h = regutils.affine_grid_backward(g, h)
                    dim2 = dim * (dim + 1)
                    g = g.reshape([*g.shape[:-2], dim2])
                    g_aff = g_aff[..., :-1, :]
                    g_aff = g_aff.reshape([*g_aff.shape[:-2], dim2])
                    g = linalg.matvec(g_aff, g)
                    if h is not None:
                        h = h.reshape([*h.shape[:-4], dim2, dim2])
                        h = g_aff.matmul(h).matmul(g_aff.transpose(-1, -2))
                        h = h.abs().sum(-1).diag_embed()
                    return g, h

                # compose with spatial gradients
                if grad or hess:
                    mugrad = moving.pull_grad(phi, rotate=False)
                    g, h = compose_grad(g, h, mugrad, gaff)

                    if loss.symmetric == 'backward':
                        g = g.neg_()
                    sumgrad = g.mul_(factor) if sumgrad is None else sumgrad.add_(g, alpha=factor)
                    if hess:
                        sumhess = h.mul_(factor) if sumhess is None else sumhess.add_(h, alpha=factor)
                sumloss = llx.mul_(factor) if sumloss is None else sumloss.add_(llx, alpha=factor)

        # TODO add regularization term
        lla = 0

        # print objective
        llx = sumloss.item()
        sumloss += lla
        lla = lla
        ll = sumloss.item()
        self.loss_value = ll
        if self.verbose and not in_line_search:
            self.n_iter += 1
            self.all_ll.append(ll)
            line = '(affine) | '
            line += f'{self.n_iter:03d} | {llx:12.6g} + {lla:12.6g} = {ll:12.6g}'
            if self.ll_prev is not None:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                line += f' | {gain:12.6g}'
            print(line, end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]


class Register:

    def __init__(self,
                 losses,                 # list[LossComponent]
                 affine=None,            # AffineModel
                 nonlin=None,            # NonLinModel
                 optim=None,             # Optimizer
                 verbose=True,           # verbosity level
                 framerate=1,            # plotting framerate
                 ):
        self.losses = losses
        self.verbose = verbose
        self.affine = affine
        self.nonlin = nonlin
        self.optim = optim
        self.framerate = framerate

    def __call__(self):
        return self.fit()

    def fit(self):
        backend = dict(device=self.losses[0].fixed.device,
                       dtype=self.losses[0].fixed.dtype)
        if self.affine is not None and self.affine.dat is None:
            self.affine = self.affine.set_dat(dim=self.losses[0].fixed.dim,
                                              **backend)
        if self.nonlin is not None and self.nonlin.dat is None:
            space = MeanSpace([loss.fixed for loss in self.losses] +
                              [loss.moving for loss in self.losses])
            self.nonlin.set_dat(space.shape, affine=space.affine, **backend)

        if self.verbose > 1:
            for loss in self.losses:
                print(loss)
            if self.affine:
                print(self.affine)
            if self.nonlin:
                print(self.nonlin)
            print(self.optim)
            print('')

        if self.verbose:
            if self.nonlin:
                print(f'{"step":8s} | {"it":3s} | {"fit":^12s} + {"nonlin":^12s} + {"affine":^12s} = {"obj":^12s} | {"gain":^12s}')
                print('-' * 89)
            else:
                print(f'{"step":8s} | {"it":3s} | {"fit":^12s} + {"affine":^12s} = {"obj":^12s} | {"gain":^12s}')
                print('-' * 74)

        step = RegisterStep(self.losses, self.affine, self.nonlin, self.verbose)
        step.framerate = self.framerate
        if self.affine and self.nonlin:
            # if isinstance(self.optim.optim[1], optm.FirstOrder):
            #     self.optim.optim[1].preconditioner = self.nonlin.greens_apply
            # elif isinstance(self.optim.optim[1].optim, optm.FirstOrder):
            #     self.optim.optim[1].optim.preconditioner = self.nonlin.greens_apply
            self.optim.iter([self.affine.dat.dat, self.nonlin.dat.dat],
                            [step.do_affine, step.do_vel])
        elif self.affine:
            self.optim.iter(self.affine.dat.dat, step.do_affine_only)
        elif self.nonlin:
            if isinstance(self.optim, optm.FirstOrder):
                self.optim.preconditioner = self.nonlin.greens_apply
            elif isinstance(self.optim.optim.optim, optm.FirstOrder):
                self.optim.preconditioner = self.nonlin.greens_apply
            self.optim.iter(self.nonlin.dat.dat, step.do_vel)

        if self.verbose:
            print('')

        return



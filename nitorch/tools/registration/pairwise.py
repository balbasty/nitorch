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
from .objects import SVFModel, ShootModel, MeanSpace, Nonlin2dModel
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


class PairwiseRegister:
    """A class that registers pairs of images"""

    def __init__(self,
                 losses,                 # list[LossComponent]
                 affine=None,            # AffineModel
                 nonlin=None,            # NonLinModel
                 optim=None,             # Optimizer
                 verbose=True,           # verbosity level
                 framerate=1,            # plotting framerate
                 figure=None,            # figure object
                 ):
        """

        Parameters
        ----------
        losses : list[LossComponent]
            A list of losses to optimize.
        affine : AffineModel, optional
            An object describing the affine transformation model
        nonlin : NonLinModel
            An object describing the nonlinear transformation model
        optim : Optimizer
            A numerical optimizer. If both `affine` and `nonlin` are
            provided, the optimizer should accept a list of tensors
            (first: affine, second: nonlin) and
            a list of closures as arguments (e.g., SequentialOptimizer)
        verbose : bool or int, default=True
            Verbosity level.
            - 0: quiet
            - 1: print iterations
            - 2: print iterations + line searches (slower)
            - 3: plot stuff (much slower).
        framerate : float, default=1
            Update plot at most `framerate` times per second.
        """
        self.losses = losses
        self.verbose = verbose
        self.affine = affine
        self.nonlin = nonlin
        self.optim = optim
        self.framerate = framerate
        self.figure = figure

    def __call__(self):
        return self.fit()

    def fit(self):
        backend = dict(device=self.losses[0].moving.device,
                       dtype=self.losses[0].moving.dtype)
        if self.affine and self.affine.parameters() is None:
            self.affine = self.affine.set_dat(dim=self.losses[0].fixed.dim,
                                              **backend)
            # self.affine.parameters().fill_(1e-12)
        if self.nonlin and self.nonlin.parameters() is None:
            space = MeanSpace([loss.fixed for loss in self.losses] +
                              [loss.moving for loss in self.losses])
            self.nonlin.set_dat(space.shape, affine=space.affine, **backend)
            # self.nonlin.parameters().fill_(1e-12)

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

        step = PairwiseRegisterStep(self.losses, self.affine, self.nonlin,
                                    self.verbose, self.framerate, self.figure)
        self.figure = step.figure

        if self.nonlin and isinstance(self.nonlin, ShootModel):
            if self.nonlin.kernel is None:
                self.nonlin.set_kernel()

        # initialize loss
        verbose, self.verbose = self.verbose, False
        if self.nonlin:
            step.do_vel(self.nonlin.parameters(), grad=False, hess=False)
            if self.affine:
                step.do_affine(self.affine.parameters(), grad=False, hess=False)
        else:
            step.do_affine_only(self.affine.parameters(), grad=False, hess=False)
        self.verbose = verbose

        step.framerate = self.framerate
        if self.affine and self.nonlin:
            if isinstance(self.optim[1], optm.FirstOrder):
                if self.nonlin.kernel is None:
                    self.nonlin.set_kernel()
                self.optim[1].preconditioner = self.nonlin.greens_apply
            self.optim.iter([self.affine.parameters(), self.nonlin.parameters()],
                            [step.do_affine, step.do_vel])
        elif self.affine:
            self.optim.iter(self.affine.dat.dat, step.do_affine_only)
        elif self.nonlin:
            if isinstance(self.optim, optm.FirstOrder):
                if self.nonlin.kernel is None:
                    self.nonlin.set_kernel()
                self.optim.preconditioner = self.nonlin.greens_apply
            self.optim.iter(self.nonlin.parameters(), step.do_vel)

        if self.verbose:
            print('')

        return step.ll


def _almost_identity(aff):
    return torch.allclose(aff, torch.eye(*aff.shape, **utils.backend(aff)))


class PairwiseRegisterStep:
    """Forward pass of Diffeo+Affine registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(
            self,
            losses,                 # list[LossComponent]
            affine=None,            # AffineModel
            nonlin=None,            # NonLinModel
            verbose=True,           # verbosity level
            framerate=1.0,          # framerate
            figure=None,            # figure object
            ):
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        self.losses = losses
        self.affine = affine
        self.nonlin = nonlin
        self.verbose = verbose

        # pretty printing
        self.n_iter = 0             # current iteration
        self.ll = None              # Current loss (total)
        self.ll_prev = None         # previous loss (total))
        self.ll_max = 0             # max loss (total))
        self.llv = 0                # last velocity penalty
        self.lla = 0                # last affine penalty
        self.all_ll = []            # all losses (total)
        self.last_step = None

        self.figure = figure
        self.framerate = framerate
        self._last_plot = 0
        if self.verbose > 1 and not self.figure:
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

        def rescale2d(x):
            if not x.dtype.is_floating_point:
                x = x.float()
            mn, mx = utils.quantile(x, [0.005, 0.995],
                                    dim=range(-2, 0), bins=1024).unbind(-1)
            mx = mx.max(mn + 1e-8)
            mn, mx = mn[..., None, None], mx[..., None, None]
            x = x.sub(mn).div_(mx-mn).clamp_(0, 1)
            return x

        if dim == 3:
            fixed = [fixed[..., fixed.shape[-1] // 2],
                     fixed[..., fixed.shape[-2] // 2, :],
                     fixed[..., fixed.shape[-3] // 2, :, :]]
            fixed = [rescale2d(f) for f in fixed]
            moving = [moving[..., moving.shape[-1] // 2],
                      moving[..., moving.shape[-2] // 2, :],
                      moving[..., moving.shape[-3] // 2, :, :]]
            moving = [rescale2d(f) for f in moving]
            warped = [warped[..., warped.shape[-1] // 2],
                      warped[..., warped.shape[-2] // 2, :],
                      warped[..., warped.shape[-3] // 2, :, :]]
            warped = [rescale2d(f) for f in warped]
            if vel is not None:
                vel = [vel[..., vel.shape[-2] // 2, :],
                       vel[..., vel.shape[-3] // 2, :, :],
                       vel[..., vel.shape[-4] // 2, :, :, :]]
                vel = [v.square().sum(-1).sqrt() for v in vel]
        else:
            fixed = [rescale2d(f) for f in fixed]
            moving = [rescale2d(f) for f in moving]
            warped = [rescale2d(f) for f in warped]
            vel = [vel.square().sum(-1).sqrt()] if vel is not None else []

        if cat:
            moving = [math.softmax(img, dim=1, implicit=True) for img in moving]
            warped = [math.softmax(img, dim=1, implicit=True) for img in warped]

        checker = []
        for f, w in zip(fixed, warped):
            patch = max([s // 8 for s in f.shape])
            patch = [max(min(patch, s), 1) for s in f.shape]
            broad_shape = utils.expanded_shape(f.shape, w.shape)
            f = f.expand(broad_shape).clone()
            w = w.expand(broad_shape)
            checker_unfold = utils.unfold(f, patch, [2*p for p in patch])
            warped_unfold = utils.unfold(w, patch, [2*p for p in patch])
            checker_unfold.copy_(warped_unfold)
            checker.append(f)

        kdim = 3 if dim == 3 else 1
        bdim = min(nb_batch, 3)
        nb_rows = kdim * bdim + 1
        nb_cols = 4 + bool(vel)

        fig = self.figure
        replot = len(fig.axes) != (nb_rows - 1) * nb_cols + 1
        replot = replot or not getattr(self, 'plt_saved', None)
        if replot:
            fig.clf()

            for b in range(bdim):
                for k in range(kdim):
                    ax = fig.add_subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 1)
                    ax.imshow(moving[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        ax.set_title('moving')
                    ax.axis('off')
                    ax = fig.add_subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 2)
                    ax.imshow(warped[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        ax.set_title('moved')
                    ax.axis('off')
                    ax = fig.add_subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 3)
                    ax.imshow(checker[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        ax.set_title('checker')
                    ax.axis('off')
                    ax = fig.add_subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 4)
                    ax.imshow(fixed[k][b, 0].cpu())
                    if b == 0 and k == 0:
                        ax.set_title('fixed')
                    ax.axis('off')
                    if vel:
                        ax = fig.add_subplot(nb_rows, nb_cols, (b + k*bdim) * nb_cols + 5)
                        d = ax.imshow(vel[k][b].cpu())
                        if b == 0 and k == 0:
                            ax.set_title('displacement')
                        ax.axis('off')
                        fig.colorbar(d, None, ax)
            ax = fig.add_subplot(nb_rows, 1, nb_rows)
            all_ll = torch.stack(self.all_ll).cpu() if self.all_ll else []
            ax.plot(range(1, len(all_ll)+1), all_ll)
            ax.set_ylabel('NLL')
            ax.set_xlabel('iteration')
            if title:
                fig.suptitle(title)

            fig.canvas.draw()
            self.plt_saved = [fig.canvas.copy_from_bbox(ax.bbox)
                              for ax in fig.axes]
            fig.canvas.flush_events()
            plt.show(block=False)

        else:
            for elem in self.plt_saved:
                fig.canvas.restore_region(elem)

            for b in range(bdim):
                for k in range(kdim):
                    j = (b + k*bdim) * nb_cols
                    fig.axes[j].images[0].set_data(moving[k][b, 0].cpu())
                    fig.axes[j+1].images[0].set_data(warped[k][b, 0].cpu())
                    fig.axes[j+2].images[0].set_data(checker[k][b, 0].cpu())
                    fig.axes[j+3].images[0].set_data(fixed[k][b, 0].cpu())
                    if vel is not None:
                        fig.axes[j+4].images[0].set_data(vel[k][b].cpu())
            all_ll = torch.stack(self.all_ll).cpu() if self.all_ll else []
            lldata = (range(1, len(all_ll)+1), all_ll)
            fig.axes[-1].lines[0].set_data(lldata)
            fig.axes[-1].relim()
            fig.axes[-1].autoscale_view()
            if title:
                fig._suptitle.set_text(title)

            for ax in fig.axes:
                if ax.images:
                    ax.draw_artist(ax.images[0])
                else:
                    ax.draw_artist(ax.lines[0])
                fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

        self.figure = fig

    def print(self, step, ll, lla=None, llv=None, in_line_search=False):

        if lla is None:
            lla = self.lla
        if llv is None:
            llv = self.llv
        llx = ll.clone()
        ll += llv + lla

        if self.verbose and (self.verbose > 1 or not in_line_search):
            if step == 'affine_only':
                step = 'affine'
                has_vel = False
            else:
                has_vel = True
            step = 'search' if in_line_search else step
            line = f'({step}) | {self.n_iter:03d} | '
            if has_vel:
                line += f'{llx:12.6g} + {llv:12.6g} + {lla:12.6g} = {ll:12.6g}'
            else:
                line += f'{llx:12.6g} + {lla:12.6g} = {ll:12.6g}'
            if self.ll_prev is not None:
                gain = self.ll_prev - ll
                # gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                if in_line_search:
                    line += ' | :D' if gain > 0 else ' | :('
                else:
                    line += f' | {gain:12.6g}'
            print(line, end='\n')

        if not in_line_search:
            self.ll = ll
            self.llv = llv
            self.all_ll.append(ll)
            self.ll_prev = self.ll
            self.ll_max = max(self.ll_max, self.ll)
            self.n_iter += 1

        return ll

    def do_vel(self, vel, grad=False, hess=False, in_line_search=False):
        """Forward pass for updating the nonlinear component"""

        sumloss = None
        sumgrad = None
        sumhess = None

        # ==============================================================
        #                     EXPONENTIATE TRANSFORMS
        # ==============================================================
        needs_inverse = any(loss.backward for loss in self.losses)
        vel0 = vel
        if needs_inverse:
            phi0, iphi0 = self.nonlin.exp2(
                vel0, recompute=True, cache_result=not in_line_search)
            ivel0 = -vel0
        else:
            phi0 = self.nonlin.exp(
                vel0, recompute=True, cache_result=not in_line_search)
            iphi0 = ivel0 = None
        if self.affine:
            recompute_affine = self.last_step != 'nonlin'
            if needs_inverse:
                aff0, iaff0 = self.affine.exp2(
                    cache_result=True, recompute=recompute_affine)
                iaff0 = iaff0.to(phi0)
            else:
                aff0 = self.affine.exp(
                    cache_result=True, recompute=recompute_affine)
            aff0 = aff0.to(phi0)
            aff_pos = self.affine.position[0].lower()
        else:
            aff_pos = 'x'
            aff0 = iaff0 = torch.eye(self.nonlin.dim + 1,
                                     dtype=phi0.dtype, device=phi0.device)
        self.last_step = 'nonlin'

        # ==============================================================
        #                     ACCUMULATE DERIVATIVES
        # ==============================================================

        has_printed = False
        for loss in self.losses:
            do_print = not (has_printed or self.verbose < 3 or in_line_search
                            or loss.backward)

            # ==========================================================
            #                     ONE LOSS COMPONENT
            # ==========================================================
            moving, fixed, factor = loss.moving, loss.fixed, loss.factor
            if loss.backward:
                phi00, aff00, vel00 = iphi0, iaff0, ivel0
            else:
                phi00, aff00, vel00 = phi0, aff0, vel0

            # ----------------------------------------------------------
            # build left and right affine
            # ----------------------------------------------------------
            aff_right = fixed.affine
            if aff_pos in 'fs':  # affine position: fixed or symmetric
                aff_right = aff00 @ aff_right
            aff_right = linalg.lmdiv(self.nonlin.affine, aff_right)
            aff_left = self.nonlin.affine
            if aff_pos in 'ms':  # affine position: moving or symmetric
                aff_left = aff00 @ aff_left
            aff_left = linalg.lmdiv(moving.affine, aff_left)

            # ----------------------------------------------------------
            # build full transform
            # ----------------------------------------------------------
            if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                aff_right = None
                phi = spatial.add_identity_grid(phi00)
                disp = phi00
            else:
                right = spatial.affine_grid(aff_right, fixed.shape)
                phi = regutils.smart_pull_grid(phi00, right)
                if do_print:
                    disp = phi.clone()
                phi += right
            if _almost_identity(aff_left) and moving.shape == self.nonlin.shape:
                aff_left = None
            else:
                phi = spatial.affine_matvec(aff_left, phi)

            # ----------------------------------------------------------
            # forward pass
            # ----------------------------------------------------------
            warped, warped_mask = moving.pull(phi, mask=True)
            if fixed.masked:
                if warped_mask is None:
                    mask = fixed.mask
                else:
                    mask = warped_mask * fixed.mask
            else:
                mask = warped_mask

            if do_print:
                disp = linalg.matvec(aff_right.inverse()[:-1, :-1], disp)
                has_printed = True
                if moving.previewed:
                    preview = moving.pull(phi, preview=True, dat=False)
                else:
                    preview = warped
                init = spatial.affine_lmdiv(moving.affine, fixed.affine)
                if _almost_identity(init) and moving.shape == fixed.shape:
                    init = moving.dat
                else:
                    init = spatial.affine_grid(init, fixed.shape)
                    initmask, init = moving.pull(init, preview=True, dat=False, mask=True)
                    if initmask is not None:
                        init = init * initmask
                dat = fixed.dat * fixed.mask if fixed.masked else fixed.dat
                preview = preview * warped_mask if warped_mask is not None else preview
                self.mov2fix(dat, init, preview, disp, dim=fixed.dim,
                             title=f'(nonlin) {self.n_iter:03d}')

            # ----------------------------------------------------------
            # derivatives wrt moving
            # ----------------------------------------------------------
            g = h = None
            loss_args = (warped, fixed.dat)
            loss_kwargs = dict(dim=fixed.dim, mask=mask)
            state = loss.loss.get_state()
            if not grad and not hess:
                llx = loss.loss.loss(*loss_args, **loss_kwargs)
            elif not hess:
                llx, g = loss.loss.loss_grad(*loss_args, **loss_kwargs)
            else:
                llx, g, h = loss.loss.loss_grad_hess(*loss_args, **loss_kwargs)
            del loss_args, loss_kwargs
            if in_line_search:
                loss.loss.set_state(state)

            # ----------------------------------------------------------
            # chain rule -> derivatives wrt phi
            # ----------------------------------------------------------
            if grad or hess:

                g, h, mugrad = self.nonlin.propagate_grad(
                    g, h, moving, phi00, aff_left, aff_right,
                    inv=loss.backward)
                g = regutils.jg(mugrad, g)
                h = regutils.jhj(mugrad, h)
                is_svf = isinstance(self.nonlin, SVFModel)
                is_svf = is_svf or (isinstance(self.nonlin, Nonlin2dModel) and
                                    isinstance(self.nonlin._model, SVFModel))
                if is_svf:
                    # propagate backward by scaling and squaring
                    g, h = spatial.exp_backward(vel00, g, h,
                                                steps=self.nonlin.steps)

                sumgrad = (g.mul_(factor) if sumgrad is None else
                           sumgrad.add_(g, alpha=factor))
                if hess:
                    sumhess = (h.mul_(factor) if sumhess is None else
                               sumhess.add_(h, alpha=factor))
            sumloss = (llx.mul_(factor) if sumloss is None else
                       sumloss.add_(llx, alpha=factor))

        # ==============================================================
        #                       REGULARIZATION
        # ==============================================================
        vgrad = self.nonlin.regulariser(vel0)
        llv = 0.5 * vel0.flatten().dot(vgrad.flatten())
        if grad:
            sumgrad += vgrad
        del vgrad

        # ==============================================================
        #                           VERBOSITY
        # ==============================================================
        sumloss = self.print('nonlin', sumloss, lla=self.lla, llv=llv,
                             in_line_search=in_line_search)

        # ==============================================================
        #                           RETURN
        # ==============================================================
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

        # ==============================================================
        #                     EXPONENTIATE TRANSFORMS
        # ==============================================================
        logaff0 = logaff
        aff_pos = self.affine.position[0].lower()
        needs_inverse = any(loss.backward for loss in self.losses)
        recompute_nonlin = self.last_step != 'affine'
        if needs_inverse:
            aff0, iaff0, gaff0, igaff0 = self.affine.exp2(
                logaff0, grad=True, cache_result=not in_line_search)
            phi0, iphi0 = self.nonlin.exp2(
                cache_result=True, recompute=recompute_nonlin)
        else:
            iaff0, igaff0, iphi0 = None, None, None
            aff0, gaff0 = self.affine.exp(
                logaff0, grad=True, cache_result=not in_line_search)
            phi0 = self.nonlin.exp(
                cache_result=True, recompute=recompute_nonlin)
        self.last_step = 'affine'

        has_printed = False
        for loss in self.losses:

            moving, fixed, factor = loss.moving, loss.fixed, loss.factor
            if loss.backward:
                phi00, aff00, gaff00 = iphi0, iaff0, igaff0
            else:
                phi00, aff00, gaff00 = phi0, aff0, gaff0

            # ----------------------------------------------------------
            # build left and right affine matrices
            # ----------------------------------------------------------
            aff_right, gaff_right = fixed.affine, None
            if aff_pos in 'fs':
                gaff_right = gaff00 @ aff_right
                gaff_right = linalg.lmdiv(self.nonlin.affine, gaff_right)
                aff_right = aff00 @ aff_right
            aff_right = linalg.lmdiv(self.nonlin.affine, aff_right)
            aff_left, gaff_left = self.nonlin.affine, None
            if aff_pos in 'ms':
                gaff_left = gaff00 @ aff_left
                gaff_left = linalg.lmdiv(moving.affine, gaff_left)
                aff_left = aff00 @ aff_left
            aff_left = linalg.lmdiv(moving.affine, aff_left)

            # ----------------------------------------------------------
            # build full transform
            # ----------------------------------------------------------
            if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                right = None
                phi = spatial.add_identity_grid(phi00)
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

            # ----------------------------------------------------------
            # forward pass
            # ----------------------------------------------------------
            warped, warped_mask = moving.pull(phi, mask=True)
            if fixed.masked:
                if warped_mask is None:
                    mask = fixed.mask
                else:
                    mask = warped_mask * fixed.mask
            else:
                mask = warped_mask

            do_print = not (has_printed or self.verbose < 3 or in_line_search
                            or loss.backward)
            if do_print:
                has_printed = True
                if moving.previewed:
                    preview = moving.pull(phi, preview=True, dat=False)
                else:
                    preview = warped
                init = spatial.affine_lmdiv(moving.affine, fixed.affine)
                if _almost_identity(init) and moving.shape == fixed.shape:
                    init = moving.dat
                else:
                    init = spatial.affine_grid(init, fixed.shape)
                    initmask, init = moving.pull(init, preview=True, dat=False, mask=True)
                    if initmask is not None:
                        init = init * initmask
                dat = fixed.dat * fixed.mask if fixed.masked else fixed.dat
                preview = preview * warped_mask if warped_mask is not None else preview
                self.mov2fix(dat, init, preview, dim=fixed.dim,
                             title=f'(affine) {self.n_iter:03d}')

            # ----------------------------------------------------------
            # derivatives wrt moving
            # ----------------------------------------------------------
            g = h = None
            loss_args = (warped, fixed.dat)
            loss_kwargs = dict(dim=fixed.dim, mask=mask)
            state = loss.loss.get_state()
            if not grad and not hess:
                llx = loss.loss.loss(*loss_args, **loss_kwargs)
            elif not hess:
                llx, g = loss.loss.loss_grad(*loss_args, **loss_kwargs)
            else:
                llx, g, h = loss.loss.loss_grad_hess(*loss_args, **loss_kwargs)
            del loss_args, loss_kwargs
            if in_line_search:
                loss.loss.set_state(state)

            # ----------------------------------------------------------
            # chain rule -> derivatives wrt Lie parameters
            # ----------------------------------------------------------

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
                    # h = h.abs().sum(-1).diag_embed()
                return g, h

            if grad or hess:
                g0, g = g, None
                h0, h = h, None
                if aff_pos in 'ms':
                    g_left = regutils.smart_push(g0, phi_right, shape=self.nonlin.shape)
                    h_left = regutils.smart_push(h0, phi_right, shape=self.nonlin.shape)
                    mugrad = moving.pull_grad(left, rotate=False)
                    g_left, h_left = compose_grad(g_left, h_left, mugrad, gaff_left)
                    g, h = g_left, h_left
                if aff_pos in 'fs':
                    g_right, h_right = g0, h0
                    mugrad = moving.pull_grad(phi, rotate=False)
                    jac = spatial.grid_jacobian(phi0, right, type='disp', extrapolate=False)
                    jac = torch.matmul(aff_left[:-1, :-1], jac)
                    mugrad = linalg.matvec(jac.transpose(-1, -2), mugrad)
                    g_right, h_right = compose_grad(g_right, h_right, mugrad, gaff_right)
                    g = g_right if g is None else g.add_(g_right)
                    h = h_right if h is None else h.add_(h_right)

                if loss.backward:
                    g = g.neg_()
                sumgrad = (g.mul_(factor) if sumgrad is None else
                           sumgrad.add_(g, alpha=factor))
                if hess:
                    sumhess = (h.mul_(factor) if sumhess is None else
                               sumhess.add_(h, alpha=factor))
            sumloss = (llx.mul_(factor) if sumloss is None else
                       sumloss.add_(llx, alpha=factor))

        # TODO add regularization term
        lla = 0

        # ==============================================================
        #                           VERBOSITY
        # ==============================================================
        sumloss = self.print('affine', sumloss, lla=lla, llv=self.llv,
                             in_line_search=in_line_search)

        # ==============================================================
        #                           RETURN
        # ==============================================================
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

        # ==============================================================
        #                     EXPONENTIATE TRANSFORMS
        # ==============================================================
        logaff0 = logaff
        aff0 = iaff0 = gaff0 = igaff0 = None
        if all(loss.backward for loss in self.losses):
            iaff0, igaff0 = self.affine.iexp(logaff0, grad=True)
        elif not any(loss.backward for loss in self.losses):
            aff0, gaff0 = self.affine.exp(logaff0, grad=True)
        else:
            aff0, iaff0, gaff0, igaff0 = self.affine.exp2(logaff0, grad=True)

        has_printed = False
        for loss in self.losses:

            moving, fixed, factor = loss.moving, loss.fixed, loss.factor
            if loss.backward:
                aff00, gaff00 = iaff0, igaff0
            else:
                aff00, gaff00 = aff0, gaff0

            # ----------------------------------------------------------
            # build full transform
            # ----------------------------------------------------------
            aff = aff00 @ fixed.affine
            aff = linalg.lmdiv(moving.affine, aff)
            gaff = gaff00 @ fixed.affine
            gaff = linalg.lmdiv(moving.affine, gaff)
            phi = spatial.affine_grid(aff, fixed.shape)

            # ----------------------------------------------------------
            # forward pass
            # ----------------------------------------------------------
            warped, mask = moving.pull(phi, mask=True)
            if fixed.masked:
                if mask is None:
                    mask = fixed.mask
                else:
                    mask = mask * fixed.mask

            do_print = not (has_printed or self.verbose < 3 or in_line_search
                            or loss.backward)
            if do_print:
                has_printed = True
                if moving.previewed:
                    preview = moving.pull(phi, preview=True, dat=False)
                else:
                    preview = warped
                init = spatial.affine_lmdiv(moving.affine, fixed.affine)
                if _almost_identity(init) and moving.shape == fixed.shape:
                    init = moving.preview
                else:
                    init = spatial.affine_grid(init, fixed.shape)
                    init = moving.pull(init, preview=True, dat=False)
                self.mov2fix(fixed.preview, init, preview, dim=fixed.dim,
                             title=f'(affine) {self.n_iter:03d}')

            # ----------------------------------------------------------
            # derivatives wrt moving
            # ----------------------------------------------------------
            g = h = None
            loss_args = (warped, fixed.dat)
            loss_kwargs = dict(dim=fixed.dim, mask=mask)
            state = loss.loss.get_state()
            if not grad and not hess:
                llx = loss.loss.loss(*loss_args, **loss_kwargs)
            elif not hess:
                llx, g = loss.loss.loss_grad(*loss_args, **loss_kwargs)
            else:
                llx, g, h = loss.loss.loss_grad_hess(*loss_args, **loss_kwargs)
            del loss_args, loss_kwargs
            if in_line_search:
                loss.loss.set_state(state)

            # ----------------------------------------------------------
            # chain rule -> derivatives wrt Lie parameters
            # ----------------------------------------------------------

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
                    # h = h.abs().sum(-1).diag_embed()
                return g, h

            # compose with spatial gradients
            if grad or hess:
                mugrad = moving.pull_grad(phi, rotate=False)
                g, h = compose_grad(g, h, mugrad, gaff)

                if loss.backward:
                    g = g.neg_()
                sumgrad = (g.mul_(factor) if sumgrad is None else
                           sumgrad.add_(g, alpha=factor))
                if hess:
                    sumhess = (h.mul_(factor) if sumhess is None else
                               sumhess.add_(h, alpha=factor))
            sumloss = (llx.mul_(factor) if sumloss is None else
                       sumloss.add_(llx, alpha=factor))

        # TODO add regularization term
        lla = 0

        # ==============================================================
        #                           VERBOSITY
        # ==============================================================
        sumloss = self.print('affine_only', sumloss, lla=lla,
                             in_line_search=in_line_search)

        # ==============================================================
        #                           RETURN
        # ==============================================================
        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]



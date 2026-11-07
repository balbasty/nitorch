"""Joint affine + nonlinear registration.

This model is the most polished one. All simpler models (affine, smalldef,
svf, shoot) can be performed by the joint model as sub-cases.

Given its flexibility, there is no simple function to call it. I used
an object-oriented representation of the different blocks (implemented in
registration.objects) to encode the images, transformations and optimizers.

The command line function `nitorch register` calls this model under the hood.

"""
from threading import Thread, Lock
from nitorch import spatial
from nitorch.core import utils, linalg, py
import torch
from . import optim as optm, utils as regutils
from .objects import (SVFModel, ShootModel, MeanSpace, Nonlin2dModel,
                      SumSimilarity, AffineModel)


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



def _new_figure():
    """New matplotlib figure, or None (with a warning) if unavailable"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    try:
        return plt.figure()
    except Exception as e:  # e.g. broken interactive backend
        import warnings
        warnings.warn(f'Cannot create a figure ({e}): live plotting is '
                      f'disabled.')
        return None

class PairwiseRegister:
    """A class that registers pairs of images"""

    def __init__(self,
                 losses,                 # list[LossComponent]
                 affine=None,            # AffineModel
                 nonlin=None,            # NonLinModel
                 optim=None,             # Optimizer
                 verbose=True,           # verbosity level
                 framerate=1,            # plotting framerate
                 figure=None,            # Matplotlib figure
                 hessian=False,          # nonlin Hessian at the solution
                 ):
        """

        Parameters
        ----------
        losses : Similarity
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
        hessian : bool, default=False
            Make sure that `nonlin.hess` holds the Hessian at the final
            parameters. The optimiser only returns it if its last step
            evaluated it there (e.g., not after a Wolfe line search);
            otherwise it is evaluated once at the end of `fit`.

        Returns
        -------
        total_loss : scalar
            Total loss of the joint problem
        """
        self.losses = SumSimilarity.sum(losses)
        self.hessian = hessian
        self.verbose = verbose
        self.affine = affine
        self.nonlin = nonlin
        self.optim = optim
        self.framerate = framerate
        self.figure = figure

    def __call__(self):
        return self.fit()

    def _set_step_sizes(self):
        """Give each optimiser that stops on the size of its updates
        (`disp_tol`) the function that measures them.

        Returns
        -------
        hooks : dict[{'affine', 'nonlin'} -> callable]
            The functions attached, by component.
        """
        hooks = {}
        targets = _step_targets(self.losses)
        affine = self.affine if not getattr(self.affine, 'frozen', False) else None
        nonlin = self.nonlin if not getattr(self.nonlin, 'frozen', False) else None
        if affine and nonlin:
            pairs = [(self.optim[0], 'affine'), (self.optim[1], 'nonlin')]
        elif affine:
            pairs = [(self.optim, 'affine')]
        elif nonlin:
            pairs = [(self.optim, 'nonlin')]
        else:
            pairs = []
        for opt, kind in pairs:
            it = getattr(opt, 'iter', None)
            if not isinstance(it, optm.OptimIterator) or not it.disp_tol:
                continue
            if kind == 'affine':
                it.step_size = _affine_step_size(self.affine, targets)
            else:
                it.step_size = _nonlin_step_size(self.nonlin, targets)
            hooks[kind] = it.step_size
        return hooks

    def fit(self):
        # Transformation parameters must be floating point, even when
        # the images are not (e.g., discretized images are integers).
        dtype = self.losses[0].moving.dtype
        if not dtype.is_floating_point:
            dtype = self.losses[0].fixed.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        backend = dict(device=self.losses[0].moving.device, dtype=dtype)
        if self.affine and self.affine.parameters() is None:
            self.affine = self.affine.set_dat(dim=self.losses[0].fixed.dim,
                                              **backend)
        if self.affine:
            self.affine.to_(**backend)
        if self.nonlin and self.nonlin.parameters() is None:
            space = MeanSpace([loss.fixed for loss in self.losses] +
                              [loss.moving for loss in self.losses])
            self.nonlin.set_dat(space.shape, affine=space.affine, **backend)
        if self.nonlin:
            self.nonlin.to_(**backend)

        if self.verbose > 1:
            print(self.losses)
            if self.affine:
                print(self.affine)
            if self.nonlin:
                print(self.nonlin)
            print(self.optim)
            print('')

        # size of the accepted updates, printed when used for early stopping
        disp_hooks = self._set_step_sizes()
        if self.verbose:
            disp = f' | {"disp (vox)":^12s}' if disp_hooks else ''
            width = 15 if disp_hooks else 0
            if self.nonlin:
                print(f'{"step":8s} | {"it":3s} | {"fit":^12s} + {"nonlin":^12s} + {"affine":^12s} = {"obj":^12s} | {"gain":^12s}' + disp)
                print('-' * (89 + width))
            else:
                print(f'{"step":8s} | {"it":3s} | {"fit":^12s} + {"affine":^12s} = {"obj":^12s} | {"gain":^12s}' + disp)
                print('-' * (74 + width))

        step = PairwiseRegisterStep(self.losses, self.affine, self.nonlin,
                                    self.verbose, self.framerate, self.figure)
        step.disp_hooks = disp_hooks
        self.figure = step.figure

        if self.nonlin and isinstance(self.nonlin, ShootModel):
            if self.nonlin.kernel is None:
                self.nonlin.set_kernel()

        # initialize loss
        if not self.nonlin:
            self.affine.position, pos = 'm', self.affine.position
        if any(loss.loss.get_state() is not None for loss in self.losses):
            # Stateful losses (warm-started EM, IRLS weights...): this
            # evaluation initialises their state before the optimiser's
            # first evaluation, as it always did.
            verbose, self.verbose = self.verbose, False
            if self.nonlin:
                step.do_vel(self.nonlin.parameters(), grad=False, hess=False)
                if self.affine:
                    step.do_affine(self.affine.parameters(), grad=False, hess=False)
            else:
                step.do_affine_only(self.affine.parameters(), grad=False, hess=False)
            self.verbose = verbose
        else:
            # Every optimiser evaluates the objective at the initial
            # parameters itself, so evaluating the data term here would be
            # a duplicate: only initialise the penalty terms that the
            # steps of a joint model take from each other.
            step.init_penalties()

        step.framerate = self.framerate
        if self.affine and self.nonlin:
            if getattr(self.affine, 'frozen', False):
                nonlin_optim = self.optim
            elif getattr(self.nonlin, 'frozen', False):
                nonlin_optim = None
            else:
                nonlin_optim = self.optim[1]
            if isinstance(nonlin_optim, optm.FirstOrder):
                if self.nonlin.kernel is None:
                    self.nonlin.set_kernel()
                nonlin_optim.preconditioner = self.nonlin.greens_apply
            if getattr(self.affine, 'frozen', False):
                out = self.optim.iter(self.nonlin.parameters(), step.do_vel)
            elif getattr(self.nonlin, 'frozen', False):
                out = self.optim.iter(self.affine.parameters(), step.do_affine)
            else:
                out = self.optim.iter(
                    [self.affine.parameters(), self.nonlin.parameters()],
                    [step.do_affine, step.do_vel]
                )
        elif self.affine:
            out = self.optim.iter(self.affine.parameters(), step.do_affine_only)
        elif self.nonlin:
            if isinstance(self.optim, optm.FirstOrder):
                if self.nonlin.kernel is None:
                    self.nonlin.set_kernel()
                self.optim.preconditioner = self.nonlin.greens_apply
            out = self.optim.iter(self.nonlin.parameters(), step.do_vel)

        if self.verbose:
            print('')

        if self.affine and not self.nonlin:
            self.affine.position = pos

        if self.affine:
            self.affine.grad = None
            self.affine.hess = None
            if len(out) > 2:
                grad = out[2]
                if isinstance(grad, (list, tuple)):
                    grad = grad[0]
                self.affine.grad = grad
            if len(out) > 3:
                hess = out[3]
                if isinstance(hess, (list, tuple)):
                    hess = hess[0]
                self.affine.hess = hess

        if self.nonlin:
            self.nonlin.grad = None
            self.nonlin.hess = None
            if len(out) > 2:
                grad = out[2]
                if isinstance(grad, (list, tuple)):
                    grad = grad[-1]
                self.nonlin.grad = grad
            if len(out) > 3:
                hess = out[3]
                if isinstance(hess, (list, tuple)):
                    hess = hess[-1]
                self.nonlin.hess = hess
            if (self.hessian and self.nonlin.hess is None
                    and not getattr(self.nonlin, 'frozen', False)):
                # not returned by the optimiser (e.g., the last step was a
                # Wolfe line search, which does not evaluate it): evaluate
                # it at the final parameters
                verbose, step.verbose = step.verbose, 0
                _, _, self.nonlin.hess = step.do_vel(
                    self.nonlin.parameters(), grad=True, hess=True)
                step.verbose = verbose

        return step.ll


# ----------------------------------------------------------------------
#   Size of an accepted update, for displacement-based early stopping
#   (#266): RMS of the displacement that the update causes, over the
#   voxels of the images on which the losses are evaluated, each measured
#   in the voxel coordinates of its own image (current pyramid level).
#
#   The "targets" are the (image, direction) pairs on whose grid a loss is
#   computed: each loss's `fixed` image, warped by the forward transform,
#   or by the inverse one if `loss.backward` (the reversed half of a
#   symmetric loss, whose `fixed` is the original moving image).
# ----------------------------------------------------------------------

def _image_weights(image):
    """Weights of an image's voxels: its mask if any, else ones."""
    dat = image.dat
    if image.masked:
        w = image.mask.to(device=dat.device, dtype=torch.double)
        while w.dim() > image.dim:      # channels: valid in any channel
            w = w.amax(0)
        return (w > 0).double()
    return torch.ones(image.shape, dtype=torch.double, device=dat.device)


def _step_targets(losses):
    """Distinct (image, backward) pairs on whose grid the losses are
    evaluated."""
    targets, seen = [], set()
    for loss in losses:
        key = (id(loss.fixed), bool(loss.backward))
        if key not in seen:
            seen.add(key)
            targets.append((loss.fixed, bool(loss.backward)))
    return targets


def _affine_step_size(affine, targets):
    """Exact RMS displacement of an affine update, in closed form.

    For a target image with voxel-to-world matrix `F` (linear part `F3`)
    and homogeneous world coordinates `x~` of its (masked) voxels, the
    displacement of a voxel, in that image's voxel coordinates, is
    `F3^-1 (A1 - A0) x~` (A = forward transform, or its inverse for a
    backward target), so that its mean square is `tr(R E[x~ x~'] R')`,
    with `R = F3^-1 (A1 - A0)`. The second moments of each target grid
    are computed once.
    """
    if not isinstance(targets, (list, tuple)):
        targets = [(targets, False)]
    stats = []
    for image, backward in targets:
        w = _image_weights(image)
        dim = w.dim()
        total = w.sum()
        coords = [torch.arange(n, dtype=torch.double, device=w.device).reshape(
                  [n if d == i else 1 for i in range(dim)])
                  for d, n in enumerate(w.shape)]
        mom = torch.zeros([dim + 1, dim + 1], dtype=torch.double,
                          device=w.device)
        mom[-1, -1] = 1
        for d in range(dim):
            wd = w * coords[d]
            mom[d, -1] = mom[-1, d] = wd.sum() / total
            for e in range(d, dim):
                mom[d, e] = mom[e, d] = (wd * coords[e]).sum() / total
            del wd
        vox2world = image.affine.to(mom)
        mom = vox2world.matmul(mom).matmul(vox2world.T)       # world moments
        world2vox = torch.linalg.inv(vox2world[:dim, :dim])
        stats.append((backward, total, mom, world2vox))
    total = sum(t for _, t, _, _ in stats)

    def step_size(prev, new):
        q0 = getattr(prev, 'dat', prev)
        q1 = getattr(new, 'dat', new)
        rms2 = 0
        for backward, count, mom, world2vox in stats:
            exp = affine.dat.iexp if backward else affine.dat.exp
            delta = exp(q1).to(mom) - exp(q0).to(mom)
            delta = world2vox.matmul(delta[..., :dim, :])
            ms = delta.matmul(mom).matmul(delta.transpose(-1, -2))
            rms2 += count * ms.diagonal(0, -1, -2).sum(-1).mean()
        return (rms2 / total).clamp_min(0).sqrt().item()
    return step_size


def _nonlin_step_size(nonlin, targets):
    """RMS of the update of the nonlinear parameters, over the target
    images, in each image's voxel coordinates.

    The parameters live on the nonlinear lattice (voxel-to-world `L`), in
    lattice voxels: an update `dv` is `L3 dv` in world units and
    `F3^-1 L3 dv` in the voxels of a target image (voxel-to-world `F`).
    Each lattice voxel is weighted by the number of (masked) target voxels
    it covers.

    smalldef: the parameter is the displacement: exact.
    svf, shoot: the parameter is a velocity; the change of the displacement
    is `exp(v + dv) - exp(v) ~= dv`, transported and stretched by the flow,
    to first order (and `-dv` for the inverse, for backward targets). The
    RMS of `dv` is used.
    """
    if not isinstance(targets, (list, tuple)):
        targets = [(targets, False)]
    lattice = list(nonlin.shape)
    lat2world = nonlin.affine.double()
    dim = len(lattice)
    stats = []
    for image, _ in targets:
        w = _image_weights(image)
        vox2world = image.affine.double().to(w.device)
        lat2world = lat2world.to(w.device)
        if list(w.shape) == lattice and torch.allclose(vox2world, lat2world):
            weights = w
        else:
            # fraction of each lattice voxel that falls in the target image
            lat2vox = torch.linalg.inv(vox2world).matmul(lat2world)
            grid = spatial.affine_grid(lat2vox.to(w), lattice)
            weights = spatial.grid_pull(w[None], grid, bound='zero',
                                        extrapolate=False, interpolation=1)[0]
        # number of target voxels per lattice voxel
        weights = weights * (lat2world[:dim, :dim].det().abs()
                             / vox2world[:dim, :dim].det().abs())
        # lattice voxels -> target voxels
        lat2tgt = torch.linalg.inv(vox2world[:dim, :dim]).matmul(
            lat2world[:dim, :dim])
        stats.append((weights, lat2tgt))
    total = sum(w.sum() for w, _ in stats)

    def step_size(prev, new):
        delta = (new - prev).reshape([-1, *lattice, dim])
        rms2 = 0
        for weights, lat2tgt in stats:
            d = delta.matmul(lat2tgt.T.to(delta)).square_().sum(-1)
            rms2 += (d * weights.to(d)).sum() / len(d)
        return (rms2 / total).clamp_min(0).sqrt().item()
    return step_size


def _almost_identity(aff):
    return torch.allclose(aff, torch.eye(*aff.shape, **utils.backend(aff)))


def _backward_position(pos):
    """Side of the affine in the inverse transform.

    (A o phi)^-1 = phi^-1 o A^-1: an affine applied on the moving side of
    the forward transform is applied on the fixed side of its inverse
    (and vice versa). A symmetric affine stays on both sides.
    """
    return {'m': 'f', 'f': 'm'}.get(pos, pos)


class SampledVoxels:
    """The voxels of a fixed image on which a loss is evaluated: all of
    them, or a fixed random subset.

    Either way, it gives the fixed data and mask at these voxels and maps
    them through an affine, so that the forward pass does not depend on
    whether it samples.

    A subset is drawn uniformly, without replacement, among the voxels
    of the image (those in its mask, if it has one), and laid out as an
    image of shape `(n, 1, ..., 1)`, so that pointwise losses can be
    evaluated on it as on any image. Each sampled voxel is weighted by
    the inverse of its probability of being drawn (Horvitz-Thompson), so
    that sums over the subset are unbiased estimates of sums over the
    image. If there are fewer candidate voxels than samples, the image
    is used as is.

    Attributes
    ----------
    sampled : bool
        Whether the voxels are a subset of the image
    n, ncand : int
        Number of voxels used, and of candidate voxels (in the mask)
    grid : (n, 1, ..., 1, dim) tensor or None
        Voxel coordinates of the samples (None: the image lattice)
    dat : (C, *spatial) or (C, n, 1, ..., 1) tensor
        Image values at the voxels
    mask : (C|1, *spatial) or (C|1, n, 1, ..., 1) tensor or None
        Weights of the voxels (mask, x inverse sampling probability)
    """

    def __init__(self, image, samples=None, generator=None):
        self.image = image
        self.sampled = False
        self.grid = None
        self.dat = image.dat
        self.mask = image.mask if image.masked else None
        self.n = self.ncand = py.prod(image.shape)
        if not samples:
            return

        dim, shape, dat = image.dim, list(image.shape), image.dat
        inside = None
        if image.masked:
            inside = image.mask
            while inside.dim() > dim:  # channels: valid in any channel
                inside = inside.amax(0)
            inside = (inside > 0).flatten().nonzero().squeeze(-1)
            self.n = self.ncand = len(inside)
        if samples >= self.ncand:
            return

        n, ncand, dtype = int(samples), self.ncand, image._float_dtype
        idx = torch.randperm(ncand, generator=generator)[:n]
        idx = idx.sort().values.to(dat.device)  # sorted: local memory access
        if inside is not None:
            idx = inside[idx]

        layout = [n] + [1] * (dim - 1)
        coords, rem = [], idx
        for size in reversed(shape):
            coords.append(rem % size)
            rem = rem // size
        coords = torch.stack(coords[::-1], -1).to(dtype)
        self.grid = coords.reshape([*layout, dim])
        self.dat = dat.reshape([len(dat), -1])[:, idx]
        self.dat = self.dat.reshape([len(dat), *layout])
        weight = ncand / n
        if image.masked:
            mask = image.mask.to(dtype)
            mask = mask.reshape([len(mask), -1])[:, idx].mul_(weight)
        else:
            mask = torch.full([1, n], weight, dtype=dtype, device=dat.device)
        self.mask = mask.reshape([len(mask), *layout])
        self.n, self.sampled = n, True

    def coords(self, like):
        """Voxel coordinates of the samples (None: the image lattice)"""
        return None if self.grid is None else self.grid.to(like)

    def affine_grid(self, aff):
        """Map the voxels through a voxel-to-voxel affine"""
        if self.grid is None:
            return spatial.affine_grid(aff, self.image.shape)
        return spatial.affine_matvec(aff, self.coords(aff))

    def loss_kwargs(self, loss):
        """Options of a loss evaluated on these voxels: on a subset,
        histogram losses whose bins are automatic take the bins of the
        whole image, not those of the subset"""
        if self.sampled and hasattr(loss, 'resolved_bins') and not loss.bins:
            return dict(bins=loss.resolved_bins(self.image.dat,
                                                self.image.dim))
        return {}


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
            framerate=1,
            figure=None,
            ):
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
        self.disp_hooks = {}        # size of an update, by component
        self._disp_prev = {}        # parameters at the last printed row
        self._samples = {}          # voxel samples, by fixed image
        self._sample_rng = None     # their (seeded) random generator

        self.framerate = framerate
        self.figure = figure
        self._last_plot = 0
        self._plot_lock = Lock()
        self._plot_pending = None
        if self.verbose > 2 and not self.figure:
            self.figure = _new_figure()

    def mov2fix(self, *args, **kwargs):
        if not self.figure:  # matplotlib unavailable: no live plot
            return
        thread = Thread(target=(lambda: self._mov2fix(*args, **kwargs)))
        if self._plot_lock.locked():
            self._plot_pending = thread
        else:
            thread.run()

    def _mov2fix(self, fixed, moving, warped, vel=None, title=None, lam=None):
        """Plot registration live"""

        self._plot_lock.acquire()
        self._plot_pending = None

        import time
        # tic = self._last_plot
        # toc = time.time()
        # if toc - tic < 1/self.framerate:
        #     return

        import matplotlib.pyplot as plt

        warped = warped.detach()
        if vel is not None:
            vel = vel.detach()

        dim = moving.dim() - 1
        while fixed.dim() < dim + 1:
            fixed = fixed[None]
        moving = moving[0]
        fixed = fixed[0]
        warped = warped[0]
        if lam is not None:
            lam = lam[0]

        def rescale2d(x):
            if not x.dtype.is_floating_point:
                x = x.float()
            mask = (x == 0).any(dim=list(range(x.ndim-2)))
            xmasked = x[~mask]
            mn, mx = utils.quantile(xmasked, [0.005, 0.995],
                                    dim=range(-2, 0), bins=1024).unbind(-1)
            mx = mx.max(mn + 1e-8)
            mn, mx = mn[..., None, None], mx[..., None, None]
            x = x.sub(mn).div_(mx-mn).clamp_(0, 1)
            return x

        if dim == 3:
            fixed = [fixed[:, :, fixed.shape[-1] // 2],
                     fixed[:, fixed.shape[-2] // 2, :],
                     fixed[fixed.shape[-3] // 2, :, :]]
            fixed = [rescale2d(f) for f in fixed]
            moving = [moving[:, :, moving.shape[-1] // 2],
                      moving[:, moving.shape[-2] // 2, :],
                      moving[moving.shape[-3] // 2, :, :]]
            moving = [rescale2d(f) for f in moving]
            warped = [warped[:, :, warped.shape[-1] // 2],
                      warped[:, warped.shape[-2] // 2, :],
                      warped[warped.shape[-3] // 2, :, :]]
            warped = [rescale2d(f) for f in warped]
            if vel is not None:
                vel = [vel[:, :, vel.shape[-2] // 2, :],
                       vel[:, vel.shape[-3] // 2, :, :],
                       vel[vel.shape[-4] // 2, :, :, :]]
                vel = [v.square().sum(-1).sqrt() for v in vel]
            if lam is not None:
                lam = [lam[:, :, lam.shape[-1] // 2],
                       lam[:, lam.shape[-2] // 2, :],
                       lam[lam.shape[-3] // 2, :, :]]
                lam = [rescale2d(f) for f in lam]
        else:
            fixed = [rescale2d(fixed)]
            moving = [rescale2d(moving)]
            warped = [rescale2d(warped)]
            vel = [vel.square().sum(-1).sqrt()] if vel is not None else []
            lam = [rescale2d(lam)] if lam is not None else []

        if vel:
            vmax = max(v.max().item() for v in vel)
        if lam:
            lmax = max(x.max().item() for x in lam)

        checker = []
        for f, w in zip(fixed, warped):
            patch = max([s // 8 for s in f.shape])
            patch = [max(min(patch, s), 1) for s in f.shape]
            broad_shape = utils.expanded_shape(f.shape, w.shape)
            f = f.expand(broad_shape).clone()
            w = w.expand(broad_shape)
            # red
            checker_unfold = utils.unfold(f, patch, [2*p for p in patch])
            warped_unfold = utils.unfold(w, patch, [2*p for p in patch])
            checker_unfold.copy_(warped_unfold)
            # black
            fblack = f[tuple(slice(p, None) for p in patch)]
            wblack = w[tuple(slice(p, None) for p in patch)]
            if fblack.numel() and wblack.numel():
                checker_unfold = utils.unfold(fblack, patch, [2*p for p in patch])
                warped_unfold = utils.unfold(wblack, patch, [2*p for p in patch])
                checker_unfold.copy_(warped_unfold)
            checker.append(f)

        kdim = len(fixed)
        nb_rows = kdim + 1
        nb_cols = 4 + bool(vel) + bool(lam)

        imkwargs = dict(interpolation='nearest')

        def bbox2rect(text):
            (left, bottom, width, height) = text.get_tightbbox().bounds
            (_, _, figwidth, figheight) = text.figure.bbox.bounds
            return (left / figwidth, bottom / figheight, width / figwidth, height / figheight)

        def add_title_axis(fig, title):
            _title = fig.suptitle(title)
            ax: plt.Axes = fig.add_axes(bbox2rect(_title))
            ax.axis("off")
            ax.text(0, 0, title, animated=True, font=_title.get_fontproperties())
            fig.suptitle("")
            return ax

        all_ll = torch.stack(self.all_ll).cpu() if self.all_ll else []
        lldata = (range(1, len(all_ll)+1), all_ll)

        fig = self.figure
        replot = len(getattr(self, 'axes_saved', [])) != (nb_rows - 1) * nb_cols + 1 + bool(title)
        if replot:
            fig.clf()

            axes = []
            for k in range(kdim):
                ax = fig.add_subplot(nb_rows, nb_cols, k * nb_cols + 1)
                axes += [ax]
                ax.imshow(moving[k].cpu(), **imkwargs)
                if k == 0:
                    ax.set_title('moving')
                ax.axis('image')
                ax.set_xticks([])
                ax.set_yticks([])
                ax = fig.add_subplot(nb_rows, nb_cols, k * nb_cols + 2)
                axes += [ax]
                ax.imshow(warped[k].cpu(), **imkwargs)
                if k == 0:
                    ax.set_title('moved')
                ax.axis('image')
                ax.set_xticks([])
                ax.set_yticks([])
                ax = fig.add_subplot(nb_rows, nb_cols, k * nb_cols + 3)
                axes += [ax]
                ax.imshow(checker[k].cpu(), **imkwargs)
                if k == 0:
                    ax.set_title('checker')
                ax.axis('image')
                ax.set_xticks([])
                ax.set_yticks([])
                ax = fig.add_subplot(nb_rows, nb_cols, k * nb_cols + 4)
                axes += [ax]
                ax.imshow(fixed[k].cpu(), **imkwargs)
                if k == 0:
                    ax.set_title('fixed')
                ax.axis('image')
                ax.set_xticks([])
                ax.set_yticks([])
                if vel:
                    ax = fig.add_subplot(nb_rows, nb_cols, k * nb_cols + 5)
                    axes += [ax]
                    d = ax.imshow(vel[k].cpu(), **imkwargs, vmin=0, vmax=vmax)
                    if k == 0:
                        ax.set_title('displacement')
                    ax.axis('image')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        v0 = d
                if lam:
                    ax = fig.add_subplot(nb_rows, nb_cols, k * nb_cols + 5 + bool(vel))
                    axes += [ax]
                    d = ax.imshow(lam[k].cpu(), **imkwargs, vmin=0, vmax=lmax)
                    if k == 0:
                        ax.set_title('precision')
                    ax.axis('image')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        l0 = d

            if title:
                axtitle = add_title_axis(fig, title)
                axes += [axtitle]

            ax = fig.add_subplot(nb_rows, 1, nb_rows)
            axes += [ax]
            ax.plot(*lldata, animated=True)
            ax.set_ylabel('NLL')
            ax.set_xlabel('iteration')
            ax.set_xticks([])
            ax.set_yticks([])

            if vel or lam:
                fig.subplots_adjust(right=0.9)
                if vel and not lam:
                    vbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
                elif lam and not vel:
                    lbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
                else:
                    vbar_ax = fig.add_axes([0.92, 0.10, 0.025, 0.35])
                    lbar_ax = fig.add_axes([0.92, 0.55, 0.025, 0.35])
                if vel:
                    self._vbar = fig.colorbar(v0, cax=vbar_ax)
                    self._vbar_ax = vbar_ax
                if lam:
                    self._lbar = fig.colorbar(l0, cax=lbar_ax)
                    self._lbar_ax = lbar_ax

            # fig.canvas.draw()
            plt.show(block=False)
            self._plt_background = fig.canvas.copy_from_bbox(fig.bbox)
            self.axes_saved = axes

        # --- Actually plot the stuff by blitting ---
        fig.canvas.restore_region(self._plt_background)

        for k in range(kdim):
            j = k * nb_cols
            self.axes_saved[j].images[0].set_data(moving[k].cpu())
            self.axes_saved[j+1].images[0].set_data(warped[k].cpu())
            self.axes_saved[j+2].images[0].set_data(checker[k].cpu())
            self.axes_saved[j+3].images[0].set_data(fixed[k].cpu())
            if vel:
                self.axes_saved[j+4].images[0].set_data(vel[k].cpu())
                self.axes_saved[j+4].images[0].set_clim(0, vmax)
                j += 1
            if lam:
                self.axes_saved[j+4].images[0].set_data(lam[k].cpu())
                self.axes_saved[j+4].images[0].set_clim(0, lmax)

        self.axes_saved[-1].lines[0].set_data(lldata)
        self.axes_saved[-1].relim()
        self.axes_saved[-1].autoscale_view()

        if title:
            self.axes_saved[-2].texts[0].set_text(title)

        if vel or lam:
            if vel:
                self._vbar.update_ticks()
                self._vbar.update_normal()
            if lam:
                self._lbar.update_ticks()
                self._lbar.update_normal()
            fig.canvas.draw()

        for ax in self.axes_saved:
            ax.draw_artist(ax.patch)
            if not ax.texts:
                for spine in ax.spines.values():
                    ax.draw_artist(spine)
            for elem in (ax.images or []):
                ax.draw_artist(elem)
            for elem in (ax.lines or []):
                ax.draw_artist(elem)
            for elem in (ax.texts or []):
                ax.draw_artist(elem)
            fig.canvas.blit(ax.bbox)

        fig.canvas.flush_events()

        self.figure = fig
        self._last_plot = time.time()

        # Run next ploting thread if there is one
        self._plot_lock.release()
        if self._plot_pending:
            self._plot_pending.run()

    def _disp(self, step, param):
        """Size of the update since the last printed row of this component
        (None on its first row), and remember the current parameters."""
        kind = 'nonlin' if step == 'nonlin' else 'affine'
        hook = self.disp_hooks.get(kind)
        if hook is None or param is None:
            return None
        prev = self._disp_prev.get(kind)
        disp = None if prev is None else hook(prev, param)
        self._disp_prev[kind] = optm.base._clone(param)
        return disp

    def print(self, step, ll, lla=None, llv=None, in_line_search=False,
              param=None):
        # Assembles the total objective (data + llv + lla) and records it:
        # must be called exactly once per evaluation, whatever `verbose`.
        # `param`: parameters of the component (for the `disp` column,
        # printed when the size of the updates is used for early stopping).

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
            show_disp = bool(getattr(self, 'disp_hooks', None)) and not in_line_search
            if show_disp and self.ll_prev is None:
                line += ' | ' + ' ' * 12
            if show_disp:
                disp = self._disp(step if has_vel else 'affine', param)
                line += f' | {disp:12.6g}' if disp is not None else ' | ' + ' ' * 12
            print(line, end='\r' if in_line_search or self.verbose < 3 else '\n')

        if not in_line_search:
            self.ll = ll
            self.llv = llv
            self.lla = lla
            self.all_ll.append(ll)
            self.ll_prev = self.ll
            self.ll_max = max(self.ll_max, self.ll)
            self.n_iter += 1

        return ll

    def init_penalties(self):
        """Initialise the penalties that each step of a joint model takes
        from the other component (`do_affine` uses the last velocity
        penalty `llv`, `do_vel` the last affine penalty `lla`), without
        evaluating the data term."""
        if not (self.affine and self.nonlin):
            return
        self.llv = self._nonlin_penalty(self.nonlin.parameters())
        if self.affine.penalty is not None:
            self.lla = self._affine_penalty(self.affine.parameters())

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
            pos = _backward_position(aff_pos) if loss.backward else aff_pos

            # ----------------------------------------------------------
            # build left and right affine
            # ----------------------------------------------------------
            aff_mov = moving.affine.to(aff00)
            aff_nonlin = self.nonlin.affine.to(aff00)
            aff_right = fixed.affine.to(aff00)
            if pos in 'fs':  # affine position: fixed or symmetric
                aff_right = aff00 @ aff_right
            aff_right = linalg.lmdiv(aff_nonlin, aff_right)
            aff_left = aff_nonlin
            if pos in 'ms':  # affine position: moving or symmetric
                aff_left = aff00 @ aff_left
            aff_left = linalg.lmdiv(aff_mov, aff_left)

            # ----------------------------------------------------------
            # build full transform
            # ----------------------------------------------------------
            right = None
            if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                aff_right = None
                psi = spatial.add_identity_grid(phi00)
                disp = phi00
            else:
                right = spatial.affine_grid(aff_right, fixed.shape)
                psi = regutils.smart_pull_grid(phi00, right, bound='zero')
                if do_print:
                    disp = psi.clone()
                psi += right
            if _almost_identity(aff_left) and moving.shape == self.nonlin.shape:
                aff_left = None
            else:
                psi = spatial.affine_matvec(aff_left, psi)

            # ----------------------------------------------------------
            # forward pass
            # ----------------------------------------------------------
            warped, warped_mask = moving.pull(psi, mask=True)
            if fixed.masked:
                if warped_mask is None:
                    mask = fixed.mask
                else:
                    mask = warped_mask * fixed.mask
            else:
                mask = warped_mask

            if do_print:
                has_printed = True
                if aff_right is not None:
                    disp = linalg.matvec(aff_right.inverse()[:-1, :-1], disp)
                else:   # disp is phi00 itself: do not scale it in place
                    disp = disp.clone()
                disp *= spatial.voxel_size(fixed.affine)
                lam = None
                if hasattr(loss.loss, 'irls'):
                    # lam estimated by the last evaluation: no IRLS loop
                    lam = loss.loss.weighted_lam(warped, fixed.dat, mask,
                                                 warped.dim()-1)
                    if mask is not None:
                        lam *= mask
                if moving.previewed:
                    preview = moving.pull(psi, preview=True, dat=False)
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
                dat = fixed.preview * fixed.mask if fixed.masked else fixed.preview
                preview = preview * warped_mask if warped_mask is not None else preview
                self.mov2fix(dat, init, preview, disp, lam=lam,
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
            # chain rule -> derivatives wrt the displacement
            # ----------------------------------------------------------
            if grad or hess:
                # reuse the full warp `psi` and the dense right affine
                # (`right` is None iff `aff_right` is None)
                g, h, mugrad = self.nonlin.propagate_grad(
                    g, h, moving, phi00, aff_left, right,
                    inv=loss.backward, psi=psi)
                del psi, right
                if mugrad is not None:
                    g = regutils.jg(mugrad, g)
                    h = regutils.jhj(mugrad, h)
                else:
                    # already wrt the displacement: channels last
                    g = utils.fast_movedim(g, 0, -1)
                    h = utils.fast_movedim(h, 0, -1) if h is not None else h
                is_svf = isinstance(self.nonlin, SVFModel)
                is_svf = is_svf or (isinstance(self.nonlin, Nonlin2dModel) and
                                    isinstance(self.nonlin._model, SVFModel))
                if is_svf:
                    # propagate backward by scaling and squaring
                    g, h = spatial.exp_backward(vel00, g, h,
                                                steps=self.nonlin.steps,
                                                bound=self.nonlin.bound)

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
        llv = self._nonlin_penalty(vel0, sumgrad if grad else None)

        # ==============================================================
        #                           VERBOSITY
        # ==============================================================
        sumloss = self.print('nonlin', sumloss, lla=self.lla, llv=llv,
                             in_line_search=in_line_search, param=vel)

        # ==============================================================
        #                           RETURN
        # ==============================================================
        out = [sumloss]
        if grad:
            if hasattr(self.nonlin, 'projection'):
                P = self.nonlin.projection()
                sumgrad = linalg.matvec(P.to(sumgrad), sumgrad)
            out.append(sumgrad)
        if hess:
            if hasattr(self.nonlin, 'projection'):
                sumhess = self.nonlin.load(sumhess)
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]

    def _nonlin_penalty(self, vel, grad=None):
        """Regularisation of the nonlinear parameters: `0.5 * <v, L v>`.

        Its gradient (L v) is added to `grad` in place; its Hessian (L) is
        not materialised (the nonlinear solvers apply it themselves).
        Returns the value.
        """
        vgrad = self.nonlin.regulariser(vel)
        if grad is not None:
            grad += vgrad
        return 0.5 * vel.flatten().dot(vgrad.flatten())

    def _affine_penalty(self, logaff, grad=None, hess=None):
        """Gaussian prior N(0, P^-1) on the model-basis log-parameters.

        P is diagonal: a scalar lambda (P = lambda I) or a vector of
        per-parameter precisions. Value: 1/2 q' P q. Its gradient (P q)
        and Hessian (P) live in the model basis; they are mapped to the
        optimised basis (`-g`) with the Gram matrix M of `_optim_grad`:
        M.T (P q) and M.T P M (lambda M.T q and lambda M.T M for a scalar).
        `grad` and `hess` are updated in place. Returns the value.
        """
        if isinstance(logaff, AffineModel.Parameters):
            # optimisers pass the model's parameters on their first
            # evaluation, and (model-basis) tensors afterwards
            logaff = logaff.dat
        penalty = self.affine.penalty
        lla = logaff * penalty
        if grad is not None or hess is not None:
            par = self.affine.parameters()
            gram = None
            if par.optim not in (None, par.basis):
                gram = par._gram(lla)
            if grad is not None:
                grad += lla if gram is None else gram.T.matmul(lla)
            if hess is not None:
                if gram is None:
                    hess.diagonal(0, -1, -2).add_(penalty)
                else:
                    hess += (gram.T * penalty).matmul(gram)
        return 0.5 * lla.dot(logaff)

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
            do_print = not (has_printed or self.verbose < 3 or in_line_search
                            or loss.backward)

            moving, fixed, factor = loss.moving, loss.fixed, loss.factor
            if loss.backward:
                phi00, aff00, gaff00 = iphi0, iaff0, igaff0
            else:
                phi00, aff00, gaff00 = phi0, aff0, gaff0
            pos = _backward_position(aff_pos) if loss.backward else aff_pos

            # ----------------------------------------------------------
            # build left and right affine matrices
            # ----------------------------------------------------------
            aff_nonlin = self.nonlin.affine.to(aff00)
            aff_mov = moving.affine.to(aff00)
            aff_fix = fixed.affine.to(aff00)
            aff_right, gaff_right = aff_fix, None
            if pos in 'fs':
                gaff_right = gaff00 @ aff_right
                gaff_right = linalg.lmdiv(aff_nonlin, gaff_right)
                aff_right = aff00 @ aff_right
            aff_right = linalg.lmdiv(aff_nonlin, aff_right)
            aff_left, gaff_left = aff_nonlin, None
            if pos in 'ms':
                gaff_left = gaff00 @ aff_left
                gaff_left = linalg.lmdiv(aff_mov, gaff_left)
                aff_left = aff00 @ aff_left
            aff_left = linalg.lmdiv(aff_mov, aff_left)

            # ----------------------------------------------------------
            # build full transform
            # ----------------------------------------------------------
            if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                right = None
                psi = spatial.add_identity_grid(phi00)
            else:
                right = spatial.affine_grid(aff_right, fixed.shape)
                psi = regutils.smart_pull_grid(phi00, right, bound='zero')
                psi += right
            phi_right = psi  # fixed lattice -> nonlin voxels
            if not (_almost_identity(aff_left) and
                    moving.shape == self.nonlin.shape):
                psi = spatial.affine_matvec(aff_left, psi)

            # ----------------------------------------------------------
            # forward pass
            # ----------------------------------------------------------
            warped, warped_mask = moving.pull(psi, mask=True)
            if fixed.masked:
                if warped_mask is None:
                    mask = fixed.mask
                else:
                    mask = warped_mask * fixed.mask
            else:
                mask = warped_mask

            if do_print:
                has_printed = True
                lam = None
                if hasattr(loss.loss, 'irls'):
                    # lam estimated by the last evaluation: no IRLS loop
                    lam = loss.loss.weighted_lam(warped, fixed.dat, mask,
                                                 warped.dim()-1)
                    if mask is not None:
                        lam *= mask
                if moving.previewed:
                    preview = moving.pull(psi, preview=True, dat=False)
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
                dat = fixed.preview * fixed.mask if fixed.masked else fixed.preview
                preview = preview * warped_mask if warped_mask is not None else preview
                self.mov2fix(dat, init, preview, lam=lam,
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

            def compose_grad(g, h, g_mu, g_aff, grid=None):
                """
                g, h : gradient/Hessian of loss wrt moving image
                g_mu : spatial gradients of moving image
                g_aff : gradient of affine matrix wrt Lie parameters
                grid : coordinates the affine is applied to
                       (default: the identity grid of the fixed lattice)
                returns g, h: gradient/Hessian of loss wrt Lie parameters
                """
                # Note that `h` can be `None`, but the functions I
                # use deal with this case correctly.
                dim = g_mu.shape[-1]
                g = regutils.jg(g_mu, g)
                h = regutils.jhj(g_mu, h)
                g, h = regutils.affine_grid_backward(g, h, grid=grid)
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
                # spatial gradients of the moving image, at the warped
                # fixed voxels (moving voxel space)
                mugrad = moving.pull_grad(psi, rotate=False)
                if pos in 'ms':
                    # warped(x) = mu(L phi_right(x)), so
                    # dwarped(x)/dL = grad_mu(psi(x)) [phi_right(x); 1]':
                    # computed on the fixed lattice, at the points
                    # `phi_right(x)`. (Pushing `g` to the nonlin lattice
                    # first and taking `grad_mu` and the coordinates at
                    # its nodes is exact only if `phi_right` is the
                    # identity -- never the case in the symmetric
                    # position, whose right affine is the fixed half.)
                    g, h = compose_grad(g0, h0, mugrad, gaff_left,
                                        grid=phi_right)
                del phi_right
                if pos in 'fs':
                    g_right, h_right = g0, h0
                    jac = spatial.grid_jacobian(phi00, right, type='disp',
                                              bound='zero', extrapolate=True)
                    jac = torch.matmul(aff_left[:-1, :-1], jac)
                    mugrad = linalg.matvec(jac.transpose(-1, -2), mugrad)
                    g_right, h_right = compose_grad(g_right, h_right, mugrad, gaff_right)
                    g = g_right if g is None else g.add_(g_right)
                    h = h_right if h is None else h.add_(h_right)
                if pos == 's' and h is not None:
                    # Both halves of the affine move the same voxels, so
                    # the Gauss-Newton Hessian has cross terms
                    # sum(a_mov a_fix'), dropped above. Since
                    # (a+b)(a+b)' <= 2(aa' + bb'), twice the sum of the
                    # two halves majorises it. As both halves are nearly
                    # equal, this is also tight, whereas the plain sum
                    # is about half the Gauss-Newton Hessian.
                    h = h.mul_(2)

                if loss.backward:
                    g = g.neg_()
                sumgrad = (g.mul_(factor) if sumgrad is None else
                           sumgrad.add_(g, alpha=factor))
                if hess:
                    sumhess = (h.mul_(factor) if sumhess is None else
                               sumhess.add_(h, alpha=factor))
            sumloss = (llx.mul_(factor) if sumloss is None else
                       sumloss.add_(llx, alpha=factor))

        if self.affine.penalty is not None:
            lla = self._affine_penalty(logaff, sumgrad if grad else None,
                                       sumhess if hess else None)
            # (the value is added to `sumloss` by `self.print`)
        else:
            lla = 0

        # ==============================================================
        #                           VERBOSITY
        # ==============================================================
        sumloss = self.print('affine', sumloss, lla=lla, llv=self.llv,
                             in_line_search=in_line_search, param=logaff)

        # ==============================================================
        #                           RETURN
        # ==============================================================
        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]

    def _sampled_voxels(self, loss, all_voxels=False):
        """Voxels of the fixed image on which a loss of the affine-only
        fit is evaluated (see `SampledVoxels`).

        A subset is drawn the first time it is needed and kept by this
        step (i.e., for one pyramid level), so that the objective is the
        same function across iterations and line searches. The seed is
        fixed, so that a run is reproducible.
        """
        n = None if all_voxels else getattr(self.affine, 'samples', None)
        if n == 'auto':
            n = self._auto_samples(loss)
        if n and getattr(loss.loss, 'spatial', False):
            if not self._samples.get('warned'):
                import warnings
                warnings.warn(f'{type(loss.loss).__name__} depends on the '
                              f'spatial arrangement of the voxels: it is '
                              f'evaluated on all voxels (samples ignored).')
                self._samples['warned'] = True
            n = None
        key = (id(loss.fixed), n or None)
        if key not in self._samples:
            if n and self._sample_rng is None:
                self._sample_rng = torch.Generator().manual_seed(0)
            self._samples[key] = SampledVoxels(loss.fixed, n, self._sample_rng)
        return self._samples[key]

    def _auto_samples(self, loss):
        """Number of samples of `samples='auto'` (see `auto_samples`)"""
        from .objects import auto_samples
        nb_params = len(self.affine.parameters())
        cells = loss.loss.histogram_cells(loss.moving.dat, loss.fixed.dat,
                                          loss.fixed.dim)
        return auto_samples(nb_params, cells)

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
        for i, loss in enumerate(self.losses):
            do_print = not (has_printed or self.verbose < 3 or in_line_search
                            or loss.backward)

            moving, fixed, factor = loss.moving, loss.fixed, loss.factor
            if loss.backward:
                aff00, gaff00 = iaff0, igaff0
            else:
                aff00, gaff00 = aff0, gaff0

            # ----------------------------------------------------------
            # build full transform
            # ----------------------------------------------------------
            aff_mov = moving.affine.to(aff00)
            aff_fix = fixed.affine.to(aff00)
            aff = aff00 @ aff_fix
            aff = linalg.lmdiv(aff_mov, aff)
            gaff = gaff00 @ aff_fix
            gaff = linalg.lmdiv(aff_mov, gaff)

            # ----------------------------------------------------------
            # forward pass (on all fixed voxels, or on a subset)
            # ----------------------------------------------------------
            def forward(voxels):
                psi = voxels.affine_grid(aff)
                warped, warped_mask = moving.pull(psi, mask=True)
                if voxels.mask is None:
                    mask = warped_mask
                elif warped_mask is None:
                    mask = voxels.mask
                else:
                    mask = warped_mask * voxels.mask
                return psi, warped, warped_mask, mask

            voxels = self._sampled_voxels(loss)
            psi, warped, warped_mask, mask = forward(voxels)
            grid = voxels.coords(aff)

            if do_print:
                has_printed = True
                # the plot shows all voxels
                if not voxels.sampled:
                    ppsi, pwarped, pwarped_mask, pmask = \
                        psi, warped, warped_mask, mask
                else:
                    pvoxels = self._sampled_voxels(loss, all_voxels=True)
                    ppsi, pwarped, pwarped_mask, pmask = forward(pvoxels)
                lam = None
                if hasattr(loss.loss, 'irls'):
                    # lam estimated by the last evaluation: no IRLS loop
                    lam = loss.loss.weighted_lam(pwarped, fixed.dat, pmask,
                                                 pwarped.dim()-1)
                    if pmask is not None:
                        lam *= pmask
                if moving.previewed:
                    preview = moving.pull(ppsi, preview=True, dat=False)
                else:
                    preview = pwarped
                init = spatial.affine_lmdiv(moving.affine, fixed.affine)
                if _almost_identity(init) and moving.shape == fixed.shape:
                    init = moving.dat
                else:
                    init = spatial.affine_grid(init, fixed.shape)
                    initmask, init = moving.pull(init, preview=True, dat=False, mask=True)
                    if initmask is not None:
                        init = init * initmask
                dat = fixed.preview * fixed.mask if fixed.masked else fixed.preview
                preview = preview * pwarped_mask if pwarped_mask is not None else preview
                self.mov2fix(dat, init, preview, lam=lam,
                             title=f'(affine) {self.n_iter:03d}')
                del ppsi, pwarped, pwarped_mask, pmask

            # ----------------------------------------------------------
            # derivatives wrt moving
            # ----------------------------------------------------------
            g = h = None
            loss_args = (warped, voxels.dat)
            loss_kwargs = dict(dim=fixed.dim, mask=mask,
                               **voxels.loss_kwargs(loss.loss))
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

            def compose_grad(g, h, g_mu, g_aff, grid=None):
                """
                g, h : gradient/Hessian of loss wrt moving image
                g_mu : spatial gradients of moving image
                g_aff : gradient of affine matrix wrt Lie parameters
                grid : coordinates the affine is applied to
                       (default: the identity grid of the fixed lattice)
                returns g, h: gradient/Hessian of loss wrt Lie parameters
                """
                # Note that `h` can be `None`, but the functions I
                # use deal with this case correctly.
                dim = g_mu.shape[-1]
                g = regutils.jg(g_mu, g)
                h = regutils.jhj(g_mu, h)
                g, h = regutils.affine_grid_backward(g, h, grid=grid)
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
                mugrad = moving.pull_grad(psi, rotate=False)
                g, h = compose_grad(g, h, mugrad, gaff, grid=grid)

                if loss.backward:
                    g = g.neg_()
                sumgrad = (g.mul_(factor) if sumgrad is None else
                           sumgrad.add_(g, alpha=factor))
                if hess:
                    sumhess = (h.mul_(factor) if sumhess is None else
                               sumhess.add_(h, alpha=factor))
            sumloss = (llx.mul_(factor) if sumloss is None else
                       sumloss.add_(llx, alpha=factor))

        if self.affine.penalty is not None:
            lla = self._affine_penalty(logaff, sumgrad if grad else None,
                                       sumhess if hess else None)
            # (the value is added to `sumloss` by `self.print`)
        else:
            lla = 0

        # ==============================================================
        #                           VERBOSITY
        # ==============================================================
        sumloss = self.print('affine_only', sumloss, lla=lla,
                             in_line_search=in_line_search, param=logaff)

        # ==============================================================
        #                           RETURN
        # ==============================================================
        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]

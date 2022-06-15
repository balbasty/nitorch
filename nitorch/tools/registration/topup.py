import copy

from nitorch import spatial
from nitorch.core import py, utils, linalg
from . import optim as optm
from .losses import make_loss


BND = 'dct2'


def _deform_1d(img, disp, grad=False):
    img = utils.movedim(img, 0, -2)
    disp = disp.unsqueeze(-1)
    disp = spatial.add_identity_grid(disp)
    wrp = spatial.grid_pull(img, disp, bound=BND, extrapolate=True)
    wrp = utils.movedim(wrp, -2, 0)
    if not grad:
        return wrp, None
    grd = spatial.grid_grad(img, disp, bound=BND, extrapolate=True)
    grd = utils.movedim(grd.squeeze(-1), -2, 0)
    return wrp, grd


def _diff_1d(disp):
    return spatial.diff1d(disp, dim=-1, bound=BND, side='c')


def _div_1d(disp):
    return spatial.div1d(disp, dim=-1, bound=BND, side='c')


def _exp_1d(vel, model='smalldef'):
    if model == 'svf':
        phi, jac = spatial.exp1d_forward(vel, bound=BND, jacobian=True)
        iphi, ijac = spatial.exp1d_forward(-vel, bound=BND, jacobian=True)
    else:
        phi, iphi = vel, -vel
        jac = _diff_1d(vel)
        ijac = 1 - jac
        jac += 1
    return phi, iphi, jac, ijac


class TopUpFlexiStep:
    """One step of TopUpFlexi

    Assumes:
    - `pos` and `neg` have a channel dimension
    - the distortion direction is the last dimension
    """

    def __init__(self, pos, neg, loss, reg, verbose=False, model='smalldef',
                 modulation=True, mask=None):
        self.pos = pos
        self.neg = neg
        self.mask = mask
        self.loss = loss
        self.reg = reg
        self.model = model
        self.modulation = modulation

        self.verbose = verbose
        self.n_iter = 0
        self.ll = float('inf')

    def __call__(self, vel, grad=False, hess=False, in_line_search=False):

        vel = vel[0]

        phi, iphi, jac, ijac = _exp_1d(vel, model=self.model)

        pos, grad_pos = _deform_1d(self.pos, phi, grad=grad)
        del phi

        neg, grad_neg = _deform_1d(self.neg, iphi, grad=grad)
        del iphi

        if self.modulation:
            pos *= jac
            neg *= ijac
            if self.model == 'svf':
                grad_pos *= jac
                grad_neg *= ijac

        g = ig = h = ih = None
        state = self.loss.get_state()
        if grad and hess:
            ll, g, h = self.loss.loss_grad_hess(pos, neg, mask=self.mask)
            ill, ig, ih = self.loss.loss_grad_hess(neg, pos, mask=self.mask)
        elif grad:
            ll, g = self.loss.loss_grad(pos, neg, mask=self.mask)
            ill, ig = self.loss.loss_grad(neg, pos, mask=self.mask)
        else:
            ll = self.loss.loss(pos, neg, mask=self.mask)
            ill = self.loss.loss(neg, pos, mask=self.mask)
        if in_line_search:
            self.loss.set_state(state)

        ll += ill
        ll /= 2
        if grad:

            if self.modulation:
                pos /= jac
                neg /= ijac

            # move channel channels to the end so that we can use `dot`
            g = utils.movedim(g, 0, -1)
            ig = utils.movedim(ig, 0, -1)
            pos = utils.movedim(pos, 0, -1)
            neg = utils.movedim(neg, 0, -1)
            grad_pos = utils.movedim(grad_pos, 0, -1)
            grad_neg = utils.movedim(grad_neg, 0, -1)

            g0, ig0 = g, ig

            g = linalg.dot(g0, grad_pos)
            if self.modulation:
                g = g.mul_(jac)
                g0 = linalg.dot(g0, pos)
                if self.model == 'svf':
                    g0.mul_(jac)
                g += _div_1d(g0)
            ig = linalg.dot(ig0, grad_neg)
            if self.modulation:
                ig = ig.mul_(ijac)
                ig0 = linalg.dot(ig0, neg)
                if self.model == 'svf':
                    ig0.mul_(ijac)
                ig += _div_1d(ig0)
            del g0, ig0

            if hess:
                h = utils.movedim(h, 0, -1)
                ih = utils.movedim(ih, 0, -1)
                h0, ih0 = h, ih

                grad_pos.square_()
                grad_neg.square_()

                h = linalg.dot(grad_pos, h0)
                if self.modulation:
                    jac.square_()
                    h = h.mul_(jac)
                    h0 = linalg.dot(pos, h0).square_()
                    if self.model == 'svf':
                        h0.mul_(jac)
                    h += _div_1d(_div_1d(h0))
                ih = linalg.dot(grad_neg, ih0)
                if self.modulation:
                    ijac.square_()
                    ih = ih.mul_(ijac)
                    ih0 = linalg.dot(neg, ih0).square_()
                    if self.model == 'svf':
                        ih0.mul_(ijac).mul_(ijac)
                    ih += _div_1d(_div_1d(ih0))
                del h0, ih0

            if self.model == 'svf':
                g, h = spatial.exp1d_backward(vel, g, h, bound=BND)
                ig, ih = spatial.exp1d_backward(-vel, ig, ih, bound=BND)

            g = g.sub_(ig).div_(2)
            g = g[None]
            if hess:
                h = h.add_(ih).div_(2)
                h = h[None]

        del ig, ih, grad_pos, grad_neg, jac, ijac
        vel = vel[None]

        # add regularization term
        vgrad = self.reg(vel)
        llv = 0.5 * vel.flatten().dot(vgrad.flatten())
        if grad:
            g += vgrad
        del vgrad

        # print objective
        if self.verbose and (self.verbose > 1 or not in_line_search):
            ll_prev = self.ll
            if in_line_search:
                line = '(search) | '
            else:
                line = '(topup)  | '
            line += f'{self.n_iter:03d} | {ll.item():12.6g} + {llv.item():12.6g} = {ll.item() + llv.item():12.6g}'
            if not in_line_search:
                self.ll = ll.item() + llv.item()
                self.n_iter += 1
                gain = (ll_prev - self.ll)
                line += f' | {gain:12.6g}'
            print(line, end='\r')

        ll += llv
        out = [ll]
        if grad:
            out.append(g)
        if hess:
            out.append(h)
        return tuple(out) if len(out) > 1 else out[0]


def topup_apply(pos, neg, vel, dim=-1, model='smalldef', modulation=True):
    """Apply a topup correction

    Parameters
    ----------
    pos : ([C], *spatial) tensor
        Images with positive readout polarity
    neg : ([C], *spatial) tensor
        Images with negative readout polarity
    vel : (*spatial) tensor
        1D displacement or velocity field
    dim : int, default=-1
        Readout dimension
    model : {'smalldef', 'svf'}, default='smalldef'
        Deformation model

    Returns
    -------
    pos : ([C], *spatial) tensor
        Images with positive polarity, unwarped
    neg : ([C], *spatial) tensor
        Images with negative polarity, unwarped

    """
    ndim = vel.dim()
    dim = (dim - ndim) if dim >= 0 else dim

    no_batch_pos = pos.dim() == ndim
    if no_batch_pos:
        pos = pos[None]
    pos = utils.movedim(pos, dim, -1)
    no_batch_neg = neg.dim() == ndim
    if no_batch_neg:
        neg = neg[None]
    neg = utils.movedim(neg, dim, -1)
    vel = utils.movedim(vel, dim, -1)

    phi, iphi, jac, ijac = _exp_1d(vel, model=model)
    pos, _ = _deform_1d(pos, phi)
    neg, _ = _deform_1d(neg, iphi)
    if modulation:
        pos *= jac
        neg *= ijac
    del phi, iphi, jac, ijac

    pos = utils.movedim(pos, -1, dim)
    neg = utils.movedim(neg, -1, dim)
    if no_batch_pos:
        pos = pos[0]
    if no_batch_neg:
        neg = neg[0]

    return pos, neg


def topup_fit(pos, neg, dim=-1, loss='mse', lam=1, vx=1, ndim=3, mask=None,
              model='smalldef', penalty='bending', modulation=True,
              max_iter=50, tolerance=1e-4, verbose=True, vel=None):
    """TOPUP correction with flexible objective function

    Parameters
    ----------
    pos : ([C], *spatial) tensor
        Images with positive readout polarity
    neg : ([C], *spatial) tensor
        Images with negative readout polarity
    dim : int, default=-1
        Readout dimension
    loss : {'mse', 'mad', 'cc'} or RegistrationLoss, default='mse'
        Loss function
    lam : float, default=1
        Regularisation factor
    ndim : int, default=3
        Number of spatial dimensions
    model : {'smalldef', 'svf'}, default='smalldef'
        Deformation model
    penalty : {'membrane', 'bending'}, default='bending'
        Type of regularization
    modulation : bool, default=True
        Include Jacobian modulation
    verbose : bool, default=True
        Print progress
    vel : (*spatial) tensor, otpional
        Initial guess

    Returns
    -------
    vel : (*spatial) tensor
        1D displacement (if 'smalldef') or velocity (if 'svf') field

    """
    dim = (dim - ndim) if dim >= 0 else dim

    vx = utils.make_vector(vx, ndim, dtype=pos.dtype).tolist()
    vx = py.move_elem(vx, dim, -1)
    no_batch_pos = pos.dim() == ndim
    if no_batch_pos:
        pos = pos[None]
    pos = utils.movedim(pos, dim, -1)
    no_batch_neg = neg.dim() == ndim
    if no_batch_neg:
        neg = neg[None]
    neg = utils.movedim(neg, dim, -1)
    if mask is not None:
        mask = utils.movedim(mask, dim, -1)
    if isinstance(loss, str):
        loss = make_loss(loss)

    lam = lam / pos.shape[-ndim:].numel()

    def regulariser(v):
        return spatial.regulariser(v, dim=ndim, **{penalty: 1},
                                   factor=lam, voxel_size=vx, bound=BND)

    step = TopUpFlexiStep(pos, neg, loss, regulariser, verbose,
                          model=model, modulation=modulation, mask=mask)
    optim = optm.FieldCG(factor=lam, voxel_size=vx, **{penalty: 1},
                         bound=BND, max_iter=4)
    optim.search = 'wolfe'
    optim.iter = optm.OptimIterator(max_iter=max_iter, tol=tolerance,
                                    stop='diff')

    if vel is None:
        vel = pos.new_zeros(pos.shape[-ndim:])
    else:
        vel = utils.movedim(vel, dim, -1)

    optim.iter(vel[None], step)
    if verbose:
        print('')

    vel = utils.movedim(vel, -1, dim)
    return vel


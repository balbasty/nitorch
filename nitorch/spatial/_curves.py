import torch
from nitorch.core import utils, linalg, py
from nitorch.spatial import grid_pull, grid_push, grid_grad, spline_coeff, identity_grid
import time
import math as pymath
from ._solvers import restrict, prolong


def make_bound3_(mat, bound):
    """Enforce boundary conditions of a regularisation matrix (cubic)"""
    elems = mat[0, :4]

    if bound == 'dct2':
        mat[0, 0] += elems[1]
        mat[0, 1] += elems[2]
        mat[1, 0] += elems[2]
        mat[0, 2] += elems[3]
        mat[2, 0] += elems[3]  # sym
        mat[1, 1] += elems[3]
        mat[-1, -1] += elems[1]
        mat[-1, -2] += elems[2]
        mat[-2, -1] += elems[2]  # sym
        mat[-1, -3] += elems[3]
        mat[-3, -1] += elems[3]  # sym
        mat[-2, -2] += elems[3]
    elif bound == 'dct1':
        mat[0, 1] += elems[1]
        mat[0, 2] += elems[2]
        mat[2, 0] += elems[2]  # sym
        mat[0, 3] += elems[3]
        mat[3, 0] += elems[3]  # sym
        mat[1, 2] += elems[3]
        mat[-1, -2] += elems[1]
        mat[-1, -3] += elems[2]
        mat[-3, -1] += elems[2]  # sym
        mat[-1, -4] += elems[3]
        mat[-3, -2] += elems[3]  # sym
        mat[-2, -3] += elems[3]
    elif bound == 'dft':
        mat[0, -1] += elems[1]
        mat[0, -2] += elems[2]
        mat[-2, 0] += elems[2]
        mat[0, -3] += elems[3]
        mat[-3, 0] += elems[3]  # sym
        mat[1, -2] += elems[3]
        mat[-1, 0] += elems[1]
        mat[-1, 1] += elems[2]
        mat[1, -1] += elems[2]
        mat[-1, 2] += elems[3]
        mat[2, -1] += elems[3]  # sym
        mat[-2, 1] += elems[3]


def make_matrix3(n, elems, bound, **backend):
    """Build a Toeplitz regularisation matrix"""
    mat = torch.zeros([n, n], **backend)
    if n < 3:
        # TODO?
        return mat
    # Make Toeplitz matrix
    for i, e in enumerate(elems):
        mat.diagonal(i).fill_(e)
        if i > 0:
            mat.diagonal(-i).fill_(e)
    # Boundary conditions
    make_bound3_(mat, bound)
    # Jacobian scaling ([0, N-1] to [0, 1])
    mat /= (n - 1)
    return mat


def membrane3(n, bound='dct2', **backend):
    """Build a membrane regulariser for cubic splines"""
    elems = [2/3, -1/6, -7/60, -1/20]
    return make_matrix3(n, elems, bound, **backend)


def bending3(n, bound='dct2', **backend):
    """Build a bending regulariser for cubic splines"""
    elems = [8/3, -3/2, 0, 1/6]
    return make_matrix3(n, elems, bound, **backend)


class BSplineCurve:
    """A smooth N-D curve parameterized by B-splines"""

    def __init__(self, waypoints, order=3, radius=1):
        """

        Parameters
        ----------
        waypoints : (N, D) tensor
            List of waypoints, that the curve will interpolate.
        order : int, default=3
            Order of the encoding B-splines
        radius : float or (N,) tensor
            Radius of the curve at each waypoint.
        """
        waypoints = torch.as_tensor(waypoints)
        if not waypoints.dtype.is_floating_point:
            waypoints = waypoints.to(torch.get_default_dtype())
        self.waypoints = waypoints
        self.order = order
        self.bound = 'dct2'
        self.coeff = spline_coeff(waypoints, interpolation=self.order,
                                  bound=self.bound, dim=0)
        if not isinstance(radius, (int, float)):
            radius = torch.as_tensor(radius, **utils.backend(waypoints))
        self.radius = radius
        if torch.is_tensor(radius):
            self.coeff_radius = spline_coeff(radius, interpolation=self.order,
                                             bound=self.bound, dim=0)

    def update_waypoints(self):
        """Convert coefficients into waypoints"""
        t = torch.linspace(0, 1, len(self.coeff), **utils.backend(self.coeff))
        p = self.eval_position(t)
        if p.shape == self.waypoints.shape:
            self.waypoints.copy_(p)
        else:
            self.waypoints = p

    def update_radius(self):
        """Convert coefficients into radii"""
        if not hasattr(self, 'coeff_radius'):
            return
        t = torch.linspace(0, 1, len(self.coeff_radius),
                           **utils.backend(self.coeff_radius))
        r = self.eval_radius(t)
        if torch.is_tensor(self.radius) and r.shape == self.radius.shape:
            self.radius.copy_(r)
        else:
            self.radius = r

    def restrict(self, from_shape, to_shape=None):
        """Apply transform == to a restriction of the underlying grid"""
        to_shape = to_shape or [pymath.ceil(s/2) for s in from_shape]
        shifts = [0.5 * (frm / to - 1)
                  for frm, to in zip(from_shape, to_shape)]
        scales = [frm / to for frm, to in zip(from_shape, to_shape)]
        shifts = torch.as_tensor(shifts, **utils.backend(self.waypoints))
        scales = torch.as_tensor(scales, **utils.backend(self.waypoints))
        self.waypoints.sub_(shifts).div_(scales)
        self.coeff.sub_(shifts).div_(scales)
        self.radius.div_(scales.prod().pow_(1/len(scales)))
        self.coeff_radius.div_(scales.prod().pow_(1/len(scales)))

    def prolong(self, from_shape, to_shape=None):
        """Apply transform == to a prolongation of the underlying grid"""
        to_shape = to_shape or [2*s for s in from_shape]
        from_shape, to_shape = to_shape, from_shape
        shifts = [0.5 * (frm / to - 1)
                  for frm, to in zip(from_shape, to_shape)]
        scales = [frm / to for frm, to in zip(from_shape, to_shape)]
        shifts = torch.as_tensor(shifts, **utils.backend(self.waypoints))
        scales = torch.as_tensor(scales, **utils.backend(self.waypoints))
        self.waypoints.mul_(scales).add_(shifts)
        self.coeff.mul_(scales).add_(shifts)
        self.radius.mul_(scales.prod().pow_(1/len(scales)))
        self.coeff_radius.mul_(scales.prod().pow_(1/len(scales)))

    def eval_position(self, t):
        """Evaluate the position at a given (batched) time"""
        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.waypoints) - 1)

        # interpolate
        y = self.coeff.T                      # [D, K]
        t = t.unsqueeze(-1)                   # [N, 1]
        x = grid_pull(y, t, interpolation=self.order, bound=self.bound)
        x = x.T                               # [N, D]
        x = x.reshape([*shape, x.shape[-1]])
        return x

    def eval_radius(self, t):
        """Evaluate the radius at a given (batched) time"""
        if not torch.is_tensor(self.radius):
            return self.radius

        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.coeff_radius) - 1)

        # interpolate
        y = self.coeff_radius                 # [K]
        t = t.unsqueeze(-1)                   # [N, 1]
        x = grid_pull(y, t, interpolation=self.order, bound=self.bound)
        x = x.reshape(shape)
        return x

    def grad_position(self, t):
        """Gradient of the evaluated position wrt time"""
        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.waypoints) - 1)

        # interpolate
        y = self.coeff.T                          # [D, K]
        t = t.unsqueeze(-1)                       # [N, 1]
        g = grid_grad(y, t, interpolation=self.order, bound=self.bound)
        g = g.squeeze(-1).T                       # [N, D]
        g = g.reshape([*shape, g.shape[-1]])
        g *= (len(self.waypoints) - 1)
        return g

    def eval_grad_position(self, t):
        """Evaluate position and its gradient wrt time"""
        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.waypoints) - 1)

        # interpolate
        y = self.coeff.T                          # [D, K]
        t = t.unsqueeze(-1)                       # [N, 1]
        x = grid_pull(y, t, interpolation=self.order, bound=self.bound)
        x = x.T                                   # [N, D]
        g = grid_grad(y, t, interpolation=self.order, bound=self.bound)
        g = g.squeeze(-1).T                       # [N, D]

        x = x.reshape([*shape, x.shape[-1]])
        g = g.reshape([*shape, g.shape[-1]])
        g *= (len(self.waypoints) - 1)
        return x, g

    def push_position(self, x, t):
        """Push gradient into the control points
        (= differentiate wrt control points)"""
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.coeff) - 1)

        x = x.reshape(-1, x.shape[-1]).T  # [D, N]
        t = t.unsqueeze(-1)               # [N, 1]
        y = grid_push(x, t, [len(self.coeff)],
                      bound=self.bound, interpolation=self.order)
        y = y.T                           # [K, D]
        return y

    def push_radius(self, x, t):
        """Push gradient into the radius control points
        (= differentiate wrt radius control points)"""
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.coeff_radius) - 1)

        x = x.flatten()                   # [N]
        t = t.unsqueeze(-1)               # [N, 1]
        y = grid_push(x, t, [len(self.coeff_radius)],
                      bound=self.bound, interpolation=self.order)
        return y


def min_dist_table(x, s, steps=128):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    This function performs a global search across a list of values

    Parameters
    ----------
    steps : int or sequence, default=100
        Number of values evaluated, or explicit list of values to evaluate.

    Returns
    -------
    t : (...) tensor
        Coordinate of the closest point
    d : (...) tensor
        Minimum distance between each point and the curve

    """
    # initialize using a discrete search
    if isinstance(steps, int):
        all_t = torch.linspace(0, 1, steps, **utils.backend(x))
    else:
        all_t = utils.make_vector(steps, **utils.backend(x))
    t = x.new_zeros(x.shape[:-1])
    d = x.new_empty(x.shape[:-1]).fill_(float('inf'))
    for t1 in all_t:
        x1 = s.eval_position(t1)
        d1 = x1 - x
        d1 = d1.square_().sum(-1).sqrt_()
        t = torch.where(d1 < d, t1, t)
        d = torch.min(d, d1)
    return t, d


def min_dist_quad(x, s, max_iter=16, tol=1e-3, step=0.01,
                  init=None, init_kwargs=None):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    This function uses quadratic optimization to estimate the distance map.

    Parameters
    ----------
    x : (*shape, dim) tensor
        Coordinates
    s : BSplineCurve
        Parameterized curve
    max_iter : int, default=2**16
        Maximum number of iterations
    tol : float, default=1e-6
        Tolerance for early stopping
    step : float, default=0.1
        Step around the initial value, used to fit the first quadratic curve.
    init : (*shape) tensor or 'table', default='table'
        Initial guess for `t`.
        If 'table', initialize using a table search.

    Returns
    -------
    t : (*shape) tensor
        Coordinate of the closest point
    d : (*shape) tensor
        Minimum distance between each point and the curve

    """
    def dist(t):
        return s.eval_position(t).sub_(x).square_().sum(-1).sqrt_()

    if init is None:
        # initialize using a discrete search
        t, d = min_dist_table(x, s, **(init_kwargs or {}))
    else:
        t = init.clone()

    brent = Brent(tol=tol, max_iter=max_iter, step=step)
    return brent.fit(t, dist)


def min_dist_gn(x, s, max_iter=16, tol=1e-3, init=None, init_kwargs=None):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    This function uses Gauss-Newton optimization to estimate the distance map.

    Parameters
    ----------
    x : (*shape, dim) tensor
        Coordinates
    s : BSplineCurve
        Parameterized curve
    max_iter : int, default=2**16
        Maximum number of iterations
    tol : float, default=1e-6
        Tolerance for early stopping
    init : (*shape) tensor or {'table', 'quad'}, default='table'
        Initial guess for `t`.
        If 'table', initialize using a table search.
        If 'quad', initialize using a quadratic optimizer.

    Returns
    -------
    t : (*shape) tensor
        Coordinate of the closest point
    d : (*shape) tensor
        Minimum distance between each point and the curve

    """
    if init is None or init == 'quad':
        # initialize using a quadratic search
        t, d = min_dist_quad(x, s, **(init_kwargs or {}))
    elif init == 'table':
        # initialize using a discrete search
        t, d = min_dist_table(x, s, **(init_kwargs or {}))
    else:
        t = init.clone()
        d = s.eval_position(t) - x
        d = d.square_().sum(-1).sqrt_()

    # Fine tune using Gauss-Newton optimization
    nll = d.square_().sum(-1)
    # print(f'{0:03d} {nll.sum().item():12.6g}')
    for n_iter in range(1, max_iter+1):
        # compute the distance between x and s(t) + gradients
        d, g = s.eval_grad_position(t)
        d.sub_(x)
        h = linalg.dot(g, g)
        g = linalg.dot(g, d)
        h.add_(1e-3)
        g.div_(h)

        # Perform GN step (with line search)
        t0 = t.clone()
        nll0 = nll
        armijo = 1
        success = torch.zeros_like(t, dtype=torch.bool)
        for n_ls in range(12):
            t = torch.where(success, t, t0 - armijo * g).clamp_(0, 1)
            d = s.eval_position(t).sub_(x)
            nll = d.square().sum(-1)
            success = success.logical_or_(nll < nll0)
            if success.all():
                break
            armijo /= 2
        t = torch.where(success, t, t0)
        if not success.any():
            break

        # print(f'{n_iter:03d} '
        #       f'{nll.sum().item()/nll.numel():12.6g} '
        #       f'{(nll0 - nll).sum().item()/t.numel():6.3g} ')
        if (nll0 - nll).sum() < tol * t.numel():
            break

    d = s.eval_position(t).sub_(x)
    d = d.square_().sum(-1).sqrt_()
    return t, d


def min_dist(x, s, method='gn', **kwargs):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    Parameters
    ----------
    x : (*shape, dim) tensor
        Coordinates
    s : BSplineCurve
        Parameterized curve
    method : {'table', 'quad', 'gn}, default='gn'
        If 'table', use a table search.
        If 'quad', use a quadratic optimizer (Brent).
        If 'gn', use a Gauss-Newton optimizer

    Returns
    -------
    t : (*shape) tensor
        Coordinate of the closest point
    d : (*shape) tensor
        Minimum distance between each point and the curve

    References
    ----------
    ..[1]   "Robust and efficient computation of the closest point on a
            spline curve"
            Hongling Wang, Joseph Kearney, Kendall Atkinson
            Proc. 5th International Conference on Curves and Surfaces (2002)
    """
    method = method[0].lower()
    if method == 'g':
        return min_dist_gn(x, s, **kwargs)
    elif method == 'q':
        return min_dist_quad(x, s, **kwargs)
    else:
        return min_dist_table(x, s, **kwargs)


def dist_to_prob(d, r, tiny=0):
    if torch.is_tensor(d):
        d = d.clone()
    if torch.is_tensor(r):
        r = r.clone()
    return dist_to_prob_(d, r, tiny)


def dist_to_prob_(d, r, tiny=0):
    d = torch.as_tensor(d)
    r = radius_to_prec_(r)
    d.square_().mul_(r).mul_(-0.5)
    d.exp_()
    if tiny:
        d.mul_(1-2*tiny).add_(tiny)
    return d


def radius_to_prec(r):
    if torch.is_tensor(r):
        r = r.clone()
    return radius_to_prec_(r)


def radius_to_prec_(r):
    r *= 2            # diameter
    r /= 2.355        # standard deviation
    if torch.is_tensor(r):
        r.square_()       # variance
        r.reciprocal_()   # precision
    else:
        r = 1 / (r*r)
    return r


def draw_curve(shape, s, mode='gaussian', tiny=0, **kwargs):
    """Draw a BSpline curve

    Parameters
    ----------
    shape : list[int]
    s : BSplineCurve
    mode : {'binary', 'gaussian'}

    Returns
    -------
    x : (*shape) tensor
        Drawn curve

    """
    x = identity_grid(shape, **utils.backend(s.waypoints))
    t, d = min_dist(x, s, **kwargs)
    r = s.eval_radius(t)
    if mode[0].lower() == 'b':
        return d <= r
    else:
        return dist_to_prob(d, r, tiny)


def draw_curves(shape, curves, mode='gaussian', fast=0, **kwargs):
    """Draw multiple BSpline curves

    Parameters
    ----------
    shape : list[int]
    s : list[BSplineCurve]
    mode : {'binary', 'gaussian'}
    fast : float, default=0

    Returns
    -------
    x : (*shape) tensor
        Drawn curve
    lab : (*shape) tensor[int]
        Label of closest curve

    """
    mode = mode[0].lower()
    if mode == 'g':
        return draw_curves_gaussian(shape, curves, fast, **kwargs)
    elif mode == 'b':
        return draw_curves_binary(shape, curves, fast, **kwargs)
    else:
        raise ValueError(f'Unknown mode {mode}')


def draw_curves_gaussian(shape, curves, fast=0, **kwargs):
    if fast:
        return draw_curves_gaussian_fast(shape, curves, fast, **kwargs)

    curves = list(curves)
    locations = identity_grid(shape, **utils.backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)
    sum_prob = locations.new_ones(shape)
    max_prob = locations.new_zeros(shape)

    count = 0
    while curves:
        curve = curves.pop(0)
        count += 1
        time, dist = min_dist(locations, curve, **kwargs)
        radius = curve.eval_radius(time)
        prob = dist_to_prob(dist, radius)
        label.masked_fill_(prob > max_prob, count)
        max_prob = torch.maximum(max_prob, prob)
        sum_prob *= prob.neg_().add_(1)  # probability of no vessel
    sum_prob = sum_prob.neg_().add_(1)   # probability of at least one vessel

    return sum_prob, label


def draw_curves_gaussian_fast(shape, curves, threshold, **kwargs):
    curves = list(curves)
    locations = identity_grid(shape, **utils.backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)
    sum_prob = locations.new_ones(shape)
    max_prob = locations.new_zeros(shape)
    prob = locations.new_zeros(shape)

    count = 0
    while curves:
        curve = curves.pop(0)
        count += 1

        # initialize distance from table and only process
        # points that are close enough from the curve
        time, dist = min_dist_table(locations, curve)
        mask = dist < threshold
        sublocations = locations[mask, :]

        kwargs.setdefault('init_kwargs', {})
        kwargs['init_kwargs']['init'] = time[mask]
        time, dist = min_dist(sublocations, curve, **kwargs)
        radius = curve.eval_radius(time)
        subprob = dist_to_prob(dist, radius)
        prob.zero_()[mask] = subprob

        label.masked_fill_(prob > max_prob, count)
        max_prob = torch.maximum(max_prob, prob)
        sum_prob *= prob.neg_().add_(1)  # probability of no vessel
    sum_prob = sum_prob.neg_().add_(1)   # probability of at least one vessel

    return sum_prob, label


def draw_curves_binary(shape, curves, fast=0, **kwargs):
    if fast:
        return draw_curves_binary_fast(shape, curves, fast, **kwargs)

    curves = list(curves)
    locations = identity_grid(shape, **utils.backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)

    count = 0
    while curves:
        curve = curves.pop(0)
        count += 1
        time, dist = min_dist(locations, curve, **kwargs)
        radius = curve.eval_radius(time)
        is_vessel = dist <= radius
        label.masked_fill_(is_vessel, count)

    return label > 0, label


def draw_curves_binary_fast(shape, curves, threshold, **kwargs):
    curves = list(curves)
    locations = identity_grid(shape, **utils.backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)

    count = label.new_zeros([])
    while curves:
        curve = curves.pop(0)
        count += 1

        # initialize distance from table and only process
        # points that are close enough from the curve
        time, dist = min_dist_table(locations, curve)
        mask = dist < threshold
        sublocations = locations[mask, :]

        kwargs.setdefault('init_kwargs', {})
        kwargs['init_kwargs']['init'] = time[mask]
        time, dist = min_dist(sublocations, curve, **kwargs)
        radius = curve.eval_radius(time)
        is_vessel = dist <= radius
        label[mask] = torch.where(is_vessel, count, label[mask])

    return label > 0, label


def fit_curve_joint(f, s, max_iter=128, tol=1e-8):
    """Fit the curve that maximizes the joint probability p(f) * p(s)

    Parameters
    ----------
    f : (*shape) tensor
        Observed grid of binary labels or smooth probabilities.
    s : BSplineCurve
        Initial curve (will be modified in-place)
    max_iter : int, default=128
    tol : float, default=1e-8

    Returns
    -------
    s : BSplineCurve
        Fitted curve

    """

    x = identity_grid(f.shape, **utils.backend(s.coeff))

    max_iter_position = 10
    max_iter_radius = 3
    sumf = f.sum(dtype=torch.double)

    def get_nll(e):
        if f.dtype is torch.bool:
            return sumf + e.sum(dtype=torch.double) - 2 * e[f].sum(dtype=torch.double)
        else:
            return sumf + e.sum(dtype=torch.double) - 2 * (e*f).sum(dtype=torch.double)

    start = time.time()
    nll = float('inf')
    for n_iter in range(max_iter):

        nll0_prev = nll

        for n_iter_position in range(max_iter_position):

            t, d = min_dist(x, s)
            p = s.eval_position(t).sub_(x)  # residuals
            r = s.eval_radius(t)
            r = torch.as_tensor(r, **utils.backend(x))
            e = dist_to_prob(d, r)
            nll_prev = nll
            nll = get_nll(e)
            lam = radius_to_prec(r)

            # gradient of the categorical term
            g = e * (1 - 2*f) * (-lam)
            g = g.unsqueeze(-1)
            lam = lam.unsqueeze(-1)
            # e = e.unsqueeze(-1)
            # h = g.abs() + e * (lam * p).square()
            h = g.abs() * (1 + lam * p.square())
            g = g * p

            # push
            g = s.push_position(g, t)
            h = s.push_position(h, t)
            g.div_(h)
            s.coeff -= g

            # print('position', n_iter, n_iter_position,
            #       nll.item(), (nll_prev - nll).item() / f.numel())
            if nll_prev - nll < tol * f.numel():
                break

        for n_iter_position in range(max_iter_radius):

            alpha = (2.355/2)**2
            t, d = min_dist(x, s)
            r = s.eval_radius(t)
            r = torch.as_tensor(r, **utils.backend(x))
            e = dist_to_prob(d, r)
            d = d.square_()
            nll_prev = nll
            nll = get_nll(e)

            # gradient of the categorical term
            g = e * (1 - 2*f) * (alpha * d / r.pow(3))
            h = g.abs() * (alpha * d / r.pow(3)) * (1 + 3 / r)

            # push
            g = s.push_radius(g, t)
            h = s.push_radius(h, t)
            g.div_(h)
            s.coeff_radius -= g
            s.coeff_radius.clamp_min_(0.5)

            # print('radius', n_iter, n_iter_position,
            #       nll.item(), (nll_prev - nll).item() / f.numel())
            if nll_prev - nll < tol * f.numel():
                break

        if not n_iter % 10:
            print(n_iter, nll.item(), (nll0_prev - nll).item() / f.numel())
        if abs(nll0_prev - nll) < tol * f.numel():
            print('Converged')
            break

    stop = time.time()
    print(stop-start)


def plot_nll(nll, pred=None, ref=None, waypoints=None,
             f=0, elem=None, show_img=True):

    import matplotlib.pyplot as plt
    if not isinstance(f, plt.Figure):
        f = plt.figure(f)

    if not isinstance(nll, (list, tuple)):
        nll = [nll]
    nll = [nll1.item() if torch.is_tensor(nll1) else nll1
           for nll1 in nll]

    show_img = show_img and (pred is not None)

    if show_img and ref.dim() == 3:
        ref = ref[..., ref.shape[-1]//2]
        pred = pred[..., pred.shape[-1]//2]

    if not elem:
        if show_img:
            plt.subplot(2, 2, 1)
            plt.imshow(ref)
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(pred)
            plt.axis('off')
            if waypoints is not None:
                if not isinstance(waypoints, (list, tuple)):
                    waypoints = [waypoints]
                for wp in waypoints:
                    plt.plot(wp[:, 1], wp[:, 0], 'r')
            plt.subplot(1, 2, 2)
        for nll1 in nll:
            plt.plot([1], [nll1])
        if len(nll) > 1:
            plt.plot([1], [sum(nll)], linewidth=2)
        f.canvas.draw()
        elem = [f.canvas.copy_from_bbox(ax.bbox) for ax in f.axes]
        f.canvas.flush_events()
        plt.show(block=False)
    else:
        f.canvas.draw()
        for e in elem:
            f.canvas.restore_region(e)

        if show_img:
            ax = f.axes[0]
            dat = ax.images[0].get_array()
            if dat.shape != ref.shape:
                ax.images[0].set_data(ref)
            ax = f.axes[1]
            dat = ax.images[0].get_array()
            if dat.shape != pred.shape:
                ax.images[0].set_extent(
                    (-0.5, pred.shape[0]-0.5, pred.shape[1]-0.5, -0.5))
            ax.images[0].set_data(pred)
            if waypoints is not None:
                if not isinstance(waypoints, (list, tuple)):
                    waypoints = [waypoints]
                for i, wp in enumerate(waypoints):
                    ax.lines[i].set_data((wp[:, 1], wp[:, 0]))
            ax.relim()

        ax = f.axes[-1]
        lines = ax.lines
        if len(nll) > 1:
            lines = lines[:-1]
        for line, nll1 in zip(lines, nll):
            iter, all_nll1 = line.get_data()
            iter = [*iter, iter[-1]+1]
            all_nll1 = [*all_nll1, nll1]
            line.set_data(iter, all_nll1)
        if len(nll) > 1:
            line = ax.lines[-1]
            iter, all_nll = line.get_data()
            iter = [*iter, iter[-1]+1]
            all_nll = [*all_nll, sum(nll)]
            line.set_data(iter, all_nll)
        ax.relim()
        ax.autoscale_view()

        for ax in f.axes[:-1]:
            ax.draw_artist(ax.images[0])
            for line in ax.lines:
                ax.draw_artist(line)
            f.canvas.blit(ax.bbox)
        ax = f.axes[-1]
        for line in ax.lines:
            ax.draw_artist(line)
        f.canvas.blit(ax.bbox)
        f.canvas.flush_events()

    return f, elem


def fit_curve_cat(f, s, lam=0, gamma=0, vx=1, max_iter=8, tol=1e-8, max_levels=4):
    """Fit the curve that maximizes the categorical likelihood

    Parameters
    ----------
    f : (*shape) tensor
        Observed grid of binary labels or smooth probabilities.
    s : BSplineCurve
        Initial curve (will be modified in-place)

    Other Parameters
    ----------------
    lam : float, default=0
        Centerline regularization (bending)
    gamma : float, default=0
        Radius regularization (membrane)
    vx : float, default=1
        Voxel size
    max_iter : int, default=128
        Maximum number of iterations per level
        (This will me multiplied by 2 at each resolution level, such that
        more iterations are used at coarser levels).
    tol : float, default=1e-8
        Unused
    max_levels : int, default=4
        Number of multi-resolution levels.

    Returns
    -------
    s : BSplineCurve
        Fitted curve

    """
    TINY = 1e-6
    fig = elem = None

    max_iter_position = 8
    max_iter_radius = 4

    backend = utils.backend(s.coeff)
    vx = utils.make_vector(vx, f.dim(), **backend)
    vx0 = vx.clone()
    n0 = f.numel()

    # Build pyramid by restriction
    shapes = [f.shape]
    images = [f]
    vxs = [vx]
    for n_level in range(max_levels-1):
        shape = [pymath.ceil(s / 2) for s in shapes[-1]]
        if all(s == 1 for s in shape):
            break
        shapes.append(shape)
        images.append(restrict(f.unsqueeze(-1), shapes[-1]).squeeze(-1))
        s.restrict(shapes[-2], shapes[-1])
        vx = vx * (torch.as_tensor(shapes[-2], **backend) /
                   torch.as_tensor(shapes[-1], **backend))
        vxs.append(vx)

    start = time.time()

    shape = None
    level = len(images) + 1
    while images:
        level -= 1
        print('-'*16, 'level', level, '-'*16)

        if shape is not None:
            s.prolong(shape, shapes[-1])
        f, shape, vx = images.pop(-1), shapes.pop(-1), vxs.pop(-1)
        scl = vx.prod() / vx0.prod()
        x = identity_grid(f.shape, **backend)
        if lam:
            L = lam * bending3(len(s.coeff), **backend)
            reg = L.matmul(s.coeff).mul_(vx.square())
            reg = 0.5 * (s.coeff * reg).sum(dtype=torch.double)
        else:
            reg = 0
        if gamma:
            Lr = gamma * membrane3(len(s.coeff_radius), **backend)
            Lr /= vx.prod().pow_(1/len(vx)).square_()
            reg_radius = Lr.matmul(s.coeff_radius)
            reg_radius = 0.5 * (s.coeff_radius * reg_radius).sum(dtype=torch.double)
        else:
            reg_radius = 0

        def get_nll(e):
            ie = (1 - e).log()
            e = e.log()
            if f.dtype is torch.bool:
                ll = e[f].sum(dtype=torch.double) + ie[~f].sum(dtype=torch.double)
            else:
                ll = (e * f).sum(dtype=torch.double) + (ie * (1 - f)).sum(dtype=torch.double)
            ll = -ll
            return ll

        nll = float('inf')
        max_iter_level = max_iter * 2 ** ((level-1)//2)
        for n_iter in range(max_iter_level):

            nll0_prev = nll

            for n_iter_position in range(max_iter_position):

                t, d = min_dist(x, s)
                p = s.eval_position(t).sub_(x)  # residuals
                r = s.eval_radius(t)
                r = torch.as_tensor(r, **utils.backend(x))
                e = dist_to_prob(d, r, tiny=TINY)
                nll_prev = nll
                nll = get_nll(e)
                prec = radius_to_prec(r)

                # gradient of the categorical term
                omf = (1 - f) if f.dtype.is_floating_point else f.bitwise_not()
                ome = (1 - e)
                g = (omf / ome - 1) * (-prec)
                h = omf * e / ome.square()
                g = g.unsqueeze(-1)
                h = h.unsqueeze(-1)
                prec = prec.unsqueeze(-1)

                acc = 0.5
                h = h * (prec * p).square()
                if acc != 1:
                    h += (1 - acc) * g.abs()
                g = g * p

                # push
                g = s.push_position(g, t)
                h = s.push_position(h, t)

                # resolution scale
                g *= scl
                h *= scl
                nll *= scl

                # regularisation + solve
                if lam:
                    reg = L.matmul(s.coeff).mul_(vx.square())
                    g += reg
                    reg = 0.5 * (s.coeff * reg).sum(dtype=torch.double)
                    # h += L[1, :].abs().sum()
                    g = torch.stack([linalg.lmdiv(h1.diag() + (v1*v1)*L, g1[:, None])[:, 0]
                                     for v1, g1, h1 in zip(vx, g.T, h.T)], -1)
                else:
                    g.div_(h)
                    reg = 0
                s.coeff.sub_(g)
                # s.coeff.clamp_min_(0)
                # for d, sz in enumerate(f.shape):
                #     s.coeff[:, d].clamp_max_(sz-1)

                fig, elem = plot_nll([nll, reg, reg_radius], e, f, s.waypoints, fig, elem)
                nll = nll + reg + reg_radius
                print('position', n_iter, n_iter_position,
                      nll.item(), (nll_prev - nll).item() / n0)
                s.update_waypoints()
                # if nll_prev - nll < tol * f.numel():
                #     break

            if level < 3:
                max_iter_radius_level = max_iter_radius
            else:
                max_iter_radius_level = 0
            for n_iter_radius in range(max_iter_radius_level):

                alpha = (2.355/2)**2
                t, d = min_dist(x, s)
                r = s.eval_radius(t)
                r = torch.as_tensor(r, **utils.backend(x))
                e = dist_to_prob(d, r, TINY)
                d = d.square_()
                nll_prev = nll
                nll = get_nll(e)

                # gradient of the categorical term
                omf = (1 - f) if f.dtype.is_floating_point else f.bitwise_not()
                ome = (1 - e)
                alpha = alpha * d / r.pow(3)
                g = (omf / ome - 1) * alpha

                acc = 0
                h = omf * e / ome.square()
                h *= alpha.square()
                if acc != 1:
                    h += (1 - acc) * g.abs() * 3 / r

                # push
                g = s.push_radius(g, t)
                h = s.push_radius(h, t)

                # resolution scale
                g *= scl
                h *= scl
                nll *= scl

                # regularisation + solve
                if gamma:
                    reg_radius = Lr.matmul(s.coeff_radius)
                    g += reg_radius
                    reg_radius = 0.5 * (s.coeff_radius * reg_radius).sum(dtype=torch.double)
                    g = linalg.lmdiv(h.diag() + L, g[:, None])[:, 0]
                else:
                    g.div_(h)
                    reg_radius = 0

                # solve
                s.coeff_radius -= g
                s.coeff_radius.clamp_min_(0.5)

                fig, elem = plot_nll([nll, reg, reg_radius], e, f, s.waypoints, fig, elem)
                nll = nll + reg + reg_radius
                print('radius', n_iter, n_iter_radius,
                      nll.item(), (nll_prev - nll).item() / n0)
                s.update_radius()
                # if nll_prev - nll < tol * f.numel():
                #     break

            if not n_iter % 10:
                print(n_iter, nll.item(), (nll0_prev - nll).item() / n0)
            # if nll0_prev - nll < tol * f.numel():
            #     print('Converged')
            #     break

    stop = time.time()
    print(stop-start)


def _draw_curves_inv(shape, s, tiny=0):
    """prod_k (1 - p_k)"""
    s = list(s)
    x = identity_grid(shape, **utils.backend(s[0].waypoints))
    s1 = s.pop(0)
    t, d = min_dist(x, s1)
    r = s1.eval_radius(t)
    c = dist_to_prob(d, r, tiny=tiny).neg_().add_(1)
    while s:
        s1 = s.pop(0)
        t, d = min_dist(x, s1)
        r = s1.eval_radius(t)
        c.mul_(dist_to_prob(d, r, tiny=tiny).neg_().add_(1))
    return c


def fit_curves_cat(f, s, vx=1, max_iter=8, tol=1e-8, max_levels=4):
    """Fit the set of curves that maximizes a Categorial likelihood

    Parameters
    ----------
    f : (*shape) tensor
        Observed grid of binary labels or smooth probabilities.
    s : list[BSplineCurve]
        Initial curves (will be modified in-place)

    Returns
    -------
    s : list[BSplineCurve]
        Fitted curves

    """

    TINY = 1e-6
    fig = elem = None
    backend = utils.backend(s[0].coeff)

    max_iter_position = 8
    max_iter_radius = 4

    vx = utils.make_vector(vx, f.dim(), **backend)
    vx0 = vx.clone()
    n0 = f.numel()

    # Build pyramid by restriction
    shapes = [f.shape]
    images = [f]
    vxs = [vx]
    for n_level in range(max_levels-1):
        shape = [pymath.ceil(s / 2) for s in shapes[-1]]
        if all(s == 1 for s in shape):
            break
        shapes.append(shape)
        images.append(restrict(f.unsqueeze(-1), shapes[-1]).squeeze(-1))
        vx = vx * (torch.as_tensor(shapes[-2], **backend) /
                   torch.as_tensor(shapes[-1], **backend))
        vxs.append(vx)
        for s1 in s:
            s1.restrict(shapes[-2], shapes[-1])

    start = time.time()

    shape = None
    level = len(images) + 1
    while images:
        level -= 1
        print('-'*16, 'level', level, '-'*16)

        if shape is not None:
            for s1 in s:
                s1.prolong(shape, shapes[-1])
        f, shape, vx = images.pop(-1), shapes.pop(-1), vxs.pop(-1)
        x = identity_grid(f.shape, **backend)
        scl = vx.prod() / vx0.prod()

        def get_nll(e):
            ie = (1 - e).log()
            e = e.log()
            if f.dtype is torch.bool:
                ll = e[f].sum(dtype=torch.double) + ie[~f].sum(dtype=torch.double)
            else:
                ll = (e * f).sum(dtype=torch.double) + (ie * (1 - f)).sum(dtype=torch.double)
            return -ll

        nll = float('inf')
        max_iter_level = max_iter * 2 ** ((level-1)//2)
        for n_iter in range(max_iter_level):

            nll0_prev = nll

            for n_curve in range(len(s)):

                s0 = s[n_curve]
                s1 = s[:n_curve] + s[n_curve+1:]
                ie1 = _draw_curves_inv(f.shape, s1, TINY)

                for n_iter_position in range(max_iter_position):

                    t, d = min_dist(x, s0)
                    p = s0.eval_position(t).sub_(x)  # residuals
                    r = s0.eval_radius(t)
                    r = torch.as_tensor(r, **utils.backend(x))
                    e0 = dist_to_prob(d, r, TINY)
                    ome0 = 1 - e0
                    e = 1 - ome0 * ie1
                    nll_prev = nll
                    nll = get_nll(e)
                    lam = radius_to_prec(r)

                    # gradient of the categorical term
                    g = (1 - f/e) * e0 / ome0 * (-lam)
                    h = (e0 / ome0).square() * (1 - e) / e
                    g = g.unsqueeze(-1)
                    h = h.unsqueeze(-1)
                    lam = lam.unsqueeze(-1)

                    acc = 0.5
                    h = h * (lam * p).square()
                    if acc != 1:
                        h += (1 - acc) * g.abs()
                    g = g * p

                    # push
                    g = s0.push_position(g, t)
                    h = s0.push_position(h, t)

                    g *= scl
                    h *= scl
                    nll *= scl

                    g.div_(h)
                    s0.coeff -= g

                    wp = [ss.waypoints for ss in s]
                    fig, elem = plot_nll(nll, e, f, wp, fig, elem)
                    print('position', n_iter, n_curve, n_iter_position,
                          nll.item(), (nll_prev - nll).item() / n0)
                    s0.update_waypoints()
                    # if nll_prev - nll < tol * f.numel():
                    #     break

                if level < 3:
                    max_iter_radius_level = max_iter_radius
                else:
                    max_iter_radius_level = 0
                for n_iter_radius in range(max_iter_radius_level):

                    alpha = (2.355/2)**2
                    t, d = min_dist(x, s0)
                    r = s0.eval_radius(t)
                    r = torch.as_tensor(r, **utils.backend(x))
                    e0 = dist_to_prob(d, r)
                    ome0 = 1 - e0
                    e = 1 - ome0 * ie1
                    d = d.square_()
                    nll_prev = nll
                    nll = get_nll(e)

                    # gradient of the categorical term
                    alpha = alpha * d / r.pow(3)
                    g = (1 - f/e) * e0 / ome0 * alpha
                    h = e0 / ome0.square()

                    acc = 0
                    h *= alpha.square()
                    if acc != 1:
                        h += (1 - acc) * g.abs() * 3 / r

                    # push
                    g = s0.push_radius(g, t)
                    h = s0.push_radius(h, t)

                    g *= scl
                    h *= scl
                    nll *= scl

                    g.div_(h)
                    s0.coeff_radius -= g
                    s0.coeff_radius.clamp_min_(0.5)

                    wp = [ss.waypoints for ss in s]
                    fig, elem = plot_nll(nll, e, f, wp, fig, elem)
                    print('radius', n_iter, n_curve, n_iter_radius,
                          nll.item(), (nll_prev - nll).item() / n0)
                    s0.update_radius()
                    # if nll_prev - nll < tol * f.numel():
                    #     break

            if not n_iter % 10:
                print(n_iter, nll.item(), (nll0_prev - nll).item() / n0)
            # if abs(nll0_prev - nll) < tol * f.numel():
            #     print('Converged')
            #     break

        stop = time.time()
        print(stop-start)


class Brent:
    """Batched 1D optimizer (derivative-free)"""

    gold = (1 + pymath.sqrt(5))/2
    igold = 1 - (pymath.sqrt(5) - 1)/2
    tiny = 1e-8

    def __init__(self, tol=1e-9, max_iter=128, step=0.1):
        """

        Parameters
        ----------
        tol : float, defualt=1e-9
            Tolerance for early stopping
        max_iter : int, default=128
            Maximum number of iterations
        step : float, default=0.1
            Initial step size used to compute the initial bracket
        """
        self.tol = tol
        self.max_iter = max_iter
        self.step = step

    def fit(self, init, closure):
        """

        Parameters
        ----------
        init : tensor
            Initial estimate of the minimum
        closure : callable(tensor) -> tensor
            Function that computes the loss

        Returns
        -------
        a : tensor
            Location of the minimum
        f : tensor
            Value of the minimum


        """
        bracket = self.bracket(init, closure(init), closure)
        return self.search_in_bracket(bracket, closure)

    def bracket(self, a0, f0, closure):
        """Bracket the minimum

        Parameters
        ----------
        a0 : Initial parameter
        f0 : Initial value
        closure : callable(a) -> evaluate function at `a`

        Returns
        -------
        a0, a1, a2, f0, f1, f2
            (a1, f1) is the current estimate of the minimum location and value
            a1 is in (a0, a2) or (a2, a0)
            f1 is lower than both f0 and f2
        """
        @torch.jit.script
        def quad_min(a0, a1, a2, f0, f1, f2):
            y01 = a0 * a0 - a1 * a1
            y02 = a0 * a0 - a2 * a2
            y12 = a1 * a1 - a2 * a2
            a01 = a0 - a1
            a02 = a0 - a2
            a12 = a1 - a2
            a = (f0 * y12 + f1 * y02 + f2 * y01) / (f0 * a12 + f1 * a02 + f2 * a01)
            s = f0 / (y01 * y02) + f1 / (y01 * y12) + f2 / (y02 * y12)
            return a / 2, s

        a1 = a0 + self.step
        f1 = closure(a1)

        # sort such that f1 < f0
        mask = f1 > f0
        a0, a1 = torch.where(mask, a1, a0), torch.where(mask, a0, a1)
        f0, f1 = torch.where(mask, f1, f0), torch.where(mask, f0, f1)

        a2 = (a1 - a0).mul_(self.gold).add_(a1)
        f2 = closure(a2)

        mask0 = f2 < f1
        while mask0.any():
            # fit quadratic polynomial
            delta0 = a2 - a1
            delta, s = quad_min(a0, a1, a2, f0, f1, f2)
            delta = delta.sub_(a1).minimum((1 + self.gold) * delta0)
            a = torch.where(s > 0, a1 + delta, a2 + self.gold * delta0)

            # check progress and update bracket
            # f2 < f1 < f0 so (assuming unicity) the minimum is in
            # (a1, a2) or (a2, inf)
            f = closure(a)
            mask1 = ((a1 < a) == (a < a2))
            mask2 = mask0 & mask1 & (f < f2)
            a0, a1 = torch.where(mask2, a1, a0), torch.where(mask2, a, a1)
            f0, f1 = torch.where(mask2, f1, f0), torch.where(mask2, f, f1)
            mask2 = mask0 & mask1 & (f2 < f)
            a2 = torch.where(mask2, a, a2)
            f2 = torch.where(mask2, f, f2)
            # shift by one point
            mask2 = mask0 & ~mask1
            a0, a1, a2 = (torch.where(mask2, a1, a0),
                          torch.where(mask2, a2, a1),
                          torch.where(mask2, a, a2))
            f0, f1, f2 = (torch.where(mask2, f1, f0),
                          torch.where(mask2, f2, f1),
                          torch.where(mask2, f, f2))
            mask0 = f2 < f1

        return a0, a1, a2, f0, f1, f2

    def search_in_bracket(self, bracket, closure):
        """

        Parameters
        ----------
        bracket : tuple[tensor] = (a0, a1, a2, f0, f1, f2)
            Estimate (a1, f1) and bracket [(a0, f0), (a2, f2)] returned
            by the `bracket` function.
        closure : callable(a) -> f
            Function that computes the objective function.

        Returns
        -------
        a : tensor
            Location of the minimum
        f : tensor
            Value of the minimum

        """

        @torch.jit.script
        def quad_min(a0, a1, a2, f0, f1, f2):
            y01 = a0 * a0 - a1 * a1
            y02 = a0 * a0 - a2 * a2
            y12 = a1 * a1 - a2 * a2
            a01 = a0 - a1
            a02 = a0 - a2
            a12 = a1 - a2
            a = (f0 * y12 + f1 * y02 + f2 * y01) / (f0 * a12 + f1 * a02 + f2 * a01)
            s = f0 / (y01 * y02) + f1 / (y01 * y12) + f2 / (y02 * y12)
            return a / 2, s

        @torch.jit.script
        def sort(a0, a1, a2, f0, f1, f2):
            mask = f2 < f1
            a1, a2 = torch.where(mask, a2, a1), torch.where(mask, a1, a2)
            f1, f2 = torch.where(mask, f2, f1), torch.where(mask, f1, f2)
            mask = f1 < f0
            a0, a1 = torch.where(mask, a1, a0), torch.where(mask, a0, a1)
            f0, f1 = torch.where(mask, f1, f0), torch.where(mask, f0, f1)
            mask = f2 < f1
            a1, a2 = torch.where(mask, a2, a1), torch.where(mask, a1, a2)
            f1, f2 = torch.where(mask, f2, f1), torch.where(mask, f1, f2)
            return a0, a1, a2, f0, f1, f2

        b0, b1 = bracket[0].clone(), bracket[2].clone()
        mask = b1 < b0
        b0, b1 = torch.where(mask, b1, b0), torch.where(mask, b0, b1)
        a0, a1, a2, f0, f1, f2 = sort(*bracket)  # sort by values

        d = torch.full_like(a0, float('inf'))
        d0 = d.clone()
        for n_iter in range(1, self.max_iter+1):

            mask_stop = abs(a0 - 0.5 * (b0 + b1)) + 0.5 * (b1 - b0) <= 2 * self.tol
            if mask_stop.all():
                return a0, f0

            d1, d0 = d0, d

            # fit quadratic polynomial
            a, s = quad_min(a0, a1, a2, f0, f1, f2)
            d = a - a0

            tiny = self.tiny * (1 + 2 * a0.abs())
            mask_nomin = (s < 0) | (abs(d) > abs(d1)/2) | ~((b0 + tiny < a) & (a < b1 - tiny))
            mask_nomin2 = a0 > 0.5 * (b0 + b1)
            d = torch.where(mask_nomin & mask_nomin2, (b0 - a0) * self.igold, d)
            d = torch.where(mask_nomin & ~mask_nomin2, (b1 - a) * self.igold, d)
            a = torch.where(mask_nomin, a0 + d, a)

            # check progress and update bracket
            f = closure(a)
            # f < f0 < f1 < f2
            mask = f < f0
            mask2 = a < a0
            b0, b1 = (torch.where(mask & ~mask2, a0, b0),
                      torch.where(mask & mask2, a0, b1))
            a0, a1, a2 = (torch.where(mask, a, a0),
                          torch.where(mask, a0, a1),
                          torch.where(mask, a1, a2))
            f0, f1, f2 = (torch.where(mask, f, f0),
                          torch.where(mask, f0, f1),
                          torch.where(mask, f1, f2))

            mask = f0 < f
            b0, b1 = (torch.where(mask & mask2, a, b0),
                      torch.where(mask & ~mask2, a, b1))
            # f0 < f < f1 < f2
            mask = (f0 < f) & (f < f1)
            a1, a2 = torch.where(mask, a, a1), torch.where(mask, a1, a2)
            f1, f2 = torch.where(mask, f, f1), torch.where(mask, f1, f2)
            mask = (f1 < f) & (f < f2)
            a2 = torch.where(mask, a, a2)
            f2 = torch.where(mask, f, f2)

        return a0, f0

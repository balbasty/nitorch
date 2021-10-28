from timeit import default_timer as timer
import math as pymath
from nitorch.core.optim import plot_convergence
from nitorch.core import linalg, math, utils, py
from nitorch.plot import plot_mixture_fit
from nitorch import spatial
import torch

default_warp_reg = {'absolute': 0,
                    'membrane': 0.001,
                    'bending': 0.5,
                    'lame': (0.05, 0.2), }


def uniseg(x, w=None, aff=None, bias=True, warp=True, mixing=True,
           nb_classes=2, prior=None, prior_aff=None, lam_bias=1e6, lam_warp=None,
           max_iter=30, tol=1e-4, verbose=1, show_fit=False):

    # prepare inputs
    batch, channel, *spatial = x.shape
    dim = (x.dim() - 2)
    if w is not None:
        w = w.expand([batch, 1, *spatial])
    else:
        w = [None] * batch
    if aff is not None:
        aff = aff.expand([batch, dim+1, dim+1])
    else:
        aff = [None] * batch

    # loop over batch elements
    models = []
    z = []
    lb = []
    for b in range(batch):
        model = UniSeg(
            dim, nb_classes, bias=bias, warp=warp, mixing=mixing,
            prior=prior, prior_aff=prior_aff, lam_bias=lam_bias, lam_warp=lam_warp,
            max_iter=max_iter, tol=tol, verbose=verbose, show_fit=show_fit,
        )

        z1, lb1 = model.fit(x[b], w[b], aff=aff[b])
        models.append(model)
        z.append(z1)
        lb.append(lb1)

    # stack results
    parameters = {}
    parameters['mean'] = torch.stack([model.mu for model in models])
    parameters['cov'] = torch.stack([model.sigma for model in models])
    if warp:
        parameters['warp'] = torch.stack([model.alpha for model in models])
    if bias:
        parameters['bias'] = torch.stack([model.beta.exp() for model in models])
    if mixing:
        parameters['mixing'] = torch.stack([model.gamma for model in models])

    return torch.stack(z), torch.stack(lb), parameters


class SpatialMixture:
    """Mixture model with spatially varying priors"""

    _tiny = 1e-32
    _tinyish = 1e-3

    def __init__(self, nb_classes=6,
                 bias=True, warp=True, mixing=True, prior=None, prior_aff=None,
                 lam_bias=1, lam_warp=1, lam_mixing=100,
                 bias_acceleration=0.9, warp_acceleration=0.9, spacing=3,
                 max_iter=30, tol=1e-4, max_iter_intensity=8,
                 max_iter_cluster=20, max_iter_bias=1, max_iter_warp=3,
                 verbose=1, verbose_bias=0, verbose_warp=0, show_fit=False):
        """
        Parameters
        ----------
        nb_classes : int or sequence[int], default=2
            Number of components in the mixture (including background).
            If a sequence, defines the number of components within each
            "prior" class; e.g. `[5, 3, 3, 2, 2, 1]`
        bias : bool, default=True
            Optimize bias field
        warp : bool, default=True
            Optimize warping field
        mixing : bool, default=True
            Optimize stationary mixing proportions
        prior : (nb_classes, [*spatial]) tensor, optional
            GMM mixing proportions. Stationary and learned by default.
        prior_aff : (D+1, D+1) tensor, optionsl
            Orientation matrix of the prior.
        lam_bias : float, default=1
            Regularization factor of the bias field.
        lam_warp : float or dict, default=1
            Regularization of the warping field.
            If a scalar, multiplies the default parameters, which are:
            {'membrane': 1e-3, 'bending': 0.5, 'lame': (0.05, 0.2)}
        lam_mixing : float, default=100
            Dirichlet regularization of the mixing proportions

        Other Parameters
        ----------------
        max_iter : int, default=30
            Maximum number of outer (coordinate descent) iterations
        tol : float, default=1e-4
            Tolerance for early stopping
        max_iter_intensity : int, default=8
            Maximum number of GMM+Bias (EM) iterations
        max_iter_cluster : int, default=20
            Maximum number of GMM iterations
        max_iter_bias : int, default=1
            Maximum number of Bias (Gauss-Newton) iterations
        max_iter_warp : int, default=3
            Maximum number of Warp (Gauss-Newton) iterations
        bias_acceleration : float, default=0.9
        warp_acceleration : float, default=0.9
        spacing : float, default=3
            Distance (in mm) between sampled points when optimizing parameters
        verbose : int, default=1
            0: None.
            1: Print summary when finished.
            2: 1 + print convergence.
            3: 1 + 2 + Log-likelihood plot.
        verbose_bias : int, default=0
        verbose_warp : int, default=0
        show_fit : bool or int, default=False

        """
        # Convert nb_classes to a list of cluster indices
        if isinstance(nb_classes, int):
            lkp = list(range(nb_classes))
        else:
            nb_classes = py.make_list(nb_classes)
            lkp = [n for n, k in enumerate(nb_classes) for _ in range(k)]
        self.lkp = lkp

        if prior is not None:
            implicit_in = len(prior) < self.nb_classes
            prior = math.logit(prior, implicit=(implicit_in, True), dim=0)
            if prior_aff is None:
                prior_aff = spatial.affine_default(prior.shape[1:])
        self.prior = prior
        self.prior_aff = prior_aff
        self.lam_bias = lam_bias * 1e6
        if isinstance(lam_warp, (int, float)):
            factor_warp = lam_warp
            lam_warp = {k: v*factor_warp for k, v in default_warp_reg.items()}
        elif not lam_warp:
            lam_warp = default_warp_reg
        self.lam_warp = lam_warp
        self.lam_mixing = lam_mixing
        self.warp_acceleration = warp_acceleration
        self.bias_acceleration = bias_acceleration
        self.bias = bias
        self.warp = warp
        self.mixing = mixing

        if prior is None:
            self.warp = False

        # Optimization
        self.spacing = spacing
        self.max_iter = max_iter
        self.max_iter_intensity = max_iter_intensity
        self.max_iter_cluster = max_iter_cluster
        self.max_iter_bias = max_iter_bias
        self.max_iter_warp = max_iter_warp
        self.max_ls_warp = 12
        self.tol = tol

        if not self.bias:
            self.max_iter_bias = 0
        if not self.warp:
            self.max_iter_warp = 0

        # Verbosity
        self.verbose = verbose
        self.verbose_bias = verbose_bias
        self.verbose_warp = verbose_warp
        self.show_fit = show_fit
        if self.show_fit is True:
            self.show_fit = 1

        self.lam = []

    @property
    def nb_classes(self):
        return max(x for x in self.lkp) + 1

    @property
    def nb_clusters(self):
        return len(self.lkp)

    @property
    def nb_clusters_per_class(self):
        return [self.lkp.count(k) for k in range(self.nb_clusters)]

    # Functions
    def fit(self, X, W=None, aff=None, **kwargs):
        """ Fit mixture model.

        Parameters
        ----------
        X : (C, *spatial) tensor
            Observed data
                C = num channels
                *spatial = spatial dimensions. Can be [x], [x,y] or [x,y,z]
        W : (*spatial) tensor, optional
            Observation weights.
        aff : (D+1, D+1) tensor, optional
            Orientation matrix

        Other Parameters
        ----------------
        alpha : (*spatial, D) tensor, default=0
            Initial displacement
        beta : (C, *spatial) tensor, default=0
            Initial log-bias
        gamma : (K,) tensor, default=1/K
            Initial mixing proportions
        kappa : (K,) tensor, default=1/Ki
            Initial within-class mixing proportions

        Returns
        -------
        Z : (K, [*spatial]) tensor
            Class responsibilities

        """
        with torch.random.fork_rng():
            torch.random.manual_seed(1)
            return self._fit(X, W, aff, **kwargs)

    def _fit(self, X, W=None, aff=None, **kwargs):
        if self.verbose > 0:
            t0 = timer()  # Start timer
            if X.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(X.device)

        if not X.dtype.is_floating_point:
            print(f'Data type {X.dtype} not supported - converting to '
                  f'{torch.get_default_dtype()}.')
            X = X.to(torch.get_default_dtype())

        # Try to guess the number of spatial dimensions
        if self.prior is not None:
            dim = self.prior.dim() - 1
        elif aff is not None:
            dim = aff.shape[-1] - 1
        else:
            dim = X.dim() - 1
        if X.dim() == dim:
            # add a channel dimensions
            X = X[None]
        elif not dim:
            # can only be 1D -> add a channel dimensions
            X = X[None]

        # Prepare mask/weight
        W0 = W
        W = X.isfinite()
        W.bitwise_and_(X != 0)
        if not W.all():
            X = X.clone()
            X[~W] = 0
        if W0 is not None:
            W = W.to(W.dtype).mul_(W0.to(W.device))
        if W.dtype is torch.bool:
            W = W.all(dim=0, keepdim=True)
        else:
            W = W.prod(dim=0, keepdim=True)

        # default affine
        if aff is None:
            aff = spatial.affine_default(X.shape[1:])

        # Subsample
        X0, W0, aff0 = X, W, aff
        if self.spacing:
            vx = spatial.voxel_size(aff)
            factor = (vx / self.spacing).tolist()
            X, aff = spatial.resize(X[None], factor=factor, affine=aff)
            X = X[0]
            W = utils.unsqueeze(W, 0, max(0, X.dim()-W.dim()))
            W = spatial.resize(W[None].to(X.dtype), factor=factor)[0]

        # Initialise model parameters
        self._init_parameters(X, W, aff, **kwargs)

        # EM loop
        Z, lb = self._em(X, W, aff)

        # Estimate Z at highest resolution
        if self.spacing:
            X = X0
            W = W0
            M = None
            if self.beta is not None:
                self.beta = spatial.reslice(
                    self.beta, aff, aff0, X.shape[1:],
                    bound='dct2', interpolation=3,
                    prefilter=True, extrapolate=True)
                X = self.beta.exp().mul_(X)
            if self.prior is not None:
                self.alpha = utils.movedim(self.alpha, -1, 0)
                self.alpha = spatial.reslice(
                    self.alpha, aff, aff0, X.shape[1:],
                    bound='dft', interpolation=3,
                    prefilter=True, extrapolate=True)
                self.alpha = utils.movedim(self.alpha, 0, -1)
                factor = spatial.voxel_size(aff) / spatial.voxel_size(aff0)
                self.alpha *= factor
                M = self.warp_tpm(aff=aff0)
            Z, _ = self.e_step(X, W, M)

        # Print algorithm info
        if self.verbose > 0:
            v = (f'Algorithm finished in {len(lb)} iterations, '
                 f'log-likelihood = {lb[-1].item()}, '
                 f'runtime: {timer() - t0:0.1f} s, '
                 f'device: {X.device}, ')
            if X.device.type == 'cuda':
                vram_peak = torch.cuda.max_memory_allocated(X.device)
                vram_peak = int(vram_peak / 2 ** 20)
                v += f'peak VRAM: {vram_peak} MB'
            print(v)
        if self.verbose >= 3:
            _ = plot_convergence(lb, xlab='Iteration number',
                                 fig_title='Model lower bound',
                                 fig_num=self.show_fit)
        # Plot mixture fit
        if self.show_fit:
            self._plot_fit(X, W, fig_num=self.show_fit+1)

        return Z

    def e_step(self, X, W=None, M=None, combine=False):
        """Perform the Expectation step

        Parameters
        ----------
        X : (C, *spatial) tensor
            Observed image
        W : (*spatial) tensor, optional
            Weights
        M : (K, *spatial) tensor, optional
            Non-stationary prior

        Returns
        -------
        Z : (K, *spatial) tensor
            Responsibilities
        lb : tensor

        """
        if combine:
            Z, lb = self.e_step(X, W, M)
            Z = self._z_combine(Z)
            return Z, lb

        if self.verbose >= 3:
            print('compute Z')

        M = M.log() if M is not None else None
        gamma = self.gamma.log() if self.gamma is not None else None
        N = X.shape[1:]
        Z = X.new_zeros((self.nb_clusters, *N))
        # --- likelihood ---
        for k, k0 in enumerate(self.lkp):
            Z[k] += self._log_likelihood(X, cluster=k)
            # --- prior (stationary component) ---
            Z[k] += gamma[k0]
            # --- prior (nonstationary component) ---
            if M is not None:
                Z[k] += M[k0]
        # --- softmax ---
        Z, lb = math.softmax_lse(Z + self._tinyish, lse=True, weights=W, dim=0)
        return Z, lb

    def _z_combine(self, Z0):
        # Dumb way to compute the "classes" responsibilities.
        # There might be a better way that combines computing cluster-wise
        # log-likelihood and combining them.
        if self.nb_clusters == self.nb_classes:
            return Z0
        Z = Z0.new_zeros((self.nb_classes, *Z0.shape[1:]))
        for k0, k in enumerate(self.lkp):
            Z[k] += Z0[k0]
        return Z

    def _em(self, X, W=None, aff=None):
        """EM loop for fitting GMM.

        Parameters
        ----------
        X : (C, *spatial) tensor
            Observed image
        W : (*spatial) tensor, optional
            Weights

        Returns
        -------
        Z : (K, *spatial) tensor
            Responsibilities
        lb : list[float]
            Lower bound at each iteration.

        """
        vx = spatial.voxel_size(aff) if aff is not None else None
        nW = W.sum() if W is not None else X[1:].numel()

        M = G = None
        if self.prior is not None:
            if self.warp:
                M, G = self.warp_tpm(aff=aff, grad=True)
            else:
                M = self.warp_tpm(aff=aff)

        XB = X
        if self.beta is not None:
            XB = self.beta.exp().mul_(X)

        all_lb = []
        all_all_lb = []
        lb = -float('inf')
        for n_iter in range(self.max_iter):  # EM loop
            if self.verbose >= 2:
                print(f'n_iter: {n_iter + 1}')
            olb_em = lb

            for n_iter_intensity in range(self.max_iter_intensity):
                olb_intensity = lb

                for n_iter_cluster in range(self.max_iter_cluster):
                    # ======
                    # E-step
                    # ======
                    olb = lb
                    Z, lb = self.e_step(XB, W, M)
                    lb += self._lb_parameters()
                    if self.verbose >= 3:
                        print(f'{n_iter:02d} | {n_iter_intensity:02d} | {n_iter_cluster:02d} | '
                              f'pre gmm: {lb.item():12.6g}')
                        all_all_lb.append(lb)
                        self._plot_lb(all_all_lb, X, Z, M)

                    # ==================
                    # M-step - Intensity
                    # ==================
                    Z = Z if W is None else Z*W
                    ss0, ss1, ss2 = self._suffstats(XB, Z)
                    self._update_intensity(ss0, ss1, ss2)

                    # ===========================
                    # M-step - Mixing proportions
                    # ===========================
                    if self.mixing:
                        self._update_mixing(ss0, M, W)

                    if n_iter_cluster > 1 and lb-olb < self.tol * nW:
                        break

                for n_iter_bias in range(self.max_iter_bias):
                    # ======
                    # E-step
                    # ======
                    olb = lb
                    Z, lb = self.e_step(XB, W, M)
                    lb += self._lb_parameters()
                    if self.verbose >= 3:
                        print(f'{n_iter:02d} | {n_iter_intensity:02d} | {n_iter_bias:02d} | '
                              f'pre bias: {lb.item():12.6g}')
                        all_all_lb.append(lb)
                        self._plot_lb(all_all_lb, X, Z, M)

                    # =============
                    # M-step - Bias
                    # =============
                    self._update_bias(XB, Z, W, vx=vx)
                    XB = torch.exp(self.beta, out=XB).mul_(X)

                    if n_iter_bias > 1 and lb-olb < self.tol * nW:
                        break

                if n_iter_intensity > 1 and lb - olb_intensity < 2 * self.tol * nW:
                    break

            for n_iter_warp in range(self.max_iter_warp):
                # ======
                # E-step
                # ======
                olb = lb
                Z, lb = self.e_step(XB, W, M, combine=True)
                lb += self._lb_parameters()
                if self.verbose >= 3:
                    print(f'{n_iter:02d} | {n_iter_warp:02d} | {"":2s} | '
                          f'pre warp: {lb.item():12.6g}')
                    all_all_lb.append(lb)
                    self._plot_lb(all_all_lb, X, Z, M)

                # =============
                # M-step - Warp
                # =============
                M = self._modulate_prior_(M)
                self._update_warp(Z, M, G, aff=aff)
                M, G = self.warp_tpm(aff=aff, grad=True)

                if n_iter_warp > 1 and lb - olb < self.tol * nW:
                    break

            # ==================
            # Log-likelihood
            # ==================
            all_lb.append(lb)
            if n_iter > 1 and lb - olb_em < 2 * self.tol * nW:
                if self.verbose > 0:
                    print('converged')
                break

        all_lb = torch.stack(all_lb)
        return Z, all_lb

    def _modulate_prior_(self, M):
        r"""Module non-stationary prior with stationary prior

        /!\ This is done in place

        Parameters
        ----------
        M : (K, *spatial) tensor

        Returns
        -------
        M : (K, *spatial) tensor

        """
        M = utils.movedim(M, 0, -1)
        M *= self.gamma
        M /= M.sum(-1, keepdim=True)
        M = utils.movedim(M, -1, 0)
        return M

    def _suffstats(self, X, Z):
        """ Compute sufficient statistics.

        Parameters
        ----------
        X : (C, *spatial) tensor
            Observed data modulated by the current bias field
        Z : (K, *spatial) tensor
            (Weighted) Responsibilities

        Returns
        -------
        ss0 : (K,) tensor
            0th moment (B, K).
        ss1 : (K, C) tensor
            1st moment
        ss2 : (K, C, C) tensor
            2nd moment

        """
        def mul(*x, out=None):
            x = list(x)
            out = x.pop(0).clone() if out is None else out.copy_(x.pop(0))
            while x:
                out.mul_(x.pop(0))
            return out

        def reduce(*x, buffer=None):
            return mul(*x, out=buffer).sum(dtype=torch.double)

        C = X.shape[0]
        K = Z.shape[0]

        ss1 = X.new_zeros((K, C), dtype=torch.double)
        ss2 = X.new_zeros((K, C, C), dtype=torch.double)

        # Compute 0th moment
        ss0 = torch.sum(Z.reshape(K, -1), dim=-1, dtype=torch.double)

        # Compute 1st and 2nd moments
        buffer = torch.empty_like(X[0])
        for k in range(K):
            for c in range(C):
                ss1[k, c] = reduce(X[c], Z[k], buffer=buffer)
                ss2[k, c, c] = reduce(X[c], X[c], Z[k], buffer=buffer)
                for cc in range(c + 1, C):
                    ss2[k, c, cc] = reduce(X[c], X[cc], Z[k], buffer=buffer)
                    ss2[k, cc, c] = ss2[k, c, cc]

        return ss0, ss1, ss2

    def _plot_fit(self, X, W, fig_num):
        """ Plot mixture fit."""
        if hasattr(self, 'get_means_variances'):
            mp = self.gamma
            mu, var = self.get_means_variances()
            log_pdf = lambda x, k, c: self._log_likelihood(x, k, c)
            plot_mixture_fit(X, log_pdf, mu, var, mp, fig_num, W)

    def _plot_lb(self, lb, X, Z, M=None):
        import matplotlib.pyplot as plt
        from nitorch.plot.colormaps import prob_to_rgb

        plt.figure(666)
        if X.dim()-1 == 3:
            X1 = X[:, :, :, X.shape[-1]//2]
            X2 = X[:, :, X.shape[-2]//2, :]
            X3 = X[:, X.shape[-3]//2, :, :]
            Z1 = Z[:, :, :, Z.shape[-1]//2]
            Z2 = Z[:, :, Z.shape[-2]//2, :]
            Z3 = Z[:, Z.shape[-3]//2, :, :]
            if len(Z) > self.nb_classes:
                Z1 = self._z_combine(Z1)
                Z2 = self._z_combine(Z2)
                Z3 = self._z_combine(Z3)
            ncol = len(X) + 1 + (M is not None) + 2*(self.beta is not None)
            for c in range(len(X)):
                mn = X[c].min()
                mx = X[c].max()
                plt.subplot(4, ncol, c+1)
                plt.imshow(X1[c], vmin=mn, vmax=0.8*mx)
                plt.axis('off')
                plt.title('Original')
                plt.subplot(4, ncol, ncol+c+1)
                plt.imshow(X2[c], vmin=mn, vmax=0.8*mx)
                plt.axis('off')
                plt.subplot(4, ncol, 2*ncol+c+1)
                plt.imshow(X3[c], vmin=mn, vmax=0.8*mx)
                plt.axis('off')

            offset = 0
            if self.beta is not None:
                B = self.beta
                B1 = B[:, :, :, B.shape[-1]//2].exp()
                B2 = B[:, :, B.shape[-2]//2, :].exp()
                B3 = B[:, B.shape[-3]//2, :, :].exp()
                X1 = X1 * B1
                X2 = X2 * B2
                X3 = X3 * B3
                for c in range(len(B)):
                    plt.subplot(4, ncol, len(X)+c+1)
                    plt.imshow(X1[c], vmin=mn, vmax=0.8*mx)
                    plt.axis('off')
                    plt.title('Corrected')
                    plt.subplot(4, ncol, ncol+len(X)+c+1)
                    plt.imshow(X2[c], vmin=mn, vmax=0.8*mx)
                    plt.axis('off')
                    plt.subplot(4, ncol, 2*ncol+len(X)+c+1)
                    plt.imshow(X3[c], vmin=mn, vmax=0.8*mx)
                    plt.axis('off')

                    plt.subplot(4, ncol, len(X)+c+2)
                    plt.imshow(B1[c])
                    plt.axis('off')
                    plt.title('Bias')
                    plt.subplot(4, ncol, ncol+len(X)+c+2)
                    plt.imshow(B2[c])
                    plt.axis('off')
                    plt.subplot(4, ncol, 2*ncol+len(X)+c+2)
                    plt.imshow(B3[c])
                    plt.axis('off')
                    plt.colorbar()
                offset = 2*len(B)

            plt.subplot(4, ncol, len(X)+offset+1)
            plt.imshow(prob_to_rgb(Z1))
            plt.axis('off')
            plt.title('Resp.')
            plt.subplot(4, ncol, ncol+len(X)+offset+1)
            plt.imshow(prob_to_rgb(Z2))
            plt.axis('off')
            plt.subplot(4, ncol, 2*ncol+len(X)+offset+1)
            plt.imshow(prob_to_rgb(Z3))
            plt.axis('off')

            if M is not None:
                M1 = self._modulate_prior_(M[:, :, :, M.shape[-1]//2].clone())
                M2 = self._modulate_prior_(M[:, :, M.shape[-2]//2, :].clone())
                M3 = self._modulate_prior_(M[:, M.shape[-3]//2, :, :].clone())
                plt.subplot(4, ncol, len(X)+2+offset)
                plt.imshow(prob_to_rgb(M1))
                plt.axis('off')
                plt.title('Prior')
                plt.subplot(4, ncol, ncol+len(X)+2+offset)
                plt.imshow(prob_to_rgb(M2))
                plt.axis('off')
                plt.subplot(4, ncol, 2*ncol+len(X)+2+offset)
                plt.imshow(prob_to_rgb(M3))
                plt.axis('off')

            plt.subplot(4, 1, 4)
            plt.plot(lb)
            plt.show()

    def _lb_parameters(self):
        return getattr(self, '_lb_warp', 0) + getattr(self, '_lb_bias', 0)

    # Implement in child classes
    def get_means_variances(self, *a, **k):
        pass

    def _log_likelihood(self, *a, **k):
        pass

    def _init_parameters(self, X, W=None, aff=None, **kwargs):
        C = X.shape[0]
        N = X.shape[1:]
        dim = X.dim() - 1
        backend = utils.backend(X)

        # spatial deformation coefficients
        alpha = kwargs.pop('alpha', None)
        if alpha is not None:
            self.alpha = torch.as_tensor(alpha, **backend).expand([*N, dim]).clone()
        elif self.warp:
            self.alpha = torch.zeros([*N, dim], **backend)
        else:
            self.alpha = None

        # bias field coefficients
        beta = kwargs.pop('beta', None)
        if beta is not None:
            self.beta = torch.as_tensor(beta, **backend).expand([C, *N]).clone()
        elif self.bias:
            self.beta = torch.zeros([C, *N], **backend)
        else:
            self.beta = None

        # mixing prop (extra)
        backend['dtype'] = torch.double
        gamma = kwargs.pop('gamma', None)
        if gamma is not None:
            self.gamma = torch.as_tensor(gamma, **backend).expand(self.nb_classes).clone()
        else:
            self.gamma = torch.ones(self.nb_classes, **backend)
            self.gamma /= self.nb_classes

        # mixing prop (intra)
        backend['dtype'] = torch.double
        kappa = kwargs.pop('kappa', None)
        if kappa is not None:
            self.kappa = torch.as_tensor(kappa, **backend).expand(self.nb_clusters).clone()
        else:
            self.kappa = torch.ones(self.nb_clusters, **backend)
            for k, K1 in enumerate(self.nb_clusters_per_class):
                offset = sum(self.nb_clusters_per_class[:k])
                self.kappa[offset:offset+K1] /= K1

    def _update_mixing(self, ss0, M=None, W=None):
        """
        Parameters
        ----------
        ss0 : (K,) tensor
            Zero-th statistic
        M : (K, *spatial) tensor, optional
            Non-stationary prior
        W : (*spatial) tensor, optional
            Observation weights
        """
        if self.verbose >= 3:
            print('update mixing')

        if M is not None:
            gamma = self.gamma.to(M.dtype)
            M = M.reshape(self.nb_classes, -1)
            ssm = linalg.dot(M.T, gamma).reciprocal_()
            if W is not None:
                ssm *= W.flatten()
            ssm = linalg.matvec(M, ssm)
            ssm = ssm.to(self.gamma.dtype)
        else:
            ssm = None

        kappa = ss0.clone()
        gamma = ss0.new_empty(self.nb_classes)
        Nw = ss0.sum()
        for k in range(self.nb_classes):
            mask = [(l == k) for l in self.lkp]
            mask = torch.as_tensor(mask, dtype=torch.bool, device=kappa.device)
            norm = kappa[mask].sum()
            kappa = torch.where(mask, kappa/(norm + self._tiny), kappa)
            if ssm is None:
                gamma[k] = norm / Nw
            else:
                gamma[k] = norm + self.lam_mixing
                gamma[k] /= ssm[k] + self.lam_mixing * self.nb_classes
        gamma /= gamma.sum()
        self.kappa.copy_(kappa)
        self.gamma.copy_(gamma)

    def _update_warp(self, Z, M, G, aff=None):
        """

        Parameters
        ----------
        Z : (K, *spatial) tensor
        M : (K, *spatial) tensor
        G : (K, *spatial, D) tensor
        vx : (D,) vector_like, optional

        """
        if self.verbose >= 3:
            print('update warp')
        vx = spatial.voxel_size(aff)

        def mul(*x, out=None):
            x = list(x)
            if out is None:
                out = x.pop(0).clone()
            else:
                out.copy_(x.pop(0))
            while x:
                out.mul_(x.pop(0))
            return out

        M0 = M
        M = M[1:]
        if len(Z) == self.nb_classes:
            Z = Z[1:]
        if len(G) == self.nb_classes:
            G = G[1:]
        shape = Z.shape[1:]
        dim = len(shape)
        a = self.warp_acceleration

        g = Z.new_zeros([*shape, dim])
        H = Z.new_zeros([*shape, dim*(dim+1)//2])
        H0 = 1 - 1/self.nb_classes
        H1 = -1/self.nb_classes
        buffer = torch.empty_like(g)
        for k in range(self.nb_classes-1):
            g.addcmul_((M[k] - Z[k]).unsqueeze(-1), G[k])
            if a:
                H0 = M[k].clone().addcmul_(M[k], M[k], value=-1).mul_(a)
                H0 += (1 - a) * (1 - 1/self.nb_classes)
                H0 = H0.unsqueeze(-1)
            # --- diagonal K / diagonal D ---
            H[..., :dim] += mul(G[k], G[k], H0, out=buffer)
            # --- diagonal K / off-diagonal D ---
            if a:
                H0 = H0.squeeze(-1)
            count = 0
            for d in range(dim):
                for dd in range(d+1, dim):
                    H[..., dim+count] += mul(G[k, ..., d], G[k, ..., dd], H0,
                                             out=buffer[..., 0])
                    count += 1
            # --- off-diagonal K ---
            for l in range(k+1, self.nb_classes-1):
                if a:
                    H1 = (M[k] * M[l]).mul_(a)
                    H1 += (1 - a) / self.nb_classes
                    H1.neg_()
                    H1 = H1.unsqueeze(-1)
                # --- off-diagonal K / diagonal D ---
                H[..., :dim].add_(mul(G[k], G[l], H1), alpha=2)
                # --- off-diagonal K / off-diagonal D ---
                if a:
                    H1 = H1.squeeze(-1)
                count = 0
                for d in range(dim):
                    for dd in range(d+1, dim):
                        H[..., dim+count] += mul(G[k, ..., d], G[l, ..., dd], H1,
                                                 out=buffer[..., 0])
                        H[..., dim+count] += mul(G[l, ..., d], G[k, ..., dd], H1,
                                                 out=buffer[..., 0])
                        count += 1

        La = spatial.regulariser_grid(self.alpha, **self.lam_warp, voxel_size=vx)
        aLa = self.alpha.flatten().dot(La.flatten())
        g += La

        delta = spatial.solve_grid_fmg(H, g, **self.lam_warp, voxel_size=vx)
        if not self.max_ls_warp:
            self.alpha -= delta
            self._lb_warp = -0.5 * aLa
            return

        # line search
        dLd = spatial.regulariser_grid(delta, **self.lam_warp, voxel_size=vx)
        dLd = delta.flatten().dot(dLd.flatten())
        dLa = delta.flatten().dot(La.flatten())
        armijo, prev_armijo = 1, 0
        M = M0.log()
        ll0 = M[0].sum() + (Z * (M[1:] - M[:1])).sum()
        success = False
        for n_ls in range(self.max_ls_warp):
            self.alpha.sub_(delta, alpha=armijo - prev_armijo)
            M = self._modulate_prior_(self.warp_tpm(aff=aff)).log_()
            ll = M[0].sum() + (Z * (M[1:] - M[:1])).sum()
            if ll + armijo * (dLa - 0.5 * armijo * dLd) > ll0:
                success = True
                break
            prev_armijo, armijo = armijo, armijo/2
        if not success:
            self.alpha.add_(delta, alpha=armijo)
        else:
            self._lb_warp = (getattr(self, '_lb_warp', 0) +
                             armijo * (dLa - 0.5 * armijo * dLd))

    def _update_intensity(self, *a, **k):
        pass

    def _update_bias(self, *a, **k):
        pass

    def warp_tpm(self, aff=None, grad=False, mode='softmax'):
        """

        Parameters
        ----------
        aff : (D+1, D+1) tensor, optional
        grad : bool, default=False
        mode : {'softmax', 'log_softmax', 'logit', 'mask', 'mask8'}, default='softmax'

        Returns
        -------
        mask : (1, *shape) tensor, if mode in ('mask', 'mask8')
        logit or prob or logprob : (K, *shape) if mode not in ('mask', 'mask8')
        grad : (K, *shape, D) tensor, if grad is True

        """
        if self.prior is None:
            return None
        dim = self.alpha.shape[-1]
        grid = spatial.add_identity_grid(self.alpha)
        if aff is not None:
            aff = torch.matmul(self.prior_aff.inverse(), aff)
            grid = spatial.affine_matvec(aff, grid)
        mask = self.prior.new_ones(self.prior.shape[1:])
        if mode == 'mask8':
            # remove bottom 5 mm
            if self.prior_aff is not None:
                vx = spatial.voxel_size(self.prior_aff)
            else:
                vx = grid.new_ones([grid.dim[-1]])
            offset = (5 / vx).floor().int().tolist()
            offset = [slice(o, None) for o in offset]
            mask[(Ellipsis, *offset)] = 0
        mask = spatial.grid_pull(mask, grid, bound='dct2', extrapolate=False)
        if mode.startswith('mask'):
            return mask[None]

        M = spatial.grid_pull(self.prior, grid, bound='dct2', extrapolate=False)
        M[:, mask == 0] = M.min()  # uncomment for informative out-of-FOV
        if mode == 'softmax':
            M = math.softmax(M, dim=0, implicit=(True, False))
        elif mode == 'log_softmax':
            M = math.log_softmax(M, dim=0, implicit=(True, False))
        if not grad:
            return M
        G = spatial.grid_grad(self.prior, grid, bound='dct2', extrapolate=False)
        aff = aff.to(**utils.backend(G))
        G = linalg.matvec(aff[:dim, :dim].T, G)  # rotate spatial gradients
        return M, G


class UniSeg(SpatialMixture):
    """
    Unified Segmentation model, based on multivariate Gaussian Mixture Model (GMM).
    """

    def __init__(self, *args, wishart=True, **kwargs):
        # wishart : bool or 'preproc8', default=True
        #   If True, use a Wishart prior derived from global suffstats.
        #   If 'preproc8', the bottom of the template FOV is discarded
        #   when estimating these suffstats.
        super().__init__(*args, **kwargs)
        self.wishart = wishart

    def get_means_variances(self):
        return self.mu, self.sigma

    def _lb_parameters(self):
        return super()._lb_parameters() + getattr(self, '_lb_intensity', 0)

    def _log_likelihood(self, X, cluster=0):
        """ Log-probability density function (pdf) of the standard normal
            distribution, evaluated at the values in X.

        Parameters
        ----------
        X : (C, *spatial) tensor
            Observed data modulated by the current bias field
        cluster : int, default=0
            Index of mixture component.

        Returns
        -------
        log_pdf : (*spatial) tensor

        """
        mu = self.mu[cluster]
        sigma = self.sigma[cluster]

        C = X.shape[0]

        chol = linalg.cholesky(sigma)
        log_det = chol.diag().log().sum().mul(2).to(**utils.backend(X))
        chol = chol.inverse()
        chol.to(**utils.backend(X))
        mu.to(**utils.backend(X))

        log_pdf = (X.reshape(C, -1).T - mu)
        log_pdf = linalg.matvec(chol, log_pdf).square_()
        log_pdf += log_det
        log_pdf *= -0.5
        log_pdf += self.kappa[cluster].log()
        log_pdf = log_pdf.T.reshape(X.shape[1:])

        # NOTE: I do not include `C * log(2 * pi)`, since it is common
        #   to all classes and does not have an impact on the softmax or
        #   any of the lower-bound components.

        return log_pdf

    def _init_wishart(self, X, W=None, aff=None):
        """Estimate a diagonal wishart prior from sufficient statistics"""
        if not self.wishart:
            self.wishart = None
            return

        # 1) Compute mask of voxels to include
        # - If a TPM exists, we only use voxels that fall in its FOV
        # - If, furthermore, self.wishart == 'preproc8', we exclude the
        #   neck (assumed to be the bottom 5 mm of the FOV)
        # - Otherwise, we just use the weight map
        mode = 'mask8' if self.wishart == 'preproc8' else 'mask'
        if self.prior is not None:
            M = self.warp_tpm(mode=mode, aff=aff)
            if W is not None:
                M *= W
        elif W is not None:
            M = W
        else:
            M = X.new_ones([1]*X.dim())
        while M.dim() < X.dim():
            M = M[None]

        # 2) Estimate the (diagonal) variance of the data, assuming
        #    a single Gaussian.
        ss0, ss1, ss2 = self._suffstats(X, W)
        ss0 = ss0[0]
        ss1 = ss1[0]
        ss2 = ss2[0].diag()
        scale = ss2 / ss0 - (ss1/ss0).square()

        # 3) Fiddle with degrees of freedom
        scale /= max(self.nb_clusters_per_class) ** 2
        df = len(X)
        self.wishart = (scale.diag(), df)

    def _init_parameters(self, X, W=None, aff=None, **kwargs):
        """Initialise all parameters"""
        super()._init_parameters(X, W, aff, **kwargs)  # alpha, beta, gamma
        self._init_wishart(X, W, aff)

        K = self.nb_clusters
        C = X.shape[0]
        dim = X.dim() - 1
        backend = dict(device=X.device, dtype=torch.double)

        if self.prior is not None:
            # init from suffstats
            self.mu = torch.empty((K, C), **backend)
            self.sigma = torch.empty((K, C, C), **backend)
            M = self.warp_tpm(aff=aff)
            if W is not None:
                M *= W
            ss0, ss1, ss2 = self._suffstats(X, M)
            self._update_intensity(ss0, ss1, ss2)

            # if user-defined mu/sigma, use them instead
            mu = kwargs.get('mu', None)
            if mu is not None:
                self.mu = torch.as_tensor(mu, **backend).expand(K, C).clone()
            sigma = kwargs.get('sigma', None)
            if sigma is not None:
                self.sigma = torch.as_tensor(sigma, **backend).expand(K, C, C).clone()

        else:
            reduced_dim = range(-dim, 0)
            mn, mx = utils.quantile(X, [0.01, 0.99], dim=reduced_dim,
                                    mask=W, bins=256).double().unbind(-1)

            mu = kwargs.pop('mu', None)
            if mu is not None:
                self.mu = torch.as_tensor(mu, **backend).expand(K, C).clone()
            else:
                self.mu = torch.empty((K, C), **backend)
                for c in range(C):
                    self.mu[:, c] = torch.linspace(mn[c], mx[c], K, **backend)

            sigma = kwargs.pop('sigma', None)
            if sigma is not None:
                self.sigma = torch.as_tensor(sigma, **backend).expand(K, C, C).clone()
            else:
                self.sigma = torch.empty((K, C, C), **backend)
                for c in range(C):
                    self.sigma[:, c, c] = (mx[c] - mn[c]) / K**2

    def _update_intensity(self, ss0, ss1, ss2):
        """ Update GMM means and variances

        Parameters
        ----------
        ss0 : (K,) tensor)
            0th moment
        ss1 : (K, C) tensor
            1st moment
        ss2 : (K, C, C) tensor
            2nd moment

        """
        if self.verbose >= 3:
            print('update gmm')

        # update means
        ss0 = ss0.unsqueeze(-1)
        mu = ss1 / ss0

        # update covariances
        ss0 = ss0.unsqueeze(-1)
        if not self.wishart:
            sigma = ss2 / ss0 - linalg.outer(mu, mu)
        else:
            scale, df = self.wishart
            sigma = df * scale + ss2 - linalg.outer(ss1, ss1) / ss0
            sigma /= (ss0 + df)

        if len(mu) < len(self.mu):
            # Heuristic to split a single Gaussians into multiple Gaussians.
            # We do this at initialization time, whereas in preproc8, a
            # full round of GMM+Bias is performed with one Gaussian per class,
            # before splitting happens.
            mu0 = mu
            sigma0 = sigma
            ss00 = ss0
            mu = torch.empty_like(self.mu)
            sigma = torch.empty_like(self.sigma)
            ss0 = ss00.new_empty([self.nb_clusters, 1, 1])
            for k in range(self.nb_classes):
                mask = [k1 == k for k1 in self.lkp]
                K1 = sum(mask)
                w = 1. / (1 + pymath.exp(-(K1 - 1) * 0.25)) - 0.5
                chol = linalg.cholesky(sigma0[k]).diag()
                noise = torch.randn(K1, **utils.backend(mu)) * w
                mu[mask] = chol * noise[:, None] + mu0[k]
                sigma[mask] = sigma0[k] * (1 - w)
                ss0[mask] = ss00[k] / K1

        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

        if self.wishart is not None:
            # update lower bound: KL between inverse wishart,
            # keeping only terms that depend on sigma
            chol = linalg.cholesky(self.sigma)
            logdet = chol.diagonal(0, -1, -2).log().sum(-1)
            tr = linalg.trace(torch.matmul(scale, self.sigma.inverse()))
            # tr = linalg.trace(torch.cholesky_solve(scale, chol))
            ss0 = ss0[..., 0, 0]
            lb = tr * (df + ss0 - sigma.shape[-1] - 1) / (df + ss0) \
               - logdet * (df - sigma.shape[-1] - 1)
            lb = lb.sum()
            self._lb_intensity = -0.5*lb
        else:
            self._lb_intensity = 0

    def _update_bias(self, X, Z, W=None, vx=None):
        """
        Update bias field for all channels and images in batch, in-place.

        Parameters
        ----------
        X : (C, *spatial) tensor
            Observed image modulated by the current bias field
        Z : (K, *spatial) tensor
            (Unweighted) responsibilities
        W : (*spatial) tensor, optional
            Observation weights
        vx : (D,) vector_like
            Voxel size

        """
        if self.verbose >= 3:
            print('update bias')

        C, *N = X.shape
        K = self.nb_clusters

        X = X.reshape([C, -1]).T
        Z = Z.reshape([K, -1])
        if W is not None:
            W = W.reshape([-1])

        lb = 0
        for c in range(C):
            # Solve for each channel individually
            g = X.new_zeros(X.shape[0])
            H = X.new_zeros(X.shape[0])
            for k in range(K):
                sigma = self.sigma[k]
                mu = self.mu[k].to(**utils.backend(X))
                # Determine each class' contribution to gradient and Hessian
                lam = sigma.inverse()[c].to(**utils.backend(X))
                g.addcmul_(Z[k], linalg.dot(X - mu, lam))
                H.addcmul_(Z[k], lam[c])
            g.mul_(X[:, c]).sub_(1)
            H.mul_(X[:, c].square()).add_(1)
            if self.bias_acceleration < 1:
                a = 1 - self.bias_acceleration
                H.add_(g.abs(), alpha=a)  # new robust Hessian

            if W is not None:
                g *= W
                H *= W
            g = g.reshape([1, *N])
            H = H.reshape([1, *N])

            Lb = spatial.regulariser(self.beta[None, c],
                                     bending=self.lam_bias,
                                     voxel_size=vx)
            lb += self.beta[c].flatten().dot(Lb.flatten())
            g += Lb

            delta = spatial.solve_field_fmg(H, g,
                                            bending=self.lam_bias,
                                            voxel_size=vx,
                                            verbose=self.verbose_bias)
            self.beta[c] -= delta[0]

        self.beta -= self.beta.mean()
        self._lb_bias = -0.5*lb + self.beta.sum()

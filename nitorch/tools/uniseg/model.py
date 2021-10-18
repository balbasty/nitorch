import math
from nitorch.core.utils import isin
from timeit import default_timer as timer
from nitorch.core.optim import get_gain, plot_convergence
from nitorch.core.math import besseli, softmax_lse
from nitorch.plot import plot_mixture_fit
from nitorch.spatial import regulariser, solve_field_fmg
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.functional import one_hot


torch.backends.cudnn.benchmark = True


class Mixture:
    # A mixture model.
    def __init__(self, num_class=2, mp=None):
        """
        num_class (int or sequence[int], optional): Number of mixture components. Defaults to 2.
            sequence[int] used for case of multiple gaussians per class, e.g. [5,3,3,2,2,1]
        mp (torch.tensor): GMM mixing proportions.
        lam (torch.tensor): Regularisation.

        """
        self.K = num_class
        if mp is not None:
            self.mp = mp
            self.prior = True
        else:
            self.mp = []
            self.prior = False
        self.lam = []
        self.dev = ''  # PyTorch device
        self.dt = ''  # PyTorch data type

    # Functions
    def fit(self, X, verbose=1, max_iter=10000, tol=1e-8, fig_num=1, W=None,
            show_fit=False, bias=True, verbose_bias=0, penalty_bias=1e16, class_weight=True, delay=10):
        """ Fit mixture model.

        Args:
            X (torch.tensor): Observed data (B, C, [Spatial]).
                B = num batch size of independent patients to process
                C = num channels
                [Spatial] = spatial dimensions. Can be [x], [x,y] or [x,y,z]
            verbose (int, optional) Display progress. Defaults to 1.
                0: None.
                1: Print summary when finished.
                2: 1 + print convergence.
                3: 1 + 2 + Log-likelihood plot.
            max_iter (int, optional) Maxmimum number of algorithm iterations.
                Defaults to 10000.
            tol (int, optional): Convergence threshold. Defaults to 1e-8.
            fig_num (int, optional): Defaults to 1.
            W (torch.tensor, optional): Observation weights (N, 1). Defaults to no weights.
            show_fit (bool, optional): Plot mixture fit, defaults to False.

        Returns:
            Z (torch.tensor): Responsibilities (B, C, [Spatial]).

        """
        if verbose:
            t0 = timer()  # Start timer
        self.verbose_bias = verbose_bias

        # Set random seed
        torch.manual_seed(1)

        if X.dtype not in [torch.float16, torch.float32, torch.float64]:
            print('Data type {} not supported - converting to single-precision float.'.format(X.dtype))
            X = X.float()

        self.dev = X.device
        self.dt = X.dtype

        if len(X.shape) == 1:
            X = X[:, None]

        B = X.shape[0]  # Number of observations
        C = X.shape[1]  # Number of channels
        N = X.shape[:2]
        K = self.K  # Number of components
        if isinstance(K, list):
            K = len(K)

        if W is not None:  # Observation weights given
            W = torch.reshape(W, N)

        # Initialise model parameters
        self._init_par(X)

        self.class_weight = class_weight

        # Compute a regularisation value
        self.lam = torch.zeros((B, C), dtype=self.dt, device=self.dev)
        for b in range(B):
            for c in range(C):
                if W is not None:
                    self.lam[b, c] = (torch.sum(X[b, c] * W.flatten()) / (torch.sum(W) * K)) ** 2
                else:
                    self.lam[b, c] = (torch.sum(X[b, c]) / K) ** 2

        # EM loop
        Z, lb = self._em(X, max_iter=max_iter, tol=tol, verbose=verbose, W=W, bias=bias, penalty_bias=penalty_bias, delay=delay)

        # Print algorithm info
        if verbose >= 1:
            print('Algorithm finished in {} iterations, '
                  'log-likelihood = {}, '
                  'runtime: {:0.1f} s, '
                  'device: {}'.format(len(lb[-1]), lb[-1,-1], timer() - t0, self.dev))
        if verbose >= 3:
            _ = plot_convergence(lb, xlab='Iteration number',
                                 fig_title='Model lower bound', fig_num=fig_num)
        # Plot mixture fit
        if show_fit:
            self._plot_fit(X, W, fig_num=fig_num + 1)

        return Z
    
    def _em(self, X, max_iter, tol, verbose, W, bias=True, warp=True, penalty_bias=1e16, delay=10):
        """ EM loop for fitting GMM.

        Args:
            X (torch.tensor): (B, C, [Spatial]).
            max_iter (int)
            tol (int)
            verbose (int)
            W (torch.tensor): ([Spatial]).

        Returns:
            Z (torch.tensor): Responsibilities (B, K, [Spatial]).
            lb (list): Lower bound at each iteration.

        """

        # Init
        B = X.shape[0]
        C = X.shape[1]
        N = X.shape[2:]
        K = self.K
        dtype = self.dt
        device = self.dev
        if len(X.shape)-1 == len(self.mp.shape) or len(self.mp.shape) == 1:
            self.mp = torch.stack(B*[self.mp], dim=0)
            # , dtype=dtype, device=device)
        tiny = torch.tensor(1e-32, dtype=dtype, device=device)
        tinyish = torch.tensor(1e-3, dtype=dtype, device=device)

        # Start EM algorithm
        if isinstance(K, list):
            Z = torch.zeros((B, len(K), *N), dtype=dtype, device=device)  # responsibility
        else:
            Z = torch.zeros((B, K, *N), dtype=dtype, device=device)  # responsibility
        lb = torch.zeros((B, max_iter), dtype=torch.float64, device=device)
        for b in range(B):
            for n_iter in range(max_iter):  # EM loop
                if verbose >= 2:
                    print('n_iter: {}'.format(n_iter+1))
                # ==========
                # Segmentation
                # ==========
                #
                # ==========
                # E-step
                # ==========
                # Product Rule
                if isinstance(K, list):
                    for k in K:
                        for j in k:
                            Z[b, k] += self._log_likelihood(X, j, b=b)
                        Z[b, k] += torch.log(self.gamma[b,k] * self.mp[b, k] + tinyish)
                else:
                    for k in range(K):
                        Z[b, k] = torch.log(self.gamma[b,k] * self.mp[b,k] + tinyish) + self._log_likelihood(X, k, b=b)

                # Get responsibilities
                Z[b], dlb = softmax_lse(Z[b]+tinyish, lse=True, weights=W, dim=0)

                # Objective function and convergence related
                lb[b, n_iter] = dlb.float()
                gain = get_gain(lb[b, :n_iter + 1])
                if verbose >= 2:
                    print('Segmentation - lb: {}, gain: {}'
                        .format(lb[b, n_iter], gain))
                if gain < tol:
                    last_iter = n_iter
                    break  # Finished
                elif n_iter+1==max_iter:
                    last_iter = max_iter

                if W is not None:  # Weight responsibilities
                    Z[b] = Z[b] * W

                # ==========
                # M-step
                # ==========
                # Compute sufficient statistics
                ss0, ss1, ss2 = self._suffstats(X, Z)

                # Update class weighting
                if self.class_weight:
                    self.gamma = (Z.reshape(B, K, -1)/((self.mp+tinyish).reshape(1, 6, -1)/(self.gamma.reshape(1, 6, 1)*(self.mp+tinyish).reshape(1, 6, -1)).sum(1, keepdim=True))).sum(-1)
                    for b in range(B):
                        self.gamma[b] /= self.gamma[b].sum()

                # Update prior
                if not self.prior:
                    if W is not None:
                        self.mp = ss0 / torch.sum(W, dim=0, dtype=torch.float64)
                    else:
                        self.mp = ss0 / N.numel()
                    
                # Update model specific parameters
                self._update(ss0, ss1, ss2)

                if bias and n_iter+1>delay:
                    # ==========
                    # Bias-field
                    # ==========
                    #
                    # ==========
                    # E-step
                    # ==========
                    # Product Rule
                    if isinstance(K, list):
                        for k in K:
                            for j in k:
                                Z[b, k] += self._log_likelihood(X, j, b=b)
                            Z[b, k] += torch.log(self.gamma[b,k] * self.mp[b, k] + tinyish)
                    else:
                        for k in range(K):
                            Z[b, k] = torch.log(self.gamma[b,k] * self.mp[b,k] + tinyish) + self._log_likelihood(X, k, b=b)

                    # Get responsibilities
                    Z[b], dlb = softmax_lse(Z[b]+tinyish, lse=True, weights=W, dim=0)

                    if W is not None:  # Weight responsibilities
                        Z[b] = Z[b] * W

                    # ==========
                    # M-step
                    # ==========
                    # Update bias field
                    self._update_bias(X, Z, penalty=penalty_bias)

        return Z, lb[:, :last_iter + 1]
    
    def _init_mp(self, dtype=torch.float64):
        """ Initialise mixing proportions: mp

        """
        # Mixing proportions
        if self.mp == []:
            self.mp = torch.ones(self.K, dtype=dtype, device=self.dev)/self.K

    def _suffstats(self, X, Z):
        """ Compute sufficient statistics.

        Args:
            X (torch.tensor): Observed data (B, C, [Spatial]).
            Z (torch.tensor): Responsibilities (B, K, [Spatial]).

        Returns:
            ss0 (torch.tensor): 0th moment (B, K).
            ss1 (torch.tensor): 1st moment (B, C, K).
            ss2 (torch.tensor): 2nd moment (B, C, C, K).

        """
        N = X.shape[2:].numel()
        B = X.shape[0]
        C = X.shape[1]
        K = Z.shape[1]
        device = self.dev
        tiny = torch.tensor(1e-32, dtype=torch.float64, device=device)

        # Suffstats
        ss1 = torch.zeros((B, C, K), dtype=torch.float64, device=device)
        ss2 = torch.zeros((B, C, C, K), dtype=torch.float64, device=device)

        # Compute 0th moment
        ss0 = torch.sum(Z.reshape(B, K, -1), dim=-1, dtype=torch.float64) + tiny

        # Compute 1st and 2nd moments
        for b in range(B):
            for k in range(K):
                # 1st
                ss1[b, :, k] = torch.sum(Z[b, k].reshape(N, 1) * (X[b]*self.beta[b].exp()).reshape(-1, C),
                                    dim=0, dtype=torch.float64)

                # 2nd
                for c1 in range(C):
                    ss2[b, c1, c1, k] = \
                        torch.sum(Z[b, k] * (X[b, c1]*self.beta[b, c1].exp()) ** 2, dtype=torch.float64)
                    for c2 in range(c1 + 1, C):
                        ss2[b, c1, c2, k] = \
                            torch.sum(Z[b, k] * ((X[b, c1]*self.beta[b, c1].exp()) * (X[b, c2]*self.beta[b, c2].exp())),
                                    dtype=torch.float64)
                        ss2[b, c2, c1, k] = ss2[b, c1, c2, k]

        return ss0, ss1, ss2

    def _plot_fit(self, X, W, fig_num):
        """ Plot mixture fit.

        """
        mp = self.mp
        mu, var = self.get_means_variances()
        log_pdf = lambda x, k, c: self._log_likelihood(x, k, c)
        plot_mixture_fit(X, log_pdf, mu, var, mp, fig_num, W)
    
    # Implement in child classes
    def get_means_variances(self): pass

    def _log_likelihood(self): pass

    def _init_par(self): pass

    def _update(self): pass

    def _update_bias(self): pass

    def _update_warp(self): pass

    # Static methods
    @staticmethod
    def apply_mask(X):
        """ Mask tensor, removing zeros and non-finite values.

        Args:
            X (torch.tensor): Observed data (N0, C).

        Returns:
            X_msk (torch.tensor): Observed data (N, C), where N < N0.
            msk (torch.tensor): Logical mask (N, 1).

        """
        dtype = X.dtype
        device = X.device
        C = X.shape[1]
        msk = (X != 0) & (torch.isfinite(X))
        msk = torch.sum(msk, dim=1) == C

        N = torch.sum(msk != 0)
        X_msk = torch.zeros((N, C), dtype=dtype, device=device)
        for c in range(C):
            X_msk[:, c] = X[msk, c]

        return X_msk, msk

    @staticmethod
    def full_resp(Z, msk, dm=[]):
        """ Converts masked responsibilities to full.

        Args:
            Z (torch.tensor): Masked responsibilities (N, K).
            msk (torch.tensor): Mask of original data (N0, 1).
            dm (torch.Size, optional): Reshapes Z_full using dm. Defaults to [].

        Returns:
            Z_full (torch.tensor): Full responsibilities (N0, K).

        """
        N0 = len(msk)
        K = Z.shape[1]
        Z_full = torch.zeros((N0, K), dtype=Z.dtype, device=Z.device)
        for k in range(K):
            Z_full[msk, k] = Z[:, k]
        if len(dm) >= 3:
            Z_full = torch.reshape(Z_full, (dm[0], dm[1], dm[2], K))

        return Z_full

    @staticmethod
    def maximum_likelihood(Z):
        """ Return maximum likelihood map.

        Args:
            Z (torch.tensor): Responsibilities (B, K, [Spatial]).

        Returns:
            (torch.tensor): Maximum likelihood map (B, 1, [Spatial]).

        """
        return torch.argmax(Z, dim=1)


class UniSeg(Mixture):
    # Unified Segmentation model, based on multivariate Gaussian Mixture Model (GMM).
    def __init__(self, num_class=2, prior=None, mu=None, Cov=None, alpha=None, beta=None, gamma=None):
        """
        mu (torch.tensor): GMM means (C, K).
        Cov (torch.tensor): GMM covariances (C, C, K).

        """
        super(UniSeg, self).__init__(num_class=num_class, mp=prior)
        self.mu = mu
        self.Cov = Cov
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_means_variances(self):
        """
        Return means and variances.

        Returns:
            (torch.tensor): Means (C, K).
            (torch.tensor): Covariances (C, C, K).

        """
        return self.mu, self.Cov

    def _log_likelihood(self, X, k=0, c=None, b=None):
        """ Log-probability density function (pdf) of the standard normal
            distribution, evaluated at the values in X.

        Args:
            X (torch.tensor): Observed data (B, C, Spatial).
            k (int, optional): Index of mixture component. Defaults to 0.
            c (int, optional): Index of channel. Defaults to None.
            b (int, optional): Index of batch. Defaults to None.

        Returns:
            log_pdf (torch.tensor): (N, 1).

        """
        B = X.shape[0]
        C = X.shape[1]
        device = X.device
        dtype = X.dtype
        pi = torch.tensor(math.pi, dtype=dtype, device=device)
        if b is not None:
            if c is not None:
                beta = self.beta[b,c]
                Cov = self.Cov[b, c, c, k].reshape(1, 1).cpu()
                mu = self.mu[b, c, k].reshape(1).cpu()
            else:
                beta = self.beta[b]
                Cov = self.Cov[b, :, :, k].reshape(C, C) # shape (C, C)
                mu = self.mu[b, :, k].reshape(C) # shape C
            if C == 1:
                chol_Cov = torch.sqrt(Cov)
                log_det_Cov = torch.log(chol_Cov[0, 0])
            else:
                chol_Cov = torch.cholesky(Cov)
                log_det_Cov = torch.sum(torch.log(torch.diag(chol_Cov)))
                chol_Cov = chol_Cov.inverse()
            chol_Cov = chol_Cov.type(dtype)
            mu = mu.type(dtype)
            if C == 1:
                diff = ((X[b]*self.beta[b].exp()).reshape(C, -1) - mu.reshape(C, 1))/chol_Cov
            else:
                diff = torch.tensordot((X[b]*self.beta[b].exp()).reshape(C, -1) - mu.reshape(C, 1), chol_Cov, dims=([0],[1])).permute(1,0)
            log_pdf = (- (C / 2) * torch.log(2 * pi) - log_det_Cov.unsqueeze(dim=-1) - 0.5 * torch.sum(diff**2, dim=0, keepdim=True)).reshape((-1,)+tuple(X.shape[2:]))
            log_pdf += beta
        else:
            if c is not None:
                beta = self.beta[:,c]
                Cov = torch.ones((B, 1, 1), device=device, dtype=dtype)
                mu = torch.ones((B, 1), device=device, dtype=dtype)
                chol_Cov = torch.ones((B, 1, 1), device=device, dtype=dtype)
                log_det_Cov = torch.ones((B, 1), device=device, dtype=dtype)
                diff = torch.ones((B, 1,)+(X.shape[2:].numel(),), device=device, dtype=dtype)
            else:
                beta = self.beta
                Cov = torch.ones((B, C, C), device=device, dtype=dtype)
                mu = torch.ones((B, C), device=device, dtype=dtype)
                chol_Cov = torch.ones((B, C, C), device=device, dtype=dtype)
                log_det_Cov = torch.ones((B, C), device=device, dtype=dtype)
                diff = torch.ones((B, C,)+(X.shape[2:].numel(),), device=device, dtype=dtype)
            for b in range(B):
                if c is not None:
                    Cov[b] = self.Cov[b, c, c, k].reshape(1, 1).cpu()
                    mu[b] = self.mu[b, c, k].reshape(1).cpu()
                else:
                    Cov[b] = self.Cov[b, :, :, k].reshape(C, C) # shape (C, C)
                    mu[b] = self.mu[b, :, k].reshape(C) # shape C
                if C == 1:
                    chol_Cov[b] = torch.sqrt(Cov[b])
                    log_det_Cov[b] = torch.log(chol_Cov[b, 0, 0])
                else:
                    chol_Cov[b] = torch.cholesky(Cov[b])
                    log_det_Cov[b] = torch.sum(torch.log(torch.diag(chol_Cov[b])))
                    chol_Cov[b] = chol_Cov[b].inverse()
            chol_Cov = chol_Cov.type(dtype)
            mu = mu.type(dtype)
            if C == 1:
                diff = ((X*self.beta.exp()).reshape(B, C, -1) - mu.reshape(B, C, 1))/chol_Cov
            else:
                for b in range(B):
                    diff[b] = torch.tensordot((X[b]*self.beta[b].exp()).reshape(C, -1) - mu[b].reshape(C, 1), chol_Cov[b], dims=([0],[1])).permute(1,0)
            log_pdf = (- (C / 2) * torch.log(2 * pi) - log_det_Cov.unsqueeze(dim=-1) - 0.5 * torch.sum(diff**2, dim=1, keepdim=True)).reshape((B, -1,)+tuple(X.shape[2:]))
            log_pdf += beta
        return log_pdf

    def _init_par(self, X):
        """ Initialise GMM specific parameters: mu, Cov

        """
        dtype = torch.float64
        if isinstance(self.K, (list, tuple)):
            K = len(self.K)
        else:
            K = self.K
        B = X.shape[0]
        C = X.shape[1]
        mn = torch.min(X.reshape(B,C,-1), dim=2)[0]
        mx = torch.max(X.reshape(B,C,-1), dim=2)[0]

        # Init mixing prop
        self._init_mp(dtype)

        if self.gamma is None:
            self.gamma = torch.ones((B,K), dtype=dtype, device=self.dev) / K
    
        if self.mu is None:
            # means
            self.mu = torch.zeros((B, C, K), dtype=dtype, device=self.dev)
        if self.Cov is None:
            # covariance
            self.Cov = torch.zeros((B, C, C, K), dtype=dtype, device=self.dev)
            for b in range(B):
                for c in range(C):
                    # rng = torch.linspace(start=mn[c], end=mx[c], steps=K, dtype=dtype, device=self.dev)
                    # num_neg = sum(rng < 0)
                    # num_pos = sum(rng > 0)
                    # rng = torch.arange(-num_neg, num_pos, dtype=dtype, device=self.dev)
                    # self.mu[c, :] = torch.reshape((rng * (mx[c] - mn[c]))/(K + 1), (1, K))
                    self.mu[b, c, :] = torch.reshape(torch.linspace(mn[b, c], mx[b, c], K, dtype=dtype, device=self.dev), (1, K))
                    self.Cov[b, c, c, :] = \
                        torch.reshape(torch.ones(K, dtype=dtype, device=self.dev)
                                    * ((mx[b, c] - mn[b, c])/(K))**2, (1, 1, 1, K))
        if self.alpha is None:
            # spatial deformation coefficients
            self.alpha = torch.zeros_like(X, dtype=dtype, device=self.dev)
        if self.beta is None:
            # bias field coefficients
            self.beta = torch.zeros_like(X, dtype=dtype, device=self.dev)

    def _update(self, ss0, ss1, ss2):
        """ Update GMM means and variances

        Args:
            ss0 (torch.tensor): 0th moment (B, K).
            ss1 (torch.tensor): 1st moment (B, C, K).
            ss2 (torch.tensor): 2nd moment (B, C, C, K).

        """
        B = ss1.shape[0]
        C = ss1.shape[1]
        K = ss1.shape[2]

        # Update means and covariances
        for b in range(B):
            for k in range(K):
                # Update mean
                self.mu[b, :, k] = 1/ss0[b, k] * ss1[b, :, k]

                # Update covariance
                self.Cov[b, :, :, k] = ss2[b, :, :, k] / ss0[b, k] \
                    - torch.ger(self.mu[b, :, k], self.mu[b, :, k])

    def _update_bias(self, X, Z, penalty=1e16, vx=1):
        """
        Update bias field for all channels and images in batch, in-place.

        Args:
            X (torch.tensor): Image (B, C, Spatial).
            Z (torch.tensor): Responsibilites (B, C, Spatial).
            penalty (float, default=1e16): Penalty on bending energy.
            vx (int or float, default=1): Voxel size.
        
        """
        B = X.shape[0]
        C = X.shape[1]
        if isinstance(self.K, list):
            K = len(self.K)
        else:
            K = self.K
        beta = self.beta
        bX = X * beta.exp()
        for b in range(B):
            # Extract single item from batch
            bX_ = bX[b].reshape(-1)
            Z_ = Z[b].reshape(K, -1)
            mu = self.mu[b]
            Cov = self.Cov[b]
            for c in range(C):
                # Solve for each channel individually
                g = 0
                H = 0
                for k in range(K):
                    # Determine each class' contribution to gradient and Hessian
                    lam = Cov[...,k].inverse()[c]
                    g += Z_[k] * ((bX_ - mu[...,k]) * lam).sum(dim=-1)
                    H += Z_[k] * bX_[c].square() * lam[c]

                g *= bX_[c]
                g -= 1
                H += 1 + g.abs()

                g = g.reshape((1,)+tuple(X.shape[2:]))
                H = H.reshape((1,)+tuple(X.shape[2:]))

                g += regulariser(beta[b, c].reshape((1,)+tuple(X.shape[2:])),
                                bending=penalty, voxel_size=vx)             

                if self.verbose_bias > 0:
                    print('Bias correction, L1 norm values: Gradient = {:.4g}, Hessian = {:.4g}'.format(g.abs().sum(), H.abs().sum()))

                beta[b, c] -= solve_field_fmg(H, g, bending=penalty, voxel_size=vx, verbose=self.verbose_bias).reshape(X.shape[2:])

        beta -= beta.mean()
        # Update bias field in-place
        self.beta = beta

    def sample(self, mu=None, Cov=None, alpha=False, beta=False, seg=None, label_sampling='cat'):
        """ Sample new image with GMM parameters and (eventually) bias and warp parameters
        
        Args:
            mu (torch.tensor or sequence[torch.tensor]): means (B, C, K)
            Cov (torch.tensor or sequence[torch.tensor]): covariance (B, C, C, K)
            alpha (torch.tensor): warping parameters (B, C, Spatial)
            beta (torch.tensor): bias-field parameters (B, C, Spatial)
            seg (torch.tensor): tissue class maps to use for image generation (B, C, Spatial)
            label_sampling (string): method to sample tissue classes:
                * 'cat' (default): sample from categorical distribution with segmentation labels as prior
                * 'pv': mean intensity across all classes, weighted by segmentation labels
        
        """
        if mu == None:
            mu = self.mu
        if Cov == None:
            Cov = self.Cov
        if seg == None:
            seg = self.mp
        B = mu.shape[0]
        C = mu.shape[1]
        K = mu.shape[2]
        dim = len(seg.shape[2:])
        device = seg.device
        dtype = seg.dtype
        X = torch.zeros((B, C,)+tuple(seg.shape[2:]), device=device, dtype=dtype)
        if isinstance(self.K, list):
            for b in range(B):
                sample = []
                ix = 0
                for k_ in range(len(self.K)):
                    K_ = self.K[k_]
                    pdf_ = [MultivariateNormal(mu[b,...,ix:ix+k], Cov[b,...,ix:ix+k]) for k in range(K_)]
                    sample.append(torch.stack([dist.rsample(seg.shape[2:]) for dist in pdf_]).sum(dim=-1)) # expect shape (C, Spatial)
                    ix += K_
                sample = torch.stack(sample) # expect shape (K, C, Spatial)
                if dim==3:
                    sample = sample.permute(1,0,2,3,4)
                elif dim==2:
                    sample = sample.permute(1,0,2,3)
                if label_sampling=='pv':
                    X[b] = (seg[b].unsqueeze(dim=0) * sample).sum(dim=1) # multiply (1, K, Spatial) * (C, K, Spatial) and sum over K
                elif label_sampling=='cat':
                    if dim==3:
                        seg_ = seg[b].permute(1,2,3,0)
                    elif dim==2:
                        seg_ = seg[b].permute(1,2,0)
                    seg_ = one_hot(Categorical(seg_).sample()) # (Spatial, K)
                    if dim==3:
                        seg_ = seg_.permute(3,0,1,2)
                    elif dim==2:
                        seg_ = seg_.permute(2,0,1)
                    X[b] = (seg_.unsqueeze(dim=0) * sample).sum(dim=1) # multiply (1, K, Spatial) * (C, K, Spatial) and sum over K
        else:
            for b in range(B):
                pdf = [MultivariateNormal(mu[b,...,k], Cov[b,...,k]) for k in range(K)]
                sample = torch.stack([dist.rsample(seg.shape[2:]) for dist in pdf]) # expect shape (K, Spatial, C)
                if dim==3:
                    sample = sample.permute(4,0,1,2,3)
                elif dim==2:
                    sample = sample.permute(3,0,1,2)
                if label_sampling=='pv':
                    X[b] = (seg[b].unsqueeze(dim=0) * sample).sum(dim=1) # multiply (1, K, Spatial) * (C, K, Spatial) and sum over K
                elif label_sampling=='cat':
                    if dim==3:
                        seg_ = seg[b].permute(1,2,3,0)
                    elif dim==2:
                        seg_ = seg[b].permute(1,2,0)
                    seg_ = one_hot(Categorical(seg_).sample()) # (Spatial, K)
                    if dim==3:
                        seg_ = seg_.permute(3,0,1,2)
                    elif dim==2:
                        seg_ = seg_.permute(2,0,1)
                    X[b] = (seg_.unsqueeze(dim=0) * sample).sum(dim=1) # multiply (1, K, Spatial) * (C, K, Spatial) and sum over K
        # if alpha:
        #     if alpha is None:
        #         alpha = self.alpha
        if isinstance(beta, (torch.Tensor, type(None))):
            if beta is None:
                beta = self.beta
            X *= beta.exp()
        return X

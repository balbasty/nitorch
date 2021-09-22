import math
from timeit import default_timer as timer
from ..core.optim import get_gain, plot_convergence
from ..core.math import besseli, softmax_lse
from ..plot import plot_mixture_fit
import torch


torch.backends.cudnn.benchmark = True


class Mixture:
    # A mixture model.
    def __init__(self, num_class=2, prior=[]):
        """
        num_class (int, optional): Number of mixture components. Defaults to 2.
        mp (torch.tensor): GMM mixing proportions.
        lam (torch.tensor): Regularisation.

        """
        self.K = num_class
        self.mp = prior
        if prior == []:
            self.prior = False
        else:
            self.prior = True
        self.lam = []
        self.dev = ''  # PyTorch device
        self.dt = ''  # PyTorch data type

    # Functions
    def fit(self, X, verbose=1, max_iter=10000, tol=1e-8, fig_num=1, W=None,
            show_fit=False):
        """ Fit mixture model.

        Args:
            X (torch.tensor): Observed data (N, C).
                N = num observations per channel
                C = num channels
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
            Z (torch.tensor): Responsibilities (N, K).

        """
        if verbose:
            t0 = timer()  # Start timer

        # Set random seed
        torch.manual_seed(1)

        self.dev = X.device
        self.dt = X.dtype

        if len(X.shape) == 1:
            X = X[:, None]

        N = X.shape[0]  # Number of observations
        C = X.shape[1]  # Number of channels
        K = self.K  # Number of components

        if W is not None:  # Observation weights given
            W = torch.reshape(W, (N, 1))

        # Initialise model parameters
        self._init_par(X)

        # Compute a regularisation value
        self.lam = torch.zeros(C, dtype=self.dt, device=self.dev)
        for c in range(C):
            if W is not None:
                self.lam[c] = (torch.sum(X[:, c] * W.flatten()) / (torch.sum(W) * K)) ** 2
            else:
                self.lam[c] = (torch.sum(X[:, c]) / K) ** 2

        # EM loop
        Z, lb = self._em(X, max_iter=max_iter, tol=tol, verbose=verbose, W=W)

        # Print algorithm info
        if verbose >= 1:
            print('Algorithm finished in {} iterations, '
                  'log-likelihood = {}, '
                  'runtime: {:0.1f} s, '
                  'device: {}'.format(len(lb), lb[-1], timer() - t0, self.dev))
        if verbose >= 3:
            _ = plot_convergence(lb, xlab='Iteration number',
                                 fig_title='Model lower bound', fig_num=fig_num)
        # Plot mixture fit
        if show_fit:
            self._plot_fit(X, W, fig_num=fig_num + 1)

        return Z
    
    def _em(self, X, max_iter, tol, verbose, W):
        """ EM loop for fitting GMM.

        Args:
            X (torch.tensor): (N, C).
            max_iter (int)
            tol (int)
            verbose (int)
            W (torch.tensor): (N, 1).

        Returns:
            Z (torch.tensor): Responsibilities (N, K).
            lb (list): Lower bound at each iteration.

        """

        # Init
        N = X.shape[0]
        C = X.shape[1]
        K = self.K
        dtype = self.dt
        device = self.dev
        tiny = torch.tensor(1e-32, dtype=dtype, device=device)
        tinyish = torch.tensor(1e-3, dtype=dtype, device=device)

        # Start EM algorithm
        Z = torch.zeros((N, K), dtype=dtype, device=device)  # responsibility
        lb = torch.zeros(max_iter, dtype=torch.float64, device=device)
        for n_iter in range(max_iter):  # EM loop
            # ==========
            # E-step
            # ==========
            # Product Rule
            for k in range(K):
                Z[:, k] = torch.log(self.mp[...,k] + tinyish) + self._log_likelihood(X, k)

            # Get responsibilities
            Z, dlb = softmax_lse(Z, lse=True, weights=W)

            # Objective function and convergence related
            lb[n_iter] = dlb
            gain = get_gain(lb[:n_iter + 1])
            if verbose >= 2:
                print('n_iter: {}, lb: {}, gain: {}'
                      .format(n_iter + 1, lb[n_iter], gain))
            if gain < tol:
                break  # Finished

            if W is not None:  # Weight responsibilities
                Z = Z * W

            # ==========
            # M-step
            # ==========
            # Compute sufficient statistics
            ss0, ss1, ss2 = self._suffstats(X, Z)

            # Update mixing proportions
            if not self.prior:
                if W is not None:
                    self.mp = ss0 / torch.sum(W, dim=0, dtype=torch.float64)
                else:
                    self.mp = ss0 / N

            # Update model specific parameters
            self._update(ss0, ss1, ss2)

        return Z, lb[:n_iter + 1]
    
    def _init_mp(self, dtype=torch.float64):
        """ Initialise mixing proportions: mp

        """
        # Mixing proportions
        if self.mp == []:
            self.mp = torch.ones(self.K, dtype=dtype, device=self.dev)/self.K

    def _suffstats(self, X, Z):
        """ Compute sufficient statistics.

        Args:
            X (torch.tensor): Observed data (N, C).
            Z (torch.tensor): Responsibilities (N, K).

        Returns:
            ss0 (torch.tensor): 0th moment (K).
            ss1 (torch.tensor): 1st moment (C, K).
            ss2 (torch.tensor): 2nd moment (C, C, K).

        """
        N = X.shape[0]
        C = X.shape[1]
        K = Z.shape[1]
        device = self.dev
        tiny = torch.tensor(1e-32, dtype=torch.float64, device=device)

        # Suffstats
        ss1 = torch.zeros((C, K), dtype=torch.float64, device=device)
        ss2 = torch.zeros((C, C, K), dtype=torch.float64, device=device)

        # Compute 0th moment
        ss0 = torch.sum(Z, dim=0, dtype=torch.float64) + tiny

        # Compute 1st and 2nd moments
        for k in range(K):
            # 1st
            ss1[:, k] = torch.sum(torch.reshape(Z[:, k], (N, 1)) * X,
                                  dim=0, dtype=torch.float64)

            # 2nd
            for c1 in range(C):
                ss2[c1, c1, k] = \
                    torch.sum(Z[:, k] * X[:, c1] ** 2, dtype=torch.float64)
                for c2 in range(c1 + 1, C):
                    ss2[c1, c2, k] = \
                        torch.sum(Z[:, k] * (X[:, c1] * X[:, c2]),
                                  dtype=torch.float64)
                    ss2[c2, c1, k] = ss2[c1, c2, k]

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

    def _init_par(self):
        pass

    def _update(self): pass

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
    def reshape_input(img):
        """ Reshape image to tensor with dimensions suitable as input to Mixture class.

        Args:
            img (torch.tensor): Input image. (X, Y, Z, C)

        Returns:
            X (torch.tensor): Observed data (N0, C).
            N0 (int): number of voxels in one channel
            C (int): number of channels.

        """
        dm = img.shape
        if len(dm) == 2:  # 2D
            dm = (dm[0], dm[1], dm[2])
        N0 = dm[0]*dm[1]*dm[2]
        C = dm[3]
        X = torch.reshape(img, (N0, C))

        return X, N0, C

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
            Z (torch.tensor): Responsibilities (N, K).

        Returns:
            (torch.tensor): Maximum likelihood map (N, 1).

        """
        return torch.argmax(Z, dim=3)


class UniSeg(Mixture):
    # Unified Segmentation model, based on multivariate Gaussian Mixture Model (GMM).
    def __init__(self, num_class=2, prior=[], mu=None, Cov=None):
        """
        mu (torch.tensor): GMM means (C, K).
        Cov (torch.tensor): GMM covariances (C, C, K).

        """
        super(GMM, self).__init__(num_class=num_class, prior=prior)
        self.mu = mu
        self.Cov = Cov

    def get_means_variances(self):
        """
        Return means and variances.

        Returns:
            (torch.tensor): Means (C, K).
            (torch.tensor): Covariances (C, C, K).

        """
        return self.mu, self.Cov

    def _log_likelihood(self, X, k=0, c=None):
        """ Log-probability density function (pdf) of the standard normal
            distribution, evaluated at the values in X.

        Args:
            X (torch.tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.
            c (int, optional): Index of channel. Defaults to None.

        Returns:
            log_pdf (torch.tensor): (N, 1).

        """
        C = X.shape[1]
        device = X.device
        dtype = X.dtype
        pi = torch.tensor(math.pi, dtype=dtype, device=device)
        if c is not None:
            Cov = self.Cov[c, c, k].reshape(1, 1).cpu()
            mu = self.mu[c, k].reshape(1).cpu()
        else:
            Cov = self.Cov[:, :, k]
            mu = self.mu[:, k]
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
            diff = (X - mu)/chol_Cov
        else:
            diff = torch.tensordot(X - mu, chol_Cov, dims=([1], [0]))
        log_pdf = - (C / 2) * torch.log(2 * pi) - log_det_Cov - 0.5 * torch.sum(diff**2, dim=1)
        return log_pdf

    def _init_par(self, X):
        """ Initialise GMM specific parameters: mu, Cov

        """
        dtype = torch.float64
        K = self.K
        C = X.shape[1]
        mn = torch.min(X, dim=0)[0]
        mx = torch.max(X, dim=0)[0]

        # Init mixing prop
        self._init_mp(dtype)
    
        if self.mu is None:
            # means
            self.mu = torch.zeros((C, K), dtype=dtype, device=self.dev)
        if self.Cov is None:
            # covariance
            self.Cov = torch.zeros((C, C, K), dtype=dtype, device=self.dev)
            for c in range(C):
                # rng = torch.linspace(start=mn[c], end=mx[c], steps=K, dtype=dtype, device=self.dev)
                # num_neg = sum(rng < 0)
                # num_pos = sum(rng > 0)
                # rng = torch.arange(-num_neg, num_pos, dtype=dtype, device=self.dev)
                # self.mu[c, :] = torch.reshape((rng * (mx[c] - mn[c]))/(K + 1), (1, K))
                self.mu[c, :] = torch.reshape(torch.linspace(mn[c], mx[c], K, dtype=dtype, device=self.dev), (1, K))
                self.Cov[c, c, :] = \
                    torch.reshape(torch.ones(K, dtype=dtype, device=self.dev)
                                  * ((mx[c] - mn[c])/(K))**2, (1, 1, K))

    def _update(self, ss0, ss1, ss2):
        """ Update GMM means and variances

        Args:
            ss0 (torch.tensor): 0th moment (K).
            ss1 (torch.tensor): 1st moment (C, K).
            ss2 (torch.tensor): 2nd moment (C, C, K).

        """
        C = ss1.shape[0]
        K = ss1.shape[1]

        # Update means and covariances
        for k in range(K):
            # Update mean
            self.mu[:, k] = 1/ss0[k] * ss1[:, k]

            # Update covariance
            self.Cov[:, :, k] = ss2[:, :, k] / ss0[k] \
                - torch.ger(self.mu[:, k], self.mu[:, k])
# -*- coding: utf-8 -*-
""" Mixture model class.
TODO:
    . Plot joint density.
"""


import math
from timeit import default_timer as timer
from nitorch.core import utils, py
from nitorch.core.optim import get_gain, plot_convergence
from nitorch.core.math import besseli, softmax_lse
from nitorch.plot import plot_mixture_fit
import torch


torch.backends.cudnn.benchmark = True


class Mixture:
    # A mixture model.
    def __init__(self, num_class=2):
        """
        num_class (int, optional): Number of mixture components. Defaults to 2.
        mp (torch.tensor): GMM mixing proportions.
        lam (torch.tensor): Regularisation.
        """
        self.K = num_class
        self.mp = []
        self.lam = []
        self.dev = ''  # PyTorch device
        self.dt = ''  # PyTorch data type

    # Functions
    def fit(self, X, verbose=1, max_iter=10000, tol=1e-8, fig_num=1, W=None,
            show_fit=False, samp=1):
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
            samp (int, optional): Sub-sampling, defaults to 1.
        Returns:
            Z (torch.tensor): Responsibilities (N, K).
        """
        if not isinstance(samp, int) or samp < 1:
            raise ValueError(f"samp parameter needs to be an int >= 1, got {samp}")
        
        if verbose:
            t0 = timer()  # Start timer

        # Set random seed
        torch.manual_seed(1)

        if torch.is_floating_point(X) == False:
            # Integer data type -> convert to float and add some noise
            X = X.type(torch.float)
            X += (0.001*X.max())*torch.randn_like(X)
            
        self.dev = X.device
        self.dt = X.dtype
        
        if samp > 1:
            # Sub-sample
            X0 = X[::1, :]  # Ensures copy
            X = X[::samp, :]
            if W is not None:
                W0 = W[::1, :]  # Ensures copy
                W = W[::samp, :]

        if len(X.shape) == 1:
            X = X[:, None]

        N = X.shape[0]  # Number of observations
        C = X.shape[1]  # Number of channels
        K = self.K  # Number of components

        if W is not None:  # Observation weights given
            W = torch.reshape(W, (N, 1))

        # Initialise model parameters
        self._init_par(X, W)

        # Compute a regularisation value
        self.lam = torch.zeros(C, dtype=self.dt, device=self.dev)
        for c in range(C):
            if W is not None:
                self.lam[c] = (torch.sum(X[:, c] * W.flatten()) / (torch.sum(W) * K)) ** 2
            else:
                self.lam[c] = (torch.sum(X[:, c]) / K) ** 2

        # EM loop
        Z, lb = self._em(X, max_iter=max_iter, tol=tol, verbose=verbose, W=W)

        if samp > 1:
            # Create original resolution responsibilites
            X = X0
            N = X.shape[0]
            Z = torch.zeros((N, K), dtype=self.dt, device=self.dev)
            for k in range(K):
                Z[:, k] = torch.log(self.mp[k]) + self._log_likelihood(X, k)
            if W is not None:  W = W0
            Z, _ = softmax_lse(Z, lse=True, weights=W)

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

        # Start EM algorithm
        Z = torch.zeros((N, K), dtype=dtype, device=device)  # responsibility
        lb = torch.zeros(max_iter, dtype=torch.float64, device=device)
        gain_count = 0
        for n_iter in range(max_iter):  # EM loop
            # ==========
            # E-step
            # ==========
            # Product Rule
            for k in range(K):
                Z[:, k] = torch.log(self.mp[k]) + self._log_likelihood(X, k)

            # Get responsibilities
            Z, dlb = softmax_lse(Z, lse=True, weights=W)

            # Objective function and convergence related
            lb[n_iter] = dlb
            gain = get_gain(lb[:n_iter + 1])
            if verbose >= 2:
                print('n_iter: {}, lb: {}, gain: {}'
                      .format(n_iter + 1, lb[n_iter], gain))
            if gain < tol:
                gain_count += 1
                if gain_count >= 6:
                    break  # Finished
            else:
                gain_count = 0

            if W is not None:  # Weight responsibilities
                Z = Z * W

            # ==========
            # M-step
            # ==========
            # Compute sufficient statistics
            ss0, ss1, ss2 = self._suffstats(X, Z)

            # Update mixing proportions
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
    def full_resp(Z, msk, dm=None):
        """ Converts masked responsibilities to full.
        Args:
            Z (torch.tensor): Masked responsibilities (N, K).
            msk (torch.tensor): Mask of original data (N0, ).
            dm (tuple/list, optional): Reshapes Z_full using dm.
        Returns:
            Z_full (torch.tensor): Full responsibilities (N0, K).
        """
        N0 = len(msk)
        K = Z.shape[1]
        Z_full = torch.zeros((N0, K), dtype=Z.dtype, device=Z.device)
        for k in range(K):
            Z_full[msk, k] = Z[:, k]
        if isinstance(dm, (list, tuple)):
            Z_full = torch.reshape(Z_full, tuple(dm) + (K,))

        return Z_full

    @staticmethod
    def maximum_likelihood(Z):
        """ Return maximum likelihood map.
        Args:
            Z (torch.tensor): Responsibilities (N/*spatial, K).
        Returns:
            (torch.tensor): Maximum likelihood map (N/*spatial, 1).
        """
        return torch.argmax(Z, dim=-1)


class GMM(Mixture):
    # Multivariate Gaussian Mixture Model (GMM).
    def __init__(self, num_class=2, mu=None, Cov=None):
        """
        mu (torch.tensor): GMM means (C, K).
        Cov (torch.tensor): GMM covariances (C, C, K).
        """
        super(GMM, self).__init__(num_class=num_class)
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

    def _init_par(self, X, W=None):
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


class RMM(Mixture):
    # Univariate Rician Mixture Model (RMM).
    def __init__(self, num_class=2, nu=None, sig=None):
        """
        nu (torch.tensor): "mean" parameter of each Rician (K).
        sig (torch.tensor): "standard deviation" parameter of each Rician (K).
        """
        super(RMM, self).__init__(num_class=num_class)
        self.nu = nu
        self.sig = sig

    def get_means_variances(self):
        """ Return means and variances.
        Returns:
            (torch.tensor): Means (1, K).
            (torch.tensor): Variances (1, 1, K).
        """
        K = self.K
        dtype = torch.float64
        pi = torch.tensor(math.pi, dtype=dtype, device=self.dev)

        # Laguerre polymonial for n=1/2
        Laguerre = lambda x: torch.exp(x/2) * \
            ((1 - x) * besseli(0, -x/2) - x * besseli(1, -x/2))

        # Compute means and variances
        mean = torch.zeros((1, K), dtype=dtype, device=self.dev)
        var = torch.zeros((1, 1, K), dtype=dtype, device=self.dev)
        for k in range(K):
            nu_k = self.nu[k]
            sig_k = self.sig[k]

            x = -nu_k**2/(2*sig_k**2)
            x = x.flatten()
            if x > -20:
                mean[:, k] = torch.sqrt(pi * sig_k**2/2)*Laguerre(x)
                var[:, :, k] = 2*sig_k**2 + nu_k**2 - (pi*sig_k**2/2)*Laguerre(x)**2
            else:
                mean[:, k] = nu_k
                var[:, :, k] = sig_k

        return mean, var

    def _log_likelihood(self, X, k=0, c=None):
        """
        Log-probability density function (pdf) of the Rician
        distribution, evaluated at the values in X.
        Args:
            X (torch.tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.
        Returns:
            log_pdf (torch.tensor): (N, 1).
        See also:
            https://en.wikipedia.org/wiki/Rice_distribution#Characterization
        """
        backend = dict(dtype=X.dtype, device=X.device)
        tiny = 1e-32

        # Get Rice parameters
        nu = self.nu[k].to(**backend)
        sig2 = self.sig[k].to(**backend).square()

        log_pdf = (X + tiny).log() - sig2.log() - (X.square() + nu.square()) / (2 * sig2)
        log_pdf = log_pdf + besseli(0, X * (nu / sig2), 'log')
        return log_pdf.flatten()

    def _init_par(self, X, W=None):
        """  Initialise RMM specific parameters: nu, sig
        """
        K = self.K
        mn = torch.min(X, dim=0)[0]
        mx = torch.max(X, dim=0)[0]
        dtype = torch.float64

        # Init mixing prop
        self._init_mp(dtype)

        # RMM specific
        if self.nu is None:
            self.nu = (torch.arange(K, dtype=dtype, device=self.dev)*mx)/(K + 1)
        if self.sig is None:
            self.sig = torch.ones(K, dtype=dtype, device=self.dev)*((mx - mn)/(K*10))

    def _update(self, ss0, ss1, ss2):
        """ Update RMM parameters.
        Args:
            ss0 (torch.tensor): 0th moment (K).
            ss1 (torch.tensor): 1st moment (C, K).
            ss2 (torch.tensor): 2nd moment (C, C, K).
        See also
            Koay, C.G. and Basser, P. J., Analytically exact correction scheme
            for signal extraction from noisy magnitude MR signals,
            Journal of Magnetic Resonance, Volume 179, Issue = 2, p. 317–322, (2006)
        """
        K = ss1.shape[1]
        dtype = torch.float64

        # Compute means and variances
        mu1 = torch.zeros(K, dtype=dtype, device=self.dev)
        mu2 = torch.zeros(K, dtype=dtype, device=self.dev)
        for k in range(K):
            # Update mean
            mu1[k] = 1/ss0[k] * ss1[:, k]

            # Update covariance
            mu2[k] = (ss2[:, :, k] - ss1[:, k]*ss1[:, k]/ss0[k] + self.lam*1e-3)/(ss0[k] + 1e-3)

        # Update parameters (using means and variances)
        for k in range(K):
            r = mu1[k] / mu2[k].sqrt()
            theta = math.sqrt(math.pi/(4 - math.pi))
            theta = torch.as_tensor(theta, dtype=dtype, device=self.dev).flatten()
            if r > theta:
                theta2 = theta * theta
                for i in range(256):
                    xi = besseli(0, theta2/4) * (2 + theta2) + \
                         besseli(1, theta2/4) * theta2
                    xi = xi.square() * (math.pi/8*math.exp(-theta2/2))
                    xi = (2 + theta2) - xi
                    g = (xi*(1 + r**2) - 2).sqrt()
                    if torch.abs(theta - g) < 1e-6:
                        break
                    theta = g
                if not torch.isfinite(xi):
                    xi.fill_(1)
                self.sig[k] = mu2[k].sqrt() / xi.sqrt()
                self.nu[k] = (mu1[k].square() + mu2[k]*(xi - 2)/xi).sqrt()
            else:
                self.nu[k] = 0
                self.sig[k] = 0.5*math.sqrt(2)*(mu1[k].square() + mu2[k]).sqrt()


class CMM(Mixture):
    # Chi Mixture Model (CMM).
    def __init__(self, num_class=2, dof=None, sig=None,
                 update_dof=True, update_sig=True):
        """
        Parameters
        ----------
        dof : (K,) tensor, optional
            Degrees of freedom of each Chi distribution.
        sig : (K,) tensor, optional
            Standard deviation of each Chi distribution.
        """
        super().__init__(num_class=num_class)
        self.dof = dof
        self.sig = sig
        self.update_dof = update_dof
        self.update_sig = update_sig

    def get_means_variances(self):
        """ Return means and variances.

        Returns
        -------
        means : (1, K) tensor
        variances : (1, 1, K) tensor
        """
        K = self.K
        dtype = torch.float64

        # Compute means and variances
        mean = torch.zeros((1, K), dtype=dtype, device=self.dev)
        var = torch.zeros((1, 1, K), dtype=dtype, device=self.dev)
        for k in range(K):
            dof_k = self.dof[k]

            tmp = (torch.lgamma(0.5*(dof_k+1)) - torch.lgamma(0.5*dof_k)).exp()
            if torch.isfinite(tmp):
                mean[:, k] = self.sig[k] * math.sqrt(2) * tmp
            else:
                mean[:, k] = 0
            var[:, :, k] = self.sig[k].square() * dof_k - mean[:, k].square()
        return mean, var

    def _log_likelihood(self, X, k=0, c=None):
        """
        Log-probability density function (pdf) of the Chi
        distribution, evaluated at the values in X.

        Args:
            X (torch.tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.

        Returns:
            log_pdf (torch.tensor): (N, 1).

        See also:
            https://en.wikipedia.org/wiki/Chi_distribution

        """

        N = X.shape[0]
        device = X.device
        dtype = X.dtype
        tiny = 1e-32
        log = lambda x: x.clamp_min(tiny).log_()

        # Get parameters
        dof = self.dof[k]
        dof = dof.type(dtype)
        dof = dof.to(device)
        sig2 = self.sig[k].square()
        sig2 = sig2.type(dtype)
        sig2 = sig2.to(device)

        # Use Chi distribution
        log_pdf = (1 - dof/2)*math.log(2) - 0.5*dof*log(sig2) \
                  - torch.lgamma(dof/2) + (dof - 1.)*log(X) \
                  - 0.5*X.square()/sig2.clamp_min(tiny)
        return log_pdf.flatten()

    def _init_par(self, X, W=None):
        """  Initialise CMM specific parameters: dof, sig

        """
        K = self.K
        dtype = torch.float64

        # Init mixing prop
        self._init_mp(dtype)

        if self.sig is None:
            if W is None:
                self.sig = torch.mean(X)*5
                self.sig = torch.sum((self.sig - X).square())/torch.numel(X)
                self.sig = torch.sqrt(self.sig/(K+1)*(torch.arange(1, K+1, dtype=dtype, device=self.dev)))
            else:
                self.sig = 5*(W*X).sum()/W.sum()
                self.sig = (W*(self.sig - X).square()).sum()/W.sum()
                self.sig = torch.sqrt(self.sig/(K+1)*(torch.arange(1, K+1, dtype=dtype, device=self.dev)))
        else:
            self.sig = utils.make_vector(self.sig, K, dtype=dtype, device=self.dev)

        if self.dof is None:
            self.dof = torch.full([K], 3, dtype=dtype, device=self.dev)
        else:
            self.dof = utils.make_vector(self.dof, K, dtype=dtype, device=self.dev)
        return

    def _suffstats(self, X, Z):
        """ Compute sufficient statistics.

        Args:
            X (torch.tensor): Observed data (N, 1).
            Z (torch.tensor): Responsibilities (N, K).

        Returns:
            ss0 (torch.tensor): sum(Z)          (K)
            ss1 (torch.tensor): sum(Z * log(X)) (K)
            ss2 (torch.tensor): sum(Z * X**2)   (K)

        """
        K = Z.shape[1]
        device = self.dev
        tiny = 1e-32
        X = X.flatten()
        logX = X.clamp_min(tiny).log_()

        # Suffstats
        ss0 = torch.zeros(K, dtype=torch.float64, device=device)
        ss1 = torch.zeros(K, dtype=torch.float64, device=device)
        ss2 = torch.zeros(K, dtype=torch.float64, device=device)

        # Compute sufficient statistics
        for k in range(K):
            Zk = Z[:, k]
            ss0[k] = Zk.sum(dtype=torch.float64)
            ss1[k] = (logX * Zk).sum(dim=0, dtype=torch.float64)
            ss2[k] = ((X*X) * Zk).sum(dim=0, dtype=torch.float64)
        return ss0, ss1, ss2

    def _update(self, ss0, ss1, ss2):
        """ Update CMM parameters.

        Args:
            ss0 (torch.tensor): sum(Z)          (K)
            ss1 (torch.tensor): sum(Z * log(X)) (K)
            ss2 (torch.tensor): sum(Z * X**2)   (K)

        """
        K = len(ss0)
        tiny = 1e-32

        def log(x):
            return (x+tiny).log_()

        # Update parameters (using means and variances)
        for k in range(K):

            max_iter = 10000 if self.update_dof else 1
            for i in range(max_iter):
                # print(self.sig[k].item(), self.dof[k].item())
                if self.update_sig:
                    # Closed form update of sig
                    self.sig[k] = (ss2[k]/(self.dof[k]*ss0[k])).sqrt()

                if self.update_dof:
                    # Gauss-Newton update of dof
                    gkl = torch.digamma(self.dof[k]/2) + log(2*self.sig[k].square())
                    gkl = 0.5 * ss0[k] * gkl - ss1[k]                        # gradient w.r.t. dof
                    hkl = 0.25 * ss0[k] * torch.polygamma(1, self.dof[k]/2)  # Hessian w.r.t. dof
                    self.dof[k].sub_(gkl/hkl)                                # G-N update
                    if self.dof[k] < 2:
                        self.dof[k] = 2
                        break
                    if gkl*gkl < 1e-9:
                        break

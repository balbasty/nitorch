# -*- coding: utf-8 -*-
""" Mixture model class.

TODO:
    . Plot joint density.
"""


import math
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from nitorch.optim import get_gain, plot_convergence
from nitorch.utils import softmax
import torch
from torch.distributions import MultivariateNormal as mvn

torch.backends.cudnn.benchmark = True


class Mixture:
    """ A mixture model.
    """
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

    """ Functions
    """
    def fit(self, X, verbose=1, max_iter=10000, tol=1e-6, fig_num=1, W=None,
            show_fit=False):
        """ Fit mixture model.

        Args:
            X (torch.tensor): Observed data (N, C).
                N = num observations per channel
                C = num channels
            verbose (int, optional) Display progress. Defaults to 1.
                0: None.
                1: Print summary when finished.
                2: 1 + Log-likelihood plot.
                3: 1 + 2 + print convergence.
            max_iter (int, optional) Maxmimum number of algorithm iterations.
                Defaults to 10000.
            tol (int, optional): Convergence threshold. Defaults to 1e-6.
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

        """ Initialise model parameters
        """
        self._init_par(X)

        # Compute a regularisation value
        self.lam = torch.zeros(C, dtype=self.dt, device=self.dev)
        for c in range(C):
            if W is not None:
                self.lam[c] = (torch.sum(X[:, c] * W.flatten()) / (torch.sum(W) * K)) ** 2
            else:
                self.lam[c] = (torch.sum(X[:, c]) / K) ** 2

        """ EM loop
        """
        Z, lb = self._em(X, max_iter=max_iter, tol=tol, verbose=verbose, W=W)

        # Print algorithm info
        if verbose >= 1:
            print('Algorithm finished in {} iterations, '
                  'log-likelihood = {}, '
                  'runtime: {:0.1f} s, '
                  'device: {}'.format(len(lb), lb[-1], timer() - t0, self.dev))
        if verbose >= 2:
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

        """ Init
        """
        N = X.shape[0]
        C = X.shape[1]
        K = self.K
        dtype = self.dt
        device = self.dev
        tiny = torch.tensor(1e-32, dtype=dtype, device=device)

        """ Start EM algorithm
        """
        Z = torch.zeros((N, K), dtype=dtype, device=device)  # responsibility
        lb = torch.zeros(max_iter, dtype=torch.float64, device=device)
        for n_iter in range(max_iter):  # EM loop
            """ E-step
            """
            for k in range(K):
                # Product Rule
                Z[:, k] = torch.log(self.mp[k]) + self._log_likelihood(X, k)

            # Get responsibilities
            Z, dlb = softmax(Z, W=W, get_ll=True)

            """ Objective function and convergence related
            """
            lb[n_iter] = dlb
            gain = get_gain(lb, n_iter)
            if verbose >= 3:
                print('n_iter: {}, lb: {}, gain: {}'
                      .format(n_iter + 1, lb[n_iter], gain))
            if gain < tol:
                break  # Finished

            if W is not None:  # Weight responsibilities
                Z = Z * W

            """ M-step
            """
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
        self.plot_fit(X, log_pdf, mu, var, mp, fig_num, W)
    
    """ Implement in child classes
    """
    def get_means_variances(self): pass

    def _log_likelihood(self): pass

    def _init_par(self):
        pass

    def _update(self): pass

    """ Static methods
    """
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
    def besseli(X, order=0):
        """ Approximates the modified Bessel function of the first kind,
            of either order zero or one.

        Args:
            X (torch.tensor): Input (N, 1).
            order (int, optional): 0 or 1, defaults to 0.

        Returns:
            I (torch.tensor): Modified Bessel function of the first kind (N, 1).

        See also:
            https://mathworld.wolfram.com/ModifiedBesselFunctionoftheFirstKind.html

        """
        if len(X.shape) == 1:
            X = X[:, None]
            N = X.shape[0]
        else:
            N = 1

        device = X.device
        dtype = X.dtype

        Nk = 10  # higher number, better approximation (10 seems to do just fine)
        X = X.repeat(1, Nk)
        K = torch.arange(0, Nk, dtype=dtype, device=device)
        K = K.repeat(N, 1).float()
        K_factorial = (K + 1).lgamma().exp()

        if order == 0:  # 0th order
            i = torch.sum((0.25 * X ** 2) ** K / (K_factorial ** 2), dim=1)
        else:  # First order
            i = torch.sum(
                0.5 * X * ((0.25 * X ** 2) ** K /
                           (K_factorial * torch.exp(torch.lgamma(K + 2)))), dim=1)

        return i

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

    @staticmethod
    def plot_fit(X, log_pdf, mu, var, mp, fig_num=1, W=None, title=''):
        """ Plot mixture fit.

        Args:
            X (torch.tensor): (N, C).
            fig_num (int, optional): Defaults to 1.
            W (torch.tensor, optional): Defaults to no weights.
            title (string, optional): Defaults to ''.

        """
        # Number of classes
        K = len(mp)
        # To CPU
        X = X.cpu()
        mu = mu.cpu()
        var = var.cpu()
        mp = mp.cpu()
        # Parameters
        num_sd = torch.tensor(5)
        steps = 100
        nN = 128
        inf = torch.tensor(float('inf'))
        if len(X.shape) == 1:
            X = X[:, None]
        C = X.shape[1]
        N = X.shape[0]
        # Max X
        mn_x = torch.min(X, dim=0)[0]
        mx_x = torch.max(X, dim=0)[0]

        if W is not None:
            # To CPU
            W = W.cpu()
            # Weights and observation range given
            W = torch.reshape(W, (N, 1))
            W = W / torch.sum(W)
            H = [1]
        else:
            # Make weights and observation range from data
            nX = torch.zeros(nN, C)
            W = torch.zeros(nN, C)
            H = torch.zeros(C)
            for c in range(C):
                # Bar height
                W[:, c] = torch.histc(X[:, c], bins=nN)
                # Bar start edge
                nX[:, c] = torch.linspace(start=mn_x[c], end=mx_x[c], steps=nN + 1)[:-1]
                # Bar width
                H[c] = nX[1, c] - nX[0, c]
                # Normalise height
                W[:, c] = W[:, c] / (torch.sum(W[:, c]) * H[c])
            X = nX
        # Max x and y
        mn_x = torch.min(X, dim=0)[0]
        mx_x = torch.max(X, dim=0)[0]
        mx_y = torch.max(W, dim=0)[0]
        # Plotting
        num_plt = C + 1
        num_row = math.floor(math.sqrt(num_plt))
        num_col = math.ceil(num_plt / num_row)
        plt.figure(fig_num).clear()  # Clear figure
        fig, ax = plt.subplots(num_row, num_col, num=fig_num)  # Make figure and axes
        fig.show()
        # For each channel, plot the data and the marginal density
        c = 0  # channel counter
        for row in range(num_row):  # Loop over subplot rows
            for col in range(num_col):  # Loop over subplot rows
                if c == C:
                    continue
                # Get axis
                ax_rc = ax[c] if (num_row == 1 or num_col == 1) else ax[row, col]
                # Data in bar plot
                ax_rc.bar(x=X[:, c], height=W[:, c], width=H[c], alpha=0.25)
                # Marginal density
                plot_list = []  # Store plot handles (used to set colors in bar plot for mix prop)
                for k in range(K):  # Loop over mixture components
                    width = num_sd * torch.sqrt(var[c, c, k])
                    x0 = mu[c, k] - width
                    x1 = mu[c, k] + width
                    x = torch.linspace(x0, x1, steps=steps)
                    y = mp[k]*torch.exp(log_pdf(x.reshape(steps, 1), k, c))
                    plot = ax_rc.plot(x, y)
                    plot_list.append(plot)

                ax_rc.set_xlim([mn_x[c], mx_x[c]])
                ax_rc.set_ylim([0, 0.5 * mx_y[c]])

                ax_rc.axes.get_yaxis().set_visible(False)
                ax_rc.set_title('Marginal density, C={}'.format(c + 1))
                c += 1

        # Bar plot the mixing proportions
        ax_rc = ax[c] if (num_row == 1 or num_col == 1) else ax[num_row - 1, num_col - 1]
        bp = ax_rc.bar([str(n) for n in range(1, K + 1)], mp)
        for k in range(K):
            bp[k].set_color(plot_list[k][0].get_color())
        ax_rc.axes.get_yaxis().set_visible(False)
        ax_rc.set_title('Mixing proportions')

        plt.suptitle(title)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)


class GMM(Mixture):
    """ Multivariate Gaussian Mixture Model (GMM).
    """
    def __init__(self, num_class=2):
        """
        mu (torch.tensor): GMM means (C, K).
        Cov (torch.tensor): GMM covariances (C, C, K).

        """
        super(GMM, self).__init__(num_class=num_class)
        self.mu = []
        self.Cov = []

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

        # means
        self.mu = torch.zeros((C, K), dtype=dtype, device=self.dev)
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
    """ Univariate Rician Mixture Model (RMM).
    """
    def __init__(self, num_class=2):
        """
        nu (torch.tensor): "mean" parameter of each Rician (K).
        sig (torch.tensor): "standard deviation" parameter of each Rician (K).

        """
        super(RMM, self).__init__(num_class=num_class)
        self.nu = []
        self.sig = []

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
            ((1 - x) * RMM.besseli(-x/2, order=0) - x * RMM.besseli(-x/2, order=1))

        # Compute means and variances
        mean = torch.zeros((1, K), dtype=dtype, device=self.dev)
        var = torch.zeros((1, 1, K), dtype=dtype, device=self.dev)
        for k in range(K):
            nu_k = self.nu[k]
            sig_k = self.sig[k]

            x = -nu_k**2/(2*sig_k**2)
            if x > -20:
                mean[:, k] = torch.sqrt(pi * sig_k**2/2)*Laguerre(x)
                var[:, :, k] = 2*sig_k**2 + nu_k**2 - (pi*sig_k**2/2)*Laguerre(x)**2
            else:
                mean[:, k] = nu_k
                var[:, :, k] = sig_k

        return mean, var

    def _log_likelihood(self, X, k=0, c=-1):
        """
        Log-probability density function (pdf) of the Rician
        distribution, evaluated at the values in X.

        Args:
            X (torch.tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.
            c (int, optional): Index of channel. Defaults to -1 (univariate).

        Returns:
            log_pdf (torch.tensor): (N, 1).

        See also:
            https://en.wikipedia.org/wiki/Rice_distribution#Characterization

        """
        N = X.shape[0]
        device = X.device
        dtype = X.dtype
        pi = torch.tensor(math.pi, dtype=dtype, device=self.dev)
        tiny = torch.tensor(1e-32, dtype=dtype, device=device)

        # Get Rice parameters
        nu = self.nu[k]
        sig2 = self.sig[k]**2
        nu = nu.type(dtype)
        sig2 = sig2.type(dtype)

        log_pdf = torch.zeros((N, 1), dtype=dtype, device=device)
        tmp = -(X**2 + nu**2)/(2*sig2)
        # Identify where Rice probability can be computed
        msk = (tmp > -95) & ((X * (nu / sig2)) < 85)
        # Use Rician distribution
        log_pdf[msk] = (X[msk]/sig2) * torch.exp(tmp[msk]) * RMM.besseli(X[msk] * (nu / sig2), order=0)
        # Use normal distribution
        log_pdf[~msk] = (1. / torch.sqrt(2 * pi * sig2)) \
                  * torch.exp((-0.5 / sig2) * (X[~msk] - nu)**2)

        return torch.log(log_pdf.flatten() + tiny)

    def _init_par(self, X):
        """  Initialise RMM specific parameters: nu, sig

        """
        K = self.K
        mn = torch.min(X, dim=0)[0]
        mx = torch.max(X, dim=0)[0]
        dtype = torch.float64

        # Init mixing prop
        self._init_mp(dtype)

        # RMM specific
        self.nu = (torch.arange(K, dtype=dtype, device=self.dev)*mx)/(K + 1)
        self.sig = torch.ones(K, dtype=dtype, device=self.dev)*((mx - mn)/(K))

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
        C = ss1.shape[0]
        K = ss1.shape[1]
        dtype = torch.float64
        pi = torch.tensor(math.pi, dtype=dtype, device=self.dev)

        # Compute means and variances
        mu1 = torch.zeros(K, dtype=dtype, device=self.dev)
        mu2 = torch.zeros(K, dtype=dtype, device=self.dev)
        for k in range(K):
            # Update mean
            mu1[k] = 1/ss0[k] * ss1[:, k]

            # Update covariance
            mu2[k] = (ss2[:, :, k] - ss1[:, k]*ss1[:, k]/ss0[k]
                     + self.lam*1e-3)/(ss0[k] + 1e-3)


        # Update parameters (using means and variances)
        for k in range(K):
            r = mu1[k]/torch.sqrt(mu2[k])
            theta = torch.sqrt(pi/(4 - pi))
            if r > theta:
                for i in range(256):
                    xi = 2 + theta**2 \
                        - pi/8*torch.exp(-theta**2/2)*((2 + theta**2) * RMM.besseli(theta**2/4, order=0) \
                        + theta**2*RMM.besseli(theta**2/4, order=1))**2
                    g = torch.sqrt(xi*(1 + r**2) - 2)
                    if torch.abs(theta - g) < 1e-6:
                        break
                    theta = g
                if not torch.isfinite(xi):
                    xi = 1
                self.sig[k] = torch.sqrt(mu2[k])/torch.sqrt(xi)
                self.nu[k] = torch.sqrt(mu1[k]**2 + (xi - 2)*self.sig[k]**2)
            else:
                self.nu[k] = 0
                self.sig[k] = \
                    0.5*(torch.sqrt(torch.tensor(2, dtype=dtype, device=self.dev).float())
                         *torch.sqrt(mu1[k]**2 + mu2[k]))

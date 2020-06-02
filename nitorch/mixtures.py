# -*- coding: utf-8 -*-
""" Mixture model class.

TODO:
    . Plot joint density.
    . Do log of pdf with softmax at end.
    . Implement own MVN pdf.
    . Look at using: http://deepmind.github.io/torch-distributions/.
"""


import math
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal as mvn

torch.backends.cudnn.benchmark = True


class Mixture:
    """ A mixture model.
    """
    def __init__(self, num_class=2):
        """
        num_class (int, optional): Number of mixture components. Defaults to 2.
        mp (torch.Tensor): GMM mixing proportions.
        lam (torch.Tensor): Regularisation.

        """
        self.K = num_class
        self.mp = []
        self.lam = []
        self.dev = ''  # PyTorch device
        self.dt = ''  # PyTorch data type

    """ Functions
    """
    def em(self, X, num_iter, tol, verbose, W):
        """ EM loop for fitting GMM.

        Args:
            X (torch.Tensor): (N, C).
            num_iter (int)
            tol (int)
            verbose (int)
            W (torch.Tensor): (N, 1).

        Returns:
            Z (torch.Tensor): Responsibilities (N, K).
            ll (list): Log-likelihood at each iteration.

        """

        """ Init
        """
        N = X.shape[0]
        C = X.shape[1]
        K = self.K
        dtype = self.dt
        device = self.dev
        tiny = torch.tensor(1e-16, dtype=dtype, device=device)
        inf = torch.tensor(float('inf'), dtype=dtype, device=device)

        if len(W.shape) > 0:  # Observation weights given
            tol = torch.sum(W) * tol
        else:
            tol = N * tol

        """ Start EM algorithm
        """
        Z = torch.zeros((N, K), dtype=dtype, device=device)  # responsibility
        ll = [-inf]  # log-likelihood
        for iter in range(num_iter):  # EM loop
            """ E-step
            """
            for k in range(K):
                # Product Rule
                Z[:, k] = self.mp[k]*self.pdf(X, k) + tiny

            # Sum rule
            Z_sum = torch.sum(Z, dim=1)
            Z_sum = Z_sum[:, None]

            # Bayes Rule
            Z = Z / Z_sum

            # Compute log-likelihood
            ll.append(torch.sum(torch.log(Z_sum) * W))

            if verbose >= 3:
                print('iter: {}, ll: {}, diff: {}'
                      .format(iter, ll[iter + 1], ll[iter + 1] - ll[iter]))

            if iter > 1 and (ll[iter + 1] - ll[iter]) < tol:
                break  # Finished

            # Weight responsibilities
            Z = Z * W

            """ M-step
            """
            # Compute sufficient statistics
            ss0, ss1, ss2 = self.suffstats(X, Z)

            # Update mixing proportions
            if len(W.shape) > 0:
                self.mp = ss0 / torch.sum(W, dim=0)
            else:
                self.mp = ss0 / N

            # Update model specific parameters
            self.update(ss0, ss1, ss2)

        return Z, ll

    def fit(self, X, verbose=1, num_iter=10000, tol=1e-8, fig_num=1, W=1):
        """ Fit mixture model.

        Args:
            X (torch.Tensor): Observed data (N, C).
                N = num observations per channel
                C = num channels
            verbose (int, optional) Display progress. Defaults to 1.
                0: None.
                1: Print summary when finished.
                2: 1 + Log-likelihood plot.
                3: 1 + 2 + print convergence.
            num_iter (int, optional) Maxmimum number of algorithm iterations.
                Defaults to 10000.
            tol (int, optional): Convergence threshold. Defaults to 1e-8.
            fig_num (int, optional): Defaults to 1.
            W (torch.Tensor, optional): Observation weights (N, 1). Defaults to 1 (no weights).

        Returns:
            Z (torch.Tensor): Responsibilities (N, K).

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

        if type(W) is int:  # No observation weights
            W = torch.tensor(W, dtype=self.dt, device=self.dev)
        if len(W.shape) > 0:  # Observation weights given
            W = torch.reshape(W, (N, 1))

        """ Initialise model parameters
        """
        self.init_par(X)

        # Compute a regularisation value
        self.lam = torch.zeros(C, dtype=self.dt, device=self.dev)
        for c in range(C):
            self.lam[c] = (torch.sum(X[:, c] * W.flatten()) / (torch.sum(W) * K)) ** 2

        """ EM loop
        """
        Z, ll = self.em(X, num_iter, tol, verbose, W)

        # Print algorithm info
        if verbose >= 1:
            print('Algorithm finished in {} iterations, '
                  'log-likelihood = {}, '
                  'runtime: {:0.1f} s, '
                  'device: {}'.format(len(ll) - 1, ll[-1], timer() - t0, self.dev))
        if verbose >= 2:
            self.plot_convergence(ll, xlab='Iteration number',
                                  title='Log-likelihood', fig_num=fig_num)

        return Z

    def init_mp(self):
        """ Initialise mixing proportions: mp

        """
        # Mixing proportions
        self.mp = torch.ones(self.K, dtype=self.dt, device=self.dev)/self.K

    def plot_convergence(self, vals, fig_num, xlab, title):
        """ Plot algorithm convergence.

        Args:
            vals (list): Values to plot.
            fig_num (int)
            xlab (string)
            title (string)

        """
        plt.figure(num=fig_num).clear()
        fig = plt.figure(num=fig_num)
        ax = fig.add_subplot(111)

        ax.plot(torch.arange(0, len(vals)), vals)
        ax.set_xlabel(xlab)
        ax.set_title(title)
        plt.grid()
        plt.show()

    def plot_fit(self, X, fig_num=1, W=1, suptitle=''):
        """ Plot mixture fit.

        Args:
            X (torch.Tensor): (N, C).
            fig_num (int, optional): Defaults to 1.
            W (torch.Tensor, optional): Defaults to 1 (no weights).
            suptitle (string, optional): Defaults to ''.

        """
        if type(W) is int:  # No observation weights
            W = torch.tensor(W)

        # To CPU (so can be used with matplotlib)
        X = X.cpu()
        W = W.cpu()

        # Get mixing proportions (on CPU)
        mp = self.mp.cpu()
        K = len(mp)

        # Get means and variances (on CPU)
        mu, var = self.get_means_variances()
        mu = mu.cpu()
        var = var.cpu()

        if len(X.shape) == 1:
            X = X[:, None]
        C = X.shape[1]
        N = X.shape[0]

        num_sd = torch.tensor(5)
        steps = 100
        nN = 128
        inf = torch.tensor(float('inf'))

        mn_x = torch.min(X, dim=0)[0]
        mx_x = torch.max(X, dim=0)[0]

        if len(W.shape) > 0:
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

        mx_y = torch.max(W, dim=0)[0]

        num_plt = C + 1
        num_row = math.floor(math.sqrt(num_plt))
        num_col = math.ceil(num_plt/num_row)
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
                p_list = []  # Store plot handles (used to set colors in bar plot for mix prop)
                for k in range(K):  # Loop over mixture components
                    x0 = mu[c, k] - num_sd * torch.sqrt(var[c, c, k])
                    x1 = mu[c, k] + num_sd * torch.sqrt(var[c, c, k])
                    x = torch.linspace(x0, x1, steps=steps)
                    y = mp[k] * self.pdf(x.reshape(steps, 1), k, c)
                    p = ax_rc.plot(x, y)
                    p_list.append(p)

                ax_rc.set_xlim([mn_x[c], mx_x[c]])
                ax_rc.set_ylim([0, 0.5 * mx_y[c]])

                ax_rc.axes.get_yaxis().set_visible(False)
                ax_rc.set_title('Marginal density, C={}'.format(c + 1))
                c += 1

        # Bar plot the mixing proportions
        ax_rc = ax[c] if (num_row == 1 or num_col == 1) else ax[num_row - 1, num_col - 1]
        bp = ax_rc.bar([str(n) for n in range(1, K + 1)] , mp)
        for k in range(K):
            bp[k].set_color(p_list[k][0].get_color())
        ax_rc.axes.get_yaxis().set_visible(False)
        ax_rc.set_title('Mixing proportions')

        plt.suptitle(suptitle)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    def suffstats(self, X, Z):
        """ Compute sufficient statistics.

        Args:
            X (torch.Tensor): Observed data (N, C).
            Z (torch.Tensor): Responsibilities (N, K).

        Returns:
            ss0 (torch.Tensor): 0th moment (K).
            ss1 (torch.Tensor): 1st moment (C, K).
            ss2 (torch.Tensor): 2nd moment (C, C, K).

        """
        N = X.shape[0]
        C = X.shape[1]
        K = Z.shape[1]
        device = self.dev

        # Suffstats
        ss1 = torch.zeros((C, K), dtype=torch.float64, device=device)
        ss2 = torch.zeros((C, C, K), dtype=torch.float64, device=device)

        # Compute 0th moment
        ss0 = torch.sum(Z, dim=0, dtype=torch.float64)

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

    """ Implement in child classes
    """
    def get_means_variances(self): pass

    def pdf(self): pass

    def update(self): pass

    """ Static methods
    """
    @staticmethod
    def apply_mask(X):
        """ Mask tensor, removing zeros and non-finite values.

        Args:
            X (torch.Tensor): Observed data (N0, C).

        Returns:
            X_msk (torch.Tensor): Observed data (N, C), where N < N0.
            msk (torch.Tensor): Logical mask (N, 1).

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
            X (torch.Tensor): Input (N, 1).
            order (int, optional): 0 or 1, defaults to 0.

        Returns:
            I (torch.Tensor): Modified Bessel function of the first kind (N, 1).

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
            img (torch.Tensor): Input image. (dm[0], dm[1], dm[2], C)

        Returns:
            X (torch.Tensor): Observed data (N0, C).
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
            Z (torch.Tensor): Masked responsibilities (N, K).
            msk (torch.Tensor): Mask of original data (N0, 1).
            dm (torch.Size, optional): Reshapes Z_full using dm. Defaults to [].

        Returns:
            Z_full (torch.Tensor): Full responsibilities (N0, K).

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
            Z (torch.Tensor): Responsibilities (N, K).

        Returns:
            (torch.Tensor): Maximum likelihood map (N, 1).

        """
        return torch.argmax(Z, dim=3)


class GMM(Mixture):
    """ Multivariate Gaussian Mixture Model (GMM).
    """
    def __init__(self, num_class=2):
        """
        mu (torch.Tensor): GMM means (C, K).
        Cov (torch.Tensor): GMM covariances (C, C, K).

        """
        super(GMM, self).__init__(num_class=num_class)
        self.mu = []
        self.Cov = []

    def get_means_variances(self):
        """
        Return means and variances.

        Returns:
            (torch.Tensor): Means (C, K).
            (torch.Tensor): Covariances (C, C, K).

        """
        return self.mu, self.Cov

    def init_par(self, X):
        """ Initialise GMM specific parameters: mu, Cov

        """
        self.init_mp()  # Init mixing prop

        K = self.K
        C = X.shape[1]
        mn = torch.min(X, dim=0)[0]
        mx = torch.max(X, dim=0)[0]

        # means
        self.mu = torch.zeros((C, K), dtype=self.dt, device=self.dev)
        # covariance
        self.Cov = torch.zeros((C, C, K), dtype=self.dt, device=self.dev)
        for c in range(C):
            self.mu[c, :] = \
                torch.reshape((torch.arange(K, dtype=self.dt, device=self.dev)
                               * mx[c])/(K + 1), (1, K))
            self.Cov[c, c, :] = \
                torch.reshape(torch.ones(K, dtype=self.dt, device=self.dev)
                              * ((mx[c] - mn[c])/K)**2, (1, 1, K))

    def pdf(self, X, k=0, c=-1):
        """ Probability density function (pdf) of the standard normal
            distribution, evaluated at the values in X.

        Args:
            X (torch.Tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.
            c (int, optional): Index of channel. Defaults to -1 (univariate).

        Returns:
            (torch.Tensor): (N, 1).

        """
        if c >= 0:
            return torch.exp(mvn(self.mu[c, k].reshape(1, 1).cpu(),
                                 self.Cov[c, c, k].reshape(1, 1).cpu()).log_prob(X))
        else:
            return torch.exp(mvn(self.mu[:, k], self.Cov[:, :, k]).log_prob(X))

    def update(self, ss0, ss1, ss2):
        """ Update GMM means and variances

        Args:
            ss0 (torch.Tensor): 0th moment (K).
            ss1 (torch.Tensor): 1st moment (C, K).
            ss2 (torch.Tensor): 2nd moment (C, C, K).

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
        nu (torch.Tensor): "mean" parameter of each Rician (K).
        sig (torch.Tensor): "standard deviation" parameter of each Rician (K).

        """
        super(RMM, self).__init__(num_class=num_class)
        self.nu = []
        self.sig = []

    def get_means_variances(self):
        """ Return means and variances.

        Returns:
            (torch.Tensor): Means (1, K).
            (torch.Tensor): Variances (1, 1, K).

        """
        K = self.K
        pi = torch.tensor(math.pi, dtype=self.dt, device=self.dev)

        # Laguerre polymonial for n=1/2
        Laguerre = lambda x: torch.exp(x/2) * \
            ((1 - x) * RMM.besseli(-x/2, order=0) - x * RMM.besseli(-x/2, order=1))

        # Compute means and variances
        mean = torch.zeros((1, K), dtype=self.dt, device=self.dev)
        var = torch.zeros((1, 1, K), dtype=self.dt, device=self.dev)
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

    def init_par(self, X):
        """  Initialise RMM specific parameters: nu, sig

        """
        self.init_mp()  # Init mixing prop

        K = self.K
        mn = torch.min(X, dim=0)[0]
        mx = torch.max(X, dim=0)[0]

        # RMM specific
        self.nu = (torch.arange(K, dtype=self.dt, device=self.dev)*mx)/(K + 1)
        self.sig = torch.ones(K, dtype=self.dt, device=self.dev)*((mx - mn)/(K))

    def pdf(self, X, k=0, c=-1):
        """
        Probability density function (pdf) of the Rician
        distribution, evaluated at the values in X.

        Args:
            X (torch.Tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.
            c (int, optional): Index of channel. Defaults to -1 (univariate).

        Returns:
            p (torch.Tensor): (N, 1).

        See also:
            https://en.wikipedia.org/wiki/Rice_distribution#Characterization

        """
        N = X.shape[0]
        device = X.device
        dtype = X.dtype
        pi = torch.tensor(math.pi, dtype=self.dt, device=self.dev)

        # Get Rice parameters
        nu = self.nu[k]
        sig2 = self.sig[k]**2

        p = torch.zeros((N, 1), dtype=dtype, device=device)
        tmp = -(X**2 + nu**2)/(2*sig2)
        # Identify where Rice probability can be computed
        msk = (tmp > -95) & ((X * (nu / sig2)) < 85)
        # Use Rician distribution
        p[msk] = (X[msk]/sig2) * torch.exp(tmp[msk]) * RMM.besseli(X[msk] * (nu / sig2), order=0)
        # Use normal distribution
        p[~msk] = (1. / torch.sqrt(2 * pi * sig2)) \
                  * torch.exp((-0.5 / sig2) * (X[~msk] - nu)**2)

        return p.flatten()

    def update(self, ss0, ss1, ss2):
        """ Update RMM parameters.

        Args:
            ss0 (torch.Tensor): 0th moment (K).
            ss1 (torch.Tensor): 1st moment (C, K).
            ss2 (torch.Tensor): 2nd moment (C, C, K).

        See also
            Koay, C.G. and Basser, P. J., Analytically exact correction scheme
            for signal extraction from noisy magnitude MR signals,
            Journal of Magnetic Resonance, Volume 179, Issue = 2, p. 317–322, (2006)

        """
        C = ss1.shape[0]
        K = ss1.shape[1]
        pi = torch.tensor(math.pi, dtype=self.dt, device=self.dev)

        # Compute means and variances
        mu1 = torch.zeros(K, dtype=self.dt, device=self.dev)
        mu2 = torch.zeros(K, dtype=self.dt, device=self.dev)
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
                    0.5*(torch.sqrt(torch.tensor(2, dtype=self.dt, device=self.dev).float())
                         *torch.sqrt(mu1[k]**2 + mu2[k]))

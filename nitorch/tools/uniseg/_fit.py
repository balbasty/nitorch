from timeit import default_timer as timer
import math as pymath
from nitorch.core import linalg, math, utils, py
from nitorch import spatial
from nitorch.tools.registration.utils import affine_grid_backward
import torch
from ._mrf import mrf, mrf_suffstat, mrf_covariance
from ._plot import plot_lb, plot_images_and_lb

default_warp_reg = {'absolute': 0,
                    'membrane': 0.001,
                    'bending': 0.5,
                    'lame': (0.05, 0.2),
                    }


def _softmax_lse(x, dim=-1, W=None):
    """Implicit softmax that also returns the LSE"""
    x = x.clone()
    lse, _ = torch.max(x, dim=dim, keepdim=True)
    lse.clamp_min_(0)  # don't forget the class full of zeros

    x = x.sub_(lse).exp_()
    sumval = x.sum(dim=dim, keepdim=True)
    sumval += lse.neg().exp_()  # don't forget the class full of zeros
    x /= sumval

    sumval = sumval.log_()
    lse += sumval
    if W is not None:
        if lse.numel() == 1:
            lse = lse.sum() * W.sum()
            return x, lse
        lse *= W
    lse = lse.sum(dtype=torch.float64)
    return x, lse


class SpatialMixture:
    """Mixture model with spatially varying priors"""

    _tiny = 1e-32
    _tinyish = 1e-3

    def __init__(self, nb_classes=6, prior=None, affine_prior=None,
                 do_bias=True, do_warp=True,  do_affine=True,
                 do_mixing=True, do_mrf='once', lam_prior=1,
                 lam_bias=0.1, lam_warp=0.1, lam_mixing=100, lam_mrf=10,
                 bias_acceleration=0, warp_acceleration=0.9, spacing=3,
                 max_iter=30, tol=1e-3, max_iter_intensity=8, max_iter_mrf=50,
                 max_iter_cluster=20, max_iter_bias=2, max_iter_warp=3,
                 max_iter_affine=3, max_iter_mixing=10, verbose=1, plot=0):
        """
        Parameters
        ----------
        nb_classes : int or sequence[int], default=2
            Number of components in the mixture (including background).
            If a sequence, defines the number of components within each
            "prior" class; e.g. `[5, 3, 3, 2, 2, 1]`
        prior : (nb_classes, [*spatial]) tensor, optional
            Non-stationary prior.
        affine_prior : (D+1, D+1) tensor, optional
            Orientation matrix of the prior.

        Components
        ----------
        do_bias : bool, default=True
            Optimize bias field
        do_warp : bool, default=True
            Optimize warping field
        do_affine : bool, default=True
            Optimize affine matrix
        do_mixing : bool, default=True
            Optimize stationary mixing proportions
        do_mrf : {False, 'once', 'always', 'learn' or True}, default='once'
            Include a Markov Random Field
            - 'once' : only at the end
            - 'always' : at each iteration
            - 'learn' : at each iteration and optimize its weights

        Regularization
        --------------
        lam_prior : float, default=1
            Strength of the spatial prior
        lam_bias : float, default=0.1
            Regularization factor of the bias field.
        lam_warp : float or dict, default=0.1
            Regularization of the warping field.
            If a scalar, multiplies the default parameters, which are:
            {'membrane': 1e-3, 'bending': 0.5, 'lame': (0.05, 0.2)}
        lam_mixing : float, default=100
            Dirichlet regularization of the mixing proportions
        lam_mrf : float, default=10
            Dirichlet regularization of the MRF prior.

        Performance
        -----------
        max_iter : int, default=30
            Maximum number of outer (coordinate descent) iterations
        tol : float, default=1e-3
            Tolerance for early stopping
        max_iter_intensity : int, default=8
            Maximum number of GMM+Bias (EM) iterations
        max_iter_cluster : int, default=20
            Maximum number of GMM (EM) iterations
        max_iter_mrf : int, default=50
            Maximum number of MRF (EM) iterations
        max_iter_mixing : int, default=10
            Maximum number of Mixing (EM) iterations
        max_iter_bias : int, default=1
            Maximum number of Bias (Gauss-Newton) iterations
        max_iter_warp : int, default=3
            Maximum number of Warp (Gauss-Newton) iterations
        max_iter_affine : int, default=3
            Maximum number of Affine (Gauss-Newton) iterations
        bias_acceleration : float, default=0.9
            How much to trust Fisher's Hessian
            (1 = faster but less stable, 0 = slower but more stable)
        warp_acceleration : float, default=0.9
            How much to trust Fisher's Hessian
            (1 = faster but less stable, 0 = slower but more stable)
        spacing : [list of] float, default=3
            Distance (in mm) between sampled points when optimizing parameters

        Interface
        ---------
        verbose : int, default=1
            0: None
            1: Print summary when finished
            2: Print convergence (excluding gmm)
            3: Print convergence (including gmm)
        plot : bool or int, default=False
            0: None
            1: Show lower bound and images at the end
            2: Show live lower bound
            3: Show live images (excluding gmm)
            4: Show live images (including gmm)

        """
        # Convert nb_classes to a list of cluster indices
        if isinstance(nb_classes, int):
            lkp = list(range(nb_classes))
        else:
            nb_classes = py.make_list(nb_classes)
            lkp = [n for n, k in enumerate(nb_classes) for _ in range(k)]
        self.lkp = lkp

        if not lam_prior:
            prior = None
        if prior is not None:
            implicit_in = len(prior) < self.nb_classes
            prior = prior.clamp(1e-3, 1-1e-3)
            prior = math.logit(prior, implicit=(implicit_in, True), dim=0)
            prior *= lam_prior
            if affine_prior is None:
                affine_prior = spatial.affine_default(prior.shape[1:])
        self.log_prior = prior
        self.affine_prior = affine_prior
        if isinstance(lam_warp, (int, float)):
            factor_warp = lam_warp
            lam_warp = {k: [vv*factor_warp for vv in v] if isinstance(v, (list, tuple))
                        else v*factor_warp for k, v in default_warp_reg.items()}
        elif not lam_warp:
            lam_warp = default_warp_reg
        self.lam_bias = lam_bias * 1e6
        self.lam_warp = lam_warp
        self.lam_mixing = lam_mixing
        self.lam_mrf = lam_mrf
        self.warp_acceleration = min(1., max(0., warp_acceleration))
        self.bias_acceleration = min(1., max(0., bias_acceleration))
        self.do_bias = do_bias
        self.do_warp = (prior is not None) and do_warp
        self.do_affine = (prior is not None) and do_affine
        self.do_mixing = do_mixing
        self.do_mrf = do_mrf
        if self.do_mrf is True:
            self.do_mrf = 'learn'

        if prior is None:
            self.do_warp = False

        # Optimization
        self.spacing = list(sorted(py.ensure_list(spacing)))[::-1]
        self.max_iter = max_iter
        self.max_iter_intensity = max_iter_intensity
        self.max_iter_cluster = max_iter_cluster
        self.max_iter_bias = max_iter_bias
        self.max_iter_warp = max_iter_warp
        self.max_iter_affine = max_iter_affine
        self.max_iter_mrf = max_iter_mrf
        self.max_iter_mixing = max_iter_mixing
        self.max_ls_bias = 6
        self.max_ls_warp = 12
        self.max_ls_affine = 6
        self.affine_maj = False
        self.tol = tol

        if not self.do_bias:
            self.max_iter_bias = 0
        if not self.do_warp:
            self.max_iter_warp = 0
        if not self.do_affine:
            self.max_iter_affine = 0
        if not self.do_mixing:
            self.max_iter_mixing = 0
        if self.do_mrf != 'learn':
            self.max_iter_mrf = 0
        if self.max_iter_mrf == 0 and self.max_iter_bias == 0:
            self.max_iter_intensity = max(2, self.max_iter_intensity)

        # Verbosity
        self.verbose = verbose
        self.plot = plot

    @property
    def nb_classes(self):
        return max(x for x in self.lkp) + 1

    @property
    def nb_clusters(self):
        return len(self.lkp)

    @property
    def nb_clusters_per_class(self):
        return [self.lkp.count(k) for k in range(self.nb_clusters)]

    @property
    def bias(self):
        """Return the exponentiated bias field"""
        if self.beta is None:
            return None
        return self.beta.exp()

    @property
    def warp(self):
        """Return the deformation field"""
        # if we ever implement diffeomorphic warps, this should
        # return the exponentiated displacement field
        return self.alpha

    @property
    def affine(self):
        """Return the (exponentiated) affine matrix"""
        return linalg.expm(self.eta, self.affine_basis)

    @property
    def mixing(self):
        """Return the mixing proportions"""
        if self.gamma is None:
            return None
        return math.softmax(self.gamma, 0, implicit=(True, False))

    @property
    def mrf(self):
        """Return the MRF conditional probabilities"""
        if self.psi is None:
            return None
        return math.softmax(self.psi, 0, implicit=(True, False))

    @property
    def prior(self):
        if self.log_prior is None:
            return None
        return math.softmax(self.log_prior, 0, implicit=(True, False))

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
        warp : (*spatial, D) tensor, default=0
            Initial displacement
        bias : (C, *spatial) tensor, default=1
            Initial bias field
        mixing : (K,) tensor, default=1/K
            Initial mixing proportions
        submixing : (Ki,) tensor, default=1/Ki
            Initial within-class mixing proportions
        mrf : (K, K) tensor, default=1/K
            Initial MRF conditional probabilities

        Returns
        -------
        Z : (K, [*spatial]) tensor
            Class responsibilities
        lb : () tensor
            Final lower bound

        """
        with torch.random.fork_rng([] if X.device.type == 'cpu' else
                                   [X.device]):
            torch.random.manual_seed(1)
            return self._fit(X, W, aff, **kwargs)

    def _fit(self, X, W=None, aff=None, **kwargs):
        self.figure = None
        if self.verbose > 0:
            t0 = timer()  # Start timer
            if X.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(X.device)

        if not X.dtype.is_floating_point:
            print(f'Data type {X.dtype} not supported - converting to '
                  f'{torch.get_default_dtype()}.')
            X = X.to(torch.get_default_dtype())

        # Try to guess the number of spatial dimensions
        if self.log_prior is not None:
            dim = self.log_prior.dim() - 1
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
            X.masked_fill_(~W, 0)
        if W0 is not None:
            W = W.to(W0.dtype).mul_(W0.to(W.device))
        if W.dtype is torch.bool:
            W = W.all(dim=0)
        else:
            W = W.prod(dim=0)

        # default affine
        if aff is None:
            aff = spatial.affine_default(X.shape[1:])
        aff = aff.to(X.dtype)

        # Subsample
        # NOTE: we must perform integer-sized subsampling, otherwise
        # some averaging would be introduced, which would lower the
        # observed variance and make the estimated intensity parameters
        # over-confident when applied to fully-sampled data.
        X0, W0, aff0 = X, W, aff
        prev_spacing, prev_aff = 0, aff
        for n, spacing in enumerate(self.spacing):
            if len(self.spacing) > 1 and self.verbose > 0:
                title = f'LEVEL {n+1:2d}/{len(self.spacing)}: {spacing:g} mm'
                print('#'*36)
                print('#' + title.center(34) + '#')
                print('#'*36)

            if spacing != prev_spacing:
                vx = spatial.voxel_size(aff0)
                factor = (spacing / vx).tolist()
                factor = [int(max(f, 1)//1) for f in factor]
                slicer = [slice(None, None, f) for f in factor]
                aff, _ = spatial.affine_sub(aff0, X0.shape[1:], tuple(slicer))
                X = X0[(Ellipsis, *slicer)]
                W = W0[tuple(slicer)].to(X.dtype)

            if n == 0:
                # Initialise model parameters (one gaussian per class)
                self._lkp = self.lkp
                self.lkp = list(range(self.nb_classes))
                self._init_parameters(X, W, aff, **kwargs)
            else:
                # Upsample previous parameters
                self._upsample(prev_aff, aff, X.shape[1:])
                if hasattr(self, 'wishart'):
                    self._init_wishart(X, W, aff)
                    self._update_lb_wishart()
            prev_spacing = spacing
            prev_aff = aff

            # EM loop
            Z, lb, all_lb = self._em(X, W, aff)

        # Estimate Z at highest resolution
        self._upsample(aff, aff0, X0.shape[1:])
        Z = self._final_e_step(X0, W0, aff0)

        # Print algorithm info
        if self.verbose > 0:
            v = (f'Algorithm finished in {len(lb)} iterations, '
                 f'log-likelihood = {lb[-1].item()}, '
                 f'runtime: {timer() - t0:0.1f} s, '
                 f'device: {X.device}')
            if X.device.type == 'cuda':
                vram_peak = torch.cuda.max_memory_allocated(X.device)
                vram_peak = int(vram_peak / 2 ** 20)
                v += f', peak VRAM: {vram_peak} MB'
            print(v)

        if self.plot > 0:
            M = self.warp_tpm(aff=aff0, mode='logit', shape=X0.shape[1:])
            self._plot_lb(all_lb, X0, Z, M, mode='final')

        return Z, all_lb[-1]

    def _upsample(self, aff, aff0, shape0):
        """Upsample bias and displacement to final high-resolution grid."""
        if not self.spacing:
            return
        factor = spatial.voxel_size(aff0) / spatial.voxel_size(aff)
        if self.beta is not None:
            self.beta = spatial.reslice(
                self.beta, aff, aff0, shape0,
                bound='dct2', interpolation=2,
                prefilter=False, extrapolate=True)
        if self.log_prior is not None:
            self.alpha = utils.movedim(self.alpha, -1, 0)
            self.alpha = spatial.reslice(
                self.alpha, aff, aff0, shape0,
                bound='dft', interpolation=2,
                prefilter=False, extrapolate=True)
            self.alpha = utils.movedim(self.alpha, 0, -1)
            self.alpha /= factor.to(self.alpha.device)

    def _final_e_step(self, X, W, aff):
        """Perform the final Expectation step"""
        XB = X
        if self.beta is not None:
            XB = self.beta.exp().mul_(XB)
        M = None
        if self.log_prior is not None:
            M = self.warp_tpm(aff=aff, mode='logit', shape=X.shape[1:])
        vx = spatial.voxel_size(aff)
        Z, _, _ = self.e_step(XB, W, M, vx=vx, combine=True)
        return Z

    def e_step(self, X, W=None, M=None, combine=False, vx=1):
        """Perform the Expectation step

        Parameters
        ----------
        X : (C, *spatial) tensor
            Observed image
        W : (*spatial) tensor, optional
            Weights
        M : (K-1, *spatial) tensor, optional
            Log-prior stationary prior

        Returns
        -------
        Z : (K, *spatial) tensor
            Responsibilities
        L : (K, *spatial) tensor
            MRF log-term
        lb : tensor

        """
        if combine:
            Z, L, lb = self.e_step(X, W, M, vx=vx)
            Z = self._z_combine(Z)
            return Z, L, lb

        N = X.shape[1:]
        L = X.new_empty((self.nb_clusters, *N))
        LMRF = None
        # --- likelihood ---
        for k, k0 in enumerate(self.lkp):
            L[k] = self._log_likelihood(X, cluster=k)
            if not k0:
                continue
            # --- prior (stationary component) ---
            L[k] += self.gamma[k0-1]
            # --- prior (nonstationary component) ---
            if M is not None:
                L[k] += M[k0-1]
        if self.do_mrf:
            if self.nb_clusters != self.nb_classes:
                Z = math.softmax(L, dim=0)
                Z = self._z_combine(Z)
                Lcomb = Z.log()
            else:
                Lcomb = L
                Z = math.softmax(Lcomb, dim=0)
            Z, LMRF = mrf(Z, self.psi, Lcomb, W, vx=vx, inplace=True)
            # Z0 now contains \sum_j log(pi)[j] @ Z[j]
            for k, k0 in enumerate(self.lkp):
                L[k] += LMRF[k0]
        # --- softmax ---
        Z, lb = math.softmax_lse(L, lse=True, weights=W, dim=0)
        return Z, LMRF, lb.cpu()

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
        all_lb : list[float]
            Lower bound at each step.

        """
        vx = spatial.voxel_size(aff) if aff is not None else None
        nW = W.sum().cpu() if W is not None else X[1:].numel()

        # --- prepare warped template and its gradients ----------------
        M = G = None
        if self.log_prior is not None:
            if self.do_warp:
                M, G = self.warp_tpm(aff=aff, mode='logit', shape=X.shape[1:],
                                     grad=True)
            else:
                M = self.warp_tpm(aff=aff, mode='logit', shape=X.shape[1:])

        # --- bias corrected volume ------------------------------------
        XB = X
        if self.beta is not None:
            XB = self.beta.exp().mul_(X)
        Z = None

        # --- deactivate MRF if needed ---------------------------------
        mrf = self.do_mrf
        if self.do_mrf not in ('always', 'learn'):
            mrf, self.do_mrf = self.do_mrf, False

        lw = lm = 0

        all_lb = []
        all_all_lb = []
        lb = -float('inf')
        plot_mode = None
        do_split = self.lkp != self._lkp
        for n_iter in range(self.max_iter):  # EM loop
            olb_em = lb

            for n_iter_intensity in range(self.max_iter_intensity):
                olb_intensity = lb

                if n_iter_intensity == 2 and do_split:
                    print('split')
                    do_split = False
                    self.lkp = self._lkp
                    self._split_clusters()
                    all_lb = []
                    all_all_lb = []
                    lb = -float('inf')

                for n_iter_cluster in range(self.max_iter_cluster):
                    # ======
                    # E-step
                    # ======
                    olb = lb
                    olw, olm = lw, lm
                    lw = getattr(self, '_lb_intensity', 0)
                    lm = getattr(self, '_lb_mixing', 0)
                    Z, L, lb = self.e_step(XB, W, M, vx=vx)
                    S, lse = self._make_prior(M, L, W)
                    lb -= lse
                    lb += self._lb_parameters()
                    all_all_lb.append(lb/nW)
                    if self.verbose >= 3:
                        gain = (lb - olb) / (self.tol * nW)
                        print(f'{n_iter:02d} | {n_iter_intensity:02d} | {n_iter_cluster:02d} | '
                              f'pre gmm:  {lb.item()/nW:12.6g} ({gain:.6g})')

                    # ==================
                    # M-step - Intensity
                    # ==================
                    ss0, ss1, ss2 = self._suffstats(XB, Z, W)
                    self._update_intensity(ss0, ss1, ss2)

                    if self.plot >= 4:
                        # plot after responsibilities have been combined
                        Z = self._z_combine(Z)
                        self._plot_lb(all_all_lb, X, Z, M, mode=plot_mode)
                    plot_mode = 'gmm' if plot_mode != 'bias' else 'bias'

                    # ===========================
                    # M-step - Mixing proportions
                    # ===========================
                    max_iter_mix = self.max_iter_mixing * (n_iter_intensity > 0)
                    for n_iter_mix in range(max_iter_mix):
                        if n_iter_mix > 0:
                            olb = lb
                            olw, olm = lw, lm
                            lw = getattr(self, '_lb_intensity', 0)
                            lm = getattr(self, '_lb_mixing', 0)
                            Z, L, lb = self.e_step(XB, W, M, vx=vx, combine=True)
                            S, lse = self._make_prior(M, L, W)
                            lb -= lse
                            lb += self._lb_parameters()
                            all_all_lb.append(lb/nW)
                            if self.verbose >= 3:
                                gain = (lb - olb) / (self.tol * nW)
                                print(f'{n_iter:02d} | {n_iter_intensity:02d} | {n_iter_mix:02d} | '
                                      f'pre mix:  {lb.item()/nW:12.6g} ({gain:.6g})')
                            if self.plot >= 4:
                                self._plot_lb(all_all_lb, X, Z, M, mode=plot_mode)

                        self._update_mixing(ss0, W, S)

                        if n_iter_mix > 1 and lb-olb < 0.5 * self.tol * nW:
                            break

                    if n_iter_cluster > 1 and lb-olb < self.tol * nW:
                        break

                # ============================
                # M-step - Markov Random Field
                # ============================
                max_iter_mrf = self.max_iter_mrf * (n_iter_intensity > 1)
                for n_iter_mrf in range(max_iter_mrf):
                    if n_iter_mrf > 0:
                        olb = lb
                        Z, L, lb = self.e_step(XB, W, M, vx=vx, combine=True)
                        S, lse = self._make_prior(M, L, W)
                        lb -= lse
                        lb += self._lb_parameters()
                        all_all_lb.append(lb/nW)
                        if self.verbose >= 3:
                            gain = (lb - olb) / (self.tol * nW)
                            print(f'{n_iter:02d} | {n_iter_intensity:02d} | {n_iter_mrf:02d} | '
                                  f'pre mrf:  {lb.item()/nW:12.6g} ({gain:.6g})')
                        if self.plot >= 4:
                            self._plot_lb(all_all_lb, X, Z, M, mode=plot_mode)

                    self._update_mrf(Z, W, S, vx)

                    if n_iter_mrf > 1 and lb-olb < 0.5 * self.tol * nW:
                        break

                for n_iter_bias in range(self.max_iter_bias):
                    # ======
                    # E-step
                    # ======
                    olb = lb
                    Z, L, lb = self.e_step(XB, W, M, vx=vx)
                    S, lse = self._make_prior(M, L, W)
                    lb -= lse
                    lb += self._lb_parameters()
                    all_all_lb.append(lb/nW)
                    if self.verbose >= 2:
                        gain = (lb - olb) / (self.tol * nW)
                        print(f'{n_iter:02d} | {n_iter_intensity:02d} | {n_iter_bias:02d} | '
                              f'pre bias: {lb.item()/nW:12.6g} ({gain:.6g})')
                    self._plot_lb(all_all_lb, X, self._z_combine(Z), M, mode=plot_mode)

                    # =============
                    # M-step - Bias
                    # =============
                    self._update_bias(XB, Z, W, vx=vx)
                    XB = torch.exp(self.beta, out=XB).mul_(X)
                    plot_mode = 'bias'
                    
                    if n_iter_bias > 1 and lb-olb < self.tol * nW:
                        break

                if n_iter_intensity > 1 and lb - olb_intensity < 2 * self.tol * nW:
                    break

            for n_iter_affine in range(self.max_iter_affine):
                # ======
                # E-step
                # ======
                olb = lb
                Z, L, lb = self.e_step(XB, W, M, combine=True, vx=vx)
                S, lse = self._make_prior(M, L, W)
                lb -= lse
                lb += self._lb_parameters()
                all_all_lb.append(lb/nW)
                if self.verbose >= 2:
                    gain = (lb - olb) / (self.tol * nW)
                    print(f'{n_iter:02d} | {n_iter_affine:02d} | {n_iter_affine:02d} | '
                          f'pre aff:  {lb.item()/nW:12.6g} ({gain:.6g})')
                self._plot_lb(all_all_lb, X, Z, M, mode=plot_mode)

                # ===============
                # M-step - Affine
                # ===============
                self._update_affine(Z, S, G, L, W, aff=aff)
                M, G = self.warp_tpm(aff=aff, mode='logit', shape=X.shape[1:],
                                     grad=True)
                plot_mode = 'warp'

                if n_iter_affine > 1 and lb - olb < self.tol * nW:
                    break

            for n_iter_warp in range(self.max_iter_warp):
                # ======
                # E-step
                # ======
                olb = lb
                Z, L, lb = self.e_step(XB, W, M, combine=True, vx=vx)
                S, lse = self._make_prior(M, L, W)
                lb -= lse
                lb += self._lb_parameters()
                all_all_lb.append(lb/nW)
                if self.verbose >= 2:
                    gain = (lb - olb) / (self.tol * nW)
                    print(f'{n_iter:02d} | {n_iter_warp:02d} | {n_iter_warp:02d} | '
                          f'pre warp: {lb.item()/nW:12.6g} ({gain:.6g})')
                self._plot_lb(all_all_lb, X, Z, M, mode=plot_mode)

                # =============
                # M-step - Warp
                # =============
                self._update_warp(Z, S, G, L, W, aff=aff)
                M, G = self.warp_tpm(aff=aff, mode='logit', shape=X.shape[1:],
                                     grad=True)
                plot_mode = 'warp'

                if n_iter_warp > 1 and lb - olb < self.tol * nW:
                    break

            # ==================
            # Log-likelihood
            # ==================
            all_lb.append(lb/nW)
            if n_iter > 1 and lb - olb_em < 2 * self.tol * nW:
                if self.verbose > 0:
                    print(f'converged: {(lb - olb_em) / (2 * nW):.6g}')
                break

        self.do_mrf = mrf
        all_lb = torch.stack(all_lb)            # per iteration
        all_all_lb = torch.stack(all_all_lb)    # per step
        return Z, all_lb, all_all_lb

    def _make_prior(self, M, L, W):
        """Compute softmaxed prior (atlas + mixing + mrf)"""
        if L is not None:
            L = L[1:]
        if M is not None:
            if L is not None:
                L = L + M
            else:
                L = M.clone()
        if L is not None:
            L = L.transpose(0, -1)
            L += self.gamma
            L = L.transpose(-1, 0)
        else:
            L = self.gamma.clone()
        L, lse = _softmax_lse(L, 0, W)
        return L, lse.cpu()

    def _suffstats(self, X, Z, W=None):
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
            0th moment
        ss1 : (K, C) tensor
            1st moment
        ss2 : (K, C, C) tensor
            2nd moment

        """
        C = X.shape[0]
        K = Z.shape[0]

        ss0 = X.new_empty((K,), dtype=torch.double)
        ss1 = X.new_empty((K, C), dtype=torch.double)
        ss2 = X.new_empty((K, C, C), dtype=torch.double)

        # Compute 1st and 2nd moments
        buffer = torch.empty_like(X[0])
        for k in range(K):
            ss0[k] = reduce(Z[k], W, buffer=buffer)
            for c in range(C):
                ss1[k, c] = reduce(X[c], Z[k], W, buffer=buffer)
                ss2[k, c, c] = reduce(X[c], X[c], Z[k], W, buffer=buffer)
                for cc in range(c + 1, C):
                    ss2[k, c, cc] = reduce(X[c], X[cc], Z[k], W, buffer=buffer)
                    ss2[k, cc, c] = ss2[k, c, cc]

        return ss0, ss1, ss2

    def _plot_lb(self, lb, X, Z, M=None, mode=None):
        if len(Z) > self.nb_classes:
            Z = self._z_combine(Z)
        if self.plot >= 1 and mode == 'final':
            self.plt_saved = None
            plot_images_and_lb(lb, X, Z, self.beta, M, self.alpha, self.gamma,
                               fig=getattr(self, 'figure', None))
        elif self.plot >= 3:
            self.figure, self.plt_saved = \
            plot_images_and_lb(lb, X, Z, self.beta, M, self.alpha, self.gamma,
                               mode=mode, fig=getattr(self, 'figure', None),
                               saved_elem=getattr(self, 'plt_saved', []))
        elif self.plot >= 2:
            self.figure, self.plt_saved = \
            plot_lb(lb, fig=getattr(self, 'figure', None),
                    saved_elem=getattr(self, 'plt_saved', []))

    def _lb_parameters(self):
        return (getattr(self, '_lb_warp', 0) +
                getattr(self, '_lb_bias', 0) +
                getattr(self, '_lb_mixing', 0) +
                getattr(self, '_lb_mrf', 0))

    # Implement in child classes
    def _log_likelihood(self, *a, **k):
        raise NotImplementedError('The observation log-likelihood must '
                                  'be implemented in concrete child classes')

    def _split_cluters(self):
        raise NotImplementedError('The method to split clusters must '
                                  'be implemented in concrete child classes')

    def _init_parameters(self, X, W=None, aff=None, **kwargs):
        C = X.shape[0]
        N = X.shape[1:]
        dim = X.dim() - 1
        backend = utils.backend(X)

        # change backend of pre-allocated tensors
        if self.log_prior is not None:
            self.log_prior = self.log_prior.to(**backend)

        # spatial deformation coefficients
        warp = kwargs.pop('warp', None)
        if warp is not None:
            self.alpha = torch.as_tensor(warp, **backend).expand([*N, dim]).clone()
        elif self.do_warp:
            self.alpha = torch.zeros([*N, dim], **backend)
        else:
            self.alpha = None

        # affine Lie coefficients
        affine = kwargs.pop('affine', None)
        self.affine_basis = spatial.affine_basis('affine', **backend)
        if affine is not None:
            affine = torch.as_tensor(affine, **backend)
            self.eta = spatial.affine_parameters(affine, self.affine_basis)
        elif self.do_affine:
            self.eta = torch.zeros([len(self.affine_basis)], **backend)
        else:
            self.eta = None

        # bias field coefficients (log)
        bias = kwargs.pop('bias', None)
        if bias is not None:
            self.beta = torch.as_tensor(bias, **backend).expand([C, *N]).log()
        elif self.do_bias:
            self.beta = torch.zeros([C, *N], **backend)
        else:
            self.beta = None

        # mrf coefficients (logit)
        mrf = kwargs.pop('mrf', None)
        if mrf is not None:
            psi_shape = [self.nb_classes] * 2
            self.psi = torch.as_tensor(mrf, **backend).expand(psi_shape)
            self.psi = math.logit(self.psi, 0, implicit=(False, True))
        # elif self.do_mrf == 'learn':
        #     psi_shape = [self.nb_classes - 1, self.nb_classes]
        #     self.psi = torch.zeros(psi_shape, **backend)
        elif self.do_mrf:
            self.psi = torch.eye(self.nb_classes, **backend)
            self.psi = self.psi[1:] - self.psi[:1]
        else:
            self.psi = None

        # mixing prop (extra -> logit)
        backend['dtype'] = torch.double
        gamma = kwargs.pop('gamma', None)
        if gamma is not None:
            self.gamma = torch.as_tensor(gamma, **backend).expand(self.nb_classes)
            self.gamma = math.logit(self.gamma, 0, implicit=(False, True))
        else:
            self.gamma = torch.zeros(self.nb_classes-1, **backend)

        # mixing prop (intra -> prob)
        backend['dtype'] = torch.double
        submixing = kwargs.pop('submixing', None)
        if submixing is not None:
            self.kappa = torch.as_tensor(submixing, **backend)
            self.kappa = self.kappa.expand(self.nb_clusters).clone()
        else:
            self.kappa = torch.ones(self.nb_clusters, **backend)
            for k, K1 in enumerate(self.nb_clusters_per_class):
                offset = sum(self.nb_clusters_per_class[:k])
                self.kappa[offset:offset+K1] /= K1

    def _update_mixing_old(self, ss0, M=None, W=None, Z=None):
        """
        Parameters
        ----------
        ss0 : (K,) tensor
            Zero-th statistic
        M : (K, *spatial) tensor, optional
            Non-stationary prior
        W : (*spatial) tensor, optional
            Observation weights
        Z : (K, *spatial) tensor, optional
            Responsibilities (if MRF)
        """
        if M is None and (not self.do_mrf or Z is None):
            ssm = None
        else:
            P = M

            gamma = self.gamma.to(P.dtype)
            P = P.reshape(self.nb_classes, -1)
            ssm = linalg.dot(P.T, gamma).reciprocal_()
            if W is not None:
                ssm *= W.flatten()
            ssm = linalg.matvec(P, ssm)
            ssm = ssm.to(self.gamma.dtype)

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

    def _update_mixing(self, ss0, W=None, M=None):
        """
        Parameters
        ----------
        ss0 : (K,) tensor
            Zero-th statistic
        W : (*spatial) tensor, optional
            Observation weights
        M : (K-1, *spatial) tensor, optional
            Modulated prior (atlas + mixing + MRF)
        """
        # within class mixing
        kappa = ss0.clone()
        for k in range(self.nb_classes):
            mask = [(l == k) for l in self.lkp]
            mask = torch.as_tensor(mask, dtype=torch.bool, device=kappa.device)
            norm = kappa[mask].sum()
            kappa = torch.where(mask, kappa / (norm + self._tiny), kappa)
        self.kappa.copy_(kappa)

        # across class mixing
        K = self.nb_classes
        ss0 = self._z_combine(ss0)
        A = torch.eye(K-1, **utils.backend(ss0)).sub_(1/K).mul_(0.5)
        N = ss0.sum()

        # suffstat prior
        if M.dim() == 1:
            M = N*M
        elif W is not None:
            if W.dtype is torch.bool:
                M = M[:, W].sum(-1)
            else:
                M = M.reshape([K-1, -1]).matmul(W.reshape([-1, 1]))[:, 0]
        else:
            M = M.reshape([K-1, -1]).sum(-1)

        # sum everything and solve
        delta_gamma = M - ss0[1:]
        if self.lam_mixing:  # regularization
            delta_gamma.add_(self.lam_mrf * self.gamma)
            A.diagonal(0, -1, -2).add_(self.lam_mrf)
        iA = linalg.inv(A)
        delta_gamma = linalg.matvec(iA, M - ss0[1:]) / N
        self.gamma -= delta_gamma

        if self.lam_mixing:
            self._lb_mixing = -0.5 * self.lam_mixing * self.gamma.square().sum().cpu()

    def _update_mrf(self, Z, W=None, M=None, vx=1):
        """

        Parameters
        ----------
        Z : (K, *spatial) tensor
            Responsibilities
        W : (*spatial) tensor, optional
            Observation weights
        M : (K-1, *spatial) tensor, optional
            Prior (atlas + mixing + MRF + softmax)
        vx : float, default=1
            Voxel size

        Returns
        -------
        L : (K-1, *spatial) tensor

        """

        def reduce(P, Q, W=None):
            P = P.reshape([len(P), -1])
            Q = Q.reshape([len(Q), -1])
            if W is not None:
                Q = Q * W.flatten()
            return P.matmul(Q.T)

        K = len(Z)
        A = torch.eye(K - 1, **utils.backend(Z)).sub_(1 / K).mul_(0.5)

        Zb = mrf_suffstat(Z, W, vx)
        ZZb = reduce(Zb, Zb, W) + mrf_covariance(Z, W, vx)

        delta_psi = reduce(M - Z[1:], Zb, W)
        if self.lam_mrf:  # regularization
            delta_psi.add_(self.lam_mrf * self.psi)
            A.diagonal(0, -1, -2).add_(self.lam_mrf)
            ZZb.diagonal(0, -1, -2).add_(self.lam_mrf)
        delta_psi = linalg.lmdiv(A, delta_psi)
        delta_psi = linalg.rmdiv(delta_psi, ZZb)

        self.psi -= delta_psi

        if self.lam_mrf:
            self._lb_mrf = -0.5 * self.lam_mrf * self.psi.square().sum().cpu()

    def _update_warp_base(self, Z, M, G, W=None):
        """Common bit for update_warp and update_affine

        Parameters
        ----------
        Z : (K, *spatial) tensor
            Responsibilities
        M : (K-1, *spatial) tensor
            Current modulated prior (atlas +  mixing + mrf + softmax)
        G : (K-1, *spatial, D) tensor
            Gradients of the log-TPM (eventually rotated)
        W : (*spatial) tensor, optional
            Voxel weights

        Returns
        -------
        g : (K, *spatial) tensor
            Gradient
        H : (K*(K+1)//2, *spatial) tensor
            Hessian
        ll0 : () scalar tensor
            Initial log-likelihood

        """
        def mul(*x, out=None):
            x = list(x)
            if out is None:
                out = x.pop(0).clone()
            else:
                out.copy_(x.pop(0))
            while x:
                out.mul_(x.pop(0))
            return out

        if len(M) == self.nb_classes:
            M = M[1:]
        if len(Z) == self.nb_classes:
            Z = Z[1:]
        if len(G) == self.nb_classes:
            G = G[1:]
        shape = Z.shape[1:]
        dim = len(shape)
        a = self.warp_acceleration

        # initial objective (for line search later)
        logM0 = M.sum(0).neg_().add_(1).log_()
        logM = M.log().sub_(logM0)
        if W is not None:
            ll0 = (W*logM0).sum() + logM.mul_(Z).mul_(W).sum()
        else:
            ll0 = logM0.sum() + logM.mul_(Z).sum()
        del logM

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

        if W is not None:
            g *= W.unsqueeze(-1)
            H *= W.unsqueeze(-1)

        return g, H, ll0

    def _update_warp(self, Z, M, G, L, W=None, aff=None):
        """Update dense deformation

        Parameters
        ----------
        Z : (K, *spatial) tensor
            Responsibilities
        M : (K-1, *spatial) tensor
            Current modulated prior (atlas +  mixing + mrf + softmax)
        G : (K-1, *spatial, D) tensor
            Gradients of the log-TPM
        L : (K, *spatial) tensor
            MRF log-term
        W : (*spatial) tensor, optional
            Voxel weights
        aff : (D+1,D+1) tensor, optional
            Orientation matrix

        """
        # rotate spatial gradients
        dim = G.shape[-1]
        rot = self._full_affine(aff)[:dim, :dim].T
        G = linalg.matvec(rot.to(G), G)

        if len(Z) == self.nb_classes:
            Z = Z[1:]
        g, H, ll0 = self._update_warp_base(Z, M, G, W)

        vx = spatial.voxel_size(aff)
        lam = {'factor': vx.prod(), 'voxel_size': vx, **self.lam_warp}
        La = spatial.regulariser_grid(self.alpha, **lam)
        aLa = reduce(self.alpha, La).cpu()
        g += La

        delta = spatial.solve_grid_fmg(H, g, **lam, nb_iter=4)
        if not self.max_ls_warp:
            self.alpha -= delta
            self._lb_warp = -0.5 * aLa
            return

        # line search
        dLd = spatial.regulariser_grid(delta, **lam)
        dLd = reduce(delta, dLd).cpu()
        dLa = reduce(delta, La).cpu()
        armijo, prev_armijo = 1, 0
        ll0 = ll0.cpu()
        success = False
        for n_ls in range(self.max_ls_warp):
            self.alpha.sub_(delta, alpha=armijo - prev_armijo)
            M = self.warp_tpm(aff=aff, mode='logit')
            M, _ = self._make_prior(M, L, W)
            logM0 = M.sum(0).neg_().add_(1).log_()
            logM = M.log_().sub_(logM0)
            ll = reduce(logM0, W) + reduce(logM, Z, W).cpu()
            if ll + armijo * (dLa - 0.5 * armijo * dLd) > ll0:
                success = True
                # print('success', n_ls)
                break
            prev_armijo, armijo = armijo, armijo/2
        if not success:
            # print('failure')
            self.alpha.add_(delta, alpha=armijo)
        else:
            self._lb_warp = getattr(self, '_lb_warp', 0)
            self._lb_warp += armijo * (dLa - 0.5 * armijo * dLd)

    def _update_affine(self, Z, M, G, L, W=None, aff=None):
        """Update affine transformation

        Parameters
        ----------
        Z : (K, *spatial) tensor
            Responsibilities
        M : (K-1, *spatial) tensor
            Current modulated prior (atlas +  mixing + mrf + softmax)
        G : (K-1, *spatial, D) tensor
            Gradients of the log-TPM
        L : (K, *spatial) tensor
            MRF log-term
        W : (*spatial) tensor, optional
            Voxel weights
        aff : (D+1,D+1) tensor, optional
            Orientation matrix

        """
        dim = G.shape[-1]
        if len(Z) == self.nb_classes:
            Z = Z[1:]
        g, H, ll0 = self._update_warp_base(Z, M, G, W)

        g_aff = self._affine_gradient
        aff = aff.to(g_aff)
        g_aff = linalg.lmdiv(self.affine_prior.to(g_aff), g_aff.matmul(aff))
        g_aff = g_aff[:, :-1, :].reshape([-1, dim*(dim+1)])
        g, H = affine_grid_backward(g, H)
        g = g.flatten()
        H = H.reshape([len(g), len(g)])
        g = linalg.matvec(g_aff, g.flatten())
        H = g_aff @ H @ g_aff.T
        if self.affine_maj:
            H = H.abs().sum(-1).diag_embed()

        delta = linalg.lmdiv(H, g.unsqueeze(-1)).squeeze(-1)
        if not self.max_ls_affine:
            self.eta -= delta
            return

        # line search
        armijo, prev_armijo = 1, 0
        ll0 = ll0.cpu()
        success = False
        eta0 = self.eta.clone()
        for n_ls in range(self.max_ls_affine):
            torch.sub(eta0, delta, alpha=armijo, out=self.eta)
            M = self.warp_tpm(aff=aff, mode='logit')
            M, _ = self._make_prior(M, L, W)
            logM0 = M.sum(0).neg_().add_(1).log_()
            logM = M.log_().sub_(logM0)
            ll = reduce(logM0, W) + reduce(logM, Z, W).cpu()
            if ll > ll0:
                success = True
                break
            prev_armijo, armijo = armijo, armijo/2
        if not success:
            self.eta.copy_(eta0)

    def _update_intensity(self, *a, **k):
        pass

    def _update_bias(self, *a, **k):
        pass

    @property
    def _affine_gradient(self):
        """Return the gradient of the exponentiated affine matrix wrt eta"""
        return linalg._expm(self.eta, self.affine_basis, grad_X=True)[1]

    def _full_affine_gradient(self, aff):
        """Derivative of the full affine (aff_prior \ (aff_align @ aff_dat))
        with respect to the Lie parameters if aff_align."""
        g_aff = self._affine_gradient
        g_aff = torch.matmul(g_aff, aff)
        g_aff = torch.matmul(self.affine_prior.inverse().to(aff), g_aff)
        return g_aff

    def _full_affine(self, aff):
        """Full affine matrix: aff_prior \ (aff_align @ aff_dat)"""
        if self.eta is not None:
            aff = torch.matmul(self.affine.to(aff), aff)
        aff = torch.matmul(self.affine_prior.inverse().to(aff), aff)
        return aff

    def warp_tpm(self, aff=None, grad=False, mode='softmax', shape=None):
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
        bound = 'replicate'
        # bound = 'zero'

        if self.log_prior is None:
            return None
        if self.alpha is not None:
            grid = spatial.add_identity_grid(self.alpha)
        else:
            grid = spatial.identity_grid(shape, **utils.backend(self.log_prior))
        if aff is not None:
            aff = self._full_affine(aff)
            grid = spatial.affine_matvec(aff.to(grid), grid)
        mask = self.log_prior.new_ones(self.log_prior.shape[1:])
        if mode == 'mask8':
            # remove bottom 5 mm
            if self.affine_prior is not None:
                vx = spatial.voxel_size(self.affine_prior)
            else:
                vx = grid.new_ones([grid.dim[-1]])
            offset = (5 / vx).floor().int().tolist()
            offset = [slice(o, None) for o in offset]
            mask[(Ellipsis, *offset)] = 0
        mask = spatial.grid_pull(mask, grid, bound=bound, extrapolate=False)
        if mode.startswith('mask'):
            return mask[None]

        M = spatial.grid_pull(self.log_prior, grid, bound=bound, extrapolate=True)
        # M[:, mask == 0] = M.min()  # uncomment for informative out-of-FOV
        if mode == 'softmax':
            M = math.softmax(M, dim=0, implicit=(True, False))
        elif mode == 'log_softmax':
            M = math.log_softmax(M, dim=0, implicit=(True, False))
        if not grad:
            return M
        G = spatial.grid_grad(self.log_prior, grid, bound=bound, extrapolate=True)
        return M, G


class UniSeg(SpatialMixture):
    """
    Unified Segmentation model, based on multivariate Gaussian Mixture Model (GMM).
    """

    def __init__(self, *args, wishart=True, lam_wishart=1, **kwargs):
        # wishart : bool or 'preproc8', default=True
        #   If True, use a Wishart prior derived from global suffstats.
        #   If 'preproc8', the bottom of the template FOV is discarded
        #   when estimating these suffstats.
        super().__init__(*args, **kwargs)
        self.wishart = wishart
        self.lam_wishart = lam_wishart

    @property
    def mean(self):
        return self.mu

    @property
    def covariance(self):
        return self.sigma

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
        backend = utils.backend(X)

        chol = self._chol(sigma)
        log_det = chol.diag().log().sum().mul(2).to(**backend)
        chol = chol.inverse().to(**backend)
        mu = mu.to(**backend)

        log_pdf = (X.reshape(C, -1).T - mu)
        log_pdf = linalg.matvec(chol, log_pdf).square_().sum(-1)
        log_pdf += log_det
        if self.wishart:
            df = self.wishart[1] + self.df[cluster].item()
            log_pdf += pymath.log(df) * C - mvdigamma(df/2, C)
        log_pdf *= -0.5
        log_pdf += self.kappa[cluster].log().to(**backend)
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
        if self.log_prior is not None:
            M = self.warp_tpm(mode=mode, aff=aff, shape=X.shape[1:])
            if W is not None:
                M *= W
        elif W is not None:
            M = W
        else:
            M = X.new_ones([1]*X.ndim)
        while M.ndim < X.ndim:
            M = M[None]
        Nw = X[0].numel() if W is None else W.sum(dtype=torch.double).item()

        # 2) Estimate the (diagonal) variance of the data, assuming
        #    a single Gaussian.
        ss0, ss1, ss2 = self._suffstats(X, W[None])
        ss0 = ss0[0].cpu()
        ss1 = ss1[0].cpu()
        ss2 = ss2[0].cpu().diag()
        scale = ss2 / ss0 - (ss1/ss0).square()

        # 3) Fiddle with degrees of freedom
        ndim = X.ndim - 1
        scale /= sum(self.nb_clusters_per_class) ** 2
        df = (Nw ** (1/ndim)) * self.lam_wishart
        df = max(df, len(X) - 1 + 1e-3)
        self.wishart = [scale.diag(), df]

    def _init_parameters(self, X, W=None, aff=None, **kwargs):
        """Initialise all parameters"""
        super()._init_parameters(X, W, aff, **kwargs)  # alpha, beta, gamma, kappa
        self._init_wishart(X, W, aff)

        K = self.nb_clusters
        C = X.shape[0]
        dim = X.dim() - 1
        backend = dict(device=X.device, dtype=torch.double)

        if self.log_prior is not None:
            # init from suffstats
            self.df = torch.empty((K,), **backend)
            self.mu = torch.empty((K, C), **backend)
            self.sigma = torch.empty((K, C, C), **backend)
            M = self.warp_tpm(aff=aff, shape=X.shape[1:])
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

            self.df = torch.zeros(K, **backend)
            self._update_lb_wishart()

    @staticmethod
    def _chol(x):
        if x.shape[-1] == 1:
            return x.sqrt()
        else:
            try:
                return linalg.cholesky(x)
            except RuntimeError:
                x = x.clone()
                diag = x.diagonal(0, -1, -2)
                diag.add_(diag.abs().max(), alpha=1e-3)
                return linalg.cholesky(x)

    def _split_clusters(self):
        # Heuristic to split a single Gaussians into multiple Gaussians.

        if len(self.mu) == self.nb_clusters:
            return

        mu0 = self.mu.cpu()
        sigma0 = self.sigma.cpu()
        df0 = self.df.cpu()
        kappa0 = self.kappa.cpu()
        K = self.nb_clusters

        mu = mu0.new_empty([K, *mu0.shape[1:]])
        sigma = sigma0.new_empty([K, *sigma0.shape[1:]])
        df = df0.new_empty([K, *df0.shape[1:]])
        kappa = kappa0.new_ones([K, *kappa0.shape[1:]])
        for k in range(self.nb_classes):
            mask = [k1 == k for k1 in self.lkp]
            K1 = sum(mask)
            w = 1. / (1 + pymath.exp(-(K1 - 1) * 0.25)) - 0.5
            chol = self._chol(sigma0[k]).diag()
            noise = torch.randn(K1, **utils.backend(mu)) * w
            mu[mask] = chol * noise[:, None] + mu0[k]
            sigma[mask] = sigma0[k] * (1 - w)
            df[mask] = df0[k] / K1
            kappa[mask] /= kappa[mask].sum(0, keepdim=True)

        self.mu = mu.to(self.mu)
        self.sigma = sigma.to(self.sigma)
        self.df = df.to(self.df)
        self.kappa = kappa.to(self.kappa)
        self._update_lb_wishart()

    def _update_lb_wishart(self):
        if not self.wishart:
            self._lb_intensity = 0
            return
        # update lower bound: KL between inverse wishart,
        # keeping only terms that depend on sigma
        sigma = self.sigma.cpu()
        nc = sigma.shape[-1]
        df = self.df.cpu()  # zero-th order suffstat (not true posterior df)
        sigma0, df0 = self.wishart

        # Kullbeck-Leibler divergence between inverse-Wishart distributions
        # 2*KL(q||p) = N0 * (logdet(S1) - logdet(S0)) 
        #            + N1 * tr(S1\S0)
        #            + 2 * (gammal(N0/2) - gammal(N1/2))
        #            + (N1 - N0) * digamma(N1/2) 
        #            - N1 * C
        #  
        # If we use Sigma1 = S1/N1 and Sigma0 = S0/N0, the first term becomes
        #     N0 * (logdet(Sigma1) - logdet(Sigma0)) + N0 * C * (log(N1) - log(N0))
        # and the second term becomes
        #     N0 * tr(Sigma1\Sigma0)

        chol0 = self._chol(sigma0)
        logdet0 = chol0.diag().log().sum() * 2
        lgamma0 = mvlgamma(df0/2, nc)

        lb = 0
        for k in range(len(sigma)):
            df1 = max(df0 + df[k].item(), nc-1 + 1e-3)
            sigma1 = sigma[k]
            chol1 = self._chol(sigma1)
            logdet1 = chol1.diag().log().sum() * 2
            trace = linalg.trace(torch.matmul(sigma0, sigma1.inverse()))
            # KL divergence DL(q1||q0) (x2)
            lb += (
                df0 * (logdet1 - logdet0)
                + df0 * nc * pymath.log(df1/df0)
                + df0 * trace +
                + 2 * (lgamma0 - mvlgamma(df1/2, nc))
                + (df1 - df0) * mvdigamma(df1/2, nc)
                - df1 * nc)
        self._lb_intensity = -0.5 * lb

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
        ss0 = ss0.cpu()
        ss1 = ss1.cpu()
        ss2 = ss2.cpu()

        # update means
        ss0 = ss0.unsqueeze(-1)
        mu = ss1 / ss0.clamp_min(1e-6)

        # update covariances
        ss0 = ss0.unsqueeze(-1)
        if not self.wishart:
            sigma = ss2 / ss0 - linalg.outer(mu, mu)
        else:
            sigma0, df0 = self.wishart
            sigma = df0 * sigma0 + ss2 - linalg.outer(ss1, ss1) / ss0.clamp_min(1e-6)
            sigma /= (df0 + ss0)

        self.mu.copy_(mu.to(self.mu))
        self.sigma.copy_(sigma.to(self.sigma))
        self.df.copy_(ss0[:, 0, 0].to(self.df))
        self._update_lb_wishart()

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
        C, *N = X.shape
        K = self.nb_clusters

        X = X.reshape([C, -1]).T
        Z = Z.reshape([K, -1])
        if W is not None:
            W = W.reshape([-1])
            Nw = W.sum(dtype=torch.double)
        else:
            Nw = len(X)

        g = X.new_empty(X.shape[0])
        H = X.new_empty(X.shape[0])
        a = 1 - self.bias_acceleration
        meandelta = 0
        for c in range(C):
            g.zero_()
            H.zero_()
            ll = 0
            # Solve for each channel individually
            for k in range(K):
                sigma = self.sigma[k]
                mu = self.mu[k].to(**utils.backend(X))
                # Determine each class' contribution to gradient and Hessian
                lam = sigma.inverse()[c].to(**utils.backend(X))
                g1 = linalg.matvec(lam, X - mu)
                ll += 2 * reduce(g1, X[:, c], Z[k], W)
                ll -= 2 * mu[c] * reduce(g1, Z[k], W)
                ll -= lam[c] * reduce(X[:, c], X[:, c], Z[k], W)
                ll -= lam[c] * mu[c] * mu[c] * reduce(Z[k], W)
                ll += 2 * lam[c] * mu[c] * reduce(X[:, c], Z[k], W)
                g.addcmul_(Z[k], g1)
                H.addcmul_(Z[k], lam[c])
                del g1
            ll *= -0.5
            g.mul_(X[:, c]).sub_(1)
            H.mul_(X[:, c]).mul_(X[:, c]).add_(1)
            if a > 0:
                H.add_(g.abs(), alpha=a)

            if W is not None:
                g *= W
                H *= W
            g = g.reshape([1, *N])
            H = H.reshape([1, *N])

            # TODO: Should we modulate by the relative voxel size, as we
            #       do in update_warp? Things seem to be a bit more tricky
            #       with pure bending, as resizing does not induce that much
            #       additional curvature.
            lam = {'bending': self.lam_bias, 'voxel_size': vx}
            g += spatial.regulariser(self.beta[None, c], **lam)

            delta = spatial.solve_field_fmg(H, g, **lam)
            del g, H

            dLd = spatial.regulariser(delta, **lam)
            dLb = reduce(dLd, self.beta[c]).item()
            dLd = reduce(dLd, delta).item()
            sumd = reduce(delta.flatten(), W).item()

            if self.max_ls_bias == 0:
                self.beta[c].sub_(delta[0])
                X[:, c] *= delta.flatten().exp()
                self._lb_bias = getattr(self, '_lb_bias', 0)
                self._lb_bias += (dLb - 0.5 * dLd)
                meandelta -= sumd
                continue

            # line search
            ll0 = ll.item()
            armijo, prev_armijo = 1, 0
            success = False
            for n_ls in range(self.max_ls_bias):
                X[:, c] *= delta.flatten().mul(prev_armijo - armijo).exp_()
                # compute log-likelihood of GMM
                ll = 0
                for k in range(K):
                    sigma = self.sigma[k]
                    mu = self.mu[k].to(**utils.backend(X))
                    lam = sigma.inverse()[c].to(**utils.backend(X))
                    g1 = linalg.matvec(lam, X - mu)
                    ll += 2 * reduce(g1, X[:, c], Z[k], W)
                    ll -= 2 * mu[c] * reduce(g1, Z[k], W)
                    ll -= lam[c] * reduce(X[:, c], X[:, c], Z[k], W)
                    ll -= lam[c] * mu[c] * mu[c] * reduce(Z[k], W)
                    ll += 2 * lam[c] * mu[c] * reduce(X[:, c], Z[k], W)
                    del g1
                ll *= -0.5
                ll = ll.item()
                # check improvement
                if ll + armijo * (dLb - 0.5 * armijo * dLd - sumd) > ll0:
                    success = True
                    break
                prev_armijo, armijo = armijo, armijo / 2
            if not success:
                # print('failure')
                X[:, c] *= delta.flatten().mul(armijo).exp_()
            else:
                # print('success', n_ls)
                self.beta[c].sub_(delta[0], alpha=armijo)
                self._lb_bias = getattr(self, '_lb_bias', 0)
                self._lb_bias += armijo * (dLb - 0.5 * armijo * dLd)
                meandelta -= armijo * sumd

        # zero-center bias fields + adapt mean/covariance
        meandelta /= Nw * C
        self.beta -= meandelta
        meandelta = pymath.exp(-meandelta)
        self.mu *= meandelta
        self.sigma *= meandelta * meandelta
        if self.wishart:
            self.wishart[0] *= meandelta * meandelta

# ---
# Numerically stable reduction
# ---


def multimul(*x, out=None):
    """Chained element-wise multiplication"""
    x = list(x)
    out = x.pop(0).clone() if out is None else out.copy_(x.pop(0))
    while x:
        x1 = x.pop(0)
        if x1 is not None:
            out.mul_(x1)
    return out


def reduce(*x, buffer=None):
    """Chained element-wise multiplication, followed by a stable sum"""
    return multimul(*x, out=buffer).sum(dtype=torch.double)


# ---
# Helper math functions (only defined for python scalars)
# ---

def digamma(x):
    # https://github.com/tminka/lightspeed/blob/master/digamma.m
    # --- special cases ---
    if x == float('inf') or x == float('nan'):
        return float('nan')
    if x == 0:
        return -float('inf')
    # --- negative: reflection formula ---
    if x < 0:
        return digamma(1-x) + 1/pymath.tan(-pymath.pi*x)
    # --- small: approximation ---
    small = 1e-6
    if x < small:
        d1 = -0.5772156649015328606065121  # = digamma(1)
        d2 = (pymath.pi*pymath.pi)/6
        return d1 - 1/x + d2*x
    # --- not large: reduce to digamma(x + n) where (x + n) is large 
    large = 9.5
    y = 0
    while x < large:
        y -= 1/x
        x += 1
    # --- large: Moivre's expansion ---
    s3 = 1/12
    s4 = 1/120
    s5 = 1/252
    s6 = 1/240
    s7 = 1/132
    r = 1/x
    y += pymath.log(x) - 0.5 * r
    r *= r
    y += - r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))))
    return y


def mvdigamma(x, order=1):
    y = 0
    for p in range(1, order + 1):
        y += digamma(x + (1 - p) / 2)
    return y


def mvlgamma(x, order=1):
    y = 0
    for p in range(1, order + 1):
        y += pymath.lgamma(x + (1 - p) / 2)
    return y



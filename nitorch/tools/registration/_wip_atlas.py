# from nitorch import spatial, io
# from nitorch.core import py, utils, linalg, math
# import torch
# from .utils import defaults_template, loadf, savef
# from .phantoms import demo_atlas
# from .utils import jg, jhj, defaults_velocity
# from . import plot as plt, optim as optm, losses, phantoms, utils as regutils
# from . import svf, shoot, smalldef
# import functools
#
#
# class VolumeTemplate:
#     """A utility object for fitting volumetric templates
#
#     Other possible templates could be SurfaceTemplate or ThetrahedronTemplate.
#     """
#
#     def __init__(self, shape=None, loss='mse', optim='relax', max_iter=16, lr=1,
#                  bound='dct2', penalty=None):
#         self.penalty = regutils.defaults_template(penalty)
#         self.optim = regutils.make_optim_grid(
#             optim, lr=lr, bound=bound, sub_iter=max_iter, **self.penalty)
#         self.loss = losses.make_loss(self.loss)
#         self.max_iter = max_iter
#         self.lr = lr
#         self.shape = shape
#         self.bound = bound
#
#     def get_derivatives(self, moving, fixed, grid):
#         dim = moving.dim() - 1
#         loss = losses.make_loss(self.loss)
#         warped = regutils.smart_pull(moving, grid, bound=self.bound)
#         if isinstance(self.optim, optm.SecondOrder):
#             _, *derivatives = loss.loss_grad_hess(warped, fixed)
#         else:
#             _, *derivatives = loss.loss_grad(moving, fixed)
#         derivatives = self.push_derivatives(*derivatives)
#         return derivatives
#
#     def push_derivatives(self, grid, *derivatives):
#         derivatives = list(derivatives)
#         for d in range(len(derivatives)):
#             derivatives[d] = regutils.smart_push(derivatives[d], grid,
#                                                  shape=self.shape,
#                                                  bound=self.bound)
#         return derivatives[0] if len(derivatives) == 1 else derivatives
#
#     def update(self, template, *derivatives):
#         optim = regutils.make_optim_field(self.optim, lr=self.lr,
#                                           sub_iter=self.max_iter,
#                                           **self.penalty)
#         return optim.update(template, optim.step(*derivatives))
#
#
# class _Atlas:
#     # that class does the job
#
#     def __init__(self, images, velocities, loss='mse', max_iter=10, model='svf',
#                  verbose=True, plot=False, dtype=None, device=None):
#         self.images = images
#         self.velocities = velocities
#         self.loss = loss
#         self.max_iter = max_iter
#         self.model = model
#         self.plot = plot
#         self.verbose = verbose
#         self.dtype = dtype
#         self.device = device
#
#     def fit(self):
#         self.init()
#         self.iter()
#
#     def init(self):
#         self.init_images()              # map input images
#         self.init_loss()                # instantiate an OptimizationLoss
#         self.init_deformation_model()   # instantiate a Register
#         self.init_template_model()      # instantiate a Template
#         self.init_velocities()          # initialize velocity fields
#         self.init_template()            # initialize template
#
#     def init_loss(self):
#         self.loss = losses.make_loss(self.loss, self.dim)
#
#     def init_images(self):
#
#         def get_shape(xx):
#             N = len(xx)
#             for x in xx:
#                 shape = x.shape
#                 break
#             return (N, *shape)
#
#         def get_backend(xx):
#             for x in xx:
#                 backend = utils.backend(x)
#                 break
#             return backend
#
#         self.images = [io.map(image) if isinstance(image, str) else image
#                        for image in self.images]
#         backend = get_backend(self.images)
#         self.dtype = self.dtype or backend['dtype']
#         self.device = self.device or backend['device']
#         self.backend = dict(dtype=self.dtype, device=self.device)
#         self.batch, self.channels, *self.shape = get_shape(self.images)
#         self.dim = len(self.shape)
#
#     def init_deformation_model(self):
#         model = (svf.Register if self.model == 'svf' else self.model)
#
#     def iter(self):
#
#
#
# class Atlas:
#     """Utility class to build an unbiased atlas"""
#
#     def __init__(self, loss='mse', max_iter=10, template='volume', model='svf',
#                  verbose=True, plot=False, dtype=None, device=None):
#         self.template = template
#         self.loss = loss
#         self.max_iter = max_iter
#         self.model = model
#         self.plot = plot
#         self.verbose = verbose
#         self.dtype = dtype
#         self.device = device
#
#     def __call__(self, images, velocities=None, **overload):
#         options = dict(self.__dict__)
#         options.update(overload)
#         return _Atlas(images, velocities, **options).fit()
#
#
#
# def init_template(images, loss='mse', optim='relax', velocities=None,
#                   lam=1, max_iter=1, hilbert=True, kernel=None, lr=1,
#                   **prm):
#     prm.pop('sub_iter', None)
#     defaults_template(prm)
#     prm['factor'] = lam
#
#     template = torch.zeros([])
#     if velocities is None:
#         velocities = [None] * len(images)
#
#     loss = losses.make_loss(loss)
#     optim = regutils.make_optim_field(optim, lr, sub_iter=max_iter,
#                                       kernel=kernel, **prm)
#
#     grad = 0
#     hess = 0
#     for image, vel in zip(images, velocities):
#         dim = image.dim() - 1
#         vel = regutils.smart_exp(vel)
#         image = regutils.smart_pull(image, vel, bound='dct2')
#         template = template.expand(image.shape).to(**utils.backend(images))
#         if isinstance(optim, optm.SecondOrder):
#             _, g, h = loss.loss_grad_hess(image, template, dim=dim)
#         else:
#             _, g = loss.loss_grad(image, template, dim=dim)
#         g = regutils.smart_push(g, vel, bound='dct2')
#         grad += g
#         if isinstance(optim, optm.SecondOrder):
#             h = regutils.smart_push(h, vel, bound='dct2')
#             hess += h
#
#     if hilbert and hasattr(optim, 'preconditioner'):
#         # Hilbert gradient
#         if kernel is None:
#             shape = grad.shape[-dim-1:-1]
#             kernel = spatial.greens(shape, **prm, **utils.backend(grad))
#         optim.preconditioner = lambda x: spatial.greens_apply(x, kernel)
#
#     return optim.update(template.clone(), optim.step(grad, hess))
#
#
# def update_template(template, grad, hess=None, optim='relax',
#                     lam=1, max_iter=1, hilbert=True, kernel=None, lr=1, **prm):
#     prm.pop('sub_iter', None)
#     shape = template.shape[1:]
#
#     defaults_template(prm)
#     prm['factor'] = lam
#
#     optim = regutils.make_optim_field(optim, lr, sub_iter=max_iter,
#                                       kernel=kernel, **prm)
#
#     if hilbert and hasattr(optim, 'preconditioner'):
#         # Hilbert gradient
#         if kernel is None:
#             kernel = spatial.greens(shape, **prm, **utils.backend(grad))
#         optim.preconditioner = lambda x: spatial.greens_apply(x, kernel)
#
#     # add regularisation
#     rgrad = spatial.regulariser(template, **prm)
#     ll = (template * rgrad).sum()
#     ll = ll.item()
#     grad += rgrad
#
#     return ll, optim.update(template, optim.step(grad, hess))
#
#
# def center_velocities(velocities):
#     """Subtract mean across velocities"""
#     # compute mean
#     meanvel = 0
#     for n in range(len(velocities)):
#         meanvel += loadf(velocities[n])
#     meanvel /= len(velocities)
#
#     # subtract mean
#     for n in range(len(velocities)):
#         vel = loadf(velocities[n])
#         vel -= meanvel
#         savef(vel, velocities[n])
#
#     return velocities
#
#
# def update_velocity(fixed, moving, **kwargs):
#     """Diffeomorphic registration between two images using SVFs.
#
#     Parameters
#     ----------
#     fixed : (K, *spatial) tensor
#         Fixed image
#     moving : (K, *spatial) tensor
#         Moving image
#     lam : float, default=1
#         Modulate regularisation
#     max_iter : int, default=100
#         Maximum number of Gauss-Newton or Gradient descent optimisation
#     loss : {'mse', 'cat'}, default='mse'
#         'mse': Mean-squared error
#         'cat': Categorical cross-entropy
#     optim : {'relax', 'cg', 'gd', 'gdh'}, default='relax'
#         'relax': Gauss-Newton (linear system solved by relaxation)
#         'cg': Gauss-Newton (linear system solved by conjugate gradient)
#         'gd': Gradient descent
#         'gdh': Hilbert gradient descent
#     sub_iter : int, default=16
#         Number of relax/cg iterations per GN step
#     absolute : float, default=1e-4
#         Penalty on absolute displacements
#     membrane : float, default=1e-3
#         Penalty on first derivatives
#     bending : float, default=0.2
#         Penalty on second derivatives
#     lame : (float, float), default=(0.05, 0.2)
#         Penalty on zooms and shears
#
#     Returns
#     -------
#     ll : float
#         Log-likelihood
#     gmoving : (K, *spatial) tensor
#         Gradient with respect to moving image
#     hmoving : (K*(K+1)//2, *spatial) tensor
#         Hessian with respect to moving image
#     velocity : (*spatial, dim) tensor
#         Stationary velocity field.
#
#     """
#     kwargs['verbose'] = False
#     velocity = register(fixed, moving, **kwargs)
#
#     # compute template derivatives
#     loss = kwargs.get('loss', 'mse')
#     loss = losses.make_loss(loss)
#     grid = regutils.smart_exp(velocity)
#     warped = regutils.smart_pull(moving, grid, bound='dct2', extrapolate=True)
#     ll, grad, hess = loss.loss_grad_hess(fixed, warped)
#     grad = spatial.grid_push(grad, grid, bound='dct2')
#     hess = spatial.grid_push(hess, grid, bound='dct2')
#
#     return ll, grad, hess, velocity
#
#
# def atlas(images=None, c, max_inner=8, loss='mse', plot=False,
#           prm_template=None, prm_velocity=None, dtype=None, device=None):
#     """Diffeomorphic registration between two images using SVFs.
#
#     Parameters
#     ----------
#     images : iterator of (K, *spatial) tensor
#         Batch of images
#     max_outer : int, default=10
#         Maximum number of outer (template update) iterations
#     max_inner : int, default=8
#         Maximum number of inner (registration) iterations
#     loss : {'mse', 'cat'}, default='mse'
#         'mse': Mean-squared error
#         'cat': Categorical cross-entropy
#     prm_template : dict
#         optim : str, default='relax'
#         sub_iter : int, default=8
#         lam : float, default=1
#             Modulate regularisation
#         absolute : float, default=1e-4
#             Penalty on absolute displacements
#         membrane : float, default=0.08
#             Penalty on first derivatives
#         bending : float, default=0.8
#             Penalty on second derivatives
#     prm_velocity : dict
#         optim : str, default='ogm'
#         sub_iter : int, default=8
#         lam : float, default=1
#             Modulate regularisation
#         absolute : float, default=1e-4
#             Penalty on absolute displacements
#         membrane : float, default=1e-3
#             Penalty on first derivatives
#         bending : float, default=0.2
#             Penalty on second derivatives
#         lame : (float, float), default=(0.05, 0.2)
#             Penalty on zooms and shears
#
#     Returns
#     -------
#     template : (K, *spatial) tensor
#         Template
#     vel : (N, *spatial, dim) tensor or MappedArray
#         Stationary velocity fields.
#
#     """
#     prm_velocity = prm_velocity or dict()
#     defaults_velocity(prm_velocity)
#     prm_template = prm_template or dict()
#     defaults_template(prm_template)
#     optim_velocity = dict(optim=prm_velocity.pop('optim', 'ogm'),
#                           max_iter=max_inner,
#                           sub_iter=prm_velocity.pop('sub_iter', 8),
#                           lr=prm_velocity.pop('lr', 1),
#                           ls=prm_velocity.pop('ls', 6))
#     optim_template = dict(optim=prm_template.pop('optim', 'relax'),
#                           sub_iter=prm_template.pop('sub_iter', 8),
#                           lr=prm_template.pop('lr', 1))
#     if optim_template['optim'] in ('cg', 'relax'):
#         optim_template['hilbert'] = False
#
#     def get_shape(xx):
#         N = len(xx)
#         for x in xx:
#             shape = x.shape
#             break
#         return (N, *shape)
#
#     def get_backend(xx):
#         for x in xx:
#             backend = utils.backend(x)
#             break
#         return backend
#
#     # If no inputs provided: demo "squircles"
#     if images is None:
#         images = demo_atlas(batch=8, cat=(loss == 'cat'))
#
#     backend = get_backend(images)
#     backend = dict(dtype=dtype or backend['dtype'],
#                    device=device or backend['device'])
#     N, K, *shape = get_shape(images)
#     dim = len(shape)
#
#     # allocate velocities (option to save on disk?)
#     velshape = [N, *shape, dim]
#     vels = torch.zeros(velshape, **backend)
#
#     # Greens kernel for Hilbert gradient
#     optim_velocity['kernel'] = spatial.greens(shape, **prm_velocity)
#
#     template = init_template(images, loss=loss, **optim_template, **prm_template)
#     if plot:
#         import matplotlib.pyplot as plt
#         plt.imshow(template[0])
#         plt.show()
#
#     ll_prev = None
#     for n_iter in range(max_outer):
#         # register
#         grad = 0
#         hess = 0
#         llv = 0
#         for n, image in enumerate(images):
#             vel = loadf(vels[n])
#             l, g, h, vel = update_velocity(
#                 image, template, velocity=vel, loss=loss,
#                 **optim_velocity, **prm_velocity)
#             savef(vel, vels[n])
#             llv += l
#             grad += g
#             hess += h
#         print('')
#
#         # center velocities
#         vels = center_velocities(vels)
#
#         # update template
#         llt, template = update_template(
#             template, grad, hess,
#             **optim_template, **prm_template)
#
#         if plot:
#             import matplotlib.pyplot as plt
#             plt.imshow(template[0])
#             plt.show()
#
#         llv /= images.numel()
#         llt /= images.numel()
#         ll = llt + llv
#         if ll_prev is None:
#             ll_prev = ll
#             ll_max = ll
#             print(f'{n_iter:03d} | {llv:12.6g} + {llt:12.6g} = {ll:12.6g}')
#         else:
#             gain = (ll_prev - ll) / max(abs(ll_max - ll), 1e-8)
#             ll_prev = ll
#             print(f'{n_iter:03d} | {llv:12.6g} + {llt:12.6g} = {ll:12.6g} | gain = {gain:12.6g}')
#
#     return template, vel
#

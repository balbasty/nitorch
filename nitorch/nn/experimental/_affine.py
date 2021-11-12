import torch
import torch.nn as tnn
from nitorch.nn.base import Module
from ..modules.cnn import UNet, CNN
from ..modules.spatial import GridPull, GridExp, GridResize, AffineGrid, \
    AffineExp, AffineLog, AffineClassic, \
    AffineClassicInverse
from .. import check
from nitorch import spatial
from nitorch.core.linalg import matvec
from nitorch.core.utils import unsqueeze, channel2last, last2channel
from nitorch.core.py import make_list


class AffineMorph(Module):
    """Affine registration network.

    This network builds on VoxelMorph, but replaces the U-Net by an
    encoding CNN, and the dense spatial transformer by an affine spatial
    transformer. Like VoxelMorph, this network encodes deformation on
    their tangent space: here, the Lie algebra of a variety of affine
    Lie groups is used. Affine transformation matrices are recovered
    from their Lie algebra representation using matrix exponentiation.

    * VoxelMorph:
        target |-(unet)-> velocity -(exp)-> grid -(pull)-> warped_source
        source |------------------------------------^
    * AffineMorph:
        target |-(cnn)-> lieparam -(exp)-> affine -(pull)-> warped_source
        source |-------------------------------------^
    """

    def __init__(self, dim, basis='CSO', encoder=None, stack=None,
                 kernel_size=3, interpolation='linear', bound='dct2', *,
                 _additional_input_channels=0, _additional_output_channels=0):
        """

        Parameters
        ----------
        dim : int
            Dimensionalityy of the input (1|2|3)
        basis : {'T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'}, default='CSO'
            Basis of a matrix Lie group:
                * 'T'   : Translations
                * 'SO'  : Special Orthogonal (rotations)
                * 'SE'  : Special Euclidean (translations + rotations)
                * 'D'   : Dilations (translations + isotropic scalings)
                * 'CSO' : Conformal Special Orthogonal
                          (translations + rotations + isotropic scalings)
                * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
                * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
                * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
        encoder : list[int], optional
            Number of channels after each encoding layer of the CNN.
        stack : list[int], optional
            Number of channels after each fully-connected layer of the CNN.
        kernel_size : int or list[int], default=3
            Kernel size of the UNet.
        interpolation : int, default=1
            Interpolation order.
        bound : bound_type, default='dct2'
            Boundary conditions of the image.
        """

        super().__init__()
        exp = AffineExp(dim, basis=basis)
        nb_prm = sum(
            b.shape[0] for b in exp.basis) + _additional_output_channels
        self.cnn = CNN(dim,
                       input_channels=2 + _additional_input_channels,
                       output_channels=nb_prm,
                       encoder=encoder,
                       stack=stack,
                       kernel_size=kernel_size,
                       activation=tnn.LeakyReLU(0.2),
                       final_activation=None)
        self.exp = exp
        self.grid = AffineGrid(shift=True)
        self.pull = GridPull(interpolation=interpolation,
                             bound=bound,
                             extrapolate=False)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'affine']

    def forward(self, source, target, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        affine_prm : tensor (batch,, *spatial, len(spatial))
            affine Lie parameters

        """
        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        affine_prm = self.cnn(source_and_target)
        affine_prm = affine_prm.reshape(affine_prm.shape[:2])
        affine = []
        for prm in affine_prm:
            affine.append(self.exp(prm))
        affine = torch.stack(affine, dim=0)
        grid = self.grid(affine, shape=target.shape[2:])
        deformed_source = self.pull(source, grid)

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     affine=[affine_prm])

        return deformed_source, affine_prm


class AffineMorphSemiSupervised(AffineMorph):
    """An AffineMorph network with a Categorical loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags += ['segmentation']

    def forward(self, source, target, source_seg=None, target_seg=None,
                *, _loss=None, _metric=None):

        # sanity checks
        check.dim(self.dim, source, target, source_seg, target_seg)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim + 2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        affine_prm = self.cnn(source_and_target)
        affine_prm = affine_prm.reshape(affine_prm.shape[:2])
        affine = []
        for prm in affine_prm:
            affine.append(self.exp(prm))
        affine = torch.stack(affine, dim=0)
        grid = self.grid(affine, shape=target.shape[2:])
        deformed_source = self.pull(source, grid)
        if source_seg is not None:
            if source_seg.shape[2:] != source.shape[2:]:
                grid = spatial.resize_grid(grid, shape=source_seg.shape[2:])
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = None

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     affine=[affine_prm],
                     segmentation=[deformed_source_seg, target_seg])

        if source_seg is None:
            return deformed_source, affine_prm
        else:
            return deformed_source, deformed_source_seg, affine_prm


class AffineVoxelMorph(Module):
    """Affine + diffeo registration network.
    """

    def __init__(self, dim, basis='CSO',
                 encoder=None, decoder=None, stack=None, kernel_size=3,
                 interpolation='linear', image_bound='dct2', grid_bound='dft',
                 downsample_velocity=2, *, _input_channels=2):

        super().__init__()

        resize_factor = make_list(downsample_velocity, dim)
        resize_factor = [1 / f for f in resize_factor]

        affexp = AffineExp(dim, basis=basis)
        nb_prm = sum(b.shape[0] for b in affexp.basis)
        self.cnn = CNN(dim,
                       input_channels=2,
                       output_channels=nb_prm,
                       encoder=encoder,
                       stack=stack,
                       kernel_size=kernel_size,
                       activation=tnn.LeakyReLU(0.2),
                       final_activation=None)
        self.affexp = affexp
        self.unet = UNet(dim,
                         input_channels=_input_channels,
                         output_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.resize = GridResize(interpolation=interpolation,
                                 bound=grid_bound,
                                 factor=resize_factor)
        self.velexp = GridExp(interpolation=interpolation,
                              bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound,
                             extrapolate=False)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'velocity', 'affine', 'segmentation']

    def exp(self, velocity, affine=None, displacement=False):
        """Generate a deformation grid from tangent parameters.

        Parameters
        ----------
        velocity : (batch, *spatial, nb_dim)
            Stationary velocity field
        affine : (batch, nb_prm)
            Affine parameters
        displacement : bool, default=False
            Return a displacement field (voxel to shift) rather than
            a transformation field (voxel to voxel).

        Returns
        -------
        grid : (batch, *spatial, nb_dim)
            Deformation grid (transformation or displacment).

        """
        info = {'dtype': velocity.dtype, 'device': velocity.device}

        # generate grid
        shape = velocity.shape[1:-1]
        velocity_small = self.resize(velocity, type='displacement')
        grid = self.velexp(velocity_small)
        grid = self.resize(grid, shape=shape, type='grid')

        if affine is not None:
            # exponentiate
            affine_prm = affine
            affine = []
            for prm in affine_prm:
                affine.append(self.affexp(prm))
            affine = torch.stack(affine, dim=0)

            # shift center of rotation
            affine_shift = torch.cat((
                torch.eye(self.dim, **info),
                -torch.as_tensor(shape, **info)[:, None] / 2),
                dim=1)
            affine = spatial.affine_matmul(affine, affine_shift)
            affine = spatial.affine_lmdiv(affine_shift, affine)

            # compose
            affine = unsqueeze(affine, dim=-3, ndim=self.dim)
            lin = affine[..., :self.dim, :self.dim]
            off = affine[..., :self.dim, -1]
            grid = matvec(lin, grid) + off

        if displacement:
            grid = grid - spatial.identity_grid(grid.shape[1:-1], **info)

        return grid

    def forward(self, source, target, source_seg=None, target_seg=None,
                *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        affine_prm : tensor (batch,, *spatial, len(spatial))
            affine Lie parameters

        """
        # sanity checks
        check.dim(self.dim, source, target, source_seg, target_seg)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim + 2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)

        # generate affine
        affine_prm = self.cnn(source_and_target)
        affine_prm = affine_prm.reshape(affine_prm.shape[:2])

        # generate velocity
        velocity = self.unet(source_and_target)
        velocity = channel2last(velocity)

        # generate deformation grid
        grid = self.exp(velocity, affine_prm)

        # deform
        deformed_source = self.pull(source, grid)
        if source_seg is not None:
            if source_seg.shape[2:] != source.shape[2:]:
                grid = spatial.resize_grid(grid, shape=source_seg.shape[2:])
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = None

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     velocity=[velocity],
                     segmentation=[deformed_source_seg, target_seg],
                     affine=[affine_prm])

        if deformed_source_seg is None:
            return deformed_source, velocity, affine_prm
        else:
            return deformed_source, deformed_source_seg, velocity, affine_prm


class DenseToAffine(Module):
    """Convert a dense displacement field to an affine matrix"""

    def __init__(self, shift=True):
        """

        Parameters
        ----------
        shift : bool, default=True
            Apply a shift so that the center of rotation is in the
            center of the field of view.
        """
        super().__init__()
        self.shift = shift

    def forward(self, grid, **overload):
        """

        Parameters
        ----------
        grid : (N, *spatial, dim)
            Displacement grid
        overload : dict

        Returns
        -------
        aff : (N, dim+1, dim+1)
            Affine matrix that is closest to grid in the least square sense

        """
        shift = overload.get('shift', self.shift)
        grid = torch.as_tensor(grid)
        info = dict(dtype=grid.dtype, device=grid.device)
        nb_dim = grid.shape[-1]
        shape = grid.shape[1:-1]

        if shift:
            affine_shift = torch.cat((
                torch.eye(nb_dim, **info),
                -torch.as_tensor(shape, **info)[:, None] / 2),
                dim=1)
            affine_shift = spatial.as_euclidean(affine_shift)

        # the forward model is:
        #   phi(x) = M\A*M*x
        # where phi is a *transformation* field, M is the shift matrix
        # and A is the affine matrix.
        # We can decompose phi(x) = x + d(x), where d is a *displacement*
        # field, yielding:
        #   d(x) = M\A*M*x - x = (M\A*M - I)*x := B*x
        # If we write `d(x)` and `x` as large vox*(dim+1) matrices `D`
        # and `G`, we have:
        #   D = G*B'
        # Therefore, the least squares B is obtained as:
        #   B' = inv(G'*G) * (G'*D)
        # Then, A is
        #   A = M*(B + I)/M
        #
        # Finally, we project the affine matrix to its tangent space:
        #   prm[k] = <log(A), B[k]>
        # were <X,Y> = trace(X'*Y) is the Frobenius inner product.

        def igg(identity):
            # Compute inv(g*g'), where g has homogeneous coordinates.
            #   Instead of appending ones, we compute each element of
            #   the block matrix ourselves:
            #       [[g'*g,   g'*1],
            #        [1'*g,   1'*1]]
            #    where 1'*1 = N, the number of voxels.
            g = identity.reshape([identity.shape[0], -1, nb_dim])
            nb_vox = torch.as_tensor([[[g.shape[1]]]], **info)
            sumg = g.sum(dim=1, keepdim=True)
            gg = torch.matmul(g.transpose(-1, -2), g)
            gg = torch.cat((gg, sumg), dim=1)
            sumg = sumg.transpose(-1, -2)
            sumg = torch.cat((sumg, nb_vox), dim=1)
            gg = torch.cat((gg, sumg), dim=2)
            return gg.inverse()

        def gd(identity, disp):
            # compute g'*d, where g and d have homogeneous coordinates.
            #       [[g'*d,   g'*1],
            #        [1'*d,   1'*1]]
            g = identity.reshape([identity.shape[0], -1, nb_dim])
            d = disp.reshape([disp.shape[0], -1, nb_dim])
            nb_vox = torch.as_tensor([[[g.shape[1]]]], **info)
            sumg = g.sum(dim=1, keepdim=True)
            sumd = d.sum(dim=1, keepdim=True)
            gd = torch.matmul(g.transpose(-1, -2), d)
            gd = torch.cat((gd, sumd), dim=1)
            sumg = sumg.transpose(-1, -2)
            sumg = torch.cat((sumg, nb_vox), dim=1)
            sumg = sumg.expand([d.shape[0], sumg.shape[1], sumg.shape[2]])
            gd = torch.cat((gd, sumg), dim=2)
            return gd

        def eye(d):
            x = torch.eye(d, **info)
            z = x.new_zeros([1, d], **info)
            x = torch.cat((x, z), dim=0)
            z = x.new_zeros([d + 1, 1], **info)
            x = torch.cat((x, z), dim=1)
            return x

        identity = spatial.identity_grid(shape, **info)[None, ...]
        affine = torch.matmul(igg(identity), gd(identity, grid))
        affine = affine.transpose(-1, -2) + eye(nb_dim)
        affine = affine[..., :-1, :]
        if shift:
            affine = spatial.as_euclidean(affine)
            affine = spatial.affine_matmul(affine_shift, affine)
            affine = spatial.as_euclidean(affine)
            affine = spatial.affine_rmdiv(affine, affine_shift)
        affine = spatial.affine_make_square(affine)

        return affine


class AffineMorphFromDense(Module):
    """Affine registration network.

    We predict local displacements using a UNet and then find the affine
    matrix that is the closest to this displacement field in the least
    square sense. Finally, we project the matrix to a Lie algebra.
    """

    def __init__(self, dim, group='CSO', mode='lie', encoder='leastsquares', 
                 unet=None, pull=None):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimensionality of the input.
            
        group : {'T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'}, default='CSO'
            Affine group encoded:
            
            - 'T'   : Translations
            - 'SO'  : Special Orthogonal (rotations)
            - 'SE'  : Special Euclidean (translations + rotations)
            - 'D'   : Dilations (translations + isotropic scalings)
            - 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            - 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
            - 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            - 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
                
        mode : {'lie', 'classic'}, default='lie'
            Encoding of the affine parameters.
            The basis 'SL' is only available in mode 'lie'.

        encoder : {'leastsquares', 'cnn'} or dict, default='leastsquares'
            Encoder used to transform the dense displacement field
            into affine parameters. If 'leastsquares', compute the 
            least squares affine matrix and convert it into parameters
            using `AffineLog` or `AffineClassicInverse`. If a'cnn' (or 
            a dictionary of CNN options), use an encoding CNN that takes 
            as input the displacement field and an identity grid and 
            directy outputs the affine parameters.

        unet : dict, optional
            Dictionary of UNet parameters, with fields:
            
            - kernel_size : int, default=3
                Kernel size of the UNet.
            - encoder : list[int], default=[16, 32, 32, 32]
                Number of channels after each encoding layer.
            - decoder : list[int], default=[32, 32, 32, 32, 32, 16, 16]
                Number of channels after each decoding layer.
            - batch_norm : bool, default=True
                Use batch normalization in the UNet.
            - activation : callable, default=LeakyReLU(0.2)
                Activation function.
        
        pull : dict, optional
            Dictionary of GridPull parameters, with fields:
            
            - interpolation : int, default=1
                Interpolation order.
            - bound : bound_type, default='dct2'
                Boundary conditions of the image.
            - extrapolate : bool, default=False
                Extrapolate data outside of the field of view.
        """

        # default parameters for the submodules
        
        unet = unet or dict()
        unet['encoder'] = unet.get('encoder', None)
        unet['decoder'] = unet.get('decoder', None)
        unet['kernel_size'] = unet.get('kernel_size', 3)
        unet['batch_norm'] = unet.get('batch_norm', False)
        unet['activation'] = unet.get('activation', tnn.LeakyReLU(0.2))
        
        pull = pull or dict()
        pull['interpolation'] = pull.get('interpolation', 1)
        pull['bound'] = pull.get('bound', 'dct2')
        pull['extrapolate'] = pull.get('extrapolate', False)
        
        cnn = dict()
        if isinstance(encoder, dict):
            cnn = encoder
            encoder = 'cnn'
        if encoder == 'cnn':
            cnn['encoder'] = unet.get('encoder', None)
            cnn['stack'] = unet.get('stack', None)
            cnn['kernel_size'] = cnn.get('kernel_size', 3)
            cnn['batch_norm'] = cnn.get('batch_norm', False)
            cnn['reduction'] = cnn.get('reduction', 'max')
            cnn['activation'] = cnn.get('activation', tnn.LeakyReLU(0.2))
            cnn['final_activation'] = cnn.get('final_activation', 'same')
        
        # instantiate submodules
        
        super().__init__()
        self.unet = UNet(dim,
                         input_channels=2,
                         output_channels=dim,
                         encoder=unet['encoder'],
                         decoder=unet['decoder'],
                         norm=unet['batch_norm'],
                         kernel_size=unet['kernel_size'],
                         activation=unet['activation'])
        if encoder == 'leastsquares':
            self.dense2aff = DenseToAffine(shift=True)
            if mode == 'lie':
                self.log = AffineLog(basis=group)
            else:
                self.log = AffineClassicInverse(basis=group)
            self.dense2prm = self._dense2prm_ls
        else:
            nb_prm = spatial.affine_basis_size(group, dim)
            self.cnn = CNN(dim,
                           input_channels=2*dim,
                           output_channels=nb_prm,
                           encoder=cnn['encoder'],
                           stack=cnn['stack'],
                           norm=cnn['batch_norm'],
                           kernel_size=cnn['kernel_size'],
                           reduction=cnn['reduction'],
                           activation=cnn['activation'],
                           final_activation=cnn['final_activation'])
            self.dense2prm = self._dense2prm_cnn
            
        if mode == 'lie':
            self.exp = AffineExp(dim, basis=group)
        else:
            self.exp = AffineClassic(dim, basis=group)
        self.grid = AffineGrid(shift=True)
        self.pull = GridPull(interpolation=pull['interpolation'],
                             bound=pull['bound'],
                             extrapolate=pull['extrapolate'])
        self.dim = dim
        self.group = group
        self.encoder = encoder
        self.mode = mode

        # register losses/metrics
        self.tags = ['image', 'dense', 'affine']

    @staticmethod
    def _identity(x):
        """Build an identity grid with same shape/backend as a tensor.
        The grid is built such that coordinate zero is at the center of 
        the FOV."""
        shape = x.shape[2:]
        backend = dict(dtype=x.dtype, device=x.device)
        grid = spatial.identity_grid(shape, **backend)
        grid -= torch.as_tensor(shape, **backend)/2.
        grid /= torch.as_tensor(shape, **backend)/2.
        grid = last2channel(grid[None, ...])
        return grid
    
    def _dense2prm_ls(self, x):
        """Least-squares implementation of dense2prm."""
        shape = x.shape[2:]
        backend = dict(dtype=x.dtype, device=x.device)
        x = x * torch.as_tensor(shape, **backend) / 2.
        return self.log(self.dense2aff(x))
    
    def _dense2prm_cnn(self, x):
        """CNN-based implementation of dense2prm"""
        x = last2channel(x)
        shape = x.shape[2:]
        grid = self._identity(x)
        x = torch.cat([x, grid], dim=1)
        prm = self.cnn(x)
        prm = prm.reshape(prm.shape[:2])
        return self._std_prm(prm, shape)
    
    def _std_prm(self, prm, shape):
        """Multiply generated parameters by a factor that is 
        related to their (expected) standard deviation."""
        backend = dict(dtype=prm.dtype, device=prm.device)
        trl_factor = torch.as_tensor(shape, **backend) / 2.
        rot_factor = 1. / (3. if self.mode == 'classic' else 2.)
        zom_factor = 0.5 if self.mode == 'classic' else 0.2
        shr_factor = 1.
        if self.group == 'T':
            prm = prm * trl_factor
        elif self.group == 'SO':
            prm = prm * rot_factor
        elif self.group == 'SE':
            trl = prm[:, :self.dim] * trl_factor
            rot = prm[:, self.dim:] * rot_factor
            prm = torch.cat([trl, rot], dim=1)
        elif self.group == 'D':
            trl = prm[:, :self.dim] * trl_factor
            zom = prm[:, self.dim:] * rot_factor
            prm = torch.cat([trl, zom], dim=1)
        elif self.group == 'CSO':
            trl = prm[:, :self.dim] * trl_factor
            rot = prm[:, self.dim:-1] * rot_factor
            zom = prm[:, -1:] * zom_factor
            prm = torch.cat([trl, rot, zom], dim=1)
        elif self.group == 'SL':
            trl = prm[:, :self.dim] * trl_factor
            rot = prm[:, self.dim:-(self.dim-1)] * rot_factor
            zom = prm[:, -(self.dim-1):] * zom_factor
            prm = torch.cat([trl, rot, zom], dim=1)
        elif self.group == 'GL+':
            rot = prm[:, :(self.dim)*(self.dim-1)//2] * rot_factor
            zom = prm[:, (self.dim)*(self.dim-1)//2:-((self.dim)*(self.dim-1)//2)] * zom_factor
            shr = prm[:, -((self.dim)*(self.dim-1)//2):] * shr_factor
            prm = torch.cat([rot, zom, shr], dim=1)
        elif self.group == 'Aff+':
            trl =  prm[:, :self.dim] * trl_factor
            rot = prm[:, self.dim:self.dim+(self.dim)*(self.dim-1)//2] * rot_factor
            zom = prm[:, self.dim+(self.dim)*(self.dim-1)//2:-((self.dim)*(self.dim-1)//2)] * zom_factor
            shr = prm[:, -((self.dim)*(self.dim-1)//2):] * shr_factor
            prm = torch.cat([trl, rot, zom, shr], dim=1)
        return prm
        
    def forward(self, source, target, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        affine_prm : tensor (batch,, *nb_prm)
            affine Lie/Classic parameters

        """
        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        dense = channel2last(self.unet(source_and_target))
        affprm = self.dense2prm(dense)
        affine = self.exp(affprm.double()).to(dense.dtype)
        grid = self.grid(affine, shape=target.shape[2:])
        deformed_source = self.pull(source, grid)

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     affine=[affprm],
                     dense=[dense])

        return deformed_source, affprm, dense

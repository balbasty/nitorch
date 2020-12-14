import torch
import torch.nn as tnn
from ..modules._base import Module
from ..modules._cnn import UNet, CNN
from ..modules._spatial import GridPull, GridExp, GridResize, AffineGrid, \
    AffineExp, AffineLog, AffineClassic, \
    AffineClassicInverse
from .. import check
from nitorch import spatial
from nitorch.core.linalg import matvec
from nitorch.core.utils import unsqueeze, channel2last
from nitorch.core.pyutils import make_list


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

    def __init__(self, dim, basis='CSO', mode='lie', encoder=None,
                 decoder=None,
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
        mode : {'lie', 'classic'}, default='lie'
            Encoding of the affine parameters.
            The basis 'SL' is only available in mode 'lie'.
        encoder : list[int], optional
            Number of channels after each encoding layer .
        decoder : list[int], optional
            Number of channels after each decoding layer.
        kernel_size : int or list[int], default=3
            Kernel size of the UNet.
        interpolation : int, default=1
            Interpolation order.
        bound : bound_type, default='dct2'
            Boundary conditions of the image.
        """

        super().__init__()
        self.unet = UNet(dim,
                         input_channels=2 + _additional_input_channels,
                         output_channels=dim + _additional_output_channels,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.dense2aff = DenseToAffine(shift=True)
        if mode == 'lie':
            self.log = AffineLog(basis=basis)
            self.exp = AffineExp(dim, basis=basis)
        else:
            self.log = AffineClassicInverse(basis=basis)
            self.exp = AffineClassic(dim, basis=basis)
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
        affine_prm : tensor (batch,, *nb_prm)
            affine Lie/Classic parameters

        """
        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim + 2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        dense = self.unet(source_and_target)
        dense = channel2last(dense)
        affine = self.dense2aff(dense)
        affprm = self.log(affine)    # exp(log) is not an identity because log
        affine = self.exp(affprm)    # projects on a lower dimensional space.
        grid = self.grid(affine, shape=target.shape[2:])
        deformed_source = self.pull(source, grid)

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     affine=[affprm])

        return deformed_source, affprm

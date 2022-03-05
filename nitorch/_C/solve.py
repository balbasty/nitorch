from .grid import COMPILED_BACKEND, bound_to_nitorch, inter_to_nitorch
from .utils import make_list, vector_to_list, movedim
import torch

if COMPILED_BACKEND == 'C':

    from .spatial import (
        regulariser as _c_regulariser,
        regulariser_grid as _c_regulariser_grid,
        relax_grid as _c_relax_grid,
        relax as _c_relax,
        relax_grid as _c_relax_grid,
        precond as _c_precond,
        precond_grid as _c_precond_grid,
        pcg as _c_pcg,
        pcg_grid as _c_pcg_grid,
        fmg as _c_fmg,
        fmg_grid as _c_fmg_grid)


    def c_regulariser(input, weight=None, hessian=None, dim=None,
                      absolute=0, membrane=0, bending=0, factor=1,
                      voxel_size=1, bound='dct2', output=None):
        """Apply the forward pass of a regularised linear system
                output = (hessian + regulariser) @ input

        Parameters
        ----------
        input : (N, C, *shape) tensor
        weight : (N, C|1, *shape) tensor, optional
        hessian : (N, CC, *shape) tensor, optional
            CC is one of {1, C, C*(C+1)/2}
        dim : int, optional
        absolute : [sequence of] float, default=0
        membrane : [sequence of] float, default=0
        bending : [sequence of] float, default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dct2'
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        pad_dim = 0
        if dim:
            pad_dim = dim+2 - input.dim()
            while input.dim() < dim+2: input = input.unsqueeze(0)
        if output is None:
            output = torch.Tensor()
        if weight is None:
            weight = torch.Tensor()
        else:
            pad_dim = min(pad_dim, input.dim() - weight.dim())
            while weight.dim() < input.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, input.dim() - hessian.dim())
            while hessian.dim() < input.dim(): hessian = hessian.unsqueeze(0)
        C = input.shape[1]
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = vector_to_list(absolute, float, C) or ([0.]*C)
        membrane = vector_to_list(membrane, float, C) or ([0.]*C)
        bending = vector_to_list(bending, float, C) or ([0.]*C)
        factor = vector_to_list(factor, float, C) or ([1.]*C)
        absolute = [a*f for a, f in zip(absolute, factor)]
        membrane = [m*f for m, f in zip(membrane, factor)]
        bending = [b*f for b, f in zip(bending, factor)]
        if any(bending) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_regulariser(input, output, weight, hessian,
                                absolute, membrane, bending, voxel_size, bound)
        for _ in range(pad_dim): output = output.squeeze(0)
        return output


    def c_regulariser_grid(input, weight=None, hessian=None,
                           absolute=0, membrane=0, bending=0, lame=0, factor=1,
                           voxel_size=1, bound='dft', output=None):
        """Apply the forward pass of a regularised linear system
                output = (hessian + regulariser) @ input

        Parameters
        ----------
        input : (N, *shape, D) tensor
        weight : (N, *shape) tensor, optional
        hessian : (N, *shape, DD) tensor, optional
            DD is one of {1, D, D*(D+1)/2}
        absolute : float, default=0
        membrane : float, default=0
        bending : float, default=0
        lame : (float, float), default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dft'
        output : (N, *shape, D) tensor, optional

        Returns
        -------
        output : (N, *shape, D) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        # The C++ bindings expect [N, C, *shape] so we must shuffle dimensions
        dim = input.shape[-1]
        pad_dim = dim+2 - input.dim()
        while input.dim() < dim+2: input = input.unsqueeze(0)
        input = movedim(input, -1, 1)
        if output is None:
            output = torch.Tensor()
        else:
            while output.dim() < input.dim(): output = output.unsqueeze(0)
            output = movedim(output, -1, 1)
        if weight is None:
            weight = torch.Tensor()
        else:
            weight = weight.unsqueeze(-dim-1)
            pad_dim = min(pad_dim, dim+2 - weight.dim())
            while weight.dim() < input.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, dim+2 - hessian.dim())
            while hessian.dim() < input.dim(): hessian = hessian.unsqueeze(0)
            hessian = movedim(hessian, -1, 1)
        absolute = float(absolute * factor)
        membrane = float(membrane * factor)
        bending = float(bending * factor)
        lame_shear, lame_div = make_list(lame, 2) or [0., 0.]
        lame_shear = float(lame_shear * factor)
        lame_div = float(lame_div * factor)
        if (bending or lame_shear or lame_div) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_regulariser_grid(input, output, weight, hessian, absolute,
                                     membrane, bending, lame_shear, lame_div,
                                     voxel_size, bound)
        output = movedim(output, 1, -1)
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

    def c_relax(gradient, weight=None, hessian=None, dim=None,
                absolute=0, membrane=0, bending=0, factor=1,
                voxel_size=1, bound='dct2', nb_iter=2, output=None):
        """Solve a regularised linear system by relaxation (Gauss-Seidel)
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        gradient : (N, C, *shape) tensor
        weight : (N, C|1, *shape) tensor, optional
        hessian : (N, CC, *shape) tensor, optional
            CC is one of {1, C, C*(C+1)/2}
        absolute : [sequence of] float, default=0
        membrane : [sequence of] float, default=0
        bending : [sequence of] float, default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dct2'
        nb_iter : int, default=2
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        pad_dim = 0
        if dim:
            pad_dim = dim + 2 - gradient.dim()
            while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        if output is None:
            output = torch.Tensor()
        if weight is None:
            weight = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
        C = gradient.shape[1]
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = vector_to_list(absolute, float, C) or ([0.]*C)
        membrane = vector_to_list(membrane, float, C) or ([0.]*C)
        bending = vector_to_list(bending, float, C) or ([0.]*C)
        factor = vector_to_list(factor, float, C) or ([1.]*C)
        absolute = [a*f for a, f in zip(absolute, factor)]
        membrane = [m*f for m, f in zip(membrane, factor)]
        bending = [b*f for b, f in zip(bending, factor)]
        if any(bending) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_relax(hessian, gradient, output, weight,
                          absolute, membrane, bending, voxel_size,
                          bound, int(nb_iter))
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

    def c_relax_grid(gradient, weight=None, hessian=None,
                     absolute=0, membrane=0, bending=0, lame=0, factor=1,
                     voxel_size=1, bound='dft', nb_iter=2, output=None):
        """Solve a regularised linear system by relaxation (Gauss-Seidel)
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        gradient : (N, *shape, D) tensor
        weight : (N, *shape) tensor, optional
        hessian : (N, *shape, DD) tensor, optional
            DD is one of {1, D, D*(D+1)/2}
        absolute : float, default=0
        membrane :  float, default=0
        bending :  float, default=0
        lame : (float, float), default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dft'
        nb_iter : int, default=2
        output : (N, *shape, D) tensor, optional

        Returns
        -------
        output : (N, *shape, D) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        dim = gradient.shape[-1]
        pad_dim = dim + 2 - gradient.dim()
        while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        gradient = movedim(gradient, -1, 1)
        if output is None:
            output = torch.Tensor()
        else:
            output = movedim(output, -1, 1)
        if weight is None:
            weight = torch.Tensor()
        else:
            weight = weight.unsqueeze(-dim-1)
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
            hessian = movedim(hessian, -1, 1)
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = float(absolute * factor)
        membrane = float(membrane * factor)
        bending = float(bending * factor)
        lame_shear, lame_div = make_list(lame, 2) or [0., 0.]
        lame_shear = float(lame_shear * factor)
        lame_div = float(lame_div * factor)
        if (bending or lame_shear or lame_div) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_relax_grid(hessian, gradient, output, weight, absolute,
                               membrane, bending, lame_shear, lame_div,
                               voxel_size, bound, int(nb_iter))
        output = movedim(output, 1, -1)
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

    def c_precond(gradient, weight=None, hessian=None, dim=None,
                  absolute=0, membrane=0, bending=0, factor=1,
                  voxel_size=1, bound='dct2', output=None):
        """Solve a regularised linear system by relaxation (Gauss-Seidel)
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        gradient : (N, C, *shape) tensor
        weight : (N, C|1, *shape) tensor, optional
        hessian : (N, CC, *shape) tensor, optional
            CC is one of {1, C, C*(C+1)/2}
        absolute : [sequence of] float, default=0
        membrane : [sequence of] float, default=0
        bending : [sequence of] float, default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dct2'
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        pad_dim = 0
        if dim:
            pad_dim = dim + 2 - gradient.dim()
            while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        if output is None:
            output = torch.Tensor()
        if weight is None:
            weight = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
        C = gradient.shape[1]
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = vector_to_list(absolute, float, C) or ([0.]*C)
        membrane = vector_to_list(membrane, float, C) or ([0.]*C)
        bending = vector_to_list(bending, float, C) or ([0.]*C)
        factor = vector_to_list(factor, float, C) or ([1.]*C)
        absolute = [a*f for a, f in zip(absolute, factor)]
        membrane = [m*f for m, f in zip(membrane, factor)]
        bending = [b*f for b, f in zip(bending, factor)]
        if any(bending) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_precond(hessian, gradient, output, weight,
                            absolute, membrane, bending, voxel_size,
                            bound)
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

    def c_precond_grid(gradient, weight=None, hessian=None,
                       absolute=0, membrane=0, bending=0, lame=0, factor=1,
                       voxel_size=1, bound='dft', output=None):
        """Solve a regularised linear system by relaxation (Gauss-Seidel)
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        gradient : (N, *shape, D) tensor
        weight : (N, *shape) tensor, optional
        hessian : (N, *shape, DD) tensor, optional
            DD is one of {1, D, D*(D+1)/2}
        absolute : float, default=0
        membrane :  float, default=0
        bending :  float, default=0
        lame : (float, float), default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dft'
        output : (N, *shape, D) tensor, optional

        Returns
        -------
        output : (N, *shape, D) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        dim = gradient.shape[-1]
        pad_dim = dim + 2 - gradient.dim()
        while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        gradient = movedim(gradient, -1, 1)
        if output is None:
            output = torch.Tensor()
        else:
            output = movedim(output, -1, 1)
        if weight is None:
            weight = torch.Tensor()
        else:
            weight = weight.unsqueeze(1)
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
            hessian = movedim(hessian, -1, 1)
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = float(absolute * factor)
        membrane = float(membrane * factor)
        bending = float(bending * factor)
        lame_shear, lame_div = make_list(lame, 2) or [0., 0.]
        lame_shear = float(lame_shear * factor)
        lame_div = float(lame_div * factor)
        if (bending or lame_shear or lame_div) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_precond_grid(hessian, gradient, output, weight, absolute,
                                 membrane, bending, lame_shear, lame_div,
                                 voxel_size, bound)
        output = movedim(output, 1, -1)
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

    def c_pcg(gradient, weight=None, hessian=None, dim=None,
              absolute=0, membrane=0, bending=0, factor=1,
              voxel_size=1, bound='dct2', nb_iter=16, tol=1e-4, output=None):
        """Solve a regularised linear system by conjugate gradient
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        gradient : (N, C, *shape) tensor
        weight : (N, C|1, *shape) tensor, optional
        hessian : (N, CC, *shape) tensor, optional
            CC is one of {1, C, C*(C+1)/2}
        absolute : [sequence of] float, default=0
        membrane : [sequence of] float, default=0
        bending : [sequence of] float, default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dct2'
        nb_iter : int, default=2
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        pad_dim = 0
        if dim:
            pad_dim = dim + 2 - gradient.dim()
            while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        if output is None:
            output = torch.Tensor()
        if weight is None:
            weight = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
        C = gradient.shape[1]
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = vector_to_list(absolute, float, C) or ([0.]*C)
        membrane = vector_to_list(membrane, float, C) or ([0.]*C)
        bending = vector_to_list(bending, float, C) or ([0.]*C)
        factor = vector_to_list(factor, float, C) or ([1.]*C)
        absolute = [a*f for a, f in zip(absolute, factor)]
        membrane = [m*f for m, f in zip(membrane, factor)]
        bending = [b*f for b, f in zip(bending, factor)]
        if any(bending) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_pcg(hessian, gradient, output, weight,
                        absolute, membrane, bending, voxel_size,
                        bound, int(nb_iter), float(tol))
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

    def c_pcg_grid(gradient, weight=None, hessian=None,
                   absolute=0, membrane=0, bending=0, lame=0, factor=1,
                   voxel_size=1, bound='dft', nb_iter=16, tol=1e-4, output=None):
        """Solve a regularised linear system by conjugate gradient
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        gradient : (N, *shape, D) tensor
        weight : (N, C*shape) tensor, optional
        hessian : (N, *shape, DD) tensor, optional
            DD is one of {1, D, D*(D+1)/2}
        absolute : [sequence of] float, default=0
        membrane : [sequence of] float, default=0
        bending : [sequence of] float, default=0
        lame : (float, float), default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dft'
        nb_iter : int, default=2
        output : (N, *shape, D) tensor, optional

        Returns
        -------
        output : (N, *shape, D) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        dim = gradient.shape[-1]
        pad_dim = dim + 2 - gradient.dim()
        while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        gradient = movedim(gradient, -1, 1)
        if output is None:
            output = torch.Tensor()
        else:
            output = movedim(output, -1, 1)
        if weight is None:
            weight = torch.Tensor()
        else:
            weight = weight.unsqueeze(1)
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
            hessian = movedim(hessian, -1, 1)
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = float(absolute * factor)
        membrane = float(membrane * factor)
        bending = float(bending * factor)
        lame_shear, lame_div = make_list(lame, 2) or [0., 0.]
        lame_shear = float(lame_shear * factor)
        lame_div = float(lame_div * factor)
        if (bending or lame_shear or lame_div) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_pcg_grid(hessian, gradient, output, weight,
                             absolute, membrane, bending, lame_shear, lame_div, voxel_size,
                             bound, int(nb_iter), float(tol))
        output = output.movedim(1, -1)
        for _ in range(pad_dim): output = output.squeeze(0)
        return output


    def c_fmg(hessian, gradient, weight=None, dim=None,
              absolute=0, membrane=0, bending=0, factor=1,
              voxel_size=1, bound='dct2',
              nb_cycles=2, nb_iter=2, max_levels=16,
              solver='cg', output=None):
        """Solve a regularised linear system by full multi-grid
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        hessian : (N, CC, *shape) tensor
            CC is one of {1, C, C*(C+1)/2}
        gradient : (N, C, *shape) tensor
        weight : (N, C|1, *shape) tensor, optional
        absolute : [sequence of] float, default=0
        membrane : [sequence of] float, default=0
        bending : [sequence of] float, default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dct2'
        nb_cycles : int, default=2
        nb_iter : int, default=4
        max_levels : int, default=16
        solver : {'relax', 'cg'}, default='relax'
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        pad_dim = 0
        if dim:
            pad_dim = dim + 2 - gradient.dim()
            while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        if output is None:
            output = torch.Tensor()
        if weight is None:
            weight = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
        C = gradient.shape[1]
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = vector_to_list(absolute, float, C) or ([0.]*C)
        membrane = vector_to_list(membrane, float, C) or ([0.]*C)
        bending = vector_to_list(bending, float, C) or ([0.]*C)
        factor = vector_to_list(factor, float, C) or ([1.]*C)
        absolute = [a*f for a, f in zip(absolute, factor)]
        membrane = [m*f for m, f in zip(membrane, factor)]
        bending = [b*f for b, f in zip(bending, factor)]
        if any(bending) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_fmg(hessian, gradient, output, weight,
                        absolute, membrane, bending, voxel_size,
                        bound, int(nb_cycles), int(nb_iter), int(max_levels),
                        bool(solver == 'cg'))
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

    def c_fmg_grid(hessian, gradient, weight=None,
                   absolute=0, membrane=0, bending=0, lame=0, factor=1,
                   voxel_size=1, bound='dft',
                   nb_cycles=2, nb_iter=2, max_levels=16,
                   solver='cg', output=None):
        """Solve a regularised linear system by full multi-grid
                solution = (hessian + regulariser) \ gradient

        Parameters
        ----------
        hessian : (N, *shape, DD) tensor
            DD is one of {1, D, D*(D+1)/2}
        gradient : (N, *shape, D) tensor
        weight : (N, *shape) tensor, optional
        absolute : float, default=0
        membrane : float, default=0
        bending : float, default=0
        lame : (float, float), default=0
        voxel_size : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dft'
        nb_cycles : int, default=2
        nb_iter : int, default=4
        max_levels : int, default=16
        solver : {'relax', 'cg'}, default='relax'
        output : (N, *shape, D) tensor, optional

        Returns
        -------
        output : (N, *shape, D) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        dim = gradient.shape[-1]
        pad_dim = dim + 2 - gradient.dim()
        while gradient.dim() < dim+2: gradient = gradient.unsqueeze(0)
        gradient = movedim(gradient, -1, 1)
        if output is None:
            output = torch.Tensor()
        else:
            output = movedim(output, -1, 1)
        if weight is None:
            weight = torch.Tensor()
        else:
            weight = weight.unsqueeze(1)
            pad_dim = min(pad_dim, gradient.dim() - weight.dim())
            while weight.dim() < gradient.dim(): weight = weight.unsqueeze(0)
        if hessian is None:
            hessian = torch.Tensor()
        else:
            pad_dim = min(pad_dim, gradient.dim() - hessian.dim())
            while hessian.dim() < gradient.dim(): hessian = hessian.unsqueeze(0)
            hessian = movedim(hessian, -1, 1)
        voxel_size = vector_to_list(voxel_size, float) or [1.]
        absolute = float(absolute * factor)
        membrane = float(membrane * factor)
        bending = float(bending * factor)
        lame_shear, lame_div = make_list(lame, 2) or [0., 0.]
        lame_shear = float(lame_shear * factor)
        lame_div = float(lame_div * factor)
        if (bending or lame_shear or lame_div) and weight.numel():
            raise ValueError('RLS only implemented for membrane or absolute')
        output = _c_fmg_grid(hessian, gradient, output, weight, absolute,
                             membrane, bending, lame_shear, lame_div, voxel_size,
                             bound, int(nb_cycles), int(nb_iter), int(max_levels),
                             bool(solver == 'cg'))
        output = movedim(output, 1, -1)
        for _ in range(pad_dim): output = output.squeeze(0)
        return output

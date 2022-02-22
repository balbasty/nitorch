# Fast C++/CUDA functions

This folder contains fast implementation of some of the 
spatial functions that are at the core of many of our algorithms. All
functions are implemented for 1D/2D or 3D spatial tensors and they usually
expect inputs with shape `[batch, channel, dim_x, dim_y, dim_z]`, with 
`dim_z` and `dim_y` optional. One exception concerns the grid of 
sampling coordinate in `grid_pull` and `grid_push`, which should have shape
`[batch, dim_x, dim_y, dim_z, nb_dim]`.

These functions are exposed to Python through the file `spatial.cpp` located in
the parent folder. We also have pure PyTorch implementations of these 
functions, that can be used when compilation of the C++/CUDA extension is 
not possible, but they are much (much) slower.

Currently, we implement:
- `grid_pull`: spline interpolation (order 0 to 7) of a tensor at 
   specified sampling locations.
- `grid_push`: spline "splatting" (order 0 to 7) of a tensor to
  specified sampling locations. This is the adjoint operation of
  `grid_pull`.
- `grid_grad`: sample spatial gradients at specified sampling locations.
  Spatial gradients are computed by spline interpolation (order 0 to 7).
- `resize`: spline up/down sampling of a tensor using spline 
  interpolation (order 0 to 7).
- `resize(..., do_adjoint=true)`: adjoint of `resize`, where values
  are pushed to the output grid instead of being interpolated.
- `fmg_prolongation`: A specific and stable implementation of `resize` 
  using quadratic splines. Used by the Full multi-grid algorithm.
- `fmg_restriction`: A specific and stable implementation of 
  `resize(..., do_adjoint=true)` using linear splines. 
  Used by the Full multi-grid algorithm.
- `regulariser`: Implements the forward pass of a mixture of spatial
  regulariser (absolute values, first and second derivatives). If 
  a field of symmetric matrices is provided, the matrix-vector product
  of the input with this matrix is also computed. <br \>
  In other words, computes `(H + L) * x` or `L*x` or `H*x`, 
  where `H` is the field of matrices and `L` is the spatial regulariser.
- `regulariser_grid`: Same as `regulariser` but specialized for 
  displacement fields. Same penalties plus the linear-elastic energy
  (symmetric part of the Jacobian = shears, 
  trace of the Jacobian = volume change).
- `precond`/`precond_grid`: Solves a field of symmetric linear systems, where each
  matrix is composed of a user-defined Hessian plus the diagonal of 
  the regulariser (consistent with `regulariser`). <br />
  In other words, computes `(H + diag(L)) \ x`, 
  where `H` is the field of matrices and `L` is the spatial regulariser.
- `relax`/`relax_grid`: Solves a symmetric linear system (consistent with `regulariser`) 
  by relaxation.  <br />
  In other words, iterates `x += (H + diag(L)) \ (g - (H + L)*x)`, 
  where `H` is the field of matrices and `L` is the spatial regulariser.
- `pcg`/`pcg_grid`: Solves a symmetric linear system (consistent with `regulariser`) 
  by preconditioned conjugate gradient.
- `fmg`/`fmg_grid`: Solves a symmetric linear system (consistent with `regulariser`) 
  by full multi-grid. The inner solver can be either `relax` or `pcg`.

A lot could be done to make the code nicer. To start with, since all functions
expect the same type of inputs, there are a lot of redundancies in terms of 
data handling, that currently mean a lot of copy-pasting, but could probably 
be refactored.

The other thing is `fmg` and `pcg` currently call the high-level API of 
each subfunction, which means that a bunch of checks and CPU/GPU dispatch
happen at each function call. We could instead split the API and implementation
(as in the other functions) and call lower-level implementations inside
the algorithm.


   

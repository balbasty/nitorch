#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define NI_PRECOND_GRID_DECLARE(space) \
  namespace space { \
    at::Tensor precond_grid_impl( \
      at::Tensor hessian, const at::Tensor& gradient, at::Tensor solution, at::Tensor weight, \
      double absolute, double membrane, double bending, double lame_shear, double lame_div, \
      c10::ArrayRef<double> voxel_size, BoundVectorRef bound); \
  }


namespace ni {
NI_PRECOND_GRID_DECLARE(cpu)
NI_PRECOND_GRID_DECLARE(cuda)
NI_PRECOND_GRID_DECLARE(notimplemented)
} // namespace ni

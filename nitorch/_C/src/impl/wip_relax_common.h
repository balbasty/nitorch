#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define NI_RELAX_DECLARE(space) \
  namespace space { \
    at::Tensor relax( \
      Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight, bool grid, \
      double absolute, double membrane, double bending, double lame_shear, double lame_div, \
      ArrayRef<double> factor, ArrayRef<double> voxel_size, BoundVectorRef bound); \
  }


namespace ni {
NI_RELAX_DECLARE(cpu)
NI_RELAX_DECLARE(cuda)
NI_RELAX_DECLARE(notimplemented)
} // namespace ni

#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define NI_PRECOND_DECLARE(space) \
  namespace space { \
    at::Tensor precond_impl( \
      at::Tensor hessian, const at::Tensor& gradient, at::Tensor solution, at::Tensor weight, \
      c10::ArrayRef<double> absolute, c10::ArrayRef<double> membrane, c10::ArrayRef<double> bending, \
      c10::ArrayRef<double> voxel_size, BoundVectorRef bound); \
  }


namespace ni {
NI_PRECOND_DECLARE(cpu)
NI_PRECOND_DECLARE(cuda)
NI_PRECOND_DECLARE(notimplemented)
} // namespace ni

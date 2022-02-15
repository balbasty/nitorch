#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define NI_RELAX_DECLARE(space) \
  namespace space { \
    at::Tensor relax_impl( \
      at::Tensor hessian, const at::Tensor& gradient, at::Tensor solution, at::Tensor weight, \
      c10::ArrayRef<double> absolute, c10::ArrayRef<double> membrane, c10::ArrayRef<double> bending, \
      c10::ArrayRef<double> voxel_size, BoundVectorRef bound, int64_t nb_iter); \
  }


namespace ni {
NI_RELAX_DECLARE(cpu)
NI_RELAX_DECLARE(cuda)
NI_RELAX_DECLARE(notimplemented)
} // namespace ni

#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"


#define NI_MULTIRES_DECLARE(space) \
  namespace space { \
    at::Tensor multires_impl( \
      at::Tensor source, at::Tensor target, \
      c10::ArrayRef<double> factor, BoundVectorRef bound,  \
      bool do_adjoint); \
  }


namespace ni {
NI_MULTIRES_DECLARE(cpu)
NI_MULTIRES_DECLARE(cuda)
NI_MULTIRES_DECLARE(notimplemented)
} // namespace ni

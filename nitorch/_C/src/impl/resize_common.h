#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"
#include "../interpolation.h"
#include "../grid_align.h"


#define NI_RESIZE_DECLARE(space) \
  namespace space { \
    at::Tensor resize_impl( \
      at::Tensor source, at::Tensor target, \
      c10::ArrayRef<double> factor, BoundVectorRef bound,  \
      InterpolationVectorRef interpolation, GridAlignVectorRef mode, \
      bool do_adjoint); \
  }


namespace ni {
NI_RESIZE_DECLARE(cpu)
NI_RESIZE_DECLARE(cuda)
NI_RESIZE_DECLARE(notimplemented)
} // namespace ni

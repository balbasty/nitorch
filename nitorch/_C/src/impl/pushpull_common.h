#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"
#include "../interpolation.h"
#include "utils.h"

#define NI_PUSHPULL_DECLARE(space) \
  namespace space { \
    template <typename BoundType, typename InterpolationType, typename SourceType> \
    Pair<at::Tensor> pushpull( \
      const SourceType& source, const at::Tensor& grid, \
      BoundType bound, InterpolationType interpolation, int extrapolate, \
      bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad); \
    template <typename BoundType, typename InterpolationType, typename SourceType> \
    Pair<at::Tensor> pushpull( \
      const SourceType & source, const at::Tensor& grid, const at::Tensor& target, \
      BoundType bound, InterpolationType interpolation, int extrapolate, \
      bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad); \
  }

namespace ni {
NI_PUSHPULL_DECLARE(cpu)
NI_PUSHPULL_DECLARE(cuda) 
NI_PUSHPULL_DECLARE(notimplemented)
} // namespace ni

#pragma once

#include "bounds.h"
#include "interpolation.h"
#include <ATen/ATen.h>
#include <deque>

#define NI_PUSHPULL_DECLARE(space) \
  namespace space { \
    template <typename BoundType, typename InterpolationType> \
    std::deque<at::Tensor> pushpull( \
      const at::Tensor& source, const at::Tensor& grid, \
      BoundType bound, InterpolationType interpolation, bool extrapolate, \
      bool do_pull, bool do_push, bool do_grad); \
    template <typename BoundType, typename InterpolationType, typename SourceType> \
    std::deque<at::Tensor> pushpull( \
      const SourceType & source, const at::Tensor& grid, const at::Tensor& target, \
      BoundType bound, InterpolationType interpolation, bool extrapolate, \
      bool do_pull, bool do_push, bool do_grad); \
  }

namespace ni {
NI_PUSHPULL_DECLARE(cpu)
NI_PUSHPULL_DECLARE(cuda) 
NI_PUSHPULL_DECLARE(notimplemented)
} // namespace ni
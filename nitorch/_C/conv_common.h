#pragma once

#include <ATen/ATen.h>
#include "bounds.h"
#include <deque>

#define NI_CONV_DECLARE(space)                \
  namespace space {                           \
    std::deque<at::Tensor> conv(              \
      const at::Tensor& source,               \
      const at::Tensor& weight,               \
      const at::Tensor& bias,                 \
      int groups,                             \
      BoundVectorRef bound,                   \
      c10::IntArrayRef stride,                \
      c10::IntArrayRef dilation,              \
      c10::IntArrayRef offsetlow,             \
      c10::IntArrayRef offsetup,              \
      c10::IntArrayRef center,                \
      bool do_conv,                           \
      bool do_deconv,                         \
      bool do_grad);                          \
                                              \
    template <typename SourceType>            \
    std::deque<at::Tensor> conv(              \
      const SourceType& source,               \
      const at::Tensor& weight,               \
      const at::Tensor& bias,                 \
      const at::Tensor& target,               \
      int groups,                             \
      BoundVectorRef bound,                   \
      c10::IntArrayRef stride,                \
      c10::IntArrayRef dilation,              \
      c10::IntArrayRef offsetlow,             \
      c10::IntArrayRef offsetup,              \
      c10::IntArrayRef center,                \
      bool do_conv,                           \
      bool do_deconv,                         \
      bool do_grad);                          \
  }

namespace ni {
NI_CONV_DECLARE(cpu)
NI_CONV_DECLARE(cuda)
NI_CONV_DECLARE(notimplemented)
} // namespace ni

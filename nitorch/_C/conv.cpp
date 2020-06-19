#include "conv_common.h"
#include <ATen/ATen.h>
#include <vector>
#include <deque>
#include <iostream>

#ifndef NI_WITH_CUDA
#  define cuda notimplemented
#endif

using at::Tensor;
using c10::IntArrayRef;
using std::vector;

namespace ni {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ REUSABLE CHECKS ~~~~~~~~~~~~~~~~~~~~~~~~~~
static bool isempty(const Tensor & tensor) {
  for (int64_t i = 0; i < tensor.dim(); i++)
    if (tensor.size(i) == 0) return true;
  return false;
}

#define CONV_CHECK_DEFINED(value)                                       \
  TORCH_CHECK(                                                          \
    value.defined(),                                                    \
    "(): expected " #value " not be undefined, "                        \
    "but it is ", value)
#define CONV_CHECK_OPT_STRIDED(value)                                   \
  TORCH_CHECK(                                                          \
    value.layout() == at::kStrided,                                     \
    "(): expected " #value "to have torch.strided layout, "             \
    "but it has ", value.layout())
#define CONV_CHECK_1D_2D_OR_3D(value)                                   \
  TORCH_CHECK(                                                          \
    (value.dim() == 3 || value.dim() == 4 || value.dim() == 5),         \
    "(): expected 3D, 4D or 5D " #value " but got input with sizes ",   \
    value.sizes())
#define CONV_CHECK_BIAS_SHAPE(value)                                    \
  TORCH_CHECK(                                                          \
    (value.dim() == 1),                                                 \
    "(): expected 1D " #value " but got input with sizes ",             \
    value.sizes())
#define CONV_CHECK_OPT_SAME_DEVICE(value1, value2)                      \
    TORCH_CHECK(                                                        \
    value1.device() == value2.device(),                                 \
    "(): expected " #value2 " and " #value2 " to be on same device, "   \
    "but " #value2 " is on ", value1.device(), " and " #value2          \
    " is on ", value2.device())
#define CONV_CHECK_OPT_SAME_DTYPE(value1, value2)                       \
    TORCH_CHECK(                                                        \
    value1.dtype() == value2.dtype(),                                   \
    "(): expected " #value2 " and " #value2 " to have the same dtype, " \
    "but " #value2 " has ", value1.dtype(), " and " #value2 " has ",    \
    value2.dtype())
#define CONV_CHECK_NOT_EMPTY(value)                                     \
  do { for (int64_t i = 2; i < value.dim(); i++) {                      \
    TORCH_CHECK(value.size(i) > 0,                                      \
      "(): expected " #value " to have non-empty spatial dimensions, "  \
      "but input has sizes ", value.sizes(), " with dimension ", i,     \
      " being empty"); } } while (0)
#define CONV_CHECK_CHANNEL_TARGET_COMPAT(weight, target)                \
    TORCH_CHECK(                                                        \
    weight.size(0) == target.size(1),                                   \
    "(): " #weight " and " #target " channels are not compatible, "     \
    "got " #weight " with ", weight.size(0), " channels and " #target   \
     " with ", target.size(1), " channels")
#define CONV_CHECK_CHANNEL_SOURCE_COMPAT(weight, source, groups)        \
    TORCH_CHECK(                                                        \
    weight.size(1)*groups == source.size(1),                            \
    "(): " #weight " and " #source " channels are not compatible, "     \
    "got " #weight " with ", weight.size(1), "x", groups,               \
    " channels and " #source " with ", source.size(1), " channels")
#define CONV_CHECK_BIAS_CHANNEL(bias, weight)                           \
    TORCH_CHECK(                                                        \
    bias.size(0) == weight.size(0),                                     \
    "(): " #bias " and " #weight " channels are not compatible, "       \
    "got " #bias " with ", bias.size(0), " channels and "               \
    #weight " with ", weight.size(1), " channels")
#define CONV_CHECK_LENGTH(value, dim)                                   \
  TORCH_CHECK(                                                          \
    (static_cast<int64_t>(value.size()) == dim - 2),                    \
    "(): expected ", dim, #value " elements but got ", value.size())
#define CONV_CHECK_VEC_NOT_EMPTY(value)                                 \
  TORCH_CHECK(!value.empty(), "(): expected non empty parameter " #value)


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor conv(const Tensor& input, const Tensor& weight, const Tensor& bias,
            int groups, const vector<BoundType> & bound_mode,
            IntArrayRef stride, IntArrayRef dilation,
            IntArrayRef offsetlow, IntArrayRef offsetup,
            IntArrayRef center)  {

  CONV_CHECK_DEFINED(input);
  CONV_CHECK_DEFINED(weight);
  auto input_opt  = input.options();
  auto weight_opt = weight.options();
  CONV_CHECK_OPT_STRIDED(input_opt);
  CONV_CHECK_OPT_STRIDED(weight_opt);
  CONV_CHECK_OPT_SAME_DEVICE(input_opt, weight_opt);
  CONV_CHECK_OPT_SAME_DTYPE(input_opt, weight_opt);
  CONV_CHECK_1D_2D_OR_3D(input);
  CONV_CHECK_1D_2D_OR_3D(weight);
  CONV_CHECK_CHANNEL_SOURCE_COMPAT(weight, input, groups);
  CONV_CHECK_NOT_EMPTY(input);
  CONV_CHECK_NOT_EMPTY(weight);
  if (!isempty(bias)) {
    auto bias_opt   = bias.options();
    CONV_CHECK_OPT_STRIDED(bias_opt);
    CONV_CHECK_OPT_SAME_DEVICE(input_opt, bias_opt);
    CONV_CHECK_OPT_SAME_DTYPE(input_opt, bias_opt);
    CONV_CHECK_BIAS_SHAPE(bias);
    CONV_CHECK_BIAS_CHANNEL(bias, weight);
    CONV_CHECK_NOT_EMPTY(bias);
  }

  if (input.is_cuda())
    return cuda::conv(input, weight, bias, groups, BoundVectorRef(bound_mode),
      stride, dilation, offsetlow, offsetup, center,
      true, false, false).front();
  else
    return cpu::conv(input, weight, bias, groups, BoundVectorRef(bound_mode),
      stride, dilation, offsetlow, offsetup, center,
      true, false, false).front();
}

//std::deque<Tensor>
//grid_pull_backward(const Tensor& grad, const Tensor& input, const Tensor& grid,
//                   const std::vector<BoundType> & bound_mode,
//                   const std::vector<InterpolationType> & interpolation_mode,
//                   bool extrapolate)
//{
//  if (input.is_cuda()) {
//    return cuda::conv(input, grid, grad,
//      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode),
//      extrapolate, false,
//      input.requires_grad(), false,
//      grid.requires_grad(), false);
//  } else {
//    return cpu::conv(input, grid, grad,
//      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode),
//      extrapolate, false,
//      input.requires_grad(), false,
//      grid.requires_grad(), false);
//  }
//}


} // namespace ni

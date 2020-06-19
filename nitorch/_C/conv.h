#include <ATen/ATen.h>
#include "bounds.h"
#include <tuple>
#include <vector>
#include <deque>

namespace ni {

at::Tensor conv(
  const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
  int groups, const std::vector<BoundType> & bound,
  c10::IntArrayRef stride, c10::IntArrayRef dilation,
  c10::IntArrayRef offsetlow, c10::IntArrayRef offsetup,
  c10::IntArrayRef center);



} // namespace ni

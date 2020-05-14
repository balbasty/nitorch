#include "include_first.h"
#include <ATen/ATen.h>
#include "bounds.h"
#include "interpolation.h"
#include <tuple>
#include <vector>
// #include <deque>

namespace ni {


NI_HOST at::Tensor grid_pull_cpu(
  const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation,  
  bool extrapolate);

NI_HOST std::tuple<at::Tensor,at::Tensor> grid_pull_backward_cpu(
  const at::Tensor& grad, const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  bool extrapolate);

NI_HOST at::Tensor grid_push_cpu(
  const at::Tensor& input, const at::Tensor& grid, c10::IntArrayRef source_size,
  const std::vector<BoundType> & bound,
  const std::vector<InterpolationType> & interpolation, 
  bool extrapolate);

NI_HOST std::tuple<at::Tensor,at::Tensor> grid_push_backward_cpu(
  const at::Tensor& grad, const at::Tensor& input,  const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  bool extrapolate);

} // namespace ni
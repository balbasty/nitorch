#include <ATen/ATen.h>
#include "bounds.h"
#include "interpolation.h"
#include <tuple>
#include <vector>
// #include <deque>

namespace ni {

at::Tensor grid_pull(
  const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation,  
  bool extrapolate);

std::tuple<at::Tensor,at::Tensor> grid_pull_backward(
  const at::Tensor& grad, const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  bool extrapolate);

at::Tensor grid_push(
  const at::Tensor& input, const at::Tensor& grid, c10::IntArrayRef source_size,
  const std::vector<BoundType> & bound,
  const std::vector<InterpolationType> & interpolation, 
  bool extrapolate);

std::tuple<at::Tensor,at::Tensor> grid_push_backward(
  const at::Tensor& grad, const at::Tensor& input,  const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  bool extrapolate);

} // namespace ni
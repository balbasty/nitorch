#include <ATen/ATen.h>
#include "bounds.h"
#include "interpolation.h"
#include <tuple>
#include <vector>
#include <deque>

namespace ni {

at::Tensor grid_pull(
  const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation,  
  int extrapolate);

std::deque<at::Tensor> grid_pull_backward(
  const at::Tensor& grad, const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  int extrapolate);

at::Tensor grid_push(
  const at::Tensor& input, const at::Tensor& grid, c10::IntArrayRef source_size,
  const std::vector<BoundType> & bound,
  const std::vector<InterpolationType> & interpolation, 
  int extrapolate);

std::deque<at::Tensor> grid_push_backward(
  const at::Tensor& grad, const at::Tensor& input,  const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  int extrapolate);

at::Tensor grid_count(
  const at::Tensor& grid, c10::IntArrayRef source_size,
  const std::vector<BoundType> & bound,
  const std::vector<InterpolationType> & interpolation, 
  int extrapolate);

at::Tensor grid_count_backward(
  const at::Tensor& grad, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  int extrapolate);

at::Tensor grid_grad(
  const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation,  
  int extrapolate);

std::deque<at::Tensor> grid_grad_backward(
  const at::Tensor& grad, const at::Tensor& input, const at::Tensor& grid,
  const std::vector<BoundType> & bound, 
  const std::vector<InterpolationType> & interpolation, 
  int extrapolate);



} // namespace ni

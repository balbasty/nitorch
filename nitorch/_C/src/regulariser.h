#include <ATen/ATen.h>
#include "bounds.h"
#include <vector>
#include <utility>

namespace ni {

at::Tensor regulariser(
  const at::Tensor& input, const at::Tensor& output, const at::Tensor& weight, const at::Tensor& hessian, 
  const std::vector<double> &  absolute, const std::vector<double> &  membrane, const std::vector<double> &  bending,
  const std::vector<double> & voxel_size, const std::vector<BoundType> & bound);

std::pair<at::Tensor, at::Tensor> regulariser_backward(
  const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight, const at::Tensor& hessian, 
  const std::vector<double> &  absolute, const std::vector<double> &  membrane, const std::vector<double> &  bending,
  const std::vector<double> & voxel_size, const std::vector<BoundType> & bound, bool do_input, bool do_weight);

}

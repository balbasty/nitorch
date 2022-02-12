#include <ATen/ATen.h>
#include "bounds.h"
#include <vector>
#include <utility>

namespace ni {

at::Tensor regulariser_grid(
  const at::Tensor& input, const at::Tensor& output, const at::Tensor& weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  const std::vector<double> & voxel_size, const std::vector<BoundType> & bound);

at::Tensor regulariser_grid_backward(
  const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  const std::vector<double> & voxel_size, const std::vector<BoundType> & bound,
  bool do_input, bool do_weight);

}

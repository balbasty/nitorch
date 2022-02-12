#include <ATen/ATen.h>
#include "bounds.h"
#include <vector>
#include <utility>

namespace ni {

at::Tensor relax(
  const at::Tensor& hessian, const at::Tensor& gradient,
  const at::Tensor& solution, const at::Tensor& weight, bool grid,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  const std::vector<double> & factor, const std::vector<double> & voxel_size,
  const std::vector<BoundType> & bound);

}

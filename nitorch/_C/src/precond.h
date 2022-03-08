#include <ATen/ATen.h>
#include "bounds.h"
#include <vector>
#include <utility>

namespace ni {

at::Tensor precond(
  const at::Tensor& hessian, const at::Tensor& gradient,
  const at::Tensor& solution, const at::Tensor& weight, 
  const std::vector<double> &  absolute, const std::vector<double> &  membrane, const std::vector<double> &  bending,
  const std::vector<double> & voxel_size, const std::vector<BoundType> & bound);

}

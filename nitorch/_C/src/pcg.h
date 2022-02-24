#include <ATen/ATen.h>
#include "bounds.h"
#include <vector>
#include <utility>

namespace ni {

at::Tensor pcg(
  const at::Tensor& hessian,  
  const at::Tensor& gradient,
  const at::Tensor& solution, 
  const at::Tensor& weight, 
  const std::vector<double> &  absolute, 
  const std::vector<double> &  membrane, 
  const std::vector<double> &  bending,
  const std::vector<double> &  voxel_size, 
  const std::vector<BoundType> & bound, 
  int64_t nb_iter);

at::Tensor pcg_grid(
  const at::Tensor& hessian,  
  const at::Tensor& gradient,
  const at::Tensor& solution, 
  const at::Tensor& weight, 
  double absolute, 
  double membrane, 
  double bending, 
  double lame_shear, 
  double lame_div,
  const std::vector<double> & voxel_size, 
  const std::vector<BoundType> & bound, int64_t nb_iter);

}

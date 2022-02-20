#include <ATen/ATen.h>
#include <vector>


namespace ni {
  
at::Tensor fmg(
  const at::Tensor             & hessian, 
  const at::Tensor             & gradient,
        at::Tensor               solution   = at::Tensor(),
  const at::Tensor             & weight     = at::Tensor(),
  const std::vector<double>    & absolute   = std::vector<double>(), 
  const std::vector<double>    & membrane   = std::vector<double>(), 
  const std::vector<double>    & bending    = std::vector<double>(),
  const std::vector<double>    & voxel_size = std::vector<double>(), 
  const std::vector<BoundType> & bound      = std::vector<BoundType>(),
  int64_t nb_cycles  = 2,
  int64_t nb_iter    = 2,
  int64_t max_levels = 16,
  bool    use_cg     = false);

at::Tensor fmg_grid(
  const at::Tensor             & hessian, 
  const at::Tensor             & gradient,
        at::Tensor               solution   = at::Tensor(),
  const at::Tensor             & weight     = at::Tensor(),
        double                   absolute   = 0., 
        double                   membrane   = 0., 
        double                   bending    = 0., 
        double                   lame_shear = 0., 
        double                   lame_div   = 0., 
  const std::vector<double>    & voxel_size = std::vector<double>(), 
  const std::vector<BoundType> & bound      = std::vector<BoundType>(),
  int64_t nb_cycles  = 2,
  int64_t nb_iter    = 2,
  int64_t max_levels = 16);

}
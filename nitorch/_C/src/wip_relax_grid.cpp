#include "impl/wip_relax_grid_common.h"
#include "checks.h"
#include <ATen/ATen.h>
#include <vector>
#include <utility>

#ifndef NI_WITH_CUDA
#  define cuda notimplemented
#endif

using at::Tensor;
using c10::IntArrayRef;
using c10::ArrayRef;



namespace ni {

Tensor relax_grid(
    const Tensor& hessian, const Tensor& gradient,
    const Tensor& solution, const Tensor& weight,
    double absolute, double membrane, double bending, double lame_shear, double lame_div,
    const std::vector<double> & voxel_size, const std::vector<BoundType> & bound,
    int64_t nb_iter) {

  NI_CHECK_DEFINED(gradient)
  auto gradient_opt = gradient.options();
  NI_CHECK_OPT_STRIDED(gradient_opt)
  NI_CHECK_1D_2D_OR_3D(gradient)
  NI_CHECK_NOT_EMPTY(gradient)
  NI_CHECK_VEC_NOT_EMPTY(bound);

  if (hessian.defined() && hessian.numel() > 0)
  {
    auto hessian_opt  = hessian.options();
    NI_CHECK_OPT_STRIDED(hessian_opt)
    NI_CHECK_OPT_SAME_DEVICE(gradient_opt, hessian_opt)
    NI_CHECK_OPT_SAME_DTYPE(gradient_opt, hessian_opt)
    NI_CHECK_1D_2D_OR_3D(hessian)
    NI_CHECK_NOT_EMPTY(hessian)
  }

  if (solution.defined() && solution.numel() > 0)
  {
    auto solution_opt  = solution.options();
    NI_CHECK_OPT_STRIDED(solution_opt)
    NI_CHECK_OPT_SAME_DEVICE(gradient_opt, solution_opt)
    NI_CHECK_OPT_SAME_DTYPE(gradient_opt, solution_opt)
    NI_CHECK_1D_2D_OR_3D(solution)
    NI_CHECK_NOT_EMPTY(solution)
  }

  if (weight.defined() && weight.numel() > 0)
  {
    auto weight_opt  = weight.options();
    NI_CHECK_OPT_STRIDED(weight_opt)
    NI_CHECK_OPT_SAME_DEVICE(gradient_opt, weight_opt)
    NI_CHECK_OPT_SAME_DTYPE(gradient_opt, weight_opt)
    NI_CHECK_1D_2D_OR_3D(weight)
    NI_CHECK_NOT_EMPTY(weight)
  }

  if (gradient.is_cuda())
    return cuda::relax_grid_impl(hessian, gradient, solution, weight,
        absolute, membrane, bending, lame_shear, lame_div,
        ArrayRef<double>(voxel_size), BoundVectorRef(bound), nb_iter);
  else
    return cpu::relax_grid_impl(hessian, gradient, solution, weight,
      absolute, membrane, bending, lame_shear, lame_div,
      ArrayRef<double>(voxel_size), BoundVectorRef(bound), nb_iter);
}

}

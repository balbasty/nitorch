#include "impl/regulariser_grid_common.h"
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

Tensor regulariser_grid(
    const Tensor& input, const Tensor& output, const Tensor& weight, const Tensor& hessian, 
    double absolute, double membrane, double bending, double lame_shear, double lame_div,
    const std::vector<double> & voxel_size, const std::vector<BoundType> & bound) 
{

  NI_CHECK_DEFINED(input)
  auto input_opt = input.options();
  NI_CHECK_OPT_STRIDED(input_opt)
  NI_CHECK_1D_2D_OR_3D(input)
  NI_CHECK_NOT_EMPTY(input)
  NI_CHECK_VEC_NOT_EMPTY(bound);

  if (output.defined() && output.numel() > 0)
  {
    auto output_opt  = output.options();
    NI_CHECK_OPT_STRIDED(output_opt)
    NI_CHECK_OPT_SAME_DEVICE(input_opt, output_opt)
    NI_CHECK_OPT_SAME_DTYPE(input_opt, output_opt)
    NI_CHECK_1D_2D_OR_3D(output)
    NI_CHECK_NOT_EMPTY(output)
  }

  if (weight.defined() && weight.numel() > 0)
  {
    auto weight_opt  = weight.options();
    NI_CHECK_OPT_STRIDED(weight_opt)
    NI_CHECK_OPT_SAME_DEVICE(input_opt, weight_opt)
    NI_CHECK_OPT_SAME_DTYPE(input_opt, weight_opt)
    NI_CHECK_1D_2D_OR_3D(weight)
    NI_CHECK_NOT_EMPTY(weight)
  }

  if (hessian.defined() && hessian.numel() > 0)
  {
    auto hessian_opt  = hessian.options();
    NI_CHECK_OPT_STRIDED(hessian_opt)
    NI_CHECK_OPT_SAME_DEVICE(input_opt, hessian_opt)
    NI_CHECK_OPT_SAME_DTYPE(input_opt, hessian_opt)
    NI_CHECK_1D_2D_OR_3D(hessian)
    NI_CHECK_NOT_EMPTY(hessian)
  }

  if (input.is_cuda())
    return cuda::regulariser_grid_impl(input, output, weight, hessian,
        absolute, membrane, bending, lame_shear, lame_div,
        ArrayRef<double>(voxel_size), BoundVectorRef(bound));
  else
    return cpu::regulariser_grid_impl(input, output, weight, hessian,
      absolute, membrane, bending, lame_shear, lame_div,
      ArrayRef<double>(voxel_size), BoundVectorRef(bound));
}



std::pair<Tensor,Tensor> regulariser_grid_backward(
    const Tensor& grad, const Tensor& input, const Tensor& weight, const Tensor& hessian, 
    double absolute, double membrane, double bending, double lame_shear, double lame_div,
    const std::vector<double> & voxel_size, const std::vector<BoundType> & bound, 
    bool do_input, bool do_weight) {

  Tensor grad_input = Tensor();
  Tensor grad_weight = Tensor();

  if (!(grad.defined()))
    // incoming gradient is zero -> output gradients are zero
    return std::pair<Tensor, Tensor>(grad_input, grad_weight);

  if (do_input) {
      grad_input = regulariser_grid(
        grad, Tensor(), weight, hessian,
        absolute, membrane, bending, lame_shear, lame_div,
        voxel_size, bound);
  }
  if (do_weight) {
      grad_weight = regulariser_grid( // FIXME
        grad, Tensor(), input, hessian,
        absolute, membrane, bending, lame_shear, lame_div,
        voxel_size, bound);
  }
  return std::pair<Tensor, Tensor>(grad_input, grad_weight);
}

} // namespace ni

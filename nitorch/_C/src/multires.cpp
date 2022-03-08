#include "impl/multires_common.h"
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
using std::vector;


namespace ni {

typedef at::Tensor (*ResizeFn)( \
      Tensor source, Tensor target, \
      ArrayRef<double> factor, BoundVectorRef bound,  \
      bool do_adjoint);

Tensor fmg_prolongation(const Tensor             & input, 
                        const Tensor             & output,
                        const vector<BoundType>  & bound) 
{
  NI_CHECK_DEFINED(input)
  auto input_opt = input.options();
  NI_CHECK_OPT_STRIDED(input_opt)
  NI_CHECK_1D_2D_OR_3D(input)
  NI_CHECK_NOT_EMPTY(input)

  if (output.defined() && output.numel() > 0)
  {
    auto output_opt  = output.options();
    NI_CHECK_OPT_STRIDED(output_opt)
    NI_CHECK_OPT_SAME_DEVICE(input_opt, output_opt)
    NI_CHECK_OPT_SAME_DTYPE(input_opt, output_opt)
    NI_CHECK_1D_2D_OR_3D(output)
    NI_CHECK_NOT_EMPTY(output)
  }

  ResizeFn multires_impl = input.is_cuda() ? cuda::multires_impl : cpu::multires_impl;

  return multires_impl(input, output, ArrayRef<double>(std::vector<double>({2., 2., 2.})), 
                       BoundVectorRef(bound.size() ? bound : vector<BoundType>({BoundType::DCT2})), 
                       false);
}

Tensor fmg_restriction(const Tensor             & input, 
                       const Tensor             & output,
                       const vector<BoundType>  & bound) 
{
  NI_CHECK_DEFINED(input)
  auto input_opt = input.options();
  NI_CHECK_OPT_STRIDED(input_opt)
  NI_CHECK_1D_2D_OR_3D(input)
  NI_CHECK_NOT_EMPTY(input)

  if (output.defined() && output.numel() > 0)
  {
    auto output_opt  = output.options();
    NI_CHECK_OPT_STRIDED(output_opt)
    NI_CHECK_OPT_SAME_DEVICE(input_opt, output_opt)
    NI_CHECK_OPT_SAME_DTYPE(input_opt, output_opt)
    NI_CHECK_1D_2D_OR_3D(output)
    NI_CHECK_NOT_EMPTY(output)
  }

  ResizeFn multires_impl = input.is_cuda() ? cuda::multires_impl : cpu::multires_impl;
  
  return multires_impl(input, output, ArrayRef<double>(std::vector<double>({2., 2., 2.})), 
                       BoundVectorRef(bound.size() ? bound : vector<BoundType>({BoundType::DCT2})), 
                       true);
}

} // namespace ni

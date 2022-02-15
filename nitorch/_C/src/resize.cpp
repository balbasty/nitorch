#include "impl/resize_common.h"
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
      InterpolationVectorRef interpolation, GridAlignVectorRef mode, \
      bool do_adjoint);

Tensor resize(const Tensor                     & input, 
              const Tensor                     & output,
              const vector<double>             & factor,
              const vector<BoundType>          & bound, 
              const vector<InterpolationType>  & interpolation, 
              const vector<GridAlignType>      & mode,
              bool                             do_adjoint) 
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

  at::Tensor source = do_adjoint ? input  : output;
  at::Tensor target = do_adjoint ? output : input;

  ResizeFn resize_impl = input.is_cuda() ? cuda::resize_impl : cpu::resize_impl;

  return resize_impl(source, target,
      ArrayRef<double>(       factor.size()        ? factor        : vector<double>({1.})), 
      BoundVectorRef(         bound.size()         ? bound         : vector<BoundType>({BoundType::DCT2})), 
      InterpolationVectorRef( interpolation.size() ? interpolation : vector<InterpolationType>({InterpolationType::Linear})),
      GridAlignVectorRef(     mode.size()          ? mode          : vector<GridAlignType>({GridAlignType::Center})), 
      do_adjoint);
}

Tensor prolong(const Tensor                     & input, 
               const Tensor                     & output,
               const vector<BoundType>          & bound, 
               const vector<InterpolationType>  & interpolation) 
{
  return resize(input, output,
                vector<double>({1.}),
                bound.size()          ? bound         : vector<BoundType>({BoundType::DCT2}),
                interpolation.size()  ? interpolation : vector<InterpolationType>({InterpolationType::Quadratic}),
                vector<GridAlignType>({GridAlignType::Edge}),
                false);
}

Tensor restrict(const Tensor                     & input, 
                const Tensor                     & output,
                const vector<BoundType>          & bound, 
                const vector<InterpolationType>  & interpolation) 
{
  return resize(input, output,
                vector<double>({1.}),
                bound.size()          ? bound         : vector<BoundType>({BoundType::DCT2}),
                interpolation.size()  ? interpolation : vector<InterpolationType>({InterpolationType::Linear}),
                vector<GridAlignType>({GridAlignType::Edge}),
                true);
}

Tensor resize_backward(const Tensor                     & grad, 
                       const Tensor                     & output,
                       const vector<double>             & factor,
                       const vector<BoundType>          & bound, 
                       const vector<InterpolationType>  & interpolation, 
                       const vector<GridAlignType>      & mode,
                       bool                             do_adjoint) 
{
  std::vector<double> ifactor(factor);
  for (auto it = ifactor.begin(); it != ifactor.end(); ++it)
    *it = 1./ (*it);
  return resize(grad, output, ifactor, bound, interpolation, mode, !do_adjoint);
}

Tensor prolong_backward(const Tensor                     & grad, 
                        const Tensor                     & output,
                        const vector<BoundType>          & bound, 
                        const vector<InterpolationType>  & interpolation) 
{
  vector<InterpolationType> same_inter({InterpolationType::Quadratic});
  if (interpolation.size())
    same_inter = interpolation;
  return restrict(grad, output, bound, same_inter);
}

Tensor restrict_backward(const Tensor                     & grad, 
                         const Tensor                     & output,
                         const vector<BoundType>          & bound, 
                         const vector<InterpolationType>  & interpolation) 
{
  vector<InterpolationType> same_inter({InterpolationType::Linear});
  if (interpolation.size())
    same_inter = interpolation;
  return prolong(grad, output, bound, same_inter);
}

} // namespace ni

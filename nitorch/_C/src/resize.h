#include <ATen/ATen.h>
#include "bounds.h"
#include "interpolation.h"
#include "grid_align.h"
#include <vector>
#include <utility>

namespace ni {

at::Tensor resize(
  const at::Tensor                      & input, 
  const at::Tensor                      & output,
  const std::vector<double>             & factor,
  const std::vector<BoundType>          & bound         = std::vector<BoundType>(), 
  const std::vector<InterpolationType>  & interpolation = std::vector<InterpolationType>(), 
  const std::vector<GridAlignType>      & mode          = std::vector<GridAlignType>(),
  bool                                    do_adjoint    = false);

at::Tensor prolong(
  const at::Tensor                      & input, 
  const at::Tensor                      & output        = at::Tensor(),
  const std::vector<BoundType>          & bound         = std::vector<BoundType>(),
  const std::vector<InterpolationType>  & interpolation = std::vector<InterpolationType>());

at::Tensor restrict(
  const at::Tensor                      & input, 
  const at::Tensor                      & source        = at::Tensor(),
  const std::vector<BoundType>          & bound         = std::vector<BoundType>(),
  const std::vector<InterpolationType>  & interpolation = std::vector<InterpolationType>());

at::Tensor resize_backward(
  const at::Tensor                      & grad, 
  const at::Tensor                      & output,
  const std::vector<double>             & factor,
  const std::vector<BoundType>          & bound         = std::vector<BoundType>(), 
  const std::vector<InterpolationType>  & interpolation = std::vector<InterpolationType>(), 
  const std::vector<GridAlignType>      & mode          = std::vector<GridAlignType>(),
  bool                                    do_adjoint    = false);

at::Tensor prolong_backward(
  const at::Tensor                      & grad, 
  const at::Tensor                      & output        = at::Tensor(),
  const std::vector<BoundType>          & bound         = std::vector<BoundType>(),
  const std::vector<InterpolationType>  & interpolation = std::vector<InterpolationType>());

at::Tensor restrict_backward(
  const at::Tensor                      & grad, 
  const at::Tensor                      & source        = at::Tensor(),
  const std::vector<BoundType>          & bound         = std::vector<BoundType>(),
  const std::vector<InterpolationType>  & interpolation = std::vector<InterpolationType>());

}

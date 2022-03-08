#include <ATen/ATen.h>
#include "bounds.h"
#include "interpolation.h"
#include "grid_align.h"
#include <vector>
#include <utility>

namespace ni {

at::Tensor fmg_prolongation(
  const at::Tensor                      & input, 
  const at::Tensor                      & output        = at::Tensor(),
  const std::vector<BoundType>          & bound         = std::vector<BoundType>());

at::Tensor fmg_restriction(
  const at::Tensor                      & input, 
  const at::Tensor                      & output        = at::Tensor(),
  const std::vector<BoundType>          & bound         = std::vector<BoundType>());

}

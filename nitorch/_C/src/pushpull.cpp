#include "impl/pushpull_common.h"
#include <ATen/ATen.h>
#include <vector>
#include <deque>
#include <iostream>

#ifdef NI_WITH_CUDA
#  define cudapushpull cuda::pushpull
#else
#  define cudapushpull notimplemented::pushpull
#endif

using at::Tensor;
using c10::IntArrayRef;

namespace ni {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ REUSABLE CHECKS ~~~~~~~~~~~~~~~~~~~~~~~~~~
#define PUSHPULL_CHECK_DEFINED(value)                                  \
  TORCH_CHECK(                                                         \
    value.defined(),                                                   \
    "(): expected " #value " not be undefined, but it is ", value);
#define PUSHPULL_CHECK_OPT_STRIDED(value)                              \
  TORCH_CHECK(                                                         \
    value.layout() == at::kStrided,                                    \
    "(): expected " #value "to have torch.strided layout, "            \
    "but it has ", value.layout());
#define PUSHPULL_CHECK_1D_2D_OR_3D(value)                              \
  TORCH_CHECK(                                                         \
    (value.dim() == 3 || value.dim() == 4 || value.dim() == 5),        \
    "(): expected 3D, 4D or 5D " #value " but got input with sizes ",  \
    value.sizes());
#define PUSHPULL_CHECK_GRID_COMPONENT(value, dim)                      \
  TORCH_CHECK(                                                         \
    value.size(-1) == dim - 2,                                         \
    "(): expected " #value " to have size ", dim - 2, " in last "      \
    "dimension, but got " #value " with sizes ", value.sizes());
#define PUSHPULL_CHECK_OPT_SAME_DEVICE(value1, value2)                 \
    TORCH_CHECK(                                                       \
    value1.device() == value2.device(),                                \
    "(): expected " #value1 " and " #value2 " to be on same device, "  \
    "but " #value1 " is on ", value1.device(), " and " #value2         \
    " is on ", value2.device());
#define PUSHPULL_CHECK_OPT_SAME_DTYPE(value1, value2)                  \
    TORCH_CHECK(                                                       \
    value1.dtype() == value2.dtype(),                                  \
    "(): expected " #value1 " and " #value2 " to have the same dtype," \
    " but " #value1 " has ", value1.dtype(), " and " #value2 " has ",  \
    value2.dtype());
#define PUSHPULL_CHECK_NOT_EMPTY(value)                                \
  for (int64_t i = 2; i < value.dim(); i++) {                          \
    TORCH_CHECK(value.size(i) > 0,                                     \
      "(): expected " #value " to have non-empty spatial dimensions, " \
      "but input has sizes ", value.sizes(), " with dimension ", i,    \
      " being empty"); }
#define PUSHPULL_CHECK_GRID_TARGET_COMPAT(value1, value2)              \
    TORCH_CHECK(                                                       \
    value2.size(0) == value1.size(0) &&                                \
    (value2.dim() <= 2 || value2.size(2) == value1.size(1)) &&         \
    (value2.dim() <= 3 || value2.size(3) == value1.size(2)) &&         \
    (value2.dim() <= 4 || value2.size(4) == value1.size(3)),           \
    "(): expected " #value2 " and " #value1 " to have same batch, "    \
    "width, height and (optionally) depth sizes, but got " #value2     \
    " with sizes", value2.sizes(), " and " #value1 " with sizes ",     \
    value1.sizes());
#define PUSHPULL_CHECK_LENGTH(value, dim)                              \
  TORCH_CHECK(                                                         \
    ((int64_t)(value.size()) == dim - 2),                              \
    "(): expected ", dim, #value " elements but got ", value.size());
#define PUSHPULL_CHECK_VEC_NOT_EMPTY(value)                            \
  TORCH_CHECK(!value.empty(), "(): expected non empty parameter "      \
    #value );


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PULL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_pull(const Tensor& input, const Tensor& grid,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate)  {

  PUSHPULL_CHECK_DEFINED(input)
  PUSHPULL_CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt  = grid.options();
  PUSHPULL_CHECK_OPT_STRIDED(input_opt)
  PUSHPULL_CHECK_OPT_STRIDED(grid_opt)
  PUSHPULL_CHECK_OPT_SAME_DEVICE(input_opt, grid_opt)
  PUSHPULL_CHECK_OPT_SAME_DTYPE(input_opt, grid_opt)
  PUSHPULL_CHECK_1D_2D_OR_3D(input)
  PUSHPULL_CHECK_1D_2D_OR_3D(grid)
  PUSHPULL_CHECK_GRID_COMPONENT(grid, grid.dim())
  PUSHPULL_CHECK_NOT_EMPTY(input)
  PUSHPULL_CHECK_NOT_EMPTY(grid)
  PUSHPULL_CHECK_VEC_NOT_EMPTY(bound_mode);
  PUSHPULL_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (input.is_cuda())
    return cudapushpull(input, grid, 
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/true, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/false).front();
  else
    return cpu::pushpull(input, grid, 
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/true, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/false).front();
}

std::deque<Tensor>
grid_pull_backward(const Tensor& grad, const Tensor& input, const Tensor& grid,
                   const std::vector<BoundType> & bound_mode, 
                   const std::vector<InterpolationType> & interpolation_mode, 
                   int extrapolate)
{
  if (input.is_cuda()) {
    return cudapushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,
      /*push*/input.requires_grad(), /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  } else {
    return cpu::pushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,
      /*push*/input.requires_grad(), /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PUSH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_push(const Tensor& input, const Tensor& grid,
                 IntArrayRef source_size,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate) {

  PUSHPULL_CHECK_DEFINED(input)
  PUSHPULL_CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt  = grid.options();
  PUSHPULL_CHECK_OPT_STRIDED(input_opt)
  PUSHPULL_CHECK_OPT_STRIDED(grid_opt)
  PUSHPULL_CHECK_OPT_SAME_DEVICE(input_opt, grid_opt)
  PUSHPULL_CHECK_OPT_SAME_DTYPE(input_opt, grid_opt)
  PUSHPULL_CHECK_1D_2D_OR_3D(input)
  PUSHPULL_CHECK_1D_2D_OR_3D(grid)
  PUSHPULL_CHECK_GRID_COMPONENT(grid, grid.dim())
  PUSHPULL_CHECK_NOT_EMPTY(input)
  PUSHPULL_CHECK_NOT_EMPTY(grid)
  PUSHPULL_CHECK_GRID_TARGET_COMPAT(grid, input)
  PUSHPULL_CHECK_VEC_NOT_EMPTY(bound_mode);
  PUSHPULL_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (source_size.empty())
  {
    auto size = IntArrayRef({input.dim() >= 3 ? input.size(2) : 1,
                             input.dim() >= 4 ? input.size(3) : 1,
                             input.dim() >= 5 ? input.size(4) : 1});
    if (input.is_cuda())
      return cudapushpull(size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();
    else
      return cpu::pushpull(size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();
  } 
  else 
  {
    PUSHPULL_CHECK_LENGTH(source_size, grid.dim())
    if (input.is_cuda())
      return cudapushpull(source_size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();
    else
      return cpu::pushpull(source_size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();

  }
}

std::deque<Tensor>
grid_push_backward(const Tensor& grad, const Tensor& input, const Tensor& grid,
                   const std::vector<BoundType> & bound_mode, 
                   const std::vector<InterpolationType> & interpolation_mode, 
                   int extrapolate)
{
  if (input.is_cuda()) {
    return cudapushpull(grad, grid, input,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/input.requires_grad(), /*push*/false, /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  } else {
    return cpu::pushpull(grad, grid, input,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/input.requires_grad(), /*push*/false, /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COUNT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_count(const Tensor& grid,
                 IntArrayRef source_size,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate) {

  PUSHPULL_CHECK_DEFINED(grid)
  auto grid_opt  = grid.options();
  PUSHPULL_CHECK_OPT_STRIDED(grid_opt)
  PUSHPULL_CHECK_1D_2D_OR_3D(grid)
  PUSHPULL_CHECK_GRID_COMPONENT(grid, grid.dim())
  PUSHPULL_CHECK_NOT_EMPTY(grid)
  PUSHPULL_CHECK_VEC_NOT_EMPTY(bound_mode);
  PUSHPULL_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (source_size.empty())
  {
    auto size = IntArrayRef({grid.dim() >= 3 ? grid.size(2) : 1,
                             grid.dim() >= 4 ? grid.size(3) : 1,
                             grid.dim() >= 5 ? grid.size(4) : 1});
    if (grid.is_cuda())
      return cudapushpull(size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();
    else
      return cpu::pushpull(size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();
  } 
  else 
  {
    PUSHPULL_CHECK_LENGTH(source_size, grid.dim())
    if (grid.is_cuda())
      return cudapushpull(source_size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();
    else
      return cpu::pushpull(source_size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();

  }
}

Tensor
grid_count_backward(const Tensor& grad, const Tensor& grid,
                    const std::vector<BoundType> & bound_mode, 
                    const std::vector<InterpolationType> & interpolation_mode, 
                    int extrapolate)
{
  if (grid.is_cuda()) {
    return cudapushpull(grad, grid,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/ false, /*push*/ false, /*count*/ false,
      /*grad*/ grid.requires_grad(), /*sgrad*/ false).front();
  } else {
    return cpu::pushpull(grad, grid,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/ false, /*push*/ false, /*count*/ false,
      /*grad*/ grid.requires_grad(), /*sgrad*/ false).front();
  }
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~ PULL GRADIENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_grad(const Tensor& input, const Tensor& grid,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate)  {

  PUSHPULL_CHECK_DEFINED(input)
  PUSHPULL_CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt  = grid.options();
  PUSHPULL_CHECK_OPT_STRIDED(input_opt)
  PUSHPULL_CHECK_OPT_STRIDED(grid_opt)
  PUSHPULL_CHECK_OPT_SAME_DEVICE(input_opt, grid_opt)
  PUSHPULL_CHECK_OPT_SAME_DTYPE(input_opt, grid_opt)
  PUSHPULL_CHECK_1D_2D_OR_3D(input)
  PUSHPULL_CHECK_1D_2D_OR_3D(grid)
  PUSHPULL_CHECK_GRID_COMPONENT(grid, grid.dim())
  PUSHPULL_CHECK_NOT_EMPTY(input)
  PUSHPULL_CHECK_NOT_EMPTY(grid)
  PUSHPULL_CHECK_VEC_NOT_EMPTY(bound_mode);
  PUSHPULL_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (input.is_cuda())
    return cudapushpull(input, grid, 
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/true).front();
  else
    return cpu::pushpull(input, grid, 
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/true).front();
}

std::deque<Tensor>
grid_grad_backward(const Tensor& grad, const Tensor& input, const Tensor& grid,
                   const std::vector<BoundType> & bound_mode, 
                   const std::vector<InterpolationType> & interpolation_mode, 
                   int extrapolate)
{
  if (input.is_cuda()) {
    return cudapushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,  /*push*/input.requires_grad(),
      /*count*/false, /*grad*/grid.requires_grad(), /*sgrad*/false);
  } else {
    return cpu::pushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,  /*push*/input.requires_grad(),
      /*count*/false, /*grad*/grid.requires_grad(), /*sgrad*/false);
  }
}

} // namespace ni

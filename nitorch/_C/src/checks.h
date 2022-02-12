#pragma once
#include <ATen/ATen.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ REUSABLE CHECKS ~~~~~~~~~~~~~~~~~~~~~~~~~~
#define NI_CHECK_DEFINED(value)                                        \
  TORCH_CHECK(                                                         \
    value.defined(),                                                   \
    "(): expected " #value " not be undefined, but it is ", value);
#define NI_CHECK_OPT_STRIDED(value)                                    \
  TORCH_CHECK(                                                         \
    value.layout() == at::kStrided,                                    \
    "(): expected " #value "to have torch.strided layout, "            \
    "but it has ", value.layout());
#define NI_CHECK_1D_2D_OR_3D(value)                                    \
  TORCH_CHECK(                                                         \
    (value.dim() == 3 || value.dim() == 4 || value.dim() == 5),        \
    "(): expected 3D, 4D or 5D " #value " but got input with sizes ",  \
    value.sizes());
#define NI_CHECK_GRID_COMPONENT(value, dim)                            \
  TORCH_CHECK(                                                         \
    value.size(-1) == dim - 2,                                         \
    "(): expected " #value " to have size ", dim - 2, " in last "      \
    "dimension, but got " #value " with sizes ", value.sizes());
#define NI_CHECK_OPT_SAME_DEVICE(value1, value2)                       \
    TORCH_CHECK(                                                       \
    value1.device() == value2.device(),                                \
    "(): expected " #value1 " and " #value2 " to be on same device, "  \
    "but " #value1 " is on ", value1.device(), " and " #value2         \
    " is on ", value2.device());
#define NI_CHECK_OPT_SAME_DTYPE(value1, value2)                        \
    TORCH_CHECK(                                                       \
    value1.dtype() == value2.dtype(),                                  \
    "(): expected " #value1 " and " #value2 " to have the same dtype," \
    " but " #value1 " has ", value1.dtype(), " and " #value2 " has ",  \
    value2.dtype());
#define NI_CHECK_NOT_EMPTY(value)                                      \
  for (int64_t i = 2; i < value.dim(); i++) {                          \
    TORCH_CHECK(value.size(i) > 0,                                     \
      "(): expected " #value " to have non-empty spatial dimensions, " \
      "but input has sizes ", value.sizes(), " with dimension ", i,    \
      " being empty"); }
#define NI_CHECK_GRID_TARGET_COMPAT(value1, value2)                    \
    TORCH_CHECK(                                                       \
    value2.size(0) == value1.size(0) &&                                \
    (value2.dim() <= 2 || value2.size(2) == value1.size(1)) &&         \
    (value2.dim() <= 3 || value2.size(3) == value1.size(2)) &&         \
    (value2.dim() <= 4 || value2.size(4) == value1.size(3)),           \
    "(): expected " #value2 " and " #value1 " to have same batch, "    \
    "width, height and (optionally) depth sizes, but got " #value2     \
    " with sizes", value2.sizes(), " and " #value1 " with sizes ",     \
    value1.sizes());
#define NI_CHECK_VEC_LENGTH(value, dim)                                \
  TORCH_CHECK(                                                         \
    ((int64_t)(value.size()) == dim - 2),                              \
    "(): expected ", dim, #value " elements but got ", value.size());
#define NI_CHECK_VEC_NOT_EMPTY(value)                                  \
  TORCH_CHECK(!value.empty(), "(): expected non empty parameter "      \
    #value );

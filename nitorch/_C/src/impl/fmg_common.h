#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"


#define NI_FMG_DECLARE(space) \
  namespace space { \
    at::Tensor fmg_impl(const at::Tensor & hessian, \
                const at::Tensor & gradient, \
                      at::Tensor   solution, \
                const at::Tensor & weight, \
                const c10::ArrayRef<double> &  absolute,  \
                const c10::ArrayRef<double> &  membrane,  \
                const c10::ArrayRef<double> &  bending,   \
                const c10::ArrayRef<double> &  voxel_size,  \
                const BoundVectorRef        & bound, \
                int64_t nb_cycles,  \
                int64_t nb_iter,    \
                int64_t max_levels, \
                bool use_cg); \
    at::Tensor fmg_grid_impl(const at::Tensor & hessian, \
                const at::Tensor & gradient, \
                      at::Tensor   solution, \
                const at::Tensor & weight, \
                double  absolute, \
                double  membrane, \
                double  bending,  \
                double  lame_shear, \
                double  lame_div, \
                const c10::ArrayRef<double> &  voxel_size, \
                const BoundVectorRef   & bound, \
                int64_t nb_cycles,  \
                int64_t nb_iter,    \
                int64_t max_levels, \
                bool use_cg); \
  }


namespace ni {
NI_FMG_DECLARE(cpu)
NI_FMG_DECLARE(cuda)
NI_FMG_DECLARE(notimplemented)
} // namespace ni

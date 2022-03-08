#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"


#define NI_PCG_DECLARE(space) \
  namespace space { \
    at::Tensor pcg_impl(const at::Tensor & hessian, \
                const at::Tensor & gradient, \
                      at::Tensor   solution, \
                const at::Tensor & weight, \
                const c10::ArrayRef<double> &  absolute,  \
                const c10::ArrayRef<double> &  membrane,  \
                const c10::ArrayRef<double> &  bending,   \
                const c10::ArrayRef<double> &  voxel_size,  \
                const BoundVectorRef        & bound, \
                int64_t nb_iter, double tol=0.); \
    at::Tensor pcg_grid_impl(const at::Tensor & hessian, \
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
                int64_t nb_iter, double tol=0.); \
  }


namespace ni {
NI_PCG_DECLARE(cpu)
NI_PCG_DECLARE(cuda)
NI_PCG_DECLARE(notimplemented)
} // namespace ni

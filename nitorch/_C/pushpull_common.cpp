// This file implements spline interpolation / sampling and its adjoint 
// operations. It corresponds loosely to torch's `GridSampler`.
// It handles boundary conditions and interpolation orders defined in
// `bounds.h` and `interpolation.h`. These parameters can be specified 
// per dimension.
// Isotorpic 0-th and 1-st order interpolation have their own (faster)
// implementations. Sliding boundary conditions are also implemented 
// separately.

#include "common.h"
#include "bounds_common.h"
#include "interpolation_common.h"
#include <ATen/ATen.h>
#include <tuple>
//#include <cstdio>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CPU/GPU -specific parameters
#ifdef __CUDACC__
# include <ATen/cuda/CUDAContext.h>
# include <ATen/cuda/detail/KernelUtils.h>
# include <c10/macros/Macros.h>
  using namespace at::cuda::detail;
#else
# include <ATen/Parallel.h>
  namespace {
    // This parameter specifies the minimum number of voxels that should be 
    // processed on a single processor in the parallel for loop .
    int64_t GRAIN_SIZE = static_cast<int64_t>(at::internal::GRAIN_SIZE);
  }
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using at::Tensor;
using at::TensorOptions;
using c10::IntArrayRef;

namespace ni {
NI_NAMESPACE_DEVICE { // cpu / cuda / ...

// anonymous namespace > everything inside has internal linkage
namespace {

// This parameter allows for a little bit of tolerance when considering 
// a coordinate as "out-of-bound" (if !extrapolate)
#define TINY 5e-2

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        GENERIC PUSHPULL CLASS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This class implements the bulk of the code.
// /!\ No type and shape checking is performed here.

template <typename scalar_t, typename offset_t>
class PushPullImpl {
public:

  // ~~~ CONSTRUCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST
  PushPullImpl(int dim, BoundVectorRef bound, InterpolationVectorRef interpolation, 
               bool extrapolate, bool do_pull, bool do_push, bool do_grad):
    dim(dim),
    bound0(bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound1(bound.size() > 1 ? bound[1] : 
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound2(bound.size() > 2 ? bound[2] : 
           bound.size() > 1 ? bound[1] : 
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    interpolation0(interpolation.size() > 0 ? interpolation[0] : InterpolationType::Linear),
    interpolation1(interpolation.size() > 1 ? interpolation[1] : 
                   interpolation.size() > 0 ? interpolation[0] : InterpolationType::Linear),
    interpolation2(interpolation.size() > 2 ? interpolation[2] : 
                   interpolation.size() > 1 ? interpolation[1] : 
                   interpolation.size() > 0 ? interpolation[0] : InterpolationType::Linear),
    extrapolate(extrapolate),
    do_pull(do_pull),
    do_push(do_push),
    do_grad(do_grad)
  {
    iso = interpolation0 == interpolation1 && 
          interpolation0 == interpolation2;
  }

  NI_HOST
  PushPullImpl(int dim, BoundType bound, InterpolationVectorRef interpolation, 
               bool extrapolate, bool do_pull, bool do_push, bool do_grad):
    dim(dim),
    bound0(bound),
    bound1(bound),
    bound2(bound),
    interpolation0(interpolation.size() > 0 ? interpolation[0] : InterpolationType::Linear),
    interpolation1(interpolation.size() > 1 ? interpolation[1] : 
                   interpolation.size() > 0 ? interpolation[0] : InterpolationType::Linear),
    interpolation2(interpolation.size() > 2 ? interpolation[2] : 
                   interpolation.size() > 1 ? interpolation[1] : 
                   interpolation.size() > 0 ? interpolation[0] : InterpolationType::Linear),
    extrapolate(extrapolate),
    do_pull(do_pull),
    do_push(do_push),
    do_grad(do_grad)
  {
    iso = interpolation0 == interpolation1 && 
          interpolation0 == interpolation2;
  }

  NI_HOST
  PushPullImpl(int dim, BoundVectorRef bound, InterpolationType interpolation, 
               bool extrapolate, bool do_pull, bool do_push, bool do_grad):
     dim(dim),
     bound0(bound.size() > 0 ? bound[0] : BoundType::Replicate),
     bound1(bound.size() > 1 ? bound[1] : 
            bound.size() > 0 ? bound[0] : BoundType::Replicate),
     bound2(bound.size() > 2 ? bound[2] : 
            bound.size() > 1 ? bound[1] : 
            bound.size() > 0 ? bound[0] : BoundType::Replicate),
    interpolation0(interpolation),
    interpolation1(interpolation),
    interpolation2(interpolation),
    extrapolate(extrapolate),
    do_pull(do_pull),
    do_push(do_push),
    do_grad(do_grad)
  {
    iso = interpolation0 == interpolation1 && 
          interpolation0 == interpolation2;
  }

  NI_HOST
  PushPullImpl(int dim, BoundType bound, InterpolationType interpolation, 
               bool extrapolate, bool do_pull, bool do_push, bool do_grad):
    dim(dim),
    bound0(bound),
    bound1(bound),
    bound2(bound),
    interpolation0(interpolation),
    interpolation1(interpolation),
    interpolation2(interpolation),
    extrapolate(extrapolate),
    do_pull(do_pull),
    do_push(do_push),
    do_grad(do_grad)
    {
      iso = interpolation0 == interpolation1 && 
            interpolation0 == interpolation2;
    }

  // ~~~ PUBLIC VALUE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::deque<Tensor> output;

  // NI_HOST NI_DEVICE void printInfo() const {
  //   printf("src:  [%d %d %d]\n", src_W, src_H, src_D);
  //   printf("trgt: [%d %d %d]\n", trgt_W, trgt_H, trgt_D);
  //   printf("N: %d\n", N);
  //   printf("C: %d\n", C);
  //   printf("src  -> %d\n", src_ptr);
  //   printf("trgt -> %d\n", trgt_ptr);
  //   printf("grid -> %d\n", grid_ptr);
  //   printf("push -> %d\n", push_ptr);
  //   printf("pull -> %d\n", pull_ptr);
  //   printf("grad -> %d\n", grad_ptr);
  // }

  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void ioset // Pull
  (const Tensor& source, const Tensor& grid)
  {
    init_source(source);
    init_grid(grid);
    init_output();
  }

  NI_HOST void ioset // Pull+Push+Grad
  (const Tensor& source, const Tensor& grid, const Tensor& target)
  {
    init_source(source);
    init_grid(grid);
    init_target(target);
    init_output();
  }

  NI_HOST void ioset // Push
  (IntArrayRef source_size, const Tensor& grid, const Tensor& target)
  {
    init_source(source_size);
    init_grid(grid);
    init_target(target);
    init_output();
  }

#if __CUDACC__
  NI_DEVICE void loop(int threadIdx, int blockIdx, 
                      int blockDim, int gridDim) const;
#else
  void loop() const;
#endif

  NI_HOST NI_DEVICE int64_t voxcount() const { 
    return N * trgt_D * trgt_H * trgt_W;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void init_source(const Tensor& source);
  NI_HOST void init_source(IntArrayRef source_size);
  NI_HOST void init_grid(const Tensor& source); 
  NI_HOST void init_target(const Tensor& source); 
  NI_HOST void init_output();
  NI_DEVICE void check2d(offset_t w, offset_t h, offset_t n) const;
  NI_DEVICE void check3d(offset_t w, offset_t h, offset_t d, offset_t n) const;
  NI_DEVICE void interpolate2d(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate2d_nearest(
    scalar_t x, scalar_t y,
     offset_t w, offset_t h, offset_t n) const;
  NI_DEVICE void interpolate2d_bilinear(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h,  offset_t n) const;
  NI_DEVICE void interpolate2d_sliding(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate2d_sliding_nearest(
    scalar_t x, scalar_t y, 
    offset_t w, offset_t h, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate2d_sliding_bilinear(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate3d(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate3d_nearest(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const;
  NI_DEVICE void interpolate3d_trilinear(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const;
  NI_DEVICE void interpolate3d_sliding(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate3d_sliding_nearest(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate3d_sliding_trilinear(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const {/*TODO*/}

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;
  BoundType         bound0;
  BoundType         bound1;
  BoundType         bound2;
  InterpolationType interpolation0;
  InterpolationType interpolation1;
  InterpolationType interpolation2;
  bool              iso;
  bool              extrapolate;
  bool              do_pull;
  bool              do_push;
  bool              do_grad;

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  TensorOptions src_opt;
  TensorOptions grid_opt;
  TensorOptions trgt_opt;
  offset_t N;
  offset_t C;
  offset_t src_D;
  offset_t src_H;
  offset_t src_W;
  offset_t trgt_D;
  offset_t trgt_H;
  offset_t trgt_W;
  offset_t src_sN;
  offset_t src_sC;
  offset_t src_sD;
  offset_t src_sH;
  offset_t src_sW;
  scalar_t *src_ptr;
  offset_t trgt_sN;
  offset_t trgt_sC;
  offset_t trgt_sD;
  offset_t trgt_sH;
  offset_t trgt_sW;
  scalar_t *trgt_ptr;
  offset_t grid_sN;
  offset_t grid_sC;
  offset_t grid_sD;
  offset_t grid_sH;
  offset_t grid_sW;
  scalar_t *grid_ptr;
  offset_t pull_sN;
  offset_t pull_sC;
  offset_t pull_sD;
  offset_t pull_sH;
  offset_t pull_sW;
  scalar_t *pull_ptr;
  offset_t push_sN;
  offset_t push_sC;
  offset_t push_sD;
  offset_t push_sH;
  offset_t push_sW;
  scalar_t *push_ptr;
  offset_t grad_sN;
  offset_t grad_sC;
  offset_t grad_sD;
  offset_t grad_sH;
  offset_t grad_sW;
  scalar_t *grad_ptr;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          INITIALISATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_source(const Tensor& source)
{
 N       = source.size(0);
 C       = source.size(1);
 src_D   = dim == 2 ? 1 : source.size(2);
 src_H   = source.size(dim == 2 ? 2 : 3);
 src_W   = source.size(dim == 2 ? 3 : 4);
 src_sN  = source.stride(0);
 src_sC  = source.stride(1);
 src_sD  = dim == 2 ? 0 : source.stride(2);
 src_sH  = source.stride(dim == 2 ? 2 : 3);
 src_sW  = source.stride(dim == 2 ? 3 : 4);
 src_ptr = source.data_ptr<scalar_t>();
 src_opt = source.options();
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_source(IntArrayRef source_size)
{
 src_W = source_size[0];
 src_H = source_size[1];
 src_D = dim == 2 ? 1 : source_size[2];
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_grid(const Tensor& grid)
{
  N        = grid.size(0);
  trgt_D   = dim == 2 ? 0 : grid.size(1);
  trgt_H   = grid.size(dim == 2 ? 1 : 2);
  trgt_W   = grid.size(dim == 2 ? 2 : 3);
  grid_sN  = grid.stride(0);
  grid_sD  = dim == 2 ? 0 : grid.stride(1);
  grid_sH  = grid.stride(dim == 2 ? 1 : 2);
  grid_sW  = grid.stride(dim == 2 ? 2 : 3);
  grid_sC  = grid.stride(dim == 2 ? 3 : 4);
  grid_ptr = grid.data_ptr<scalar_t>();
  grid_opt = grid.options();
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_target(const Tensor& target)
{
 N        = target.size(0);
 C        = target.size(1);
 trgt_D   = dim == 2 ? 1 : target.size(2);
 trgt_H   = target.size(dim == 2 ? 2 : 3);
 trgt_W   = target.size(dim == 2 ? 3 : 4);
 trgt_sN  = target.stride(0);
 trgt_sC  = target.stride(1);
 trgt_sD  = dim == 2 ? 0 : target.stride(2);
 trgt_sH  = target.stride(dim == 2 ? 2 : 3);
 trgt_sW  = target.stride(dim == 2 ? 3 : 4);
 trgt_ptr = target.data_ptr<scalar_t>();
 trgt_opt = target.options();
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_output()
{
  output.clear();
  if (do_pull) {
    if (dim == 2)
      output.push_back(at::empty({N, C, trgt_H, trgt_W}, src_opt));
    else
      output.push_back(at::empty({N, C, trgt_D, trgt_H, trgt_W}, src_opt));
    auto pull = output.back();
    pull_sN   = pull.stride(0);
    pull_sC   = pull.stride(1);
    pull_sD   = dim == 2 ? 0 : pull.stride(2);
    pull_sH   = pull.stride(dim == 2 ? 2 : 3);
    pull_sW   = pull.stride(dim == 2 ? 3 : 4);
    pull_ptr  = pull.data_ptr<scalar_t>();
  }
  if (do_push) {
    if (dim == 2)
      output.push_back(at::zeros({N, C, src_H, src_W}, trgt_opt));
    else
      output.push_back(at::zeros({N, C, src_D, src_H, src_W}, trgt_opt));
    auto push = output.back();
    push_sN   = push.stride(0);
    push_sC   = push.stride(1);
    push_sD   = dim == 2 ? 0 : push.stride(2);
    push_sH   = push.stride(dim == 2 ? 2 : 3);
    push_sW   = push.stride(dim == 2 ? 3 : 4);
    push_ptr  = push.data_ptr<scalar_t>();
  }

  if (do_grad) {
    if (dim == 2)
      output.push_back(at::zeros({N, src_H, src_W, 2}, grid_opt));
    else
      output.push_back(at::zeros({N, src_D, src_H, src_W, 3}, grid_opt));
    auto grad = output.back();
    grad_sN   = grad.stride(0);
    grad_sD   = dim == 2 ? 0 : grad.stride(1);
    grad_sH   = grad.stride(dim == 2 ? 1 : 2);
    grad_sW   = grad.stride(dim == 2 ? 2 : 3);
    grad_sC   = grad.stride(dim == 2 ? 3 : 4);
    grad_ptr  = grad.data_ptr<scalar_t>();

    // TODO
    // If interpolation mode is Nearest, then grad_grid is not filled in the
    // loop below.
    // if ( == 0)
    //   grad.zero_();
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  int64_t index = blockIdx * blockDim + threadIdx;
  int64_t nthreads = voxcount();
  offset_t trgt_DHW  = trgt_W * trgt_H * trgt_D;
  offset_t trgt_HW   = trgt_W * trgt_H;
  offset_t n, d, h, w;
  for (offset_t i=index; index < nthreads; index += blockDim*gridDim, i=index) {
      // Convert index: linear to sub
      n  = (i/trgt_DHW);
      d  = (i/trgt_HW) % trgt_D;
      h  = (i/trgt_W)  % trgt_H;
      w  = i % trgt_W;

      if (dim == 2)
        check2d(w, h, n);
      else
        check3d(w, h, d, n);
  }
}

#else

// This bit loops over all target voxels. We therefore need to
// convert linear indices to multivariate indices. The way I do it
// might not be optimal.
// Note that I parallelize across all voxels (wheareas ATen's grid 
// sampler is only parallelized across batches).
//
// TODO: check that the default grain size is optimal. We do quite a lot 
// of compute per voxel, so a smaller value might be better suited.
template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::loop() const
{
# if !(AT_PARALLEL_OPENMP)
    if (do_push)
    {
      // I do not have access to atomic operations so I cannot 
      // parallelize across voxels.  
      at::parallel_for(0, N, 0, [&](offset_t start, offset_t end) {
        for (offset_t n = start; n < end; ++n) {
          if (dim == 2) {
            for (offset_t h=0; h<trgt_H; ++h)
            for (offset_t w=0; w<trgt_W; ++w)
              check2d(w, h, n);
          } else {
            for (offset_t d=0; d<trgt_D; ++d)
            for (offset_t h=0; h<trgt_H; ++h)
            for (offset_t w=0; w<trgt_W; ++w)
              check3d(w, h, d, n);
          }
        }
      }); 
      return
    }
#  endif

  // Parallelize across voxels   
  offset_t trgt_NDHW = trgt_W * trgt_H * trgt_D * N;
  offset_t trgt_DHW  = trgt_W * trgt_H * trgt_D;
  offset_t trgt_HW   = trgt_W * trgt_H;
  at::parallel_for(0, trgt_NDHW, GRAIN_SIZE, 
                   [&](offset_t start, offset_t end) {
    offset_t n, d, h, w;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/trgt_DHW);
      d  = (i/trgt_HW) % trgt_D;
      h  = (i/trgt_W)  % trgt_H;
      w  = i % trgt_W;

      if (dim == 2)
        check2d(w, h, n);
      else
        check3d(w, h, d, n);
    }
  }); 
}

#endif

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        CHECK OUT-OF-BOUND
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Here, we:
// 1) read the [x,y,z] source coordinate for the current target voxel
// 3) check if the source coordinate is in bounds 

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>
::check2d(offset_t w, offset_t h, offset_t n) const
{
  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t *grid_ptr_NHW = grid_ptr + n * grid_sN
                                    + h * grid_sH 
                                    + w * grid_sW;
  scalar_t x = *grid_ptr_NHW;
  scalar_t y = grid_ptr_NHW[grid_sC];

  // Check if out-of-bound
  if (!(extrapolate || inbounds(x, src_W, static_cast<scalar_t>(TINY))
                    || inbounds(y, src_H, static_cast<scalar_t>(TINY)))) {
    if (do_pull) {
      scalar_t *pull_ptr_NCHW = pull_ptr + n * pull_sN
                                         + h * pull_sH 
                                         + w * pull_sW;
      for (offset_t c = 0; c < C; ++c, pull_ptr_NCHW += pull_sC)
        *pull_ptr_NCHW = static_cast<scalar_t>(0);
    }
    if (do_grad) {
      scalar_t * grad_ptr_NHW = grad_ptr + n * grad_sN
                                         + h * grad_sH 
                                         + w * grad_sW;
      (*grad_ptr_NHW) = static_cast<scalar_t>(0);
      grad_ptr_NHW[grad_sC] = static_cast<scalar_t>(0);
    }
    return;
  }

  // Next step
  if (bound0 == BoundType::Sliding) {
    if (iso) switch (static_cast<int>(interpolation0)) {
      case 0: return interpolate2d_sliding_nearest(x, y, w, h, n);
      case 1: return interpolate2d_sliding_bilinear(x, y, w, h, n);
    }
    return interpolate2d_sliding(x, y, w, h, n);
  } else {
    if (iso) switch (static_cast<int>(interpolation0)) {
      case 0: return interpolate2d_nearest(x, y, w, h, n);
      case 1: return interpolate2d_bilinear(x, y, w, h, n);
    } 
    return interpolate2d(x, y, w, h, n);
  }
 
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>
::check3d(offset_t w, offset_t h, offset_t d, offset_t n) const
{
  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t *grid_ptr_NDHW = grid_ptr + n * grid_sN + d * grid_sD 
                                     + h * grid_sH + w * grid_sW;
  scalar_t x = *grid_ptr_NDHW;
  scalar_t y = grid_ptr_NDHW[grid_sC];
  scalar_t z = grid_ptr_NDHW[grid_sC*2];

  // Check if out-of-bound
  if (!(extrapolate || inbounds(x, src_W, static_cast<scalar_t>(TINY))
                    || inbounds(y, src_H, static_cast<scalar_t>(TINY))
                    || inbounds(z, src_D, static_cast<scalar_t>(TINY)))) {
    if (do_pull) {
      scalar_t *pull_ptr_NCDHW = pull_ptr + n * pull_sN + d * pull_sD 
                                          + h * pull_sH + w * pull_sW;
      for (offset_t c = 0; c < C; ++c, pull_ptr_NCDHW += pull_sC)
        *pull_ptr_NCDHW = static_cast<scalar_t>(0);
    }
    if (do_grad) {
      scalar_t * grad_ptr_NDHW = grad_ptr + n * grad_sN + d * grad_sD 
                                          + h * grad_sH + w * grad_sW;
      (*grad_ptr_NDHW) = static_cast<scalar_t>(0);
      grad_ptr_NDHW[grad_sC] = static_cast<scalar_t>(0);
      grad_ptr_NDHW[grad_sC*2] = static_cast<scalar_t>(0);
    }
    return;
  }

  // Next step
  if (bound0 == BoundType::Sliding) {
    if (iso) switch (static_cast<int>(interpolation0)) {
      case 0: return interpolate3d_sliding_nearest(x, y, z, w, h, d, n);
      case 1: return interpolate3d_sliding_trilinear(x, y, z, w, h, d, n);
    }
    return interpolate3d_sliding(x, y, z, w, h, d, n);
  } else {
    if (iso) switch (static_cast<int>(interpolation0)) {
      case 0: return interpolate3d_nearest(x, y, z, w, h, d, n);
      case 1: return interpolate3d_trilinear(x, y, z, w, h, d, n);
    } 
    return interpolate3d(x, y, z, w, h, d, n);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     LINEAR INTERPOLATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate3d_trilinear(
  scalar_t x, scalar_t y, scalar_t z,
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  // Get corner pixel values from (x, y, z)
  offset_t ix0 = static_cast<offset_t>(std::floor(x));
  offset_t iy0 = static_cast<offset_t>(std::floor(y));
  offset_t iz0 = static_cast<offset_t>(std::floor(z));


  // Interpolation weights (inversely proportional to distance)
  scalar_t dx1 = x - ix0;
  scalar_t dy1 = y - iy0;
  scalar_t dz1 = z - iz0;
  scalar_t dx0 = 1. - dx1;
  scalar_t dy0 = 1. - dy1;
  scalar_t dz0 = 1. - dz1;
  scalar_t w000 = dx0 * dy0 * dz0;
  scalar_t w100 = dx1 * dy0 * dz0;
  scalar_t w010 = dx0 * dy1 * dz0;
  scalar_t w001 = dx0 * dy0 * dz1;
  scalar_t w110 = dx1 * dy1 * dz0;
  scalar_t w011 = dx0 * dy1 * dz1;
  scalar_t w101 = dx1 * dy0 * dz1;
  scalar_t w111 = dx1 * dy1 * dz1;

  // Sign (/!\ compute sign before warping indices)
  int8_t  sx1 = bound::sign(bound0, ix0+1, src_W);
  int8_t  sy1 = bound::sign(bound1, iy0+1, src_H);
  int8_t  sz1 = bound::sign(bound2, iz0+1, src_D);
  int8_t  sx0 = bound::sign(bound0, ix0,   src_W);
  int8_t  sy0 = bound::sign(bound1, iy0,   src_H);
  int8_t  sz0 = bound::sign(bound2, iz0,   src_D);
  int8_t  s000 = sx0 * sy0 * sz0;
  int8_t  s100 = sx1 * sy0 * sz0;
  int8_t  s010 = sx0 * sy1 * sz0;
  int8_t  s001 = sx0 * sy0 * sz1;
  int8_t  s110 = sx1 * sy1 * sz0;
  int8_t  s011 = sx0 * sy1 * sz1;
  int8_t  s101 = sx1 * sy0 * sz1;
  int8_t  s111 = sx1 * sy1 * sz1;

  // Warp indices
  offset_t ix1, iy1, iz1;
  ix1 = bound::index(bound0, ix0+1, src_W);
  iy1 = bound::index(bound1, iy0+1, src_H);
  iz1 = bound::index(bound2, iz0+1, src_D);
  ix0 = bound::index(bound0, ix0,   src_W);
  iy0 = bound::index(bound1, iy0,   src_H);
  iz0 = bound::index(bound2, iz0,   src_D);

  // Offsets into source volume
  offset_t o000, o100, o010, o001, o110, o011, o101, o111;

  if (do_pull || do_grad) {
    o000 = ix0*src_sW + iy0*src_sH + iz0*src_sD;
    o100 = ix1*src_sW + iy0*src_sH + iz0*src_sD;
    o010 = ix0*src_sW + iy1*src_sH + iz0*src_sD;
    o001 = ix0*src_sW + iy0*src_sH + iz1*src_sD;
    o110 = ix1*src_sW + iy1*src_sH + iz0*src_sD;
    o011 = ix0*src_sW + iy1*src_sH + iz1*src_sD;
    o101 = ix1*src_sW + iy0*src_sH + iz1*src_sD;
    o111 = ix1*src_sW + iy1*src_sH + iz1*src_sD;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_pull) {
    scalar_t *pull_ptr_NCDHW = pull_ptr + n * pull_sN + d * pull_sD 
                                        + h * pull_sH + w * pull_sW;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, pull_ptr_NCDHW += pull_sC, 
                                    src_ptr_NC     += src_sC) {
      *pull_ptr_NCDHW = bound::get(src_ptr_NC, o000, s000) * w000
                      + bound::get(src_ptr_NC, o100, s100) * w100
                      + bound::get(src_ptr_NC, o010, s010) * w010
                      + bound::get(src_ptr_NC, o110, s110) * w110
                      + bound::get(src_ptr_NC, o001, s001) * w001
                      + bound::get(src_ptr_NC, o101, s101) * w101
                      + bound::get(src_ptr_NC, o011, s011) * w011
                      + bound::get(src_ptr_NC, o111, s111) * w111;
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Grid gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {
    scalar_t gx = static_cast<scalar_t>(0);
    scalar_t gy = static_cast<scalar_t>(0);
    scalar_t gz = static_cast<scalar_t>(0);
    scalar_t *trgt_ptr_NCDHW = trgt_ptr + n * trgt_sN + d * trgt_sD 
                                        + h * trgt_sH + w * trgt_sW;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCDHW += trgt_sC, 
                                    src_ptr_NC     += src_sC) {
      scalar_t src;
      scalar_t trgt = *trgt_ptr_NCDHW;
      src = bound::get(src_ptr_NC, o000, s000) * trgt;
      gx -=       dy0 * dz0 * src;
      gy -= dx0       * dz0 * src;
      gz -= dx0 * dy0       * src;
      src = bound::get(src_ptr_NC, o100, s100) * trgt;
      gx +=       dy0 * dz0 * src;
      gy -= dx1       * dz0 * src;
      gz -= dx1 * dy0       * src;
      src = bound::get(src_ptr_NC, o010, s010) * trgt;
      gx -=       dy1 * dz0 * src;
      gy += dx0       * dz0 * src;
      gz -= dx0 * dy1       * src;
      src = bound::get(src_ptr_NC, o110, s110) * trgt;
      gx +=       dy1 * dz0 * src;
      gy += dx1       * dz0 * src;
      gz -= dx1 * dy1       * src;
      src = bound::get(src_ptr_NC, o001, s001) * trgt;
      gx -=       dy0 * dz1 * src;
      gy -= dx0       * dz1 * src;
      gz += dx0 * dy0       * src;
      src = bound::get(src_ptr_NC, o101, s101) * trgt;
      gx +=       dy0 * dz1 * src;
      gy -= dx1       * dz1 * src;
      gz += dx1 * dy0       * src;
      src = bound::get(src_ptr_NC, o011, s011) * trgt;
      gx -=       dy1 * dz1 * src;
      gy += dx0       * dz1 * src;
      gz += dx0 * dy1       * src;
      src = bound::get(src_ptr_NC, o111, s111) * trgt;
      gx +=       dy1 * dz1 * src;
      gy += dx1       * dz1 * src;
      gz += dx1 * dy1       * src;
    }

    scalar_t * grad_ptr_NDHW = grad_ptr + n * grad_sN + d * grad_sD 
                                        + h * grad_sH + w * grad_sW;
    (*grad_ptr_NDHW)         = gx;
    grad_ptr_NDHW[grad_sC]   = gy;
    grad_ptr_NDHW[grad_sC*2] = gz;
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_push) {
    // Offsets into 'push' volume
    o000 = ix0*push_sW + iy0*push_sH + iz0*push_sD;
    o100 = ix1*push_sW + iy0*push_sH + iz0*push_sD;
    o010 = ix0*push_sW + iy1*push_sH + iz0*push_sD;
    o001 = ix0*push_sW + iy0*push_sH + iz1*push_sD;
    o110 = ix1*push_sW + iy1*push_sH + iz0*push_sD;
    o011 = ix0*push_sW + iy1*push_sH + iz1*push_sD;
    o101 = ix1*push_sW + iy0*push_sH + iz1*push_sD;
    o111 = ix1*push_sW + iy1*push_sH + iz1*push_sD;

    scalar_t *trgt_ptr_NCDHW = trgt_ptr + n * trgt_sN + d * trgt_sD 
                                        + h * trgt_sH + w * trgt_sW;
    scalar_t *push_ptr_NC = push_ptr + n * push_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCDHW += trgt_sC,
                                     push_ptr_NC    += push_sC) {
      scalar_t trgt = *trgt_ptr_NCDHW;
      bound::add(push_ptr_NC, o000, w000 * trgt, s000);
      bound::add(push_ptr_NC, o100, w100 * trgt, s100);
      bound::add(push_ptr_NC, o010, w010 * trgt, s010);
      bound::add(push_ptr_NC, o110, w110 * trgt, s110);
      bound::add(push_ptr_NC, o001, w001 * trgt, s001);
      bound::add(push_ptr_NC, o101, w101 * trgt, s101);
      bound::add(push_ptr_NC, o011, w011 * trgt, s011);
      bound::add(push_ptr_NC, o111, w111 * trgt, s111);
    }
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     LINEAR INTERPOLATION 2D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate2d_bilinear(
  scalar_t x, scalar_t y,
  offset_t w, offset_t h, offset_t n) const
{
  // Get corner pixel values from (x, y, z)
  offset_t ix0 = static_cast<offset_t>(std::floor(x));
  offset_t iy0 = static_cast<offset_t>(std::floor(y));

  // Interpolation weights (inversely proportional to distance)
  scalar_t dx1 = x - ix0;
  scalar_t dy1 = y - iy0;
  scalar_t dx0 = 1. - dx1;
  scalar_t dy0 = 1. - dy1;
  scalar_t w00 = dx0 * dy0;
  scalar_t w10 = dx1 * dy0;
  scalar_t w01 = dx0 * dy1;
  scalar_t w11 = dx1 * dy1;;

  // Sign (/!\ compute sign before warping indices)
  int8_t  sx1 = bound::sign(bound0, ix0+1, src_W);
  int8_t  sy1 = bound::sign(bound1, iy0+1, src_H);
  int8_t  sx0 = bound::sign(bound0, ix0,   src_W);
  int8_t  sy0 = bound::sign(bound1, iy0,   src_H);
  int8_t  s00 = sx0 * sy0;
  int8_t  s10 = sx1 * sy0;
  int8_t  s01 = sx0 * sy1;
  int8_t  s11 = sx1 * sy1;

  // Warp indices
  offset_t ix1, iy1;
  ix1 = bound::index(bound0, ix0+1, src_W);
  iy1 = bound::index(bound1, iy0+1, src_H);
  ix0 = bound::index(bound0, ix0,   src_W);
  iy0 = bound::index(bound1, iy0,   src_H);

  // Offsets into source volume
  offset_t o00, o10, o01, o11;
  if (do_pull || do_grad) {
    o00 = ix0*src_sW + iy0*src_sH;
    o10 = ix1*src_sW + iy0*src_sH;
    o01 = ix0*src_sW + iy1*src_sH;
    o11 = ix1*src_sW + iy1*src_sH;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_pull) {
    scalar_t *pull_ptr_NCHW = pull_ptr + n * pull_sN
                                       + h * pull_sH 
                                       + w * pull_sW;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, pull_ptr_NCHW += pull_sC, 
                                    src_ptr_NC    += src_sC) {
      *pull_ptr_NCHW = bound::get(src_ptr_NC, o00, s00) * w00
                     + bound::get(src_ptr_NC, o10, s10) * w10
                     + bound::get(src_ptr_NC, o01, s01) * w01
                     + bound::get(src_ptr_NC, o11, s11) * w11;
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Grid gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {
    scalar_t gx = static_cast<scalar_t>(0);
    scalar_t gy = static_cast<scalar_t>(0);
    scalar_t *trgt_ptr_NCHW = trgt_ptr + n * trgt_sN 
                                       + h * trgt_sH 
                                       + w * trgt_sW;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCHW += trgt_sC, 
                                    src_ptr_NC    += src_sC) {
      scalar_t src;
      scalar_t trgt = *trgt_ptr_NCHW;
      src = bound::get(src_ptr_NC, o00, s00) * trgt;
      gx -=       dy0 * src;
      gy -= dx0       * src;
      src = bound::get(src_ptr_NC, o10, s10) * trgt;
      gx +=       dy0 * src;
      gy -= dx1       * src;
      src = bound::get(src_ptr_NC, o01, s01) * trgt;
      gx -=       dy1 * src;
      gy += dx0       * src;
      src = bound::get(src_ptr_NC, o11, s11) * trgt;
      gx +=       dy1 * src;
      gy += dx1       * src;
    }

    scalar_t * grad_ptr_NHW = grad_ptr + n * grad_sN 
                                       + h * grad_sH 
                                       + w * grad_sW;
    (*grad_ptr_NHW)       = gx;
    grad_ptr_NHW[grad_sC] = gy;
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_push) {
    // Offsets into 'push' volume
    o00 = ix0*push_sW + iy0*push_sH;
    o10 = ix1*push_sW + iy0*push_sH;
    o01 = ix0*push_sW + iy1*push_sH;
    o11 = ix1*push_sW + iy1*push_sH;

    scalar_t *trgt_ptr_NCHW = trgt_ptr + n * trgt_sN 
                                       + h * trgt_sH 
                                       + w * trgt_sW;
    scalar_t *push_ptr_NC = push_ptr + n * push_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCHW += trgt_sC,
                                    push_ptr_NC   += push_sC) {
      scalar_t trgt = *trgt_ptr_NCHW;
      bound::add(push_ptr_NC, o00, w00 * trgt, s00);
      bound::add(push_ptr_NC, o10, w10 * trgt, s10);
      bound::add(push_ptr_NC, o01, w01 * trgt, s01);
      bound::add(push_ptr_NC, o11, w11 * trgt, s11);
    }
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  NEAREST NEIGHBOR INTERPOLATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate3d_nearest(
  scalar_t x, scalar_t y, scalar_t z,
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  offset_t ix = static_cast<offset_t>(std::round(x));
  offset_t iy = static_cast<offset_t>(std::round(y));
  offset_t iz = static_cast<offset_t>(std::round(z));

  // Boundary condition (/!\ compute sign before warping indices)
  int8_t    sx = bound::sign(bound0, ix, src_W);
  int8_t    sy = bound::sign(bound1, iy, src_H);
  int8_t    sz = bound::sign(bound2, iz, src_D);
            ix = bound::index(bound0, ix,src_W);
            iy = bound::index(bound1, iy,src_H);
            iz = bound::index(bound2, iz,src_D);

  // Sign
  int8_t s = sz * sy * sx;

  if (do_pull) {
    offset_t  o = iz*src_sD + iy*src_sH + ix*src_sW;
    scalar_t *pull_ptr_NCDHW = pull_ptr + n * pull_sN + d * pull_sD 
                                        + h * pull_sH + w * pull_sW;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, pull_ptr_NCDHW += pull_sC, 
                                    src_ptr_NC     += src_sC)
      *pull_ptr_NCDHW = bound::get(src_ptr_NC, o, s);
  }
  if (do_push) {
    offset_t  o = iz*push_sD + iy*push_sH + ix*push_sW;
    scalar_t *trgt_ptr_NCDHW = trgt_ptr + n * trgt_sN + d * trgt_sD 
                                        + h * trgt_sH + w * trgt_sW;
    scalar_t *push_ptr_NC    = push_ptr + n * push_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCDHW += trgt_sC, 
                                    push_ptr_NC    += push_sC)
      bound::add(push_ptr_NC, o, *trgt_ptr_NCDHW, s);

  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  NEAREST NEIGHBOR INTERPOLATION 2D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate2d_nearest(
  scalar_t x, scalar_t y,
  offset_t w, offset_t h, offset_t n) const
{
  offset_t ix = static_cast<offset_t>(std::round(x));
  offset_t iy = static_cast<offset_t>(std::round(y));

  // Boundary condition (/!\ compute sign before warping indices)
  int8_t    sx = bound::sign(bound0, ix, src_W);
  int8_t    sy = bound::sign(bound1, iy, src_H);
            ix = bound::index(bound0, ix,src_W);
            iy = bound::index(bound1, iy,src_H);

  // Sign
  int8_t s = sy * sx;

  if (do_pull) {
    offset_t  o = iy*src_sH + ix*src_sW;
    scalar_t *pull_ptr_NCHW = pull_ptr + n * pull_sN 
                                       + h * pull_sH 
                                       + w * pull_sW;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, pull_ptr_NCHW += pull_sC, 
                                    src_ptr_NC    += src_sC)
      *pull_ptr_NCHW = bound::get(src_ptr_NC, o, s);
  }
  if (do_push) {
    offset_t  o = iy*push_sH + ix*push_sW;
    scalar_t *trgt_ptr_NCHW = trgt_ptr + n * trgt_sN 
                                       + h * trgt_sH 
                                       + w * trgt_sW;
    scalar_t *push_ptr_NC    = push_ptr + n * push_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCHW += trgt_sC, 
                                    push_ptr_NC   += push_sC)
      bound::add(push_ptr_NC, o, *trgt_ptr_NCHW, s);

  }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//            LINEAR INTERPOLATION 3D + SLIDING BOUNDARY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if 0

// Sliding boundary conditions only make sense if the source volume 
// is a displacement field itself. These boundary conditions are 
// different for each component of the displacement field:
// - x component: dirichlet/DST2 along W || neumann/DCT2 along H/D
// - y component: dirichlet/DST2 along H || neumann/DCT2 along W/D
// - z component: dirichlet/DST2 along D || neumann/DCT2 along W/H
// The effect is that these displacement fields are allowed to slide 
// along the edges of the field of view. This penalises translation but 
// not rotation. In contrast, circular boundary conditions penalise
// rotations but not translations.

PUSHPULL_PREFIX_DIM_BND_INT(void,3,BoundSliding,BiLinearInterpolation)
::_interpolate(offset_t x, offset_t y, offset_t z,
               offset_t w, offset_t h, offset_t d, offset_t n)
{
  // Get corner pixel values from (x, y, z)
  offset_t ix0 = static_cast<offset_t>(std::floor(x));
  offset_t iy0 = static_cast<offset_t>(std::floor(y));
  offset_t iz0 = static_cast<offset_t>(std::floor(z));

  // Interpolation weights (inversely proportional to distance)
  scalar_t dx1 = x - ix0;
  scalar_t dy1 = y - iy0;
  scalar_t dz1 = z - iz0;
  scalar_t dx0 = 1. - dx1;
  scalar_t dy0 = 1. - dy1;
  scalar_t dz0 = 1. - dz1;
  scalar_t w000 = dx0 * dy0 * dz0;
  scalar_t w100 = dx1 * dy0 * dz0;
  scalar_t w010 = dx0 * dy1 * dz0;
  scalar_t w001 = dx0 * dy0 * dz1;
  scalar_t w110 = dx1 * dy1 * dz0;
  scalar_t w011 = dx0 * dy1 * dz1;
  scalar_t w101 = dx1 * dy0 * dz1;
  scalar_t w111 = dx1 * dy1 * dz1;

  // Sign (DST condition) (/!\ compute sign before warping indices)
  int8_t  sx1c = bound::BoundDCT2::sign(ix0+1,src_W);
  int8_t  sy1c = bound::BoundDCT2::sign(iy0+1,src_H);
  int8_t  sz1c = bound::BoundDCT2::sign(iz0+1,src_D);
  int8_t  sx0c = bound::BoundDCT2::sign(ix0,  src_W);
  int8_t  sy0c = bound::BoundDCT2::sign(iy0,  src_H);
  int8_t  sz0c = bound::BoundDCT2::sign(iz0,  src_D);
  int8_t  sx1s = bound::BoundDST2::sign(ix0+1,src_W);
  int8_t  sy1s = bound::BoundDST2::sign(iy0+1,src_H);
  int8_t  sz1s = bound::BoundDST2::sign(iz0+1,src_D);
  int8_t  sx0s = bound::BoundDST2::sign(ix0,  src_W);
  int8_t  sy0s = bound::BoundDST2::sign(iy0,  src_H);
  int8_t  sz0s = bound::BoundDST2::sign(iz0,  src_D);
  int8_t  sx000 = sx0s * sy0c * sz0c;
  int8_t  sx100 = sx1s * sy0c * sz0c;
  int8_t  sx010 = sx0s * sy1c * sz0c;
  int8_t  sx001 = sx0s * sy0c * sz1c;
  int8_t  sx110 = sx1s * sy1c * sz0c;
  int8_t  sx011 = sx0s * sy1c * sz1c;
  int8_t  sx101 = sx1s * sy0c * sz1c;
  int8_t  sx111 = sx1s * sy1c * sz1c;
  int8_t  sy000 = sx0c * sy0s * sz0c;
  int8_t  sy100 = sx1c * sy0s * sz0c;
  int8_t  sy010 = sx0c * sy1s * sz0c;
  int8_t  sy001 = sx0c * sy0s * sz1c;
  int8_t  sy110 = sx1c * sy1s * sz0c;
  int8_t  sy011 = sx0c * sy1s * sz1c;
  int8_t  sy101 = sx1c * sy0s * sz1c;
  int8_t  sy111 = sx1c * sy1s * sz1c;
  int8_t  sz000 = sx0c * sy0c * sz0s;
  int8_t  sz100 = sx1c * sy0c * sz0s;
  int8_t  sz010 = sx0c * sy1c * sz0s;
  int8_t  sz001 = sx0c * sy0c * sz1s;
  int8_t  sz110 = sx1c * sy1c * sz0s;
  int8_t  sz011 = sx0c * sy1c * sz1s;
  int8_t  sz101 = sx1c * sy0c * sz1s;
  int8_t  sz111 = sx1c * sy1c * sz1s;

  // Warp indices
  offset_t ix1c, iy1c, iz1c, ix1s, iy1s, iz1s;
  ix1c = bound::BoundDCT2::index(ix0+1,src_W);
  iy1c = bound::BoundDCT2::index(iy0+1,src_H);
  iz1c = bound::BoundDCT2::index(iz0+1,src_D);
  ix0c = bound::BoundDCT2::index(ix0,  src_W);
  iy0c = bound::BoundDCT2::index(iy0,  src_H);
  iz0c = bound::BoundDCT2::index(iz0,  src_D);
  ix1s = bound::BoundDTS2::index(ix0+1,src_W);
  iy1s = bound::BoundDST2::index(iy0+1,src_H);
  iz1s = bound::BoundDST2::index(iz0+1,src_D);
  ix0s = bound::BoundDST2::index(ix0,  src_W);
  iy0s = bound::BoundDST2::index(iy0,  src_H);
  iz0s = bound::BoundDST2::index(iz0,  src_D);

  // Offsets into source volume
  offset_t ox000 = ix0c*_src_sW + iy0s*_src_sH + iz0s*_src_sD;
  offset_t ox100 = ix1c*_src_sW + iy0s*_src_sH + iz0s*_src_sD;
  offset_t ox010 = ix0c*_src_sW + iy1s*_src_sH + iz0s*_src_sD;
  offset_t ox001 = ix0c*_src_sW + iy0s*_src_sH + iz1s*_src_sD;
  offset_t ox110 = ix1c*_src_sW + iy1s*_src_sH + iz0s*_src_sD;
  offset_t ox011 = ix0c*_src_sW + iy1s*_src_sH + iz1s*_src_sD;
  offset_t ox101 = ix1c*_src_sW + iy0s*_src_sH + iz1s*_src_sD;
  offset_t ox111 = ix1c*_src_sW + iy1s*_src_sH + iz1s*_src_sD;
  offset_t oy000 = ix0s*_src_sW + iy0c*_src_sH + iz0s*_src_sD;
  offset_t oy100 = ix1s*_src_sW + iy0c*_src_sH + iz0s*_src_sD;
  offset_t oy010 = ix0s*_src_sW + iy1c*_src_sH + iz0s*_src_sD;
  offset_t oy001 = ix0s*_src_sW + iy0c*_src_sH + iz1s*_src_sD;
  offset_t oy110 = ix1s*_src_sW + iy1c*_src_sH + iz0s*_src_sD;
  offset_t oy011 = ix0s*_src_sW + iy1c*_src_sH + iz1s*_src_sD;
  offset_t oy101 = ix1s*_src_sW + iy0c*_src_sH + iz1s*_src_sD;
  offset_t oy111 = ix1s*_src_sW + iy1c*_src_sH + iz1s*_src_sD;
  offset_t oz000 = ix0s*_src_sW + iy0s*_src_sH + iz0c*_src_sD;
  offset_t oz100 = ix1s*_src_sW + iy0s*_src_sH + iz0c*_src_sD;
  offset_t oz010 = ix0s*_src_sW + iy1s*_src_sH + iz0c*_src_sD;
  offset_t oz001 = ix0s*_src_sW + iy0s*_src_sH + iz1c*_src_sD;
  offset_t oz110 = ix1s*_src_sW + iy1s*_src_sH + iz0c*_src_sD;
  offset_t oz011 = ix0s*_src_sW + iy1s*_src_sH + iz1c*_src_sD;
  offset_t oz101 = ix1s*_src_sW + iy0s*_src_sH + iz1c*_src_sD;
  offset_t oz111 = ix1s*_src_sW + iy1s*_src_sH + iz1c*_src_sD;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_pull) {
    scalar_t *pull_ptr_NCDHW =pull_ptr + n *pull_sN + d *pull_sD 
                                         + h *pull_sH + w *pull_sW;
    scalar_t *src_ptr_NC =src_ptr + n *src_sN;

    *pull_ptr_NCDHW = get(src_ptr_NC, ox000, sx000) * w000
                    + get(src_ptr_NC, ox100, sx100) * w100
                    + get(src_ptr_NC, ox010, sx010) * w010
                    + get(src_ptr_NC, ox110, sx110) * w110
                    + get(src_ptr_NC, ox001, sx001) * w001
                    + get(src_ptr_NC, ox101, sx101) * w101
                    + get(src_ptr_NC, ox011, sx011) * w011
                    + get(src_ptr_NC, ox111, sx111) * w111;
    pull_ptr_NCDHW +=pull_sC;
    src_ptr_NC     +=src_sC;
    *pull_ptr_NCDHW = get(src_ptr_NC, oy000, sy000) * w000
                    + get(src_ptr_NC, oy100, sy100) * w100
                    + get(src_ptr_NC, oy010, sy010) * w010
                    + get(src_ptr_NC, oy110, sy110) * w110
                    + get(src_ptr_NC, oy001, sy001) * w001
                    + get(src_ptr_NC, oy101, sy101) * w101
                    + get(src_ptr_NC, oy011, sy011) * w011
                    + get(src_ptr_NC, oy111, sy111) * w111;
    pull_ptr_NCDHW +=pull_sC;
    src_ptr_NC     +=src_sC;
    *pull_ptr_NCDHW = get(src_ptr_NC, oz000, sz000) * w000
                    + get(src_ptr_NC, oz100, sz100) * w100
                    + get(src_ptr_NC, oz010, sz010) * w010
                    + get(src_ptr_NC, oz110, sz110) * w110
                    + get(src_ptr_NC, oz001, sz001) * w001
                    + get(src_ptr_NC, oz101, sz101) * w101
                    + get(src_ptr_NC, oz011, sz011) * w011
                    + get(src_ptr_NC, oz111, sz111) * w111;
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_push) {
    scalar_t *trgt_ptr_NCDHW =trgt_ptr + n *trgt_sN + d *trgt_sD 
                                         + h *trgt_sH + w *trgt_sW;
    scalar_t *push_ptr_NC =push_ptr + n *push_sN;
    scalar_t trgt = *trgt_ptr_NCDHW;
    add(push_ptr_NC, ox000, sx000, w000 * trgt);
    add(push_ptr_NC, ox100, sx100, w100 * trgt);
    add(push_ptr_NC, ox010, sx010, w010 * trgt);
    add(push_ptr_NC, ox110, sx110, w110 * trgt);
    add(push_ptr_NC, ox001, sx001, w001 * trgt);
    add(push_ptr_NC, ox101, sx101, w101 * trgt);
    add(push_ptr_NC, ox011, sx011, w011 * trgt);
    add(push_ptr_NC, ox111, sx111, w111 * trgt);
    push_ptr_NC    +=push_sC;
    trgt_ptr_NCDHW +=trgt_sC;
    trgt            = *trgt_ptr_NCDHW;
    add(push_ptr_NC, oy000, sy000, w000 * trgt);
    add(push_ptr_NC, oy100, sy100, w100 * trgt);
    add(push_ptr_NC, oy010, sy010, w010 * trgt);
    add(push_ptr_NC, oy110, sy110, w110 * trgt);
    add(push_ptr_NC, oy001, sy001, w001 * trgt);
    add(push_ptr_NC, oy101, sy101, w101 * trgt);
    add(push_ptr_NC, oy011, sy011, w011 * trgt);
    add(push_ptr_NC, oy111, sy111, w111 * trgt);
    push_ptr_NC    +=push_sC;
    trgt_ptr_NCDHW +=trgt_sC;
    trgt            = *trgt_ptr_NCDHW;
    add(push_ptr_NC, oz000, sz000, w000 * trgt);
    add(push_ptr_NC, oz100, sz100, w100 * trgt);
    add(push_ptr_NC, oz010, sz010, w010 * trgt);
    add(push_ptr_NC, oz110, sz110, w110 * trgt);
    add(push_ptr_NC, oz001, sz001, w001 * trgt);
    add(push_ptr_NC, oz101, sz101, w101 * trgt);
    add(push_ptr_NC, oz011, sz011, w011 * trgt);
    add(push_ptr_NC, oz111, sz111, w111 * trgt);
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~ Grid gradient ~~~~~~~~~~~~~~~~~~~~~~~
  if do_grad) {
    scalar_t gx = static_cast<scalar_t>(0);
    scalar_t gy = static_cast<scalar_t>(0);
    scalar_t gz = static_cast<scalar_t>(0);
    scalar_t *trgt_ptr_NCDHW =trgt_ptr + n *trgt_sN + d *trgt_sD 
                                         + h *trgt_sH + w *trgt_sW;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    scalar_t src;
    scalar_t trgt = *trgt_ptr_NCDHW;
    src = get(src_ptr_NC, ox000, sx000) * trgt;
    gx +=       dy0 * dz0 * src * gix0s;
    gy += dx0       * dz0 * src * giy0c;
    gz += dx0 * dy0       * src * giz0c;
    src = get(src_ptr_NC, ox100, sx100) * trgt;
    gx +=       dy0 * dz0 * src * gix1s;
    gy += dx1       * dz0 * src * giy0c;
    gz += dx1 * dy0       * src * giz0c;
    src = get(src_ptr_NC, ox010, sx010) * trgt;
    gx +=       dy1 * dz0 * src * gix0s;
    gy += dx0       * dz0 * src * giy1c;
    gz += dx0 * dy1       * src * giz0c;
    src = get(src_ptr_NC, ox110, sx110) * trgt;
    gx +=       dy1 * dz0 * src * gix1s;
    gy += dx1       * dz0 * src * giy1c;
    gz += dx1 * dy1       * src * giz0c;
    src = get(src_ptr_NC, ox001, sx001) * trgt;
    gx +=       dy0 * dz1 * src * gix0s;
    gy += dx0       * dz1 * src * giy0c;
    gz += dx0 * dy0       * src * giz1c;
    src = get(src_ptr_NC, ox101, sx101) * trgt;
    gx +=       dy0 * dz1 * src * gix1s;
    gy += dx1       * dz1 * src * giy0c;
    gz += dx1 * dy0       * src * giz1c;
    src = get(src_ptr_NC, ox011, sx011) * trgt;
    gx +=       dy1 * dz1 * src * gix0s;
    gy += dx0       * dz1 * src * giy1c;
    gz += dx0 * dy1       * src * giz1c;
    src = get(src_ptr_NC, ox111, sx111) * trgt;
    gx +=       dy1 * dz1 * src * gix1s;
    gy += dx1       * dz1 * src * giy1c;
    gz += dx1 * dy1       * src * giz1c;
    src_ptr_NC     +=src_sC;
    trgt_ptr_NCDHW +=trgt_sC;
    trgt            = *trgt_ptr_NCDHW;
    src = get(src_ptr_NC, oy000, sy000) * trgt;
    gx +=       dy0 * dz0 * src * gix0c;
    gy += dx0       * dz0 * src * giy0s;
    gz += dx0 * dy0       * src * giz0c;
    src = get(src_ptr_NC, oy100, sy100) * trgt;
    gx +=       dy0 * dz0 * src * gix1c;
    gy += dx1       * dz0 * src * giy0s;
    gz += dx1 * dy0       * src * giz0c;
    src = get(src_ptr_NC, oy010, sy010) * trgt;
    gx +=       dy1 * dz0 * src * gix0c;
    gy += dx0       * dz0 * src * giy1s;
    gz += dx0 * dy1       * src * giz0c;
    src = get(src_ptr_NC, oy110, sy110) * trgt;
    gx +=       dy1 * dz0 * src * gix1c;
    gy += dx1       * dz0 * src * giy1s;
    gz += dx1 * dy1       * src * giz0c;
    src = get(src_ptr_NC, oy001, sy001) * trgt;
    gx +=       dy0 * dz1 * src * gix0c;
    gy += dx0       * dz1 * src * giy0s;
    gz += dx0 * dy0       * src * giz1c;
    src = get(src_ptr_NC, oy101, sy101) * trgt;
    gx +=       dy0 * dz1 * src * gix1c;
    gy += dx1       * dz1 * src * giy0s;
    gz += dx1 * dy0       * src * giz1c;
    src = get(src_ptr_NC, oy011, sy011) * trgt;
    gx +=       dy1 * dz1 * src * gix0c;
    gy += dx0       * dz1 * src * giy1s;
    gz += dx0 * dy1       * src * giz1c;
    src = get(src_ptr_NC, oy111, sy111) * trgt;
    gx +=       dy1 * dz1 * src * gix1c;
    gy += dx1       * dz1 * src * giy1s;
    gz += dx1 * dy1       * src * giz1c;
    src_ptr_NC     +=src_sC;
    trgt_ptr_NCDHW +=trgt_sC;
    trgt            = *trgt_ptr_NCDHW;
    src = get(src_ptr_NC, oz000, sz000) * trgt;
    gx +=       dy0 * dz0 * src * gix0c;
    gy += dx0       * dz0 * src * giy0c;
    gz += dx0 * dy0       * src * giz0s;
    src = get(src_ptr_NC, oz100, sz100) * trgt;
    gx +=       dy0 * dz0 * src * gix1c;
    gy += dx1       * dz0 * src * giy0c;
    gz += dx1 * dy0       * src * giz0s;
    src = get(src_ptr_NC, oz010, sz010) * trgt;
    gx +=       dy1 * dz0 * src * gix0c;
    gy += dx0       * dz0 * src * giy1c;
    gz += dx0 * dy1       * src * giz0s;
    src = get(src_ptr_NC, oz110, sz110) * trgt;
    gx +=       dy1 * dz0 * src * gix1c;
    gy += dx1       * dz0 * src * giy1c;
    gz += dx1 * dy1       * src * giz0s;
    src = get(src_ptr_NC, oz001, sz001) * trgt;
    gx +=       dy0 * dz1 * src * gix0c;
    gy += dx0       * dz1 * src * giy0c;
    gz += dx0 * dy0       * src * giz1s;
    src = get(src_ptr_NC, oz101, sz101) * trgt;
    gx +=       dy0 * dz1 * src * gix1c;
    gy += dx1       * dz1 * src * giy0c;
    gz += dx1 * dy0       * src * giz1s;
    src = get(src_ptr_NC, oz011, sz011) * trgt;
    gx +=       dy1 * dz1 * src * gix0c;
    gy += dx0       * dz1 * src * giy1c;
    gz += dx0 * dy1       * src * giz1s;
    src = get(src_ptr_NC, oz111, sz111) * trgt;
    gx +=       dy1 * dz1 * src * gix1c;
    gy += dx1       * dz1 * src * giy1c;
    gz += dx1 * dy1       * src * giz1s;

    scalar_t * grad_ptr_NDHW =grad_ptr + n *grad_sN + d *grad_sD 
                                         + h *grad_sH + w *grad_sW;
    grad_ptr_NDHW[0] = gx;
    grad_ptr_NDHW[1] = gy;
    grad_ptr_NDHW[2] = gz;
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//            LINEAR INTERPOLATION 2D + SLIDING BOUNDARY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <Coord coord_mode, typename scalar_t>
void PushPullImpl<2, Bound::Sliding, 1, coord_mode, scalar_t>
::_interpolate(offset_t x, offset_t y, offset_t z,
               offset_t w, offset_t h, offset_t d, offset_t n)
{
  // Get corner pixel values from (x, y, z)
  offset_t ix0 = static_cast<offset_t>(std::floor(x));
  offset_t iy0 = static_cast<offset_t>(std::floor(y));

  // Interpolation weights (inversely proportional to distance)
  scalar_t dx1 = x - ix0;
  scalar_t dy1 = y - iy0;
  scalar_t dx0 = 1. - dx1;
  scalar_t dy0 = 1. - dy1;
  scalar_t w00 = dx0 * dy0;
  scalar_t w10 = dx1 * dy0;
  scalar_t w01 = dx0 * dy1;
  scalar_t w11 = dx1 * dy1;

  // Sign (DST condition) (/!\ compute sign before warping indices)
  int8_t  sx1c = bound::sign<Bound::DCT2>(ix0+1,src_W);
  int8_t  sy1c = bound::sign<Bound::DCT2>(iy0+1,src_H);
  int8_t  sx0c = bound::sign<Bound::DCT2>(ix0,  src_W);
  int8_t  sy0c = bound::sign<Bound::DCT2>(iy0,  src_H);
  int8_t  sx1s = bound::sign<Bound::DST2>(ix0+1,src_W);
  int8_t  sy1s = bound::sign<Bound::DST2>(iy0+1,src_H);
  int8_t  sx0s = bound::sign<Bound::DST2>(ix0,  src_W);
  int8_t  sy0s = bound::sign<Bound::DST2>(iy0,  src_H);
  int8_t  sx00 = sx0s * sy0c;
  int8_t  sx10 = sx1s * sy0c;
  int8_t  sx01 = sx0s * sy1c;
  int8_t  sx11 = sx1s * sy1c;
  int8_t  sy00 = sx0c * sy0s;
  int8_t  sy10 = sx1c * sy0s;
  int8_t  sy01 = sx0c * sy1s;
  int8_t  sy11 = sx1c * sy1s;

  // Derivative of warping function
  offset_t gix0c, gix1c, giy0c, giy1c,
          gix0s, gix1s, giy0s, giy1s;
  if do_grad) {
    gix1c = bound::grad<Bound::DCT2>(ix0+1,src_W);
    giy1c = bound::grad<Bound::DCT2>(iy0+1,src_H);
    gix0c = bound::grad<Bound::DCT2>(ix0,  src_W);
    giy0c = bound::grad<Bound::DCT2>(iy0,  src_H);
    gix1s = bound::grad<Bound::DST2>(ix0+1,src_W);
    giy1s = bound::grad<Bound::DST2>(iy0+1,src_H);
    gix0s = bound::grad<Bound::DST2>(ix0,  src_W);
    giy0s = bound::grad<Bound::DST2>(iy0,  src_H);
  }

  // Warp indices
  offset_t ix1c, iy1c, ix1s, iy1s;
  ix1c = bound::index<Bound::DCT2>(ix0+1,src_W);
  iy1c = bound::index<Bound::DCT2>(iy0+1,src_H);
  ix0c = bound::index<Bound::DCT2>(ix0,  src_W);
  iy0c = bound::index<Bound::DCT2>(iy0,  src_H);
  ix1s = bound::index<Bound::DST2>(ix0+1,src_W);
  iy1s = bound::index<Bound::DST2>(iy0+1,src_H);
  ix0s = bound::index<Bound::DST2>(ix0,  src_W);
  iy0s = bound::index<Bound::DST2>(iy0,  src_H);

  // Offsets into source volume
  offset_t ox00 = ix0c*_src_sW + iy0s*_src_sH;
  offset_t ox10 = ix1c*_src_sW + iy0s*_src_sH;
  offset_t ox01 = ix0c*_src_sW + iy1s*_src_sH;
  offset_t ox11 = ix1c*_src_sW + iy1s*_src_sH;
  offset_t oy00 = ix0s*_src_sW + iy0c*_src_sH;
  offset_t oy10 = ix1s*_src_sW + iy0c*_src_sH;
  offset_t oy01 = ix0s*_src_sW + iy1c*_src_sH;
  offset_t oy11 = ix1s*_src_sW + iy1c*_src_sH;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if do_pull) {
    scalar_t *pull_ptr_NCHW =pull_ptr + n *pull_sN
                                        + h *pull_sH 
                                        + w *pull_sW;
    scalar_t *src_ptr_NC =src_ptr + n *src_sN;

    *pull_ptr_NCHW = get(src_ptr_NC, ox00, sx00) * w00
                   + get(src_ptr_NC, ox10, sx10) * w10
                   + get(src_ptr_NC, ox01, sx01) * w01
                   + get(src_ptr_NC, ox11, sx11) * w11;
    pull_ptr_NCHW +=pull_sC;
    src_ptr_NC    +=src_sC;
    *pull_ptr_NCHW = get(src_ptr_NC, oy00, sy00) * w00
                   + get(src_ptr_NC, oy10, sy10) * w10
                   + get(src_ptr_NC, oy01, sy01) * w01
                   + get(src_ptr_NC, oy11, sy11) * w11;
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if do_push) {
    scalar_t *trgt_ptr_NCHW =trgt_ptr + n *trgt_sN
                                        + h *trgt_sH 
                                        + w *trgt_sW;
    scalar_t *push_ptr_NC =push_ptr + n *push_sN;
    scalar_t trgt = *trgt_ptr_NCHW;
    add(push_ptr_NC, ox00, sx00, w00 * trgt);
    add(push_ptr_NC, ox10, sx10, w10 * trgt);
    add(push_ptr_NC, ox01, sx01, w01 * trgt);
    add(push_ptr_NC, ox11, sx11, w11 * trgt);
    push_ptr_NC    +=push_sC;
    trgt_ptr_NCHW +=trgt_sC;
    trgt            = *trgt_ptr_NCHW;
    add(push_ptr_NC, oy00, sy00, w00 * trgt);
    add(push_ptr_NC, oy10, sy10, w10 * trgt);
    add(push_ptr_NC, oy01, sy01, w01 * trgt);
    add(push_ptr_NC, oy11, sy11, w11 * trgt);
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~ Grid gradient ~~~~~~~~~~~~~~~~~~~~~~~
  if do_grad) {
    scalar_t gx = static_cast<scalar_t>(0);
    scalar_t gy = static_cast<scalar_t>(0);
    scalar_t *trgt_ptr_NCHW =trgt_ptr + n *trgt_sN
                                        + h *trgt_sH 
                                        + w *trgt_sW;
    scalar_t *src_ptr_NC  src_ptr + n * src_sN;
    scalar_t src;
    scalar_t trgt = *trgt_ptr_NCHW;
    src = get(src_ptr_NC, ox00, sx00) * trgt;
    gx +=       dy0 * src * gix0s;
    gy += dx0       * src * giy0c;
    src = get(src_ptr_NC, ox10, sx10) * trgt;
    gx +=       dy0 * src * gix1s;
    gy += dx1       * src * giy0c;
    src = get(src_ptr_NC, ox01, sx01) * trgt;
    gx +=       dy1 * src * gix0s;
    gy += dx0       * src * giy1c;
    src = get(src_ptr_NC, ox11, sx11) * trgt;
    gx +=       dy1 * src * gix1s;
    gy += dx1       * src * giy1c;
    src_ptr_NC     +=src_sC;
    trgt_ptr_NCHW +=trgt_sC;
    trgt            = *trgt_ptr_NCHW;
    src = get(src_ptr_NC, oy00, sy00) * trgt;
    gx +=       dy0 * src * gix0c;
    gy += dx0       * src * giy0s;
    src = get(src_ptr_NC, oy10, sy10) * trgt;
    gx +=       dy0 * src * gix1c;
    gy += dx1       * src * giy0s;
    src = get(src_ptr_NC, oy01, sy01) * trgt;
    gx +=       dy1 * src * gix0c;
    gy += dx0       * src * giy1s;
    src = get(src_ptr_NC, oy11, sy11) * trgt;
    gx +=       dy1 * src * gix1c;
    gy += dx1       * src * giy1s;

    scalar_t * grad_ptr_NDHW =grad_ptr + n *grad_sN
                                         + h *grad_sH 
                                         + w *grad_sW;
    grad_ptr_NDHW[0] = gx;
    grad_ptr_NDHW[1] = gy;
  }
}

#endif // Sliding

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void pushpull_kernel(PushPullImpl<scalar_t,offset_t> f) {
  f.loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define PUSHPULL_INSTANTIATE3(BoundType0, InterpolationType0, SourceType0) \
  template std::deque<Tensor> pushpull( \
    const SourceType0 &, const Tensor&, const Tensor&, \
    BoundType0, InterpolationType0, bool, bool, bool, bool)
#define PUSHPULL_INSTANTIATE2(BoundType0, InterpolationType0) \
  PUSHPULL_INSTANTIATE3(BoundType0, InterpolationType0, IntArrayRef); \
  PUSHPULL_INSTANTIATE3(BoundType0, InterpolationType0, Tensor); \
  template std::deque<Tensor> pushpull( \
    const Tensor&, const Tensor&, \
    BoundType0, InterpolationType0, bool, bool, bool, bool)
#define PUSHPULL_INSTANTIATE1(BoundType0) \
  PUSHPULL_INSTANTIATE2(BoundType0, InterpolationType); \
  PUSHPULL_INSTANTIATE2(BoundType0, InterpolationVectorRef)
#define PUSHPULL_INSTANTIATE \
  PUSHPULL_INSTANTIATE1(BoundType); \
  PUSHPULL_INSTANTIATE1(BoundVectorRef)

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (source, grid)
// > `bound` and `interpolation` can be single arguments or vectors.
template <typename BoundType, typename InterpolationType> 
NI_HOST
std::deque<Tensor> pushpull(
  const Tensor& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int64_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate, 
      do_pull, do_push, do_grad);
    f.ioset(source, grid);
    pushpull_kernel<<<GET_BLOCKS(f.voxcount()), CUDA_NUM_THREADS, 0, 
                      at::cuda::getCurrentCUDAStream()>>>(f);
    return f.output;
  });
}

// Three arguments (source, grid, target)
// > `bound` and `interpolation` can be single arguments or vectors.
// > `source` can be a tensor or a vector of dimensions.
template <typename BoundType, typename InterpolationType, typename SourceType> 
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int64_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate,
      do_pull, do_push, do_grad);
    f.ioset(source, grid, target);
    pushpull_kernel<<<GET_BLOCKS(f.voxcount()), CUDA_NUM_THREADS, 0, 
                      at::cuda::getCurrentCUDAStream()>>>(f);
    return f.output;
  });
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (source, grid)
// > `bound` and `interpolation` can be single arguments or vectors.
template <typename BoundType, typename InterpolationType>
NI_HOST
std::deque<Tensor> pushpull(
  const Tensor& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int32_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate, 
      do_pull, do_push, do_grad);
    f.ioset(source, grid);
    f.loop();
    auto output = f.output;
    return f.output;
  });
}

// Three arguments (source, grid, target)
// > `bound` and `interpolation` can be single arguments or vectors.
// > `source` can be a tensor or a vector of dimensions.
template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int32_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate,
      do_pull, do_push, do_grad);
    f.ioset(source, grid, target);
    f.loop();
    return f.output;
  });
}

#endif // __CUDACC__

PUSHPULL_INSTANTIATE;

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

template <typename BoundType, typename InterpolationType>
NI_HOST
std::deque<Tensor> pushpull(
  const Tensor& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_grad)
{
  throw std::logic_error("Function not implemented for this device.");
}

template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_grad)
{
  throw std::logic_error("Function not implemented for this device.");
}

PUSHPULL_INSTANTIATE;

} // namespace notimplemented

} // namespace ni
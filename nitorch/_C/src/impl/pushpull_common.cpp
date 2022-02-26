// This file implements spline interpolation / sampling and its adjoint 
// operations. It corresponds loosely to torch's `GridSampler`.
// It handles boundary conditions and interpolation orders defined in
// `bounds.h` and `interpolation.h`. These parameters can be specified 
// per dimension.
// Isotorpic 0-th and 1-st order interpolation have their own (faster)
// implementations. Sliding boundary conditions are also implemented 
// separately.

// TODO:
// . [DONE] generic 3d
// . [DONE] generic 2d
// . [DONE] generic 1d
// . sliding nearest 3d
// . sliding nearest 2d
// . sliding linear 3d
// . sliding linear 2d
// . slinding generic 3d
// . sliding generic 2d
// . [DONE] spatial gradient mode (without mutliplication with output gradient)
// . [DONE] second order gradients (backward pass for spatial gradients)
// . performance tests
// . input bound/inter are always vectors -> clean unused constructors

#include "common.h"
#include "bounds_common.h"
#include "interpolation_common.h"
#include <ATen/ATen.h>
#include <tuple>
#include <limits>
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

// maximum number of channels
// > not used in mode isotropic nearest/linear
// We override the (small) default
#undef  NI_MAX_NUM_CHANNELS
#define NI_MAX_NUM_CHANNELS 1024

// This parameter allows for a little bit of tolerance when considering 
// a coordinate as "out-of-bound" (if !extrapolate)
#define TINY 5e-2

using at::Tensor;
using at::TensorOptions;
using c10::IntArrayRef;

namespace ni {
NI_NAMESPACE_DEVICE { // cpu / cuda / ...

namespace { // anonymous namespace > everything inside has internal linkage

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        INDEXING UTILS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This class reads and sets all the parameters that will later be used
// by the algorithm in PushPullImpl. All of this is done outside of the
// implementation class so that we do not depend on generic types. The
// point is to pre-allocate all necessary tensors so that we can check
// if they're all compatible with 32 bit math. If it's the case, we can
// dispatch to a 32b cuda implementation, which might increase
// performance. Else, we use 64 bit math to compute offsets.
// (On CPU, we always use 64 bit offsets because it doesn't make a huge
// difference. It would be different if we had a vectorized
// implementation as in PyTorch).
class PushPullAllocator {
public:

  static constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();

  // ~~~ CONSTRUCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST
  PushPullAllocator(int dim, BoundVectorRef bound,
                    InterpolationVectorRef interpolation,
                    int extrapolate, bool do_pull, bool do_push,
                    bool do_count, bool do_grad, bool do_sgrad):
    dim(dim),
    bound0(bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound1(bound.size() > 1 ? bound[1] :
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound2(bound.size() > 2 ? bound[2] :
           bound.size() > 1 ? bound[1] :
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    interpolation0(interpolation.size() > 0 ? interpolation[0]
                                            : InterpolationType::Linear),
    interpolation1(interpolation.size() > 1 ? interpolation[1] :
                   interpolation.size() > 0 ? interpolation[0]
                                            : InterpolationType::Linear),
    interpolation2(interpolation.size() > 2 ? interpolation[2] :
                   interpolation.size() > 1 ? interpolation[1] :
                   interpolation.size() > 0 ? interpolation[0]
                                            : InterpolationType::Linear),
    extrapolate(extrapolate),
    do_pull(do_pull),
    do_push(do_push),
    do_count(do_count),
    do_grad(do_grad),
    do_sgrad(do_sgrad)
  {
    iso = interpolation0 == interpolation1 &&
          interpolation0 == interpolation2;
  }

  // TODO: remove constructors that take non-vector Bound/Interpolation
  //       as they are not used anymore.

  NI_HOST
  PushPullAllocator(int dim, BoundType bound, InterpolationVectorRef interpolation,
                    int extrapolate, bool do_pull, bool do_push,
                    bool do_count, bool do_grad, bool do_sgrad):
    dim(dim),
    bound0(bound),
    bound1(bound),
    bound2(bound),
    interpolation0(interpolation.size() > 0 ? interpolation[0]
                                            : InterpolationType::Linear),
    interpolation1(interpolation.size() > 1 ? interpolation[1] :
                   interpolation.size() > 0 ? interpolation[0]
                                            : InterpolationType::Linear),
    interpolation2(interpolation.size() > 2 ? interpolation[2] :
                   interpolation.size() > 1 ? interpolation[1] :
                   interpolation.size() > 0 ? interpolation[0]
                                            : InterpolationType::Linear),
    extrapolate(extrapolate),
    do_pull(do_pull),
    do_push(do_push),
    do_count(do_count),
    do_grad(do_grad),
    do_sgrad(do_sgrad)
  {
    iso = interpolation0 == interpolation1 &&
          interpolation0 == interpolation2;
  }

  NI_HOST
  PushPullAllocator(int dim, BoundVectorRef bound, InterpolationType interpolation,
                    int extrapolate, bool do_pull, bool do_push,
                    bool do_count, bool do_grad, bool do_sgrad):
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
    do_count(do_count),
    do_grad(do_grad),
    do_sgrad(do_sgrad)
  {
    iso = interpolation0 == interpolation1 &&
          interpolation0 == interpolation2;
  }

  NI_HOST
  PushPullAllocator(int dim, BoundType bound, InterpolationType interpolation,
                    int extrapolate, bool do_pull, bool do_push,
                    bool do_count, bool do_grad, bool do_sgrad):
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
    do_count(do_count),
    do_grad(do_grad),
    do_sgrad(do_sgrad)
  {
    iso = interpolation0 == interpolation1 &&
          interpolation0 == interpolation2;
  }

  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Usually used for pull:
  // - do_pull  -> return source[grid]
  // - do_push  -> fails
  // - do_grad  -> return J(source)[grid]
  // - do_sgrad -> return H(source)[grid]
  NI_HOST void ioset
  (const Tensor& source, const Tensor& grid)
  {
    init_all();
    init_source(source);
    init_grid(grid);
    init_output();
  }

  // Usually used for pull_backward:
  // - do_pull  -> return source[grid]
  // - do_push  -> return push(target, grid, source.shape)
  // - do_grad  -> return J(source)[grid]
  // - do_sgrad -> return H(source)[grid]
  NI_HOST void ioset
  (const Tensor& source, const Tensor& grid, const Tensor& target)
  {
    init_all();
    init_source(source);
    init_grid(grid);
    init_target(target);
    init_output();
  }

  // Usually used for push:
  // - do_pull  -> fails
  // - do_push  -> return push(target, grid, source_size)
  // - do_grad  -> fails
  // - do_sgrad -> fails
  NI_HOST void ioset
  (IntArrayRef source_size, const Tensor& grid, const Tensor& target)
  {
    init_all();
    init_source(source_size);
    init_grid(grid);
    init_target(target);
    init_output();
  }

  // Usually used for count:
  // - do_pull  -> fails
  // - do_push  -> return push(ones, grid, source_size)
  // - do_grad  -> fails
  // - do_sgrad -> fails
  NI_HOST void ioset
  (IntArrayRef source_size, const Tensor& grid)
  {
    init_all();
    init_source(source_size);
    init_grid(grid);
    init_output();
  }

  // We just check that all tensors that we own are compatible with 32b math
  bool canUse32BitIndexMath(int64_t max_elem=max_int32) const
  {
    return src_32b_ok  &&
           trgt_32b_ok &&
           grid_32b_ok &&
           grad_32b_ok &&
           out_32b_ok;
  }

private:

  // Copied from aten/src/ATen/native/IndexingUtils.cpp in PyTorch 1.6.
  // It is used to decide to which pointer type we should dispatch to.
  // Basically, we need to make sure that the "furthest" element we need
  // to reach is less than max_elem away.
  static bool tensorCanUse32BitIndexMath(
    const Tensor &t, int64_t max_elem=max_int32)
  {
    int64_t elements = t.numel();
    if (elements >= max_elem) {
      return false;
    }
    if (elements == 0) {
      return max_elem > 0;
    }

    int64_t offset = 0;
    int64_t linearId = elements - 1;

    // NOTE: Assumes all strides are positive, which is true for now
    for (int i = t.dim() - 1; i >= 0; --i) {
      int64_t curDimIndex = linearId % t.size(i);
      int64_t curDimOffset = curDimIndex * t.stride(i);
      offset += curDimOffset;
      linearId /= t.size(i);
    }

    if (offset >= max_elem) {
      return false;
    }

    return true;
  }

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void init_all();
  NI_HOST void init_source(const Tensor& source);
  NI_HOST void init_source(IntArrayRef source_size);
  NI_HOST void init_grid(const Tensor& grid);
  NI_HOST void init_target(const Tensor& target);
  NI_HOST void init_output();

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  InterpolationType interpolation0; // interpolation order // x|W
  InterpolationType interpolation1; // interpolation order // y|H
  InterpolationType interpolation2; // interpolation order // z|D
  bool              iso;            // isotropic interpolation?
  int               extrapolate;    // compute out-of-bound values (0 no | 1 yes | 2 different threshold)
  bool              do_pull;        // sample a volume
  bool              do_push;        // splat a volume
  bool              do_count;       // splatting weights (= jacobian determinant)
  bool              do_grad;        // backprop: gradient of grid // pull
  bool              do_sgrad;       // sample spatial gradients

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  std::deque<Tensor> output;
  TensorOptions src_opt;
  TensorOptions grid_opt;
  TensorOptions trgt_opt;
  int64_t N;
  int64_t C;
  int64_t src_X;
  int64_t src_Y;
  int64_t src_Z;
  int64_t trgt_X;
  int64_t trgt_Y;
  int64_t trgt_Z;
  int64_t trgt_K;
  int64_t src_sN;
  int64_t src_sC;
  int64_t src_sX;
  int64_t src_sY;
  int64_t src_sZ;
  bool src_32b_ok;
  void *src_ptr;
  int64_t trgt_sN;
  int64_t trgt_sC;
  int64_t trgt_sX;
  int64_t trgt_sY;
  int64_t trgt_sZ;
  int64_t trgt_sK;
  bool trgt_32b_ok;
  void *trgt_ptr;
  int64_t grid_sN;
  int64_t grid_sC;
  int64_t grid_sX;
  int64_t grid_sY;
  int64_t grid_sZ;
  bool grid_32b_ok;
  void *grid_ptr;
  int64_t out_sN;
  int64_t out_sC;
  int64_t out_sX;
  int64_t out_sY;
  int64_t out_sZ;
  int64_t out_sK; // gradient dimension
  bool out_32b_ok;
  void *out_ptr;
  int64_t grad_sN;
  int64_t grad_sC;
  int64_t grad_sX;
  int64_t grad_sY;
  int64_t grad_sZ;
  bool grad_32b_ok;
  void *grad_ptr;

  // Allow PushPullImpl's constructor to access PushPullAllocator's
  // private members.
  template <typename scalar_t, typename offset_t>
  friend class PushPullImpl;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          INITIALISATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST
void PushPullAllocator::init_all()
{
  src_opt = grid_opt = trgt_opt = TensorOptions();
  N = C   = 1L;
  src_X   = src_Y   = src_Z   = 1L;
  trgt_X  = trgt_Y  = trgt_Z  = 1L;
  trgt_K  = 0L;
  src_sN  = src_sC   = src_sX   = src_sY  = src_sZ   = 0L;
  grid_sN = grid_sC  = grid_sX  = grid_sY = grid_sZ  = 0L;
  grad_sN = grad_sC  = grad_sX  = grad_sY = grad_sZ  = 0L;
  trgt_sN = trgt_sC  = trgt_sX  = trgt_sY = trgt_sZ  = trgt_sK = 0L;
  out_sN  = out_sC   = out_sX   = out_sY  = out_sZ   = out_sK  = 0L;
  src_ptr = trgt_ptr = grid_ptr = out_ptr = grad_ptr = static_cast<float*>(0);
  src_32b_ok = trgt_32b_ok = grid_32b_ok = out_32b_ok = grad_32b_ok = true;
}

NI_HOST
void PushPullAllocator::init_source(const Tensor& source)
{
  N       = source.size(0);
  C       = source.size(1);
  src_X   = source.size(2);
  src_Y   = dim < 2 ? 1L : source.size(3);
  src_Z   = dim < 3 ? 1L : source.size(4);
  src_sN  = source.stride(0);
  src_sC  = source.stride(1);
  src_sX  = source.stride(2);
  src_sY  = dim < 2 ? 0L : source.stride(3);
  src_sZ  = dim < 3 ? 0L : source.stride(4);
  src_ptr = source.data_ptr();
  src_opt = source.options();
  src_32b_ok = tensorCanUse32BitIndexMath(source);
}

NI_HOST
void PushPullAllocator::init_source(IntArrayRef source_size)
{
  src_X = source_size[0];
  src_Y = dim < 2 ? 1L : source_size[1];
  src_Z = dim < 3 ? 1L : source_size[2];
}

NI_HOST
void PushPullAllocator::init_grid(const Tensor& grid)
{
  N        = grid.size(0);
  trgt_X   = grid.size(1);
  trgt_Y   = dim < 2 ? 1L : grid.size(2);
  trgt_Z   = dim < 3 ? 1L : grid.size(3);
  grid_sN  = grid.stride(0);
  grid_sX  = grid.stride(1);
  grid_sY  = dim < 2 ? 0L : grid.stride(2);
  grid_sZ  = dim < 3 ? 0L : grid.stride(3);
  grid_sC  = grid.stride(dim == 1 ? 2 : dim == 2 ? 3 : 4);
  grid_ptr = grid.data_ptr();
  grid_opt = grid.options();
  grid_32b_ok = tensorCanUse32BitIndexMath(grid);
}

NI_HOST
void PushPullAllocator::init_target(const Tensor& target)
{
  N        = target.size(0);
  C        = target.size(1);
  trgt_X   = target.size(2);
  trgt_Y   = dim < 2 ? 1L : target.size(3);
  trgt_Z   = dim < 3 ? 1L : target.size(4);
  trgt_K   = target.dim() == dim + 3 ? target.size(dim == 1 ? 3 :
                                                   dim == 2 ? 4 : 5)
                                     : 0L;
  trgt_sN  = target.stride(0);
  trgt_sC  = target.stride(1);
  trgt_sX  = target.stride(2);
  trgt_sY  = dim < 2 ? 0L : target.stride(3);
  trgt_sZ  = dim < 3 ? 0L : target.stride(4);
  trgt_sK  = target.dim() == dim + 3 ? target.stride(dim == 1 ? 3 :
                                                     dim == 2 ? 4 : 5)
                                     : 0L;
  trgt_ptr = target.data_ptr();
  trgt_opt = target.options();
  trgt_32b_ok = tensorCanUse32BitIndexMath(target);
}

NI_HOST
void PushPullAllocator::init_output()
{
  output.clear();
  if (do_pull) {
    if (dim == 1)
      output.push_back(at::empty({N, C, trgt_X}, src_opt));
    else if (dim == 2)
      output.push_back(at::empty({N, C, trgt_X, trgt_Y}, src_opt));
    else
      output.push_back(at::empty({N, C, trgt_X, trgt_Y, trgt_Z}, src_opt));
    auto pull = output.back();
    out_sN   = pull.stride(0);
    out_sC   = pull.stride(1);
    out_sX   = pull.stride(2);
    out_sY   = dim < 2 ? 0L : pull.stride(3);
    out_sZ   = dim < 3 ? 0L : pull.stride(4);
    out_sK   = 0L;
    out_ptr  = pull.data_ptr();
    out_32b_ok = tensorCanUse32BitIndexMath(pull);
  }
  else if (do_sgrad) {
    if (dim == 1)
      output.push_back(at::empty({N, C, trgt_X, 1}, src_opt));
    else if (dim == 2)
      output.push_back(at::empty({N, C, trgt_X, trgt_Y, 2}, src_opt));
    else
      output.push_back(at::empty({N, C, trgt_X, trgt_Y, trgt_Z, 3}, src_opt));
    auto sgrad = output.back();
    out_sN   = sgrad.stride(0);
    out_sC   = sgrad.stride(1);
    out_sX   = sgrad.stride(2);
    out_sY   = dim < 2 ? 0L : sgrad.stride(3);
    out_sZ   = dim < 3 ? 0L : sgrad.stride(4);
    out_sK   = sgrad.stride(dim == 1 ? 3 : dim == 2 ? 4 : 5);
    out_ptr  = sgrad.data_ptr();
    out_32b_ok = tensorCanUse32BitIndexMath(sgrad);

    if (iso && interpolation0 == InterpolationType::Nearest)
      sgrad.zero_();
    if (iso && interpolation0 == InterpolationType::Linear && dim == 1)
      sgrad.zero_();
  }
  else if (do_push) {
    if (dim == 1)
      output.push_back(at::zeros({N, C, src_X}, trgt_opt));
    else if (dim == 2)
      output.push_back(at::zeros({N, C, src_X, src_Y}, trgt_opt));
    else
      output.push_back(at::zeros({N, C, src_X, src_Y, src_Z}, trgt_opt));
    auto push = output.back();
    out_sN   = push.stride(0);
    out_sC   = push.stride(1);
    out_sX   = push.stride(2);
    out_sY   = dim < 2 ? 0L : push.stride(3);
    out_sZ   = dim < 3 ? 0L : push.stride(4);
    out_sK   = 0L;
    out_ptr  = push.data_ptr();
    out_32b_ok = tensorCanUse32BitIndexMath(push);
  }
  else if (do_count) {
    if (dim == 1)
      output.push_back(at::zeros({N, 1, src_X}, grid_opt));
    else if (dim == 2)
      output.push_back(at::zeros({N, 1, src_X, src_Y}, grid_opt));
    else
      output.push_back(at::zeros({N, 1, src_X, src_Y, src_Z}, grid_opt));
    auto count = output.back();
    out_sN   = count.stride(0);
    out_sC   = count.stride(1);
    out_sX   = count.stride(2);
    out_sY   = dim < 2 ? 0L : count.stride(3);
    out_sZ   = dim < 3 ? 0L : count.stride(4);
    out_sK   = 0L;
    out_ptr  = count.data_ptr();
    out_32b_ok = tensorCanUse32BitIndexMath(count);
  }
  if (do_grad) {
    if (dim == 1)
      output.push_back(at::zeros({N, trgt_X, 1}, grid_opt));
    else if (dim == 2)
      output.push_back(at::zeros({N, trgt_X, trgt_Y, 2}, grid_opt));
    else
      output.push_back(at::zeros({N, trgt_X, trgt_Y, trgt_Z, 3}, grid_opt));
    auto grad = output.back();
    grad_sN   = grad.stride(0);
    grad_sX   = grad.stride(1);
    grad_sY   = dim < 2 ? 0L : grad.stride(2);
    grad_sZ   = dim < 3 ? 0L : grad.stride(3);
    grad_sC   = grad.stride(dim == 1 ? 2 : dim == 2 ? 3 : 4);
    grad_ptr  = grad.data_ptr();
    out_32b_ok = tensorCanUse32BitIndexMath(grad);

    if (iso && interpolation0 == InterpolationType::Nearest)
      grad.zero_();
  }
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        GENERIC PUSHPULL CLASS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This class implements the bulk of the code.
// /!\ No type and shape checking is performed here.

template <typename scalar_t, typename offset_t>
class PushPullImpl {
public:

  // ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PushPullImpl(const PushPullAllocator & info):
    output(info.output),
    dim(info.dim),
    bound0(info.bound0), bound1(info.bound1), bound2(info.bound2),
    interpolation0(info.interpolation0),
    interpolation1(info.interpolation1),
    interpolation2(info.interpolation1),
    iso(info.iso), extrapolate(info.extrapolate),
    do_pull(info.do_pull), do_push(info.do_push), do_count(info.do_count),
    do_grad(info.do_grad), do_sgrad(info.do_sgrad),
    N(static_cast<offset_t>(info.N)),
    C(static_cast<offset_t>(info.C)),
    src_X(static_cast<offset_t>(info.src_X)),
    src_Y(static_cast<offset_t>(info.src_Y)),
    src_Z(static_cast<offset_t>(info.src_Z)),
    trgt_X(static_cast<offset_t>(info.trgt_X)),
    trgt_Y(static_cast<offset_t>(info.trgt_Y)),
    trgt_Z(static_cast<offset_t>(info.trgt_Z)),
    trgt_K(static_cast<offset_t>(info.trgt_K)),
    src_sN(static_cast<offset_t>(info.src_sN)),
    src_sC(static_cast<offset_t>(info.src_sC)),
    src_sX(static_cast<offset_t>(info.src_sX)),
    src_sY(static_cast<offset_t>(info.src_sY)),
    src_sZ(static_cast<offset_t>(info.src_sZ)),
    src_ptr(static_cast<scalar_t*>(info.src_ptr)),
    trgt_sN(static_cast<offset_t>(info.trgt_sN)),
    trgt_sC(static_cast<offset_t>(info.trgt_sC)),
    trgt_sX(static_cast<offset_t>(info.trgt_sX)),
    trgt_sY(static_cast<offset_t>(info.trgt_sY)),
    trgt_sZ(static_cast<offset_t>(info.trgt_sZ)),
    trgt_sK(static_cast<offset_t>(info.trgt_sK)),
    trgt_ptr(static_cast<scalar_t*>(info.trgt_ptr)),
    grid_sN(static_cast<offset_t>(info.grid_sN)),
    grid_sC(static_cast<offset_t>(info.grid_sC)),
    grid_sX(static_cast<offset_t>(info.grid_sX)),
    grid_sY(static_cast<offset_t>(info.grid_sY)),
    grid_sZ(static_cast<offset_t>(info.grid_sZ)),
    grid_ptr(static_cast<scalar_t*>(info.grid_ptr)),
    out_sN(static_cast<offset_t>(info.out_sN)),
    out_sC(static_cast<offset_t>(info.out_sC)),
    out_sX(static_cast<offset_t>(info.out_sX)),
    out_sY(static_cast<offset_t>(info.out_sY)),
    out_sZ(static_cast<offset_t>(info.out_sZ)),
    out_sK(static_cast<offset_t>(info.out_sK)),
    out_ptr(static_cast<scalar_t*>(info.out_ptr)),
    grad_sN(static_cast<offset_t>(info.grad_sN)),
    grad_sC(static_cast<offset_t>(info.grad_sC)),
    grad_sX(static_cast<offset_t>(info.grad_sX)),
    grad_sY(static_cast<offset_t>(info.grad_sY)),
    grad_sZ(static_cast<offset_t>(info.grad_sZ)),
    grad_ptr(static_cast<scalar_t*>(info.grad_ptr))
  {}


  // ~~~ PUBLIC VALUE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::deque<Tensor> output;

  // NI_HOST NI_DEVICE void printInfo() const {
  //   printf("dim: %d\n", dim);
  //   printf("do_pull:  %d\n", do_pull);
  //   printf("do_push:  %d\n", do_push);
  //   printf("do_count: %d\n", do_count);
  //   printf("do_sgrad: %d\n", do_sgrad);
  //   printf("do_grad:  %d\n", do_grad);
  //   printf("bound:         [%d %d %d]\n", static_cast<int>(bound0), 
  //     static_cast<int>(bound1), static_cast<int>(bound2));
  //   printf("interpolation: [%d %d %d]\n", static_cast<int>(interpolation0), 
  //     static_cast<int>(interpolation1), static_cast<int>(interpolation2));
  //   printf("src:  [%d %d %d]\n", src_Z, src_Y, src_X);
  //   printf("trgt: [%d %d %d (%d)]\n", trgt_Z, trgt_Y, trgt_X, trgt_K);
  //   printf("N: %d\n", N);
  //   printf("C: %d\n", C);
  //   printf("src  -> %lu\n", reinterpret_cast<std::uintptr_t>(src_ptr));
  //   printf("trgt -> %lu\n", reinterpret_cast<std::uintptr_t>(trgt_ptr));
  //   printf("grid -> %lu\n", reinterpret_cast<std::uintptr_t>(grid_ptr));
  //   printf("out  -> %lu\n", reinterpret_cast<std::uintptr_t>(out_ptr));
  //   printf("grad -> %lu\n", reinterpret_cast<std::uintptr_t>(grad_ptr));
  // }

  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
  // Loop over voxels that belong to one CUDA block
  // This function is called by the CUDA kernel
  NI_DEVICE void loop(int threadIdx, int blockIdx, 
                      int blockDim, int gridDim) const;
#else
  // Loop over all voxels
  void loop() const;
#endif

  NI_HOST NI_DEVICE int64_t voxcount() const { 
    return N * trgt_X * trgt_Y * trgt_Z;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_DEVICE void check1d(offset_t w, offset_t n) const;
  NI_DEVICE void check2d(offset_t w, offset_t h, offset_t n) const;
  NI_DEVICE void check3d(offset_t w, offset_t h, offset_t d, offset_t n) const;
  NI_DEVICE void interpolate1d(
    scalar_t x, offset_t w, offset_t n) const;
  NI_DEVICE void interpolate1d_nearest(
    scalar_t x, offset_t w, offset_t n) const;
  NI_DEVICE void interpolate1d_linear(
    scalar_t x, offset_t w, offset_t n) const;
  NI_DEVICE void interpolate1d_sliding(
    scalar_t x, offset_t w, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate1d_sliding_nearest(
    scalar_t x, offset_t w, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate1d_sliding_linear(
    scalar_t x, offset_t w, offset_t n) const {/*TODO*/}
  NI_DEVICE void interpolate2d(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h, offset_t n) const;
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
    offset_t w, offset_t h, offset_t d, offset_t n) const;
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
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  InterpolationType interpolation0; // interpolation order // x|W
  InterpolationType interpolation1; // interpolation order // y|H
  InterpolationType interpolation2; // interpolation order // z|D
  bool              iso;            // isotropic interpolation?
  int               extrapolate;    // compute out-of-bound values
  bool              do_pull;        // sample a volume
  bool              do_push;        // splat a volume
  bool              do_count;       // splatting weights (= jacobian determinant)
  bool              do_grad;        // backprop: gradient of grid // pull
  bool              do_sgrad;       // sample spatial gradients

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  offset_t N;
  offset_t C;
  offset_t src_X;
  offset_t src_Y;
  offset_t src_Z;
  offset_t trgt_X;
  offset_t trgt_Y;
  offset_t trgt_Z;
  offset_t trgt_K;
  offset_t src_sN;
  offset_t src_sC;
  offset_t src_sX;
  offset_t src_sY;
  offset_t src_sZ;
  scalar_t *src_ptr;
  offset_t trgt_sN;
  offset_t trgt_sC;
  offset_t trgt_sX;
  offset_t trgt_sY;
  offset_t trgt_sZ;
  offset_t trgt_sK;
  scalar_t *trgt_ptr;
  offset_t grid_sN;
  offset_t grid_sC;
  offset_t grid_sX;
  offset_t grid_sY;
  offset_t grid_sZ;
  scalar_t *grid_ptr;
  offset_t out_sN;
  offset_t out_sC;
  offset_t out_sX;
  offset_t out_sY;
  offset_t out_sZ;
  offset_t out_sK; // gradient dimension
  scalar_t *out_ptr;
  offset_t grad_sN;
  offset_t grad_sC;
  offset_t grad_sX;
  offset_t grad_sY;
  offset_t grad_sZ;
  scalar_t *grad_ptr;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  int64_t index = blockIdx * blockDim + threadIdx;
  int64_t nthreads = voxcount();
  offset_t trgt_XYZ  = trgt_Z * trgt_Y * trgt_X;
  offset_t trgt_YZ   = trgt_Z * trgt_Y;
  offset_t n, w, h, d;
  for (offset_t i=index; index < nthreads; index += blockDim*gridDim, i=index)
  {
      // Convert index: linear to sub
      n  = (i/trgt_XYZ);
      w  = (i/trgt_YZ) % trgt_X;
      h  = (i/trgt_Z)  % trgt_Y;
      d  = i % trgt_Z;

      if (dim == 1)
        check1d(w, n);
      else if (dim == 2)
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
//       of compute per voxel, so a smaller value might be better suited.
template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::loop() const
{
  if (!has_atomic_add<scalar_t>::value && (do_push || do_count))
  {
    // I do not have access to atomic operations so I cannot
    // parallelize across voxels.
    at::parallel_for(0, N, 0, [&](offset_t start, offset_t end) {
      for (offset_t n = start; n < end; ++n) {
        if (dim == 1) {
          for (offset_t w=0; w<trgt_X; ++w)
            check1d(w, n);
        } else if (dim == 2) {
          for (offset_t h=0; h<trgt_Y; ++h)
          for (offset_t w=0; w<trgt_X; ++w)
            check2d(w, h, n);
        } else {
          for (offset_t d=0; d<trgt_Z; ++d)
          for (offset_t h=0; h<trgt_Y; ++h)
          for (offset_t w=0; w<trgt_X; ++w)
            check3d(w, h, d, n);
        }
      }
    });
    return;
  }

  // Parallelize across voxels   
  offset_t trgt_NXYZ = trgt_Z * trgt_Y * trgt_X * N;
  offset_t trgt_XYZ  = trgt_Z * trgt_Y * trgt_X;
  offset_t trgt_YZ   = trgt_Z * trgt_Y;
  at::parallel_for(0, trgt_NXYZ, GRAIN_SIZE,
                   [&](offset_t start, offset_t end) {
    offset_t n, w, h, d;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/trgt_XYZ);
      w  = (i/trgt_YZ) % trgt_X;
      h  = (i/trgt_Z)  % trgt_Y;
      d  = i % trgt_Z;

      if (dim == 1)
        check1d(w, n);
      else if (dim == 2)
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

template <typename scalar_t, typename offset_t> NI_DEVICE
bool inbounds3d(scalar_t x, scalar_t y, scalar_t z,
                offset_t w, offset_t h, offset_t d,
                int edge)
{
  scalar_t tol = static_cast<scalar_t>(edge ? 0.5 + TINY : TINY);
  return inbounds(x, w, tol) && inbounds(y, h, tol) && inbounds(z, d, tol);
}

template <typename scalar_t, typename offset_t> NI_DEVICE
bool inbounds2d(scalar_t x, scalar_t y,
                offset_t w, offset_t h,
                int edge)
{
  scalar_t tol = static_cast<scalar_t>(edge ? 0.5 + TINY : TINY);
  return inbounds(x, w, tol) && inbounds(y, h, tol);
}

template <typename scalar_t, typename offset_t> NI_DEVICE
bool inbounds1d(scalar_t x, offset_t w, int edge)
{
  scalar_t tol = static_cast<scalar_t>(edge ? 0.5 + TINY : TINY);
  return inbounds(x, w, tol);
}

// Here, we:
// 1) read the [x,y,z] source coordinate for the current target voxel
// 3) check if the source coordinate is in bounds 


template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>
::check3d(offset_t w, offset_t h, offset_t d, offset_t n) const
{
  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t *grid_ptr_NXYZ = grid_ptr + n * grid_sN + w * grid_sX
                                     + h * grid_sY + d * grid_sZ;
  scalar_t x = *grid_ptr_NXYZ;
  scalar_t y = grid_ptr_NXYZ[grid_sC];
  scalar_t z = grid_ptr_NXYZ[grid_sC*2];

  // Check if out-of-bound
  if (!(extrapolate & 1 ||
        inbounds3d(x, y, z, src_X, src_Y, src_Z, extrapolate & 2))) {
    if (do_pull || do_sgrad) {
      scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX
                                        + h * out_sY + d * out_sZ;
      for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC) {
        *out_ptr_NCXYZ = static_cast<scalar_t>(0);
        if (do_sgrad) {
          out_ptr_NCXYZ[out_sK]   = static_cast<scalar_t>(0);
          out_ptr_NCXYZ[out_sK*2] = static_cast<scalar_t>(0);
        }
      }
    }
    if (do_grad) {
      scalar_t * grad_ptr_NXYZ = grad_ptr + n * grad_sN + w * grad_sX
                                          + h * grad_sY + d * grad_sZ;
      (*grad_ptr_NXYZ)         = static_cast<scalar_t>(0);
      grad_ptr_NXYZ[grad_sC]   = static_cast<scalar_t>(0);
      grad_ptr_NXYZ[grad_sC*2] = static_cast<scalar_t>(0);
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

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>
::check2d(offset_t w, offset_t h, offset_t n) const
{
  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t *grid_ptr_NXY = grid_ptr + n * grid_sN
                                    + w * grid_sX
                                    + h * grid_sY;
  scalar_t x = *grid_ptr_NXY;
  scalar_t y = grid_ptr_NXY[grid_sC];

  // Check if out-of-bound
  if (!(extrapolate & 1 ||
        inbounds2d(x, y, src_X, src_Y, extrapolate & 2))) {
    if (do_pull || do_sgrad) {
      scalar_t *out_ptr_NCXY = out_ptr + n * out_sN
                                       + w * out_sX
                                       + h * out_sY;
      for (offset_t c = 0; c < C; ++c, out_ptr_NCXY += out_sC) {
        *out_ptr_NCXY = static_cast<scalar_t>(0);
        if (do_sgrad)
          out_ptr_NCXY[out_sK]   = static_cast<scalar_t>(0);
      }
    }
    if (do_grad) {
      scalar_t * grad_ptr_NXY = grad_ptr + n * grad_sN
                                         + w * grad_sX
                                         + h * grad_sY;
      (*grad_ptr_NXY) = static_cast<scalar_t>(0);
      grad_ptr_NXY[grad_sC] = static_cast<scalar_t>(0);
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
::check1d(offset_t w, offset_t n) const
{
  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t *grid_ptr_NX = grid_ptr + n * grid_sN
                                   + w * grid_sX;
  scalar_t x = *grid_ptr_NX;

  // Check if out-of-bound
  if (!(extrapolate & 1 || inbounds1d(x, src_X, extrapolate & 2))) {
    if (do_pull || do_sgrad) {
      scalar_t *out_ptr_NCX = out_ptr + n * out_sN
                                      + w * out_sX;
      for (offset_t c = 0; c < C; ++c, out_ptr_NCX += out_sC) {
        *out_ptr_NCX = static_cast<scalar_t>(0);
        if (do_sgrad)
          out_ptr_NCX[out_sK]   = static_cast<scalar_t>(0);
      }
    }
    if (do_grad) {
      scalar_t * grad_ptr_NX = grad_ptr + n * grad_sN
                                        + w * grad_sX;
      (*grad_ptr_NX) = static_cast<scalar_t>(0);
      grad_ptr_NX[grad_sC] = static_cast<scalar_t>(0);
    }
    return;
  }

  // Next step
  if (bound0 == BoundType::Sliding) {
    if (iso) switch (static_cast<int>(interpolation0)) {
      case 0: return interpolate1d_sliding_nearest(x, w, n);
      case 1: return interpolate1d_sliding_linear(x, w, n);
    }
    return interpolate1d_sliding(x, w, n);
  } else {
    if (iso) switch (static_cast<int>(interpolation0)) {
      case 0: return interpolate1d_nearest(x, w, n);
      case 1: return interpolate1d_linear(x, w, n);
    }
    return interpolate1d(x, w, n);
  }

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     GENERIC INTERPOLATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate3d(
  scalar_t x, scalar_t y, scalar_t z,
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  // Get corner pixel values from (x, y, z)
  offset_t bx0, bx1, by0, by1, bz0, bz1;
  interpolation::bounds(interpolation0, x, bx0, bx1);
  interpolation::bounds(interpolation1, y, by0, by1);
  interpolation::bounds(interpolation2, z, bz0, bz1);
  offset_t dbx = bx1-bx0;
  offset_t dby = by1-by0;
  offset_t dbz = bz1-bz0;

  // Pre-compute offsets and target value
  scalar_t *src_ptr_NC0    = src_ptr  + n * src_sN;
  scalar_t *out_ptr_NC0    = out_ptr  + n * out_sN;
  scalar_t *out_ptr_NCXYZ0 = out_ptr  + n * out_sN  + w * out_sX 
                                      + h * out_sY  + d * out_sZ;
  scalar_t *trgt_ptr_NCXYZ = trgt_ptr + n * trgt_sN + w * trgt_sX 
                                      + h * trgt_sY + d * trgt_sZ;
  scalar_t target[3*NI_MAX_NUM_CHANNELS]; 
  if (trgt_ptr && (do_push || do_grad))
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXYZ += trgt_sC) {
      target[c]     = *trgt_ptr_NCXYZ;
      if (trgt_K > 0) {
        target[c+C]   = trgt_ptr_NCXYZ[trgt_sK];
        target[c+C*2] = trgt_ptr_NCXYZ[trgt_sK*2];
      }
    }

  // Initialize output
  scalar_t * out_ptr_NCXYZ = out_ptr_NCXYZ0;
  if (do_pull || do_sgrad) {
    for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC) {
      *out_ptr_NCXYZ = static_cast<scalar_t>(0);
      if (do_sgrad) {
        out_ptr_NCXYZ[out_sK]   = static_cast<scalar_t>(0);
        out_ptr_NCXYZ[out_sK*2] = static_cast<scalar_t>(0);
      }
    }
  }

  // Pre-compute indices/weights/grad
  scalar_t  wx[8],  wy[8],  wz[8]; // B-spline weights
  scalar_t  gx[8],  gy[8],  gz[8]; // B-spline derivatives
  scalar_t  hx[8],  hy[8],  hz[8]; // B-spline 2nd derivatives
  offset_t  ix[8],  iy[8],  iz[8]; // Warped indices
  uint8_t   sx[8],  sy[8],  sz[8]; // Warped indices

  {
    scalar_t *owz = static_cast<scalar_t*>(wz), 
             *ogz = static_cast<scalar_t*>(gz),
             *ohz = static_cast<scalar_t*>(hz);
    offset_t *oiz = static_cast<offset_t*>(iz);
    uint8_t  *osz = static_cast<uint8_t *>(sz);
    for (offset_t bz = bz0; bz <= bz1; ++bz) {
      scalar_t dz = z - bz;
      *(owz++)  = interpolation::fastweight(interpolation2, dz);
      if (do_grad || do_sgrad)  *(ogz++) = interpolation::fastgrad(interpolation2, dz);
      if (do_grad && trgt_sK>1) *(ohz++) = interpolation::fasthess(interpolation2, dz);
      *(osz++)  = bound::sign(bound2, bz, src_Z);
      *(oiz++)  = bound::index(bound2, bz, src_Z);
    }
  }
  {
    scalar_t *owy = static_cast<scalar_t*>(wy), 
             *ogy = static_cast<scalar_t*>(gy),
             *ohy = static_cast<scalar_t*>(hy);
    offset_t *oiy = static_cast<offset_t*>(iy);
    uint8_t  *osy = static_cast<uint8_t *>(sy);
    for (offset_t by = by0; by <= by1; ++by) {
      scalar_t dy = y - by;
      *(owy++) = interpolation::fastweight(interpolation1, dy);
      if (do_grad || do_sgrad)  *(ogy++) = interpolation::fastgrad(interpolation1, dy);
      if (do_grad && trgt_sK>1) *(ohy++) = interpolation::fasthess(interpolation1, dy);
      *(osy++)  = bound::sign(bound1, by, src_Y);
      *(oiy++)  = bound::index(bound1, by, src_Y);
    }
  }
  {
    scalar_t *owx = static_cast<scalar_t*>(wx), 
             *ogx = static_cast<scalar_t*>(gx),
             *ohx = static_cast<scalar_t*>(hx);
    offset_t *oix = static_cast<offset_t*>(ix);
    uint8_t  *osx = static_cast<uint8_t *>(sx);
    for (offset_t bx = bx0; bx <= bx1; ++bx) {
      scalar_t dx = x - bx;
      *(owx++)  = interpolation::fastweight(interpolation0, dx);
      if (do_grad || do_sgrad)  *(ogx++) = interpolation::fastgrad(interpolation0, dx);
      if (do_grad && trgt_sK>1) *(ohx++) = interpolation::fasthess(interpolation0, dx);
      *(osx++)  = bound::sign(bound0, bx, src_X);
      *(oix++)  = bound::index(bound0, bx, src_X);
    }
  }

  // Convolve coefficients with basis functions
  scalar_t ogx, ogy, ogz;
  ogx = ogy = ogz = static_cast<scalar_t>(0);
  for (offset_t k = 0; k <= dbz; ++k) {
    offset_t ooz = iz[k] * out_sZ;
    offset_t osz = iz[k] * src_sZ;
    uint8_t  szz = sz[k];
    scalar_t wzz = wz[k];
    scalar_t gzz = gz[k];
    scalar_t hzz = hz[k];
    for (offset_t j = 0; j <= dby; ++j) {
      offset_t ooyz = ooz + iy[j] * out_sY;
      offset_t osyz = osz + iy[j] * src_sY;
      uint8_t  syz  = szz * sy[j];
      scalar_t wyy  = wy[j];
      scalar_t gyy  = gy[j];
      scalar_t hyy  = hy[j];
      for (offset_t i = 0; i <= dbx; ++i) {
        offset_t ooxyz = ooyz + ix[i] * out_sX;
        offset_t osxyz = osyz + ix[i] * src_sX;
        uint8_t  sxyz  = syz  * sx[i];
        scalar_t wxx   = wx[i];
        scalar_t gxx   = gx[i];
        scalar_t hxx   = hx[i];

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (do_pull) {
          scalar_t * src_ptr_NC    = src_ptr_NC0;
          scalar_t * out_ptr_NCXYZ = out_ptr_NCXYZ0;
          for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC,
                                           src_ptr_NC    += src_sC)
            *out_ptr_NCXYZ += bound::get(src_ptr_NC, osxyz, sxyz) * 
            (wxx*wyy*wzz);
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ SGrad ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else if (do_sgrad) {
          scalar_t * src_ptr_NC    = src_ptr_NC0;
          scalar_t * out_ptr_NCXYZ = out_ptr_NCXYZ0;
          for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC,
                                           src_ptr_NC    += src_sC) {
            scalar_t src = bound::get(src_ptr_NC, osxyz, sxyz);
            *out_ptr_NCXYZ          += src * (gxx*wyy*wzz);
            out_ptr_NCXYZ[out_sK]   += src * (wxx*gyy*wzz);
            out_ptr_NCXYZ[2*out_sK] += src * (wxx*wyy*gzz);
          }
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else if (do_push) {
          if (trgt_K == 0)
          { 
            // Diff w.r.t. push/pull
            scalar_t * out_ptr_NC = out_ptr_NC0;
            for (offset_t c = 0; c < C; ++c, out_ptr_NC += out_sC)
              bound::add(out_ptr_NC, ooxyz, (wxx*wyy*wzz) * target[c], sxyz);
         }
         else 
         {
            // Diff w.r.t. sgrad
            scalar_t * out_ptr_NC = out_ptr_NC0;
            for (offset_t c = 0; c < C; ++c, out_ptr_NC += out_sC) {
              scalar_t val = (gxx*wyy*wzz) * target[c]
                           + (wxx*gyy*wzz) * target[c+C]
                           + (wxx*wyy*gzz) * target[c+C*2];
              bound::add(out_ptr_NC, ooxyz, val, sxyz);
            }
          }
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Count ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else if (do_count) {
          bound::add(out_ptr_NC0, ooxyz, (wxx*wyy*wzz), sxyz);
        }
        
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (do_grad) {
          if (trgt_K == 0)
          { 
            // Diff w.r.t. pull/push
            scalar_t * src_ptr_NC = src_ptr_NC0;
            scalar_t dot = static_cast<scalar_t>(0);
            for (offset_t c = 0; c < C; ++c, src_ptr_NC += src_sC) {
              scalar_t src = bound::get(src_ptr_NC, osxyz, sxyz);
              dot += (trgt_ptr ? src * target[c] : src);
              // trgt_ptr == 0 in the backward pass of 'count'
            }
            ogx += (gxx * wyy * wzz) * dot;
            ogy += (wxx * gyy * wzz) * dot;
            ogz += (wxx * wyy * gzz) * dot;
          }
          else
          { 
            // Diff w.r.t. sgrad
            scalar_t * src_ptr_NC = src_ptr_NC0;
            scalar_t dot0, dot1, dot2;
            dot0 = dot1 = dot2 = static_cast<scalar_t>(0);
            for (offset_t c = 0; c < C; ++c, src_ptr_NC += src_sC) {
              scalar_t src = bound::get(src_ptr_NC, osxyz, sxyz);
              dot0 += src * target[c];
              dot1 += src * target[c + C];
              dot2 += src * target[c + C*2];
            }
            ogx += (hxx * wyy * wzz) * dot0
                +  (gxx * gyy * wzz) * dot1
                +  (gxx * wyy * gzz) * dot2;
            ogy += (gxx * gyy * wzz) * dot0
                +  (wxx * hyy * wzz) * dot1
                +  (wxx * gyy * gzz) * dot2;
            ogz += (gxx * wyy * gzz) * dot0
                +  (wxx * gyy * gzz) * dot1
                +  (wxx * wyy * hzz) * dot2;
          }
        }
        
      } // x
    } // y
  } // z

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {
    scalar_t * grad_ptr_NXYZ = grad_ptr + n * grad_sN + w * grad_sX 
                                        + h * grad_sY + d * grad_sZ;
    (*grad_ptr_NXYZ)         = ogx;
    grad_ptr_NXYZ[grad_sC]   = ogy;
    grad_ptr_NXYZ[grad_sC*2] = ogz;
  }
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     GENERIC INTERPOLATION 2D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate2d(
  scalar_t x, scalar_t y,
  offset_t w, offset_t h, offset_t n) const
{
  // Get corner pixel values from (x, y)
  offset_t bx0, bx1, by0, by1;
  interpolation::bounds(interpolation0, x, bx0, bx1);
  interpolation::bounds(interpolation1, y, by0, by1);
  offset_t dbx = bx1-bx0;
  offset_t dby = by1-by0;

  // Pre-compute offsets and target value
  scalar_t *src_ptr_NC0   = src_ptr  + n * src_sN;
  scalar_t *out_ptr_NC0   = out_ptr  + n * out_sN;
  scalar_t *out_ptr_NCXY0 = out_ptr  + n * out_sN 
                                     + w * out_sX
                                     + h * out_sY;
  scalar_t *trgt_ptr_NCXY = trgt_ptr + n * trgt_sN
                                     + w * trgt_sX
                                     + h * trgt_sY;
  scalar_t target[2*NI_MAX_NUM_CHANNELS]; 
  if (trgt_ptr && (do_push || do_grad))
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXY += trgt_sC) {
      target[c]     = *trgt_ptr_NCXY;
      if (trgt_K > 0) {
        target[c+C]   = trgt_ptr_NCXY[trgt_sK];
      }
    }

  // Initialize output
  scalar_t * out_ptr_NCXY = out_ptr_NCXY0;
  if (do_pull || do_sgrad) {
    for (offset_t c = 0; c < C; ++c, out_ptr_NCXY += out_sC) {
      *out_ptr_NCXY = static_cast<scalar_t>(0);
      if (do_sgrad) {
        out_ptr_NCXY[out_sK] = static_cast<scalar_t>(0);
      }
    }
  }

  // Pre-compute indices/weights/grad
  scalar_t  wx[8],  wy[8]; // B-spline weights
  scalar_t  gx[8],  gy[8]; // B-spline derivatives
  scalar_t  hx[8],  hy[8]; // B-spline 2nd derivatives
  offset_t  ix[8],  iy[8]; // Warped indices
  uint8_t   sx[8],  sy[8]; // Warped indices

  {
    scalar_t *owy = static_cast<scalar_t*>(wy), 
             *ogy = static_cast<scalar_t*>(gy),
             *ohy = static_cast<scalar_t*>(hy);
    offset_t *oiy = static_cast<offset_t*>(iy);
    uint8_t  *osy = static_cast<uint8_t *>(sy);
    for (offset_t by = by0; by <= by1; ++by) {
      scalar_t dy = y - by;
      *(owy++) = interpolation::fastweight(interpolation1, dy);
      if (do_grad || do_sgrad)  *(ogy++) = interpolation::fastgrad(interpolation1, dy);
      if (do_grad && trgt_sK>1) *(ohy++) = interpolation::fasthess(interpolation1, dy);
      *(osy++)  = bound::sign(bound1, by, src_Y);
      *(oiy++)  = bound::index(bound1, by, src_Y);
    }
  }
  {
    scalar_t *owx = static_cast<scalar_t*>(wx), 
             *ogx = static_cast<scalar_t*>(gx),
             *ohx = static_cast<scalar_t*>(hx);
    offset_t *oix = static_cast<offset_t*>(ix);
    uint8_t  *osx = static_cast<uint8_t *>(sx);
    for (offset_t bx = bx0; bx <= bx1; ++bx) {
      scalar_t dx = x - bx;
      *(owx++)  = interpolation::fastweight(interpolation0, dx);
      if (do_grad || do_sgrad)  *(ogx++) = interpolation::fastgrad(interpolation0, dx);
      if (do_grad && trgt_sK>1) *(ohx++) = interpolation::fasthess(interpolation0, dx);
      *(osx++)  = bound::sign(bound0, bx, src_X);
      *(oix++)  = bound::index(bound0, bx, src_X);
    }
  }

  // Convolve coefficients with basis functions
  scalar_t ogx, ogy;
  ogx = ogy = static_cast<scalar_t>(0);
  for (offset_t j = 0; j <= dby; ++j) {
    offset_t ooy  = iy[j] * out_sY;
    offset_t osy  = iy[j] * src_sY;
    uint8_t  syy  = sy[j];
    scalar_t wyy  = wy[j];
    scalar_t gyy  = gy[j];
    scalar_t hyy  = hy[j];
    for (offset_t i = 0; i <= dbx; ++i) {
      offset_t ooxy = ooy + ix[i] * out_sX;
      offset_t osxy = osy + ix[i] * src_sX;
      uint8_t  sxy  = syy  * sx[i];
      scalar_t wxx  = wx[i];
      scalar_t gxx  = gx[i];
      scalar_t hxx  = hx[i];

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if (do_pull) {
        scalar_t * src_ptr_NC   = src_ptr_NC0;
        scalar_t * out_ptr_NCXY = out_ptr_NCXY0;
        for (offset_t c = 0; c < C; ++c, out_ptr_NCXY += out_sC,
                                         src_ptr_NC   += src_sC)
          *out_ptr_NCXY += bound::get(src_ptr_NC, osxy, sxy) * (wxx*wyy);
      }

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SGrad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      else if (do_sgrad) {
        scalar_t * src_ptr_NC   = src_ptr_NC0;
        scalar_t * out_ptr_NCXY = out_ptr_NCXY0;
        for (offset_t c = 0; c < C; ++c, out_ptr_NCXY += out_sC,
                                         src_ptr_NC   += src_sC) {
          scalar_t src = bound::get(src_ptr_NC, osxy, sxy);
          *out_ptr_NCXY          += src * (gxx*wyy);
          out_ptr_NCXY[out_sK]   += src * (wxx*gyy);
        }
      }

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      else if (do_push) {
        if (trgt_K == 0)
        { 
          // Diff w.r.t. push/pull
          scalar_t * out_ptr_NC = out_ptr_NC0;
          for (offset_t c = 0; c < C; ++c, out_ptr_NC += out_sC)
            bound::add(out_ptr_NC, ooxy, (wxx*wyy) * target[c], sxy);
       }
       else 
        {
          // Diff w.r.t. sgrad
          scalar_t * out_ptr_NC = out_ptr_NC0;
          for (offset_t c = 0; c < C; ++c, out_ptr_NC += out_sC) {
            scalar_t val = (gxx*wyy) * target[c]
                         + (wxx*gyy) * target[c+C];
            bound::add(out_ptr_NC, ooxy, val, sxy);
          }
       }
      }

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Count ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      else if (do_count) {
        bound::add(out_ptr_NC0, ooxy, (wxx*wyy), sxy);
      }
      
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if (do_grad) {
        if (trgt_K == 0)
        { 
          // Diff w.r.t. pull/push
          scalar_t * src_ptr_NC = src_ptr_NC0;
          scalar_t dot = static_cast<scalar_t>(0);
          for (offset_t c = 0; c < C; ++c, src_ptr_NC += src_sC) {
            scalar_t src = bound::get(src_ptr_NC, osxy, sxy);
            dot += (trgt_ptr ? src * target[c] : src);
            // trgt_ptr == 0 in the backward pass of 'count'
          }
          ogx += (gxx * wyy) * dot;
          ogy += (wxx * gyy) * dot;
        }
        else
        { 
          // Diff w.r.t. sgrad
          scalar_t * src_ptr_NC = src_ptr_NC0;
          scalar_t dot0, dot1;
          dot0 = dot1 = static_cast<scalar_t>(0);
          for (offset_t c = 0; c < C; ++c, src_ptr_NC += src_sC) {
            scalar_t src = bound::get(src_ptr_NC, osxy, sxy);
            dot0 += src * target[c];
            dot1 += src * target[c + C];
          }
          ogx += (hxx * wyy) * dot0
              +  (gxx * gyy) * dot1;
          ogy += (gxx * gyy) * dot0
              +  (wxx * hyy) * dot1;
        }
      }
      
    } // x
  } // y

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {
    scalar_t * grad_ptr_NXY = grad_ptr + n * grad_sN 
                                       + w * grad_sX
                                       + h * grad_sY;
    (*grad_ptr_NXY)         = ogx;
    grad_ptr_NXY[grad_sC]   = ogy;
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     GENERIC INTERPOLATION 1D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate1d(
  scalar_t x, offset_t w, offset_t n) const
{
  // Get corner pixel values from (x, y)
  offset_t bx0, bx1;
  interpolation::bounds(interpolation0, x, bx0, bx1);
  offset_t dbx = bx1-bx0;

  // Pre-compute offsets and target value
  scalar_t *src_ptr_NC0   = src_ptr  + n * src_sN;
  scalar_t *out_ptr_NC0   = out_ptr  + n * out_sN;
  scalar_t *out_ptr_NCX0  = out_ptr  + n * out_sN  + w * out_sX;
  scalar_t *trgt_ptr_NCX  = trgt_ptr + n * trgt_sN + w * trgt_sX;
  scalar_t target[2*NI_MAX_NUM_CHANNELS];
  if (trgt_ptr && (do_push || do_grad))
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCX += trgt_sC) {
      target[c]     = *trgt_ptr_NCX;
      if (trgt_K > 0) {
        target[c+C]   = trgt_ptr_NCX[trgt_sK];
      }
    }

  // Initialize output
  scalar_t * out_ptr_NCX = out_ptr_NCX0;
  if (do_pull || do_sgrad) {
    for (offset_t c = 0; c < C; ++c, out_ptr_NCX += out_sC) {
      *out_ptr_NCX = static_cast<scalar_t>(0);
      if (do_sgrad) {
        out_ptr_NCX[out_sK] = static_cast<scalar_t>(0);
      }
    }
  }

  // Pre-compute indices/weights/grad
  scalar_t  wx[8]; // B-spline weights
  scalar_t  gx[8]; // B-spline derivatives
  scalar_t  hx[8]; // B-spline 2nd derivatives
  offset_t  ix[8]; // Warped indices
  uint8_t   sx[8]; // Warped indices

  {
    scalar_t *owx = static_cast<scalar_t*>(wx),
             *ogx = static_cast<scalar_t*>(gx),
             *ohx = static_cast<scalar_t*>(hx);
    offset_t *oix = static_cast<offset_t*>(ix);
    uint8_t  *osx = static_cast<uint8_t *>(sx);
    for (offset_t bx = bx0; bx <= bx1; ++bx) {
      scalar_t dx = x - bx;
      *(owx++)  = interpolation::fastweight(interpolation0, dx);
      if (do_grad || do_sgrad)  *(ogx++) = interpolation::fastgrad(interpolation0, dx);
      if (do_grad && trgt_sK>1) *(ohx++) = interpolation::fasthess(interpolation0, dx);
      *(osx++)  = bound::sign(bound0, bx, src_X);
      *(oix++)  = bound::index(bound0, bx, src_X);
    }
  }

  // Convolve coefficients with basis functions
  scalar_t ogx;
  ogx = static_cast<scalar_t>(0);
  for (offset_t i = 0; i <= dbx; ++i) {
    offset_t oox = ix[i] * out_sX;
    offset_t osx = ix[i] * src_sX;
    uint8_t  sxx = sx[i];
    scalar_t wxx = wx[i];
    scalar_t gxx = gx[i];
    scalar_t hxx = hx[i];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (do_pull) {
      scalar_t * src_ptr_NC  = src_ptr_NC0;
      scalar_t * out_ptr_NCX = out_ptr_NCX0;
      for (offset_t c = 0; c < C; ++c, out_ptr_NCX += out_sC,
                                       src_ptr_NC  += src_sC)
        *out_ptr_NCX += bound::get(src_ptr_NC, osx, sxx) * wxx;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SGrad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else if (do_sgrad) {
      scalar_t * src_ptr_NC  = src_ptr_NC0;
      scalar_t * out_ptr_NCX = out_ptr_NCX0;
      for (offset_t c = 0; c < C; ++c, out_ptr_NCX += out_sC,
                                       src_ptr_NC  += src_sC) {
        scalar_t src = bound::get(src_ptr_NC, osx, sxx);
        *out_ptr_NCX += src * gxx;
      }
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else if (do_push) {
      if (trgt_K == 0)
      {
        // Diff w.r.t. push/pull
        scalar_t * out_ptr_NC = out_ptr_NC0;
        for (offset_t c = 0; c < C; ++c, out_ptr_NC += out_sC)
          bound::add(out_ptr_NC, oox, wxx * target[c], sxx);
     }
     else
      {
        // Diff w.r.t. sgrad
        scalar_t * out_ptr_NC = out_ptr_NC0;
        for (offset_t c = 0; c < C; ++c, out_ptr_NC += out_sC) {
          scalar_t val = gxx * target[c];
          bound::add(out_ptr_NC, oox, val, sxx);
        }
     }
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Count ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else if (do_count) {
      bound::add(out_ptr_NC0, oox, wxx, sxx);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (do_grad) {
      if (trgt_K == 0)
      {
        // Diff w.r.t. pull/push
        scalar_t * src_ptr_NC = src_ptr_NC0;
        scalar_t dot = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < C; ++c, src_ptr_NC += src_sC) {
          scalar_t src = bound::get(src_ptr_NC, osx, sxx);
          dot += (trgt_ptr ? src * target[c] : src);
          // trgt_ptr == 0 in the backward pass of 'count'
        }
        ogx += gxx * dot;
      }
      else
      {
        // Diff w.r.t. sgrad
        scalar_t * src_ptr_NC = src_ptr_NC0;
        scalar_t dot;
        dot = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < C; ++c, src_ptr_NC += src_sC) {
          scalar_t src = bound::get(src_ptr_NC, osx, sxx);
          dot += src * target[c];
        }
        ogx += hxx * dot;
      }
    }

  } // x

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {
    scalar_t * grad_ptr_NX = grad_ptr + n * grad_sN + w * grad_sX;
    (*grad_ptr_NX)         = ogx;
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
  int8_t  sx1 = bound::sign(bound0, ix0+1, src_X);
  int8_t  sy1 = bound::sign(bound1, iy0+1, src_Y);
  int8_t  sz1 = bound::sign(bound2, iz0+1, src_Z);
  int8_t  sx0 = bound::sign(bound0, ix0,   src_X);
  int8_t  sy0 = bound::sign(bound1, iy0,   src_Y);
  int8_t  sz0 = bound::sign(bound2, iz0,   src_Z);
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
  ix1 = bound::index(bound0, ix0+1, src_X);
  iy1 = bound::index(bound1, iy0+1, src_Y);
  iz1 = bound::index(bound2, iz0+1, src_Z);
  ix0 = bound::index(bound0, ix0,   src_X);
  iy0 = bound::index(bound1, iy0,   src_Y);
  iz0 = bound::index(bound2, iz0,   src_Z);

  offset_t o000, o100, o010, o001, o110, o011, o101, o111;

  if (do_pull || do_grad || do_sgrad) {
    // Offsets into source volume
    o000 = ix0*src_sX + iy0*src_sY + iz0*src_sZ;
    o100 = ix1*src_sX + iy0*src_sY + iz0*src_sZ;
    o010 = ix0*src_sX + iy1*src_sY + iz0*src_sZ;
    o001 = ix0*src_sX + iy0*src_sY + iz1*src_sZ;
    o110 = ix1*src_sX + iy1*src_sY + iz0*src_sZ;
    o011 = ix0*src_sX + iy1*src_sY + iz1*src_sZ;
    o101 = ix1*src_sX + iy0*src_sY + iz1*src_sZ;
    o111 = ix1*src_sX + iy1*src_sY + iz1*src_sZ;
  } else if (!(do_push || do_count)) {
    o000 = o100 = o010 = o001 = o110 = o011 = o101 = o111 = 0;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Grid gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {
    scalar_t gx = static_cast<scalar_t>(0);
    scalar_t gy = static_cast<scalar_t>(0);
    scalar_t gz = static_cast<scalar_t>(0);
    scalar_t *trgt_ptr_NCXYZ = trgt_ptr + n * trgt_sN + w * trgt_sX
                                        + h * trgt_sY + d * trgt_sZ;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;

    if (trgt_K == 0)
    {
      // backward w.r.t. push/pull
      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXYZ += trgt_sC, 
                                       src_ptr_NC     += src_sC) {
        scalar_t src;
        scalar_t trgt = trgt_ptr ? *trgt_ptr_NCXYZ : static_cast<scalar_t>(1);
        // ^ trgt_ptr == 0 during the backward pass of count
        src = bound::get(src_ptr_NC, o000, s000);
        if (trgt_ptr) src *= trgt;
        gx -=       dy0 * dz0 * src;
        gy -= dx0       * dz0 * src;
        gz -= dx0 * dy0       * src;
        src = bound::get(src_ptr_NC, o100, s100);
        if (trgt_ptr) src *= trgt;
        gx +=       dy0 * dz0 * src;
        gy -= dx1       * dz0 * src;
        gz -= dx1 * dy0       * src;
        src = bound::get(src_ptr_NC, o010, s010);
        if (trgt_ptr) src *= trgt;
        gx -=       dy1 * dz0 * src;
        gy += dx0       * dz0 * src;
        gz -= dx0 * dy1       * src;
        src = bound::get(src_ptr_NC, o110, s110);
        if (trgt_ptr) src *= trgt;
        gx +=       dy1 * dz0 * src;
        gy += dx1       * dz0 * src;
        gz -= dx1 * dy1       * src;
        src = bound::get(src_ptr_NC, o001, s001);
        if (trgt_ptr) src *= trgt;
        gx -=       dy0 * dz1 * src;
        gy -= dx0       * dz1 * src;
        gz += dx0 * dy0       * src;
        src = bound::get(src_ptr_NC, o101, s101);
        if (trgt_ptr) src *= trgt;
        gx +=       dy0 * dz1 * src;
        gy -= dx1       * dz1 * src;
        gz += dx1 * dy0       * src;
        src = bound::get(src_ptr_NC, o011, s011);
        if (trgt_ptr) src *= trgt;
        gx -=       dy1 * dz1 * src;
        gy += dx0       * dz1 * src;
        gz += dx0 * dy1       * src;
        src = bound::get(src_ptr_NC, o111, s111);
        if (trgt_ptr) src *= trgt;
        gx +=       dy1 * dz1 * src;
        gy += dx1       * dz1 * src;
        gz += dx1 * dy1       * src;
      }
    }
    else
    {
      // backward w.r.t. sgrad
      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXYZ += trgt_sC, 
                                       src_ptr_NC     += src_sC) {
        scalar_t src;
        scalar_t trgt0 = *trgt_ptr_NCXYZ,
                 trgt1 = trgt_ptr_NCXYZ[trgt_sK],
                 trgt2 = trgt_ptr_NCXYZ[trgt_sK*2];
        src = bound::get(src_ptr_NC, o000, s000);
        gx += ( dz0 * trgt1 + dy0 * trgt2) * src;
        gy += ( dz0 * trgt0 + dx0 * trgt2) * src;
        gz += ( dy0 * trgt0 + dx0 * trgt1) * src;
        src = bound::get(src_ptr_NC, o100, s100);
        gx += (-dz0 * trgt1 - dy0 * trgt2) * src;
        gy += (-dz0 * trgt0 + dx1 * trgt2) * src;
        gz += (-dy0 * trgt0 + dx1 * trgt1) * src;
        src = bound::get(src_ptr_NC, o010, s010);
        gx += (-dz0 * trgt1 + dy1 * trgt2) * src;
        gy += (-dz0 * trgt0 - dx0 * trgt2) * src;
        gz += ( dy1 * trgt0 - dx0 * trgt1) * src;
        src = bound::get(src_ptr_NC, o110, s110);
        gx += ( dz0 * trgt1 - dy1 * trgt2) * src;
        gy += ( dz0 * trgt0 - dx1 * trgt2) * src;
        gz += (-dy1 * trgt0 - dx1 * trgt1) * src;
        src = bound::get(src_ptr_NC, o001, s001);
        gx += ( dz1 * trgt1 - dy0 * trgt2) * src;
        gy += ( dz1 * trgt0 - dx0 * trgt2) * src;
        gz += (-dy0 * trgt0 - dx0 * trgt1) * src;
        src = bound::get(src_ptr_NC, o101, s101);
        gx += (-dz1 * trgt1 + dy0 * trgt2) * src;
        gy += (-dz1 * trgt0 - dx1 * trgt2) * src;
        gz += ( dy0 * trgt0 - dx1 * trgt1) * src;
        src = bound::get(src_ptr_NC, o011, s011);
        gx += (-dz1 * trgt1 - dy1 * trgt2) * src;
        gy += (-dz1 * trgt0 + dx0 * trgt2) * src;
        gz += (-dy1 * trgt0 + dx0 * trgt1) * src;
        src = bound::get(src_ptr_NC, o111, s111);
        gx += ( dz1 * trgt1 + dy1 * trgt2) * src;
        gy += ( dz1 * trgt0 + dx1 * trgt2) * src;
        gz += ( dy1 * trgt0 + dx1 * trgt1) * src;
      }
    }

    scalar_t * grad_ptr_NXYZ = grad_ptr + n * grad_sN + w * grad_sX
                                        + h * grad_sY + d * grad_sZ;
    (*grad_ptr_NXYZ)         = gx;
    grad_ptr_NXYZ[grad_sC]   = gy;
    grad_ptr_NXYZ[grad_sC*2] = gz;
  }

  if (do_push || do_count) {
    // Offsets into 'push' volume
    o000 = ix0*out_sX + iy0*out_sY + iz0*out_sZ;
    o100 = ix1*out_sX + iy0*out_sY + iz0*out_sZ;
    o010 = ix0*out_sX + iy1*out_sY + iz0*out_sZ;
    o001 = ix0*out_sX + iy0*out_sY + iz1*out_sZ;
    o110 = ix1*out_sX + iy1*out_sY + iz0*out_sZ;
    o011 = ix0*out_sX + iy1*out_sY + iz1*out_sZ;
    o101 = ix1*out_sX + iy0*out_sY + iz1*out_sZ;
    o111 = ix1*out_sX + iy1*out_sY + iz1*out_sZ;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_pull) {
    scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX
                                      + h * out_sY + d * out_sZ;
    scalar_t *src_ptr_NC    = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                     src_ptr_NC    += src_sC) {
      *out_ptr_NCXYZ = bound::get(src_ptr_NC, o000, s000) * w000
                     + bound::get(src_ptr_NC, o100, s100) * w100
                     + bound::get(src_ptr_NC, o010, s010) * w010
                     + bound::get(src_ptr_NC, o110, s110) * w110
                     + bound::get(src_ptr_NC, o001, s001) * w001
                     + bound::get(src_ptr_NC, o101, s101) * w101
                     + bound::get(src_ptr_NC, o011, s011) * w011
                     + bound::get(src_ptr_NC, o111, s111) * w111;
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SGrad ~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~
  else if (do_sgrad) {
    scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX
                                      + h * out_sY + d * out_sZ;
    scalar_t *src_ptr_NC    = src_ptr + n * src_sN;

    for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                     src_ptr_NC    += src_sC) {
      scalar_t src000 = bound::get(src_ptr_NC, o000, s000);
      scalar_t src100 = bound::get(src_ptr_NC, o100, s100);
      scalar_t src010 = bound::get(src_ptr_NC, o010, s010);
      scalar_t src110 = bound::get(src_ptr_NC, o110, s110);
      scalar_t src001 = bound::get(src_ptr_NC, o001, s001);
      scalar_t src101 = bound::get(src_ptr_NC, o101, s101);
      scalar_t src011 = bound::get(src_ptr_NC, o011, s011);
      scalar_t src111 = bound::get(src_ptr_NC, o111, s111);
      *out_ptr_NCXYZ =           - dy0 * dz0 * src000
                                 + dy0 * dz0 * src100
                                 - dy1 * dz0 * src010
                                 + dy1 * dz0 * src110
                                 - dy0 * dz1 * src001
                                 + dy0 * dz1 * src101
                                 - dy1 * dz1 * src011
                                 + dy1 * dz1 * src111;
      out_ptr_NCXYZ[out_sK] =    - dx0 * dz0 * src000
                                 - dx1 * dz0 * src100
                                 + dx0 * dz0 * src010
                                 + dx1 * dz0 * src110
                                 - dx0 * dz1 * src001
                                 - dx1 * dz1 * src101
                                 + dx0 * dz1 * src011
                                 + dx1 * dz1 * src111;
      out_ptr_NCXYZ[out_sK*2] =  - dx0 * dy0 * src000
                                 - dx1 * dy0 * src100
                                 - dx0 * dy1 * src010
                                 - dx1 * dy1 * src110
                                 + dx0 * dy0 * src001
                                 + dx1 * dy0 * src101
                                 + dx0 * dy1 * src011
                                 + dx1 * dy1 * src111;
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_push) {
    scalar_t *trgt_ptr_NCXYZ = trgt_ptr + n * trgt_sN + w * trgt_sX
                                        + h * trgt_sY + d * trgt_sZ;
    scalar_t *out_ptr_NC = out_ptr + n * out_sN;
    if (trgt_K == 0)
    {
      // Diff w.r.t. push/pull
      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXYZ += trgt_sC,
                                       out_ptr_NC     += out_sC) {
        scalar_t trgt = *trgt_ptr_NCXYZ;
        bound::add(out_ptr_NC, o000, w000 * trgt, s000);
        bound::add(out_ptr_NC, o100, w100 * trgt, s100);
        bound::add(out_ptr_NC, o010, w010 * trgt, s010);
        bound::add(out_ptr_NC, o110, w110 * trgt, s110);
        bound::add(out_ptr_NC, o001, w001 * trgt, s001);
        bound::add(out_ptr_NC, o101, w101 * trgt, s101);
        bound::add(out_ptr_NC, o011, w011 * trgt, s011);
        bound::add(out_ptr_NC, o111, w111 * trgt, s111);
      }
     }
     else 
      {
        // Diff w.r.t. sgrad
        scalar_t val;
        for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXYZ += trgt_sC,
                                         out_ptr_NC     += out_sC) {
          scalar_t trgt0 = *trgt_ptr_NCXYZ,
                   trgt1 = trgt_ptr_NCXYZ[trgt_sK],
                   trgt2 = trgt_ptr_NCXYZ[trgt_sK*2];
          val = - dy0 * dz0 * trgt0 - dx0 * dz0 * trgt1 - dx0 * dy0 * trgt2;
          bound::add(out_ptr_NC, o000, val, s000);
          val =   dy0 * dz0 * trgt0 - dx1 * dz0 * trgt1 - dx1 * dy0 * trgt2;
          bound::add(out_ptr_NC, o100, val, s100);
          val = - dy1 * dz0 * trgt0 + dx0 * dz0 * trgt1 - dx0 * dy1 * trgt2;
          bound::add(out_ptr_NC, o010, val, s010);
          val =   dy1 * dz0 * trgt0 + dx1 * dz0 * trgt1 - dx1 * dy1 * trgt2;
          bound::add(out_ptr_NC, o110, val, s110);
          val = - dy0 * dz1 * trgt0 - dx0 * dz1 * trgt1 + dx0 * dy0 * trgt2;
          bound::add(out_ptr_NC, o001, val, s001);
          val =   dy0 * dz1 * trgt0 - dx1 * dz1 * trgt1 + dx1 * dy0 * trgt2;
          bound::add(out_ptr_NC, o101, val, s101);
          val = - dy1 * dz1 * trgt0 + dx0 * dz1 * trgt1 + dx0 * dy1 * trgt2;
          bound::add(out_ptr_NC, o011, val, s011);
          val =   dy1 * dz1 * trgt0 + dx1 * dz1 * trgt1 + dx1 * dy1 * trgt2;
          bound::add(out_ptr_NC, o111, val, s111);
        }
     }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_count) {
    // Offsets into 'push' volume
    o000 = ix0*out_sX + iy0*out_sY + iz0*out_sZ;
    o100 = ix1*out_sX + iy0*out_sY + iz0*out_sZ;
    o010 = ix0*out_sX + iy1*out_sY + iz0*out_sZ;
    o001 = ix0*out_sX + iy0*out_sY + iz1*out_sZ;
    o110 = ix1*out_sX + iy1*out_sY + iz0*out_sZ;
    o011 = ix0*out_sX + iy1*out_sY + iz1*out_sZ;
    o101 = ix1*out_sX + iy0*out_sY + iz1*out_sZ;
    o111 = ix1*out_sX + iy1*out_sY + iz1*out_sZ;

    scalar_t *out_ptr_N = out_ptr + n * out_sN;
    bound::add(out_ptr_N, o000, w000, s000);
    bound::add(out_ptr_N, o100, w100, s100);
    bound::add(out_ptr_N, o010, w010, s010);
    bound::add(out_ptr_N, o110, w110, s110);
    bound::add(out_ptr_N, o001, w001, s001);
    bound::add(out_ptr_N, o101, w101, s101);
    bound::add(out_ptr_N, o011, w011, s011);
    bound::add(out_ptr_N, o111, w111, s111);
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
  scalar_t w11 = dx1 * dy1;

  // Sign (/!\ compute sign before warping indices)
  int8_t  sx1 = bound::sign(bound0, ix0+1, src_X);
  int8_t  sy1 = bound::sign(bound1, iy0+1, src_Y);
  int8_t  sx0 = bound::sign(bound0, ix0,   src_X);
  int8_t  sy0 = bound::sign(bound1, iy0,   src_Y);
  int8_t  s00 = sx0 * sy0;
  int8_t  s10 = sx1 * sy0;
  int8_t  s01 = sx0 * sy1;
  int8_t  s11 = sx1 * sy1;

  // Warp indices
  offset_t ix1, iy1;
  ix1 = bound::index(bound0, ix0+1, src_X);
  iy1 = bound::index(bound1, iy0+1, src_Y);
  ix0 = bound::index(bound0, ix0,   src_X);
  iy0 = bound::index(bound1, iy0,   src_Y);

  offset_t o00, o10, o01, o11;
  if (do_pull || do_grad || do_sgrad) {
    // Offsets into source volume
    o00 = ix0*src_sX + iy0*src_sY;
    o10 = ix1*src_sX + iy0*src_sY;
    o01 = ix0*src_sX + iy1*src_sY;
    o11 = ix1*src_sX + iy1*src_sY;
  } else if (!(do_push || do_count)) {
    o00 = o10 = o01 = o11 = 0;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Grid gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {
    scalar_t gx = static_cast<scalar_t>(0);
    scalar_t gy = static_cast<scalar_t>(0);
    scalar_t *trgt_ptr_NCXY = trgt_ptr + n * trgt_sN  
                                       + w * trgt_sX
                                       + h * trgt_sY;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;

    if (trgt_K == 0)
    {
      // backward w.r.t. push/pull
      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXY += trgt_sC, 
                                       src_ptr_NC    += src_sC) {
        scalar_t src;
        scalar_t trgt = trgt_ptr ? *trgt_ptr_NCXY : static_cast<scalar_t>(1);
        // ^ trgt_ptr == 0 during the backward pass of count
        src = bound::get(src_ptr_NC, o00, s00);
        if (trgt_ptr) src *= trgt;
        gx -=       dy0 * src;
        gy -= dx0       * src;
        src = bound::get(src_ptr_NC, o10, s10);
        if (trgt_ptr) src *= trgt;
        gx +=       dy0 * src;
        gy -= dx1       * src;
        src = bound::get(src_ptr_NC, o01, s01);
        if (trgt_ptr) src *= trgt;
        gx -=       dy1 * src;
        gy += dx0       * src;
        src = bound::get(src_ptr_NC, o11, s11);
        if (trgt_ptr) src *= trgt;
        gx +=       dy1 * src;
        gy += dx1       * src;
      }
    }
    else
    {
      // backward w.r.t. sgrad
      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXY += trgt_sC, 
                                       src_ptr_NC    += src_sC) {
        scalar_t src;
        scalar_t trgt0 = *trgt_ptr_NCXY,
                 trgt1 = trgt_ptr_NCXY[trgt_sK];
        src = bound::get(src_ptr_NC, o00, s00);
        gx += trgt1 * src;
        gy += trgt0 * src;
        src = bound::get(src_ptr_NC, o10, s10);
        gx -= trgt1 * src;
        gy -= trgt0 * src;
        src = bound::get(src_ptr_NC, o01, s01);
        gx -= trgt1 * src;
        gy -= trgt0 * src;
        src = bound::get(src_ptr_NC, o11, s11);
        gx += trgt1 * src;
        gy += trgt0 * src;
      }
    }

    scalar_t * grad_ptr_NXY = grad_ptr + n * grad_sN
                                       + w * grad_sX
                                       + h * grad_sY;
    (*grad_ptr_NXY)         = gx;
    grad_ptr_NXY[grad_sC]   = gy;
  }

  if (do_push || do_count) {
    // Offsets into 'push' volume
    o00 = ix0*out_sX + iy0*out_sY;
    o10 = ix1*out_sX + iy0*out_sY;
    o01 = ix0*out_sX + iy1*out_sY;
    o11 = ix1*out_sX + iy1*out_sY;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_pull) {
    scalar_t *out_ptr_NCXY = out_ptr + n * out_sN
                                     + w * out_sX
                                     + h * out_sY;
    scalar_t *src_ptr_NC   = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NCXY += out_sC, 
                                     src_ptr_NC   += src_sC) {
      *out_ptr_NCXY = bound::get(src_ptr_NC, o00, s00) * w00
                    + bound::get(src_ptr_NC, o10, s10) * w10
                    + bound::get(src_ptr_NC, o01, s01) * w01
                    + bound::get(src_ptr_NC, o11, s11) * w11;
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SGrad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_sgrad) {
    scalar_t *out_ptr_NCXY = out_ptr + n * out_sN
                                     + w * out_sX
                                     + h * out_sY;
    scalar_t *src_ptr_NC   = src_ptr + n * src_sN;

    for (offset_t c = 0; c < C; ++c, out_ptr_NCXY += out_sC, 
                                     src_ptr_NC   += src_sC) {
      scalar_t src00 = bound::get(src_ptr_NC, o00, s00);
      scalar_t src10 = bound::get(src_ptr_NC, o10, s10);
      scalar_t src01 = bound::get(src_ptr_NC, o01, s01);
      scalar_t src11 = bound::get(src_ptr_NC, o11, s11);
      *out_ptr_NCXY =           - dy0 * src00
                                + dy0 * src10
                                - dy1 * src01
                                + dy1 * src11;
      out_ptr_NCXY[out_sK] =    - dx0 * src00
                                - dx1 * src10
                                + dx0 * src01
                                + dx1 * src11;
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_push) {
    scalar_t *trgt_ptr_NCXY = trgt_ptr + n * trgt_sN
                                       + w * trgt_sX
                                       + h * trgt_sY;
    scalar_t *out_ptr_NC = out_ptr + n * out_sN;
    if (trgt_K == 0)
    {
      // Diff w.r.t. push/pull
      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXY += trgt_sC,
                                       out_ptr_NC    += out_sC) {
        scalar_t trgt = *trgt_ptr_NCXY;
        bound::add(out_ptr_NC, o00, w00 * trgt, s00);
        bound::add(out_ptr_NC, o10, w10 * trgt, s10);
        bound::add(out_ptr_NC, o01, w01 * trgt, s01);
        bound::add(out_ptr_NC, o11, w11 * trgt, s11);
      }
     }
     else 
      {
        // Diff w.r.t. sgrad
        scalar_t val;
        for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXY += trgt_sC,
                                         out_ptr_NC    += out_sC) {
          scalar_t trgt0 = *trgt_ptr_NCXY,
                   trgt1 = trgt_ptr_NCXY[trgt_sK];
          val = - dy0 * trgt0 - dx0 * trgt1;
          bound::add(out_ptr_NC, o00, val, s00);
          val =   dy0 * trgt0 - dx1 * trgt1;
          bound::add(out_ptr_NC, o10, val, s10);
          val = - dy1 * trgt0 + dx0 * trgt1;
          bound::add(out_ptr_NC, o01, val, s01);
          val =   dy1 * trgt0 + dx1 * trgt1;
          bound::add(out_ptr_NC, o11, val, s11);
        }
     }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_count) {
    scalar_t *out_ptr_N = out_ptr + n * out_sN;
    bound::add(out_ptr_N, o00, w00, s00);
    bound::add(out_ptr_N, o10, w10, s10);
    bound::add(out_ptr_N, o01, w01, s01);
    bound::add(out_ptr_N, o11, w11, s11);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     LINEAR INTERPOLATION 1D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate1d_linear(
  scalar_t x, offset_t w, offset_t n) const
{
  // Get corner pixel values from (x)
  offset_t ix0 = static_cast<offset_t>(std::floor(x));

  // Interpolation weights (inversely proportional to distance)
  scalar_t w1 = x - ix0;
  scalar_t w0 = 1. - w1;

  // Sign (/!\ compute sign before warping indices)
  int8_t  s1 = bound::sign(bound0, ix0+1, src_X);
  int8_t  s0 = bound::sign(bound0, ix0,   src_X);

  // Warp indices
  offset_t ix1;
  ix1 = bound::index(bound0, ix0+1, src_X);
  ix0 = bound::index(bound0, ix0,   src_X);

  
  offset_t o0, o1;
  if (do_pull || do_grad || do_sgrad) {
    // Offsets into source volume
    o0 = ix0*src_sX;
    o1 = ix1*src_sX;
  } else if (!(do_push || do_count)) {
    o0 = o1 = 0;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Grid gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_grad) {

    if (trgt_K == 0)
    {
      // backward w.r.t. push/pull
      scalar_t gx = static_cast<scalar_t>(0);
      scalar_t *trgt_ptr_NCX = trgt_ptr + n * trgt_sN + w * trgt_sX;
      scalar_t *src_ptr_NC   = src_ptr + n * src_sN;

      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCX += trgt_sC,
                                       src_ptr_NC   += src_sC) {
        scalar_t src;
        scalar_t trgt = trgt_ptr ? *trgt_ptr_NCX : static_cast<scalar_t>(1);
        // ^ trgt_ptr == 0 during the backward pass of count
        src = bound::get(src_ptr_NC, o0, s0);
        if (trgt_ptr) src *= trgt;
        gx -= src;
        src = bound::get(src_ptr_NC, o1, s1);
        if (trgt_ptr) src *= trgt;
        gx += src;
      }

      scalar_t * grad_ptr_NX = grad_ptr + n * grad_sN + w * grad_sX;
      (*grad_ptr_NX)         = gx;
    }
    else
    {
      // backward w.r.t. sgrad
      // -> zero (make sure this is done at initialization)
    }
  }

  if (do_push || do_count) {
    // Offsets into 'push' volume
    o0 = ix0*out_sX;
    o1 = ix1*out_sX;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (do_pull) {
    scalar_t *out_ptr_NCX = out_ptr + n * out_sN + w * out_sX;
    scalar_t *src_ptr_NC   = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NCX += out_sC,
                                     src_ptr_NC  += src_sC) {
      *out_ptr_NCX = bound::get(src_ptr_NC, o0, s0) * w0
                   + bound::get(src_ptr_NC, o1, s1) * w1;
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SGrad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_sgrad) {
    scalar_t *out_ptr_NCX = out_ptr + n * out_sN + w * out_sX;
    scalar_t *src_ptr_NC   = src_ptr + n * src_sN;

    for (offset_t c = 0; c < C; ++c, out_ptr_NCX += out_sC,
                                     src_ptr_NC  += src_sC) {
      *out_ptr_NCX  = bound::get(src_ptr_NC, o1, s1)
                    - bound::get(src_ptr_NC, o0, s0);
    }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_push) {
    scalar_t *trgt_ptr_NCX = trgt_ptr + n * trgt_sN + w * trgt_sX;
    scalar_t *out_ptr_NC   = out_ptr + n * out_sN;
    if (trgt_K == 0)
    {
      // Diff w.r.t. push/pull
      for (offset_t c = 0; c < C; ++c, trgt_ptr_NCX += trgt_sC,
                                       out_ptr_NC   += out_sC) {
        scalar_t trgt = *trgt_ptr_NCX;
        bound::add(out_ptr_NC, o0, w0 * trgt, s0);
        bound::add(out_ptr_NC, o1, w1 * trgt, s1);
      }
     }
     else
      {
        // Diff w.r.t. sgrad
        for (offset_t c = 0; c < C; ++c, trgt_ptr_NCX += trgt_sC,
                                         out_ptr_NC   += out_sC) {
          scalar_t trgt0 = *trgt_ptr_NCX;
          bound::add(out_ptr_NC, o0, -trgt0, s0);
          bound::add(out_ptr_NC, o1,  trgt0, s1);
        }
     }
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Push ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else if (do_count) {
    scalar_t *out_ptr_N = out_ptr + n * out_sN;
    bound::add(out_ptr_N, o0, w0, s0);
    bound::add(out_ptr_N, o1, w1, s1);
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
  int8_t    sx = bound::sign(bound0, ix, src_X);
  int8_t    sy = bound::sign(bound1, iy, src_Y);
  int8_t    sz = bound::sign(bound2, iz, src_Z);
            ix = bound::index(bound0, ix,src_X);
            iy = bound::index(bound1, iy,src_Y);
            iz = bound::index(bound2, iz,src_Z);

  // Sign
  int8_t s = sz * sy * sx;

  if (do_pull) {
    offset_t  o = iz*src_sZ + iy*src_sY + ix*src_sX;
    scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX
                                      + h * out_sY + d * out_sZ;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                     src_ptr_NC     += src_sC)
      *out_ptr_NCXYZ = bound::get(src_ptr_NC, o, s);
  }
  else if (do_push && trgt_K  == 0) {
    offset_t  o = iz*out_sZ + iy*out_sY + ix*out_sX;
    scalar_t *trgt_ptr_NCXYZ = trgt_ptr + n * trgt_sN + w * trgt_sX
                                        + h * trgt_sY + d * trgt_sZ;
    scalar_t *out_ptr_NC    = out_ptr + n * out_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXYZ += trgt_sC, 
                                    out_ptr_NC    += out_sC)
      bound::add(out_ptr_NC, o, *trgt_ptr_NCXYZ, s);
  }
  else if (do_count) {
    offset_t  o = iz*out_sZ + iy*out_sY + ix*out_sX;
    scalar_t *out_ptr_NC    = out_ptr + n * out_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NC    += out_sC)
      bound::add(out_ptr_NC, o, static_cast<scalar_t>(1), s);
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
  int8_t    sx = bound::sign(bound0, ix,  src_X);
  int8_t    sy = bound::sign(bound1, iy,  src_Y);
            ix = bound::index(bound0, ix, src_X);
            iy = bound::index(bound1, iy, src_Y);

  // Sign
  int8_t s = sy * sx;

  if (do_pull) {
    offset_t  o = iy*src_sY + ix*src_sX;
    scalar_t *out_ptr_NCXY = out_ptr + n * out_sN 
                                     + w * out_sX
                                     + h * out_sY;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NCXY += out_sC, 
                                    src_ptr_NC    += src_sC)
      *out_ptr_NCXY = bound::get(src_ptr_NC, o, s);
  }
  else if (do_push && trgt_K  == 0) {
    offset_t  o = iy*out_sY + ix*out_sX;
    scalar_t *trgt_ptr_NCXY = trgt_ptr + n * trgt_sN 
                                       + w * trgt_sX
                                       + h * trgt_sY;
    scalar_t *out_ptr_NC    = out_ptr + n * out_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCXY += trgt_sC, 
                                     out_ptr_NC    += out_sC)
      bound::add(out_ptr_NC, o, *trgt_ptr_NCXY, s);
  }
  else if (do_count) {
    offset_t  o = iy*out_sY + ix*out_sX;
    scalar_t *out_ptr_NC    = out_ptr + n * out_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NC    += out_sC)
      bound::add(out_ptr_NC, o, static_cast<scalar_t>(1), s);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  NEAREST NEIGHBOR INTERPOLATION 1D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::interpolate1d_nearest(
  scalar_t x, offset_t w, offset_t n) const
{
  offset_t i = static_cast<offset_t>(std::round(x));

  // Boundary condition (/!\ compute sign before warping indices)
  int8_t    s = bound::sign(bound0, i, src_X);
            i = bound::index(bound0, i, src_X);

  if (do_pull) {
    offset_t  o = i*src_sX;
    scalar_t *out_ptr_NCX = out_ptr + n * out_sN + w * out_sX;
    scalar_t *src_ptr_NC = src_ptr + n * src_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NCX += out_sC,
                                     src_ptr_NC  += src_sC)
      *out_ptr_NCX = bound::get(src_ptr_NC, o, s);
  }
  else if (do_push && trgt_K  == 0) {
    offset_t  o = i*out_sX;
    scalar_t *trgt_ptr_NCX = trgt_ptr + n * trgt_sN + w * trgt_sX;
    scalar_t *out_ptr_NC   = out_ptr  + n * out_sN;
    for (offset_t c = 0; c < C; ++c, trgt_ptr_NCX += trgt_sC,
                                     out_ptr_NC   += out_sC)
      bound::add(out_ptr_NC, o, *trgt_ptr_NCX, s);
  }
  else if (do_count) {
    offset_t  o = i*out_sX;
    scalar_t *out_ptr_NC    = out_ptr + n * out_sN;
    for (offset_t c = 0; c < C; ++c, out_ptr_NC    += out_sC)
      bound::add(out_ptr_NC, o, static_cast<scalar_t>(1), s);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//            LINEAR INTERPOLATION 3D + SLIDING BOUNDARY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TODO


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    BoundType0, InterpolationType0, int, bool, bool, bool, bool, bool); \
  template std::deque<Tensor> pushpull( \
    const SourceType0&, const Tensor&, \
    BoundType0, InterpolationType0, int, bool, bool, bool, bool, bool)
#define PUSHPULL_INSTANTIATE2(BoundType0, InterpolationType0) \
  PUSHPULL_INSTANTIATE3(BoundType0, InterpolationType0, IntArrayRef); \
  PUSHPULL_INSTANTIATE3(BoundType0, InterpolationType0, Tensor)
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
template <typename BoundType, typename InterpolationType, typename SourceType> 
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, int extrapolate,
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  PushPullAllocator info(grid.dim()-2, bound, interpolation, extrapolate,
                         do_pull, do_push, do_count, do_grad, do_sgrad);
  info.ioset(source, grid);

  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "pushpull", [&] {
    if (info.canUse32BitIndexMath())
    {
      PushPullImpl<scalar_t, int32_t> algo(info);
      pushpull_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(algo);
      return algo.output;
    }
    else
    {
      PushPullImpl<scalar_t, int64_t> algo(info);
      pushpull_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(algo);
      return algo.output;
    }
  });
}

// Three arguments (source, grid, target)
// > `bound` and `interpolation` can be single arguments or vectors.
// > `source` can be a tensor or a vector of dimensions.
template <typename BoundType, typename InterpolationType, typename SourceType> 
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, int extrapolate,
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  PushPullAllocator info(grid.dim()-2, bound, interpolation, extrapolate,
                         do_pull, do_push, do_count, do_grad, do_sgrad);
  info.ioset(source, grid, target);

  auto output = AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "pushpull", [&] {
    if (info.canUse32BitIndexMath())
    {
      PushPullImpl<scalar_t, int32_t> algo(info);
      pushpull_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(algo);
      return algo.output;
    }
    else
    {
      PushPullImpl<scalar_t, int64_t> algo(info);
      pushpull_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(algo);
      return algo.output;
    }
  });

  /*
  Our implementation uses more stack per thread than the available local 
  memory. CUDA probably needs to use some of the global memory to 
  compensate, but there is a bug and this memory is never freed.
  The official solution is to call cudaDeviceSetLimit to reset the 
  stack size and free that memory:
  https://forums.developer.nvidia.com/t/61314/2
  */
  cudaDeviceSetLimit(cudaLimitStackSize, 0);
  return output;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (source, grid)
// > `bound` and `interpolation` can be single arguments or vectors.
template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, int extrapolate,
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  PushPullAllocator info(grid.dim()-2, bound, interpolation, extrapolate,
                         do_pull, do_push, do_count, do_grad, do_sgrad);
  info.ioset(source, grid);

  return AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t, int64_t> algo(info);
    algo.loop();
    return algo.output;
  });
}

// Three arguments (source, grid, target)
// > `bound` and `interpolation` can be single arguments or vectors.
// > `source` can be a tensor or a vector of dimensions.
template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, int extrapolate,
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  PushPullAllocator info(grid.dim()-2, bound, interpolation, extrapolate,
                         do_pull, do_push, do_count, do_grad, do_sgrad);
  info.ioset(source, grid, target);

  return AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t, int64_t> algo(info);
    algo.loop();
    return algo.output;
  });
}

#endif // __CUDACC__

PUSHPULL_INSTANTIATE;

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, int extrapolate,
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  throw std::logic_error("Function not implemented for this device.");
}

template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, int extrapolate,
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  throw std::logic_error("Function not implemented for this device.");
}

PUSHPULL_INSTANTIATE;

} // namespace notimplemented

} // namespace ni

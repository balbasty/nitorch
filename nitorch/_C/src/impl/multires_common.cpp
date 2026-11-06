/*
FMG RESTRICTION

The restriction (do_adjoint == true) maps a fine image to a coarse image
with a gather over output (coarse) voxels (no atomics), separably in each
dimension:

  out[j] = sum_i  (1/s) * Lin((x_j - i) / s) * sign(i) * inp[fold(i)],
  x_j    = (j + 0.5) * s - 0.5,     s = fine size / coarse size,

where Lin is the linear B-spline (hat) and fold/sign implement the
boundary condition on the fine grid. FMG always restricts by a factor of
two (coarse size = ceil(fine size / 2), see fmg_common.cpp), so that
s = 2 (even sizes), s = (2m-1)/m < 2 (odd sizes) or s = 1 (size 1).
The support |x_j - i| < s <= 2 is therefore exactly covered by the four
taps floor(x_j) - 1 .. floor(x_j) + 2, and the kernel `restrict2x<D>`
uses a fixed, fully unrolled, 4^D-tap stencil. (With s > 2 the stencil
would be truncated, so multires_impl refuses such restrictions.)

The legacy kernels (restrict{1,2,3}d) compute the same operator and are
kept for comparison; they can be selected by setting the environment
variable `NI_MULTIRES_RESTRICT_LEGACY=1`.
*/
#include "common.h"                // write C++/CUDA compatible code
#include "../defines.h"            // useful macros
#include "bounds_common.h"         // boundary conditions + enum
#include "interpolation_common.h"  // interpolation weights + enum
#include "allocator.h"             // base class handling offset sizes
#include "gather_common.h"         // NI_PRAGMA_UNROLL, gather_env_flag
#include <ATen/ATen.h>             // tensors
#include <tuple>                   // needed by prepare_tensors

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

#define VEC_UNFOLD(ONAME, INAME, DEFAULT)             \
  ONAME##0(INAME.size() > 0 ? INAME[0] : DEFAULT),  \
  ONAME##1(INAME.size() > 1 ? INAME[1] :            \
           INAME.size() > 0 ? INAME[0] : DEFAULT),  \
  ONAME##2(INAME.size() > 2 ? INAME[2] :            \
           INAME.size() > 1 ? INAME[1] :            \
           INAME.size() > 0 ? INAME[0] : DEFAULT)

using at::Tensor;
using at::TensorOptions;
using c10::IntArrayRef;
using c10::ArrayRef;

namespace ni {
NI_NAMESPACE_DEVICE { // cpu / cuda / ...

namespace { // anonymous namespace > everything inside has internal linkage

const auto L = ni::InterpolationType::Linear;
const auto Q = ni::InterpolationType::Quadratic;

static NI_HOST bool multires_restrict_use_legacy()
{
  return gather_env_flag("NI_MULTIRES_RESTRICT_LEGACY");
}

// Taps of the (factor 2) restriction along one dimension, for output
// voxel j: input offsets `o` (folded, times stride), weights `w` and
// boundary signs `s` of input voxels floor(x)-1 .. floor(x)+2.
// The expressions replicate the legacy kernels exactly.
template <typename reduce_t, typename offset_t>
static NI_INLINE NI_DEVICE void restrict2x_taps(
  offset_t j, reduce_t scale, BoundType bound, offset_t size, offset_t stride,
  offset_t o[4], reduce_t w[4], int8_t s[4])
{
  reduce_t x  = (j + 0.5) * scale - 0.5;
  offset_t i1 = static_cast<offset_t>(std::floor(x));
  w[0] = interpolation::weight(L, (x - i1 + 1) / scale) / scale;
  w[1] = interpolation::weight(L, (x - i1)     / scale) / scale;
  w[2] = interpolation::weight(L, (1 + i1 - x) / scale) / scale;
  w[3] = interpolation::weight(L, (2 + i1 - x) / scale) / scale;
  NI_PRAGMA_UNROLL
  for (int k = 0; k < 4; ++k) {
    offset_t i = i1 - 1 + k;
    s[k] = bound::sign(bound, i, size);
    o[k] = bound::index(bound, i, size) * stride;
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        INDEXING UTILS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiResAllocator: public Allocator {
public:

  // ~~~ CONSTRUCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST
  MultiResAllocator(int dim, BoundVectorRef bound,
                    ArrayRef<double> scale,
                    bool do_adjoint):
    dim(dim),
    VEC_UNFOLD(bound,         bound,         BoundType::Replicate),
    VEC_UNFOLD(scale,         scale,         1.),
    do_adjoint(do_adjoint)
  {}

  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  NI_HOST void ioset
  (const Tensor& input, const Tensor& output)
  {
    init_all();
    init_input(input);
    init_output(output);
  }

  // We just check that all tensors that we own are compatible with 32b math
  bool canUse32BitIndexMath(int64_t max_elem=max_int32) const
  {
    return inp_32b_ok && out_32b_ok;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void init_all();
  NI_HOST void init_input(const Tensor& input);
  NI_HOST void init_output(const Tensor& output);

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  double            scale0;         // scale               // x|W
  double            scale1;         // scale               // y|H
  double            scale2;         // scale               // z|D
  bool              do_adjoint;     // push instead of pull

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define DECLARE_ALLOC_INFO_5D(NAME)  \
  int64_t NAME##_X;                 \
  int64_t NAME##_Y;                 \
  int64_t NAME##_Z;                 \
  int64_t NAME##_sN;                \
  int64_t NAME##_sC;                \
  int64_t NAME##_sX;                \
  int64_t NAME##_sY;                \
  int64_t NAME##_sZ;                \
  bool NAME##_32b_ok;               \
  void * NAME##_ptr;

  int64_t N;
  int64_t C;
  DECLARE_ALLOC_INFO_5D(inp)
  DECLARE_ALLOC_INFO_5D(out)

  // Allow MultiResImpl's constructor to access MultiResAllocator's
  // private members.
  template <typename scalar_t, typename offset_t, typename reduce_t>
  friend class MultiResImpl;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          INITIALISATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


NI_HOST
void MultiResAllocator::init_all()
{
  N = C = 1L;
  inp_X = inp_Y = inp_Z = 1L;
  out_X = out_Y = out_Z = 1L;
  inp_ptr = out_ptr = static_cast<float*>(0);
  inp_32b_ok = out_32b_ok = true;
}

NI_HOST
void MultiResAllocator::init_input(const Tensor& input)
{
  N       = input.size(0);
  C       = input.size(1);
  inp_X   = input.size(2);
  inp_Y   = dim < 2 ? 1L : input.size(3);
  inp_Z   = dim < 3 ? 1L : input.size(4);
  inp_sN  = input.stride(0);
  inp_sC  = input.stride(1);
  inp_sX  = input.stride(2);
  inp_sY  = dim < 2 ? 0L : input.stride(3);
  inp_sZ  = dim < 3 ? 0L : input.stride(4);
  inp_ptr = input.data_ptr();
  inp_32b_ok = tensorCanUse32BitIndexMath(input);
}

NI_HOST
void MultiResAllocator::init_output(const Tensor& input)
{
  out_X   = input.size(2);
  out_Y   = dim < 2 ? 1L : input.size(3);
  out_Z   = dim < 3 ? 1L : input.size(4);
  out_sN  = input.stride(0);
  out_sC  = input.stride(1);
  out_sX  = input.stride(2);
  out_sY  = dim < 2 ? 0L : input.stride(3);
  out_sZ  = dim < 3 ? 0L : input.stride(4);
  out_ptr = input.data_ptr();
  out_32b_ok = tensorCanUse32BitIndexMath(input);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            IMPLEMENTATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t, typename reduce_t>
class MultiResImpl {
  typedef MultiResImpl Self;
  typedef void (Self::*ResizeFn)(offset_t x, offset_t y, offset_t z, offset_t n) const;
public:

  // ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  MultiResImpl(const MultiResAllocator & info):

#define COPY_FROM_INFO(name) name(info.name)
#define COPY_FROM_INFO3(name) \
    name##0(info.name##0), name##1(info.name##1), name##2(info.name##2) 

    COPY_FROM_INFO(dim),
    COPY_FROM_INFO3(bound),
    COPY_FROM_INFO3(scale),
    COPY_FROM_INFO(do_adjoint),
    COPY_FROM_INFO(N),
    COPY_FROM_INFO(C),

#define INIT_ALLOC_INFO_5D(NAME) \
    NAME##_X(static_cast<offset_t>(info.NAME##_X)),   \
    NAME##_Y(static_cast<offset_t>(info.NAME##_Y)),   \
    NAME##_Z(static_cast<offset_t>(info.NAME##_Z)),   \
    NAME##_sN(static_cast<offset_t>(info.NAME##_sN)), \
    NAME##_sC(static_cast<offset_t>(info.NAME##_sC)), \
    NAME##_sX(static_cast<offset_t>(info.NAME##_sX)), \
    NAME##_sY(static_cast<offset_t>(info.NAME##_sY)), \
    NAME##_sZ(static_cast<offset_t>(info.NAME##_sZ)), \
    NAME##_ptr(static_cast<scalar_t*>(info.NAME##_ptr))

    INIT_ALLOC_INFO_5D(inp),
    INIT_ALLOC_INFO_5D(out),
    legacy_restrict(multires_restrict_use_legacy())
  {
#ifndef __CUDACC__
      set_resize();
#endif
  }

#ifndef __CUDACC__
  NI_HOST NI_INLINE void set_resize() 
  {
#   define ADJ 4
    uint8_t mode = dim + ADJ * do_adjoint;
    switch (mode) {
      case 1:
        resize_ = &Self::resize1d; break;
      case 2:
        resize_ = &Self::resize2d; break;
      case 3:
        resize_ = &Self::resize3d; break;
      case 1+ADJ:
        resize_ = legacy_restrict ? &Self::restrict1d
                                  : &Self::template restrict2x<1>; break;
      case 2+ADJ:
        resize_ = legacy_restrict ? &Self::restrict2d
                                  : &Self::template restrict2x<2>; break;
      case 3+ADJ:
        resize_ = legacy_restrict ? &Self::restrict3d
                                  : &Self::template restrict2x<3>; break;
      default:
        resize_ = &Self::resize3d; break;
    }
  }
#endif

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
    return N * out_X * out_Y * out_Z;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_DEVICE NI_INLINE void resize(
    offset_t w, offset_t h, offset_t d, offset_t n) const;

#define DECLARE_RESIZE(name) \
  NI_DEVICE void name( \
    offset_t w, offset_t h, offset_t d, offset_t n) const;

  DECLARE_RESIZE(resize1d)
  DECLARE_RESIZE(resize2d)
  DECLARE_RESIZE(resize3d)

  DECLARE_RESIZE(restrict1d)
  DECLARE_RESIZE(restrict2d)
  DECLARE_RESIZE(restrict3d)

  // Factor-2 restriction with a fixed 4^D-tap stencil (D = dim)
  template <int D>
  NI_DEVICE void restrict2x(
    offset_t w, offset_t h, offset_t d, offset_t n) const;

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (1 or 2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  reduce_t          scale0;         // scale               // x|W
  reduce_t          scale1;         // scale               // y|H
  reduce_t          scale2;         // scale               // z|D
  bool              do_adjoint;     // push instead of pull
#ifndef __CUDACC__
  ResizeFn          resize_;        // Pointer to resize function
#endif

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define DECLARE_STRIDE_INFO_5D(NAME) \
  offset_t NAME##_X;                \
  offset_t NAME##_Y;                \
  offset_t NAME##_Z;                \
  offset_t NAME##_sN;               \
  offset_t NAME##_sC;               \
  offset_t NAME##_sX;               \
  offset_t NAME##_sY;               \
  offset_t NAME##_sZ;               \
  scalar_t * NAME##_ptr;

  offset_t N;
  offset_t C;
  DECLARE_STRIDE_INFO_5D(inp)
  DECLARE_STRIDE_INFO_5D(out)

  bool              legacy_restrict; // use restrict{1,2,3}d (legacy)?
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::resize(
    offset_t x, offset_t y, offset_t z, offset_t n) const {
#ifdef __CUDACC__
    // dispatch
#   define ADJ 4
    uint8_t mode = dim + ADJ * do_adjoint;
    switch (mode) {
      case 1:
        return resize1d(x, y, z, n);
      case 2:
        return resize2d(x, y, z, n);
      case 3:
        return resize3d(x, y, z, n);
      case 1+ADJ:
        return legacy_restrict ? restrict1d(x, y, z, n)
                               : this->template restrict2x<1>(x, y, z, n);
      case 2+ADJ:
        return legacy_restrict ? restrict2d(x, y, z, n)
                               : this->template restrict2x<2>(x, y, z, n);
      case 3+ADJ:
        return legacy_restrict ? restrict3d(x, y, z, n)
                               : this->template restrict2x<3>(x, y, z, n);
      default:
        return resize3d(x, y, z, n);
    }
#else
    CALL_MEMBER_FN(*this, resize_)(x, y, z, n);
#endif
}

#ifdef __CUDACC__

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  offset_t index    = blockIdx * blockDim + threadIdx;
  offset_t out_YZ   =   out_Z * out_Y;
  offset_t out_XYZ  =  out_YZ * out_X;
  offset_t out_NXYZ = out_XYZ * N;
  offset_t n, w, h, d;
  for (offset_t i=index; index < out_NXYZ; index += blockDim*gridDim, i=index)
  {
      // Convert index: linear to sub
      n  = (i/out_XYZ);
      w  = (i/out_YZ) % out_X;
      h  = (i/out_Z)  % out_Y;
      d  = i % out_Z;

      resize(w, h, d, n);
  }
}

#else

// This bit loops over all output voxels. We therefore need to
// convert linear indices to multivariate indices. The way I do it
// might not be optimal.
// Note that I parallelize across all voxels (wheareas ATen's grid 
// sampler is only parallelized across batches).
//
// TODO: check that the default grain size is optimal. We do quite a lot
//       of compute per voxel, so a smaller value might be better suited.
template <typename scalar_t, typename offset_t, typename reduce_t> NI_HOST
void MultiResImpl<scalar_t,offset_t,reduce_t>::loop() const
{
  // Both prolongation and restriction gather (each output voxel is
  // written by a single thread, no atomics), so we can always
  // parallelize across output voxels, even without atomic operations.
  // Parallelize across voxels   
  offset_t out_YZ   =   out_Z * out_Y;
  offset_t out_XYZ  =  out_YZ * out_X;
  offset_t out_NXYZ = out_XYZ * N;
  at::parallel_for(0, out_NXYZ, GRAIN_SIZE,
                   [&](offset_t start, offset_t end) {
    offset_t n, w, h, d;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/out_XYZ);
      w  = (i/out_YZ) % out_X;
      h  = (i/out_Z)  % out_Y;
      d  = i % out_Z;

      resize(w, h, d, n);
    }
  }); 
}

#endif


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     QUADRATIC PROLONGATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::resize3d(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  reduce_t x = (w + 0.5) * scale0 - 0.5;
  reduce_t y = (h + 0.5) * scale1 - 0.5;
  reduce_t z = (d + 0.5) * scale2 - 0.5;
  offset_t ix1 = static_cast<offset_t>(std::floor(x+0.5));
  offset_t iy1 = static_cast<offset_t>(std::floor(y+0.5));
  offset_t iz1 = static_cast<offset_t>(std::floor(z+0.5));
  reduce_t dx1 = interpolation::weight(Q, x - ix1);
  reduce_t dy1 = interpolation::weight(Q, y - iy1);
  reduce_t dz1 = interpolation::weight(Q, z - iz1);
  reduce_t dx0 = interpolation::fastweight(Q, x - (ix1 - 1));
  reduce_t dy0 = interpolation::fastweight(Q, y - (iy1 - 1));
  reduce_t dz0 = interpolation::fastweight(Q, z - (iz1 - 1));
  reduce_t dx2 = interpolation::fastweight(Q, (ix1 + 1) - x);
  reduce_t dy2 = interpolation::fastweight(Q, (iy1 + 1) - y);
  reduce_t dz2 = interpolation::fastweight(Q, (iz1 + 1) - z);
  int8_t  sx0 = bound::sign(bound0, ix1-1, inp_X);
  int8_t  sy0 = bound::sign(bound1, iy1-1, inp_Y);
  int8_t  sz0 = bound::sign(bound2, iz1-1, inp_Z);
  int8_t  sx2 = bound::sign(bound0, ix1+1, inp_X);
  int8_t  sy2 = bound::sign(bound1, iy1+1, inp_Y);
  int8_t  sz2 = bound::sign(bound2, iz1+1, inp_Z);
  int8_t  sx1 = bound::sign(bound0, ix1,   inp_X);
  int8_t  sy1 = bound::sign(bound1, iy1,   inp_Y);
  int8_t  sz1 = bound::sign(bound2, iz1,   inp_Z);
  offset_t ix0, iy0, iz0, ix2, iy2, iz2;
  ix0 = bound::index(bound0, ix1-1, inp_X) * inp_sX;
  iy0 = bound::index(bound1, iy1-1, inp_Y) * inp_sY;
  iz0 = bound::index(bound2, iz1-1, inp_Z) * inp_sZ;
  ix2 = bound::index(bound0, ix1+1, inp_X) * inp_sX;
  iy2 = bound::index(bound1, iy1+1, inp_Y) * inp_sY;
  iz2 = bound::index(bound2, iz1+1, inp_Z) * inp_sZ;
  ix1 = bound::index(bound0, ix1,   inp_X) * inp_sX;
  iy1 = bound::index(bound1, iy1,   inp_Y) * inp_sY;
  iz1 = bound::index(bound2, iz1,   inp_Z) * inp_sZ;

  scalar_t *out_ptr_NCXYZ = out_ptr 
                          + n * out_sN + w * out_sX  
                          + h * out_sY + d * out_sZ; 
  scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN;

  for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                   inp_ptr_NC    += inp_sC) 
  {

    auto accum1d = [ix0, ix1, ix2, dx0, dx1, dx2, 
                    sx0, sx1, sx2, inp_ptr_NC]
                    (offset_t i, uint8_t s)
    {
      return static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix0, s * sx0)) * dx0
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix1, s * sx1)) * dx1
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix2, s * sx2)) * dx2;
    };

    auto accum2d = [iy0, iy1, iy2, dy0, dy1, dy2, sy0, sy1, sy2, accum1d]
                    (offset_t i, uint8_t s)
    {
      return accum1d(iy0 + i, sy0 * s) * dy0 
           + accum1d(iy1 + i, sy1 * s) * dy1 
           + accum1d(iy2 + i, sy2 * s) * dy2;
    };

    *out_ptr_NCXYZ  = static_cast<scalar_t>(accum2d(iz0, sz0) * dz0 
                                          + accum2d(iz1, sz1) * dz1 
                                          + accum2d(iz2, sz2) * dz2);
  }
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::resize2d(offset_t w, offset_t h, offset_t d, offset_t n) const
{
  reduce_t x = (w + 0.5) * scale0 - 0.5;
  reduce_t y = (h + 0.5) * scale1 - 0.5;
  offset_t ix1 = static_cast<offset_t>(std::floor(x+0.5));
  offset_t iy1 = static_cast<offset_t>(std::floor(y+0.5));
  reduce_t dx1 = interpolation::weight(Q, x - ix1);
  reduce_t dy1 = interpolation::weight(Q, y - iy1);
  reduce_t dx0 = interpolation::fastweight(Q, x - (ix1 - 1));
  reduce_t dy0 = interpolation::fastweight(Q, y - (iy1 - 1));
  reduce_t dx2 = interpolation::fastweight(Q, (ix1 + 1) - x);
  reduce_t dy2 = interpolation::fastweight(Q, (iy1 + 1) - y);
  int8_t  sx0 = bound::sign(bound0, ix1-1, inp_X);
  int8_t  sy0 = bound::sign(bound1, iy1-1, inp_Y);
  int8_t  sx2 = bound::sign(bound0, ix1+1, inp_X);
  int8_t  sy2 = bound::sign(bound1, iy1+1, inp_Y);
  int8_t  sx1 = bound::sign(bound0, ix1,   inp_X);
  int8_t  sy1 = bound::sign(bound1, iy1,   inp_Y);
  offset_t ix0, iy0, ix2, iy2;
  ix0 = bound::index(bound0, ix1-1, inp_X) * inp_sX;
  iy0 = bound::index(bound1, iy1-1, inp_Y) * inp_sY;
  ix2 = bound::index(bound0, ix1+1, inp_X) * inp_sX;
  iy2 = bound::index(bound1, iy1+1, inp_Y) * inp_sY;
  ix1 = bound::index(bound0, ix1,   inp_X) * inp_sX;
  iy1 = bound::index(bound1, iy1,   inp_Y) * inp_sY;

  scalar_t *out_ptr_NCXYZ = out_ptr 
                          + n * out_sN + w * out_sX + h * out_sY; 
  scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN;

  for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                   inp_ptr_NC    += inp_sC) 
  {

    auto accum1d = [ix0, ix1, ix2, dx0, dx1, dx2, 
                    sx0, sx1, sx2, inp_ptr_NC]
                    (offset_t i, uint8_t s)
    {
      return static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix0, s * sx0)) * dx0
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix1, s * sx1)) * dx1
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix2, s * sx2)) * dx2;
    };

    *out_ptr_NCXYZ  = static_cast<scalar_t>(accum1d(iy0, sy0) * dy0 
                                          + accum1d(iy1, sy1) * dy1 
                                          + accum1d(iy2, sy2) * dy2);
  }
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::resize1d(offset_t w, offset_t h, offset_t d, offset_t n) const
{
  reduce_t x = (w + 0.5) * scale0 - 0.5;
  offset_t ix1 = static_cast<offset_t>(std::floor(x+0.5));
  reduce_t dx1 = interpolation::weight(Q, x - ix1);
  reduce_t dx0 = interpolation::fastweight(Q, x - (ix1 - 1));
  reduce_t dx2 = interpolation::fastweight(Q, (ix1 + 1) - x);
  int8_t  sx0 = bound::sign(bound0, ix1-1, inp_X);
  int8_t  sx2 = bound::sign(bound0, ix1+1, inp_X);
  int8_t  sx1 = bound::sign(bound0, ix1,   inp_X);
  offset_t ix0, ix2;
  ix0 = bound::index(bound0, ix1-1, inp_X) * inp_sX;
  ix2 = bound::index(bound0, ix1+1, inp_X) * inp_sX;
  ix1 = bound::index(bound0, ix1,   inp_X) * inp_sX;

  scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX; 
  scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN;

  for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                   inp_ptr_NC    += inp_sC) 
  {
    *out_ptr_NCXYZ  = static_cast<scalar_t>(
        static_cast<reduce_t>(bound::get(inp_ptr_NC, ix0, sx0)) * dx0
      + static_cast<reduce_t>(bound::get(inp_ptr_NC, ix1, sx1)) * dx1
      + static_cast<reduce_t>(bound::get(inp_ptr_NC, ix2, sx2)) * dx2);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//              LINEAR RESTRICTION (FACTOR 2, 4^D TAPS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Same operator and same order of operations as the legacy kernels
// restrict{1,2,3}d below (unused dimensions use a single unit tap).
template <typename scalar_t, typename offset_t, typename reduce_t>
template <int D>
NI_DEVICE void MultiResImpl<scalar_t,offset_t,reduce_t>::restrict2x(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  constexpr int KY = D > 1 ? 4 : 1;
  constexpr int KZ = D > 2 ? 4 : 1;
  offset_t ox[4], oy[4], oz[4];
  reduce_t wx[4], wy[4], wz[4];
  int8_t   sx[4], sy[4], sz[4];
  restrict2x_taps(w, scale0, bound0, inp_X, inp_sX, ox, wx, sx);
  if (D > 1) {
    restrict2x_taps(h, scale1, bound1, inp_Y, inp_sY, oy, wy, sy);
  } else {
    oy[0] = 0; wy[0] = static_cast<reduce_t>(1); sy[0] = 1;
  }
  if (D > 2) {
    restrict2x_taps(d, scale2, bound2, inp_Z, inp_sZ, oz, wz, sz);
  } else {
    oz[0] = 0; wz[0] = static_cast<reduce_t>(1); sz[0] = 1;
  }

  scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX
                                    + h * out_sY + d * out_sZ;
  const scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN;

  for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC,
                                   inp_ptr_NC    += inp_sC)
  {
    reduce_t acc = static_cast<reduce_t>(0);
    NI_PRAGMA_UNROLL
    for (int kz = 0; kz < KZ; ++kz) {
      reduce_t accz = static_cast<reduce_t>(0);
      NI_PRAGMA_UNROLL
      for (int ky = 0; ky < KY; ++ky) {
        const offset_t oyz = oz[kz] + oy[ky];
        const int8_t   syz = static_cast<int8_t>(sz[kz] * sy[ky]);
        reduce_t accy = static_cast<reduce_t>(0);
        NI_PRAGMA_UNROLL
        for (int kx = 0; kx < 4; ++kx)
          accy += static_cast<reduce_t>(bound::get(
                    inp_ptr_NC, oyz + ox[kx],
                    static_cast<int8_t>(syz * sx[kx]))) * wx[kx];
        accz += accy * wy[ky];
      }
      acc += accz * wz[kz];
    }
    *out_ptr_NCXYZ = static_cast<scalar_t>(acc);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                LINEAR RESTRICTION 3D (LEGACY)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::restrict3d(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  reduce_t x = (w + 0.5) * scale0 - 0.5;
  reduce_t y = (h + 0.5) * scale1 - 0.5;
  reduce_t z = (d + 0.5) * scale2 - 0.5;
  offset_t ix1 = static_cast<offset_t>(std::floor(x));
  offset_t iy1 = static_cast<offset_t>(std::floor(y));
  offset_t iz1 = static_cast<offset_t>(std::floor(z));
  reduce_t dx1 = interpolation::weight(L, (x - ix1) / scale0) / scale0;
  reduce_t dy1 = interpolation::weight(L, (y - iy1) / scale1) / scale1;
  reduce_t dz1 = interpolation::weight(L, (z - iz1) / scale2) / scale2;
  reduce_t dx0 = interpolation::weight(L, (x - ix1 + 1) / scale0) / scale0;
  reduce_t dy0 = interpolation::weight(L, (y - iy1 + 1) / scale1) / scale1;
  reduce_t dz0 = interpolation::weight(L, (z - iz1 + 1) / scale2) / scale2;
  reduce_t dx2 = interpolation::weight(L, (1 + ix1 - x) / scale0) / scale0;
  reduce_t dy2 = interpolation::weight(L, (1 + iy1 - y) / scale1) / scale1;
  reduce_t dz2 = interpolation::weight(L, (1 + iz1 - z) / scale2) / scale2;
  reduce_t dx3 = interpolation::weight(L, (2 + ix1 - x) / scale0) / scale0;
  reduce_t dy3 = interpolation::weight(L, (2 + iy1 - y) / scale1) / scale1;
  reduce_t dz3 = interpolation::weight(L, (2 + iz1 - z) / scale2) / scale2;
  int8_t  sx3 = bound::sign(bound0, ix1+2, inp_X);
  int8_t  sy3 = bound::sign(bound1, iy1+2, inp_Y);
  int8_t  sz3 = bound::sign(bound2, iz1+2, inp_Z);
  int8_t  sx2 = bound::sign(bound0, ix1+1, inp_X);
  int8_t  sy2 = bound::sign(bound1, iy1+1, inp_Y);
  int8_t  sz2 = bound::sign(bound2, iz1+1, inp_Z);
  int8_t  sx0 = bound::sign(bound0, ix1-1, inp_X);
  int8_t  sy0 = bound::sign(bound1, iy1-1, inp_Y);
  int8_t  sz0 = bound::sign(bound2, iz1-1, inp_Z);
  int8_t  sx1 = bound::sign(bound0, ix1,   inp_X);
  int8_t  sy1 = bound::sign(bound1, iy1,   inp_Y);
  int8_t  sz1 = bound::sign(bound2, iz1,   inp_Z);
  offset_t ix0, ix2, ix3, iy0, iy2, iy3, iz0, iz2, iz3;
  ix3 = bound::index(bound0, ix1+2, inp_X) * inp_sX;
  iy3 = bound::index(bound1, iy1+2, inp_Y) * inp_sY;
  iz3 = bound::index(bound2, iz1+2, inp_Z) * inp_sZ;
  ix2 = bound::index(bound0, ix1+1, inp_X) * inp_sX;
  iy2 = bound::index(bound1, iy1+1, inp_Y) * inp_sY;
  iz2 = bound::index(bound2, iz1+1, inp_Z) * inp_sZ;
  ix0 = bound::index(bound0, ix1-1, inp_X) * inp_sX;
  iy0 = bound::index(bound1, iy1-1, inp_Y) * inp_sY;
  iz0 = bound::index(bound2, iz1-1, inp_Z) * inp_sZ;
  ix1 = bound::index(bound0, ix1,   inp_X) * inp_sX;
  iy1 = bound::index(bound1, iy1,   inp_Y) * inp_sY;
  iz1 = bound::index(bound2, iz1,   inp_Z) * inp_sZ;

  scalar_t *out_ptr_NCXYZ = out_ptr
                          + n * out_sN + w * out_sX 
                          + h * out_sY + d * out_sZ;
  scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN;


  for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                   inp_ptr_NC    += inp_sC) 
  {

    auto accum1d = [ix0, ix1, ix2, ix3, dx0, dx1, dx2, dx3, 
                    sx0, sx1, sx2, sx3, inp_ptr_NC]
                    (offset_t i, uint8_t s)
    {
      return static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix0, s * sx0)) * dx0
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix1, s * sx1)) * dx1
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix2, s * sx2)) * dx2
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix3, s * sx3)) * dx3;
    };

    auto accum2d = [iy0, iy1, iy2, iy3, dy0, dy1, dy2, dy3, 
                    sy0, sy1, sy2, sy3, accum1d]
                    (offset_t i, uint8_t s)
    {
      return accum1d(iy0 + i, sy0 * s) * dy0 
           + accum1d(iy1 + i, sy1 * s) * dy1 
           + accum1d(iy2 + i, sy2 * s) * dy2 
           + accum1d(iy3 + i, sy3 * s) * dy3;
    };

    *out_ptr_NCXYZ  = static_cast<scalar_t>(accum2d(iz0, sz0) * dz0 
                                          + accum2d(iz1, sz1) * dz1 
                                          + accum2d(iz2, sz2) * dz2 
                                          + accum2d(iz3, sz3) * dz3);
  }
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::restrict2d(offset_t w, offset_t h, offset_t d, offset_t n) const 
{
  reduce_t x = (w + 0.5) * scale0 - 0.5;
  reduce_t y = (h + 0.5) * scale1 - 0.5;
  offset_t ix1 = static_cast<offset_t>(std::floor(x));
  offset_t iy1 = static_cast<offset_t>(std::floor(y));
  reduce_t dx1 = interpolation::weight(L, (x - ix1) / scale0) / scale0;
  reduce_t dy1 = interpolation::weight(L, (y - iy1) / scale1) / scale1;
  reduce_t dx0 = interpolation::weight(L, (x - ix1 + 1) / scale0) / scale0;
  reduce_t dy0 = interpolation::weight(L, (y - iy1 + 1) / scale1) / scale1;
  reduce_t dx2 = interpolation::weight(L, (1 + ix1 - x) / scale0) / scale0;
  reduce_t dy2 = interpolation::weight(L, (1 + iy1 - y) / scale1) / scale1;
  reduce_t dx3 = interpolation::weight(L, (2 + ix1 - x) / scale0) / scale0;
  reduce_t dy3 = interpolation::weight(L, (2 + iy1 - y) / scale1) / scale1;
  int8_t  sx3 = bound::sign(bound0, ix1+2, inp_X);
  int8_t  sy3 = bound::sign(bound1, iy1+2, inp_Y);
  int8_t  sx2 = bound::sign(bound0, ix1+1, inp_X);
  int8_t  sy2 = bound::sign(bound1, iy1+1, inp_Y);
  int8_t  sx0 = bound::sign(bound0, ix1-1, inp_X);
  int8_t  sy0 = bound::sign(bound1, iy1-1, inp_Y);
  int8_t  sx1 = bound::sign(bound0, ix1,   inp_X);
  int8_t  sy1 = bound::sign(bound1, iy1,   inp_Y);
  offset_t ix0, ix2, ix3, iy0, iy2, iy3;
  ix3 = bound::index(bound0, ix1+2, inp_X) * inp_sX;
  iy3 = bound::index(bound1, iy1+2, inp_Y) * inp_sY;
  ix2 = bound::index(bound0, ix1+1, inp_X) * inp_sX;
  iy2 = bound::index(bound1, iy1+1, inp_Y) * inp_sY;
  ix0 = bound::index(bound0, ix1-1, inp_X) * inp_sX;
  iy0 = bound::index(bound1, iy1-1, inp_Y) * inp_sY;
  ix1 = bound::index(bound0, ix1,   inp_X) * inp_sX;
  iy1 = bound::index(bound1, iy1,   inp_Y) * inp_sY;

  scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX + h * out_sY;
  scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN;


  for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                   inp_ptr_NC    += inp_sC) 
  {

    auto accum1d = [ix0, ix1, ix2, ix3, dx0, dx1, dx2, dx3, 
                    sx0, sx1, sx2, sx3, inp_ptr_NC]
                    (offset_t i, uint8_t s)
    {
      return static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix0, s * sx0)) * dx0
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix1, s * sx1)) * dx1
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix2, s * sx2)) * dx2
           + static_cast<reduce_t>(bound::get(inp_ptr_NC, i + ix3, s * sx3)) * dx3;
    };

    *out_ptr_NCXYZ  = static_cast<scalar_t>(accum1d(iy0, sy0) * dy0 
                                          + accum1d(iy1, sy1) * dy1 
                                          + accum1d(iy2, sy2) * dy2 
                                          + accum1d(iy3, sy3) * dy3);
  }
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void MultiResImpl<scalar_t,offset_t,reduce_t>::restrict1d(offset_t w, offset_t h, offset_t d, offset_t n) const 
{
  reduce_t x = (w + 0.5) * scale0 - 0.5;
  offset_t ix1 = static_cast<offset_t>(std::floor(x));
  reduce_t dx1 = interpolation::weight(L, (x - ix1) / scale0) / scale0;
  reduce_t dx0 = interpolation::weight(L, (x - ix1 + 1) / scale0) / scale0;
  reduce_t dx2 = interpolation::weight(L, (1 + ix1 - x) / scale0) / scale0;
  reduce_t dx3 = interpolation::weight(L, (2 + ix1 - x) / scale0) / scale0;
  int8_t  sx3 = bound::sign(bound0, ix1+2, inp_X);
  int8_t  sx2 = bound::sign(bound0, ix1+1, inp_X);
  int8_t  sx0 = bound::sign(bound0, ix1-1, inp_X);
  int8_t  sx1 = bound::sign(bound0, ix1,   inp_X);
  offset_t ix0, ix2, ix3;
  ix3 = bound::index(bound0, ix1+2, inp_X) * inp_sX;
  ix2 = bound::index(bound0, ix1+1, inp_X) * inp_sX;
  ix0 = bound::index(bound0, ix1-1, inp_X) * inp_sX;
  ix1 = bound::index(bound0, ix1,   inp_X) * inp_sX;

  scalar_t *out_ptr_NCXYZ = out_ptr + n * out_sN + w * out_sX;
  scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN;


  for (offset_t c = 0; c < C; ++c, out_ptr_NCXYZ += out_sC, 
                                   inp_ptr_NC    += inp_sC) 
  {
    *out_ptr_NCXYZ  = static_cast<scalar_t>(
        static_cast<reduce_t>(bound::get(inp_ptr_NC, ix0, sx0)) * dx0
      + static_cast<reduce_t>(bound::get(inp_ptr_NC, ix1, sx1)) * dx1
      + static_cast<reduce_t>(bound::get(inp_ptr_NC, ix2, sx2)) * dx2
      + static_cast<reduce_t>(bound::get(inp_ptr_NC, ix3, sx3)) * dx3);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t, typename reduce_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void resize_kernel(MultiResImpl<scalar_t,offset_t,reduce_t> * f) {
  f->loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST NI_INLINE void
check_same_nonspatial(Tensor input, Tensor output)
{
  bool same_nonspatial = (input.dim()   == output.dim())    &&
                         (input.size(0) == output.size(0))  &&
                         (input.size(1) == output.size(1));
  if (!same_nonspatial) {
    std::string const msg = "Source and output should have the same "
                            "batch and channel shapes but found dims "
                          + std::to_string(input.dim()) + " vs " 
                          + std::to_string(output.dim()) + " and shapes ["
                          + std::to_string(input.size(0)) + " " 
                          + std::to_string(input.size(1)) + "] vs ["
                          + std::to_string(output.size(0)) + " " 
                          + std::to_string(output.size(1)) + "].";
    throw std::invalid_argument(msg);
  }
}

NI_HOST NI_INLINE Tensor
prepare_tensor(Tensor input, Tensor output, ArrayRef<double> factor, bool do_adjoint)
{
  bool output_defined = (output.defined() && output.numel() > 0);

  if (output_defined) {
    check_same_nonspatial(input, output);
    return output;
  }

  double fx = factor.size() > 0 ? factor[0] : 1.;
  double fy = factor.size() > 1 ? factor[1] : fx;
  double fz = factor.size() > 2 ? factor[2] : fy;
  if (do_adjoint) {
    fx = 1./fx;
    fy = 1./fy;
    fz = 1./fz;
  }

  int64_t dim = input.dim() - 2;
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t X = input.size(2);
  int64_t Y = dim > 1 ? input.size(3) : 1L;
  int64_t Z = dim > 2 ? input.size(4) : 1L;

  if (do_adjoint) { // RESTRICT
    int64_t Xs =           static_cast<int64_t>(std::ceil(static_cast<double>(X) * fx));
    int64_t Ys = dim > 1 ? static_cast<int64_t>(std::ceil(static_cast<double>(Y) * fy)) : 1L;
    int64_t Zs = dim > 2 ? static_cast<int64_t>(std::ceil(static_cast<double>(Z) * fz)) : 1L;
    output = at::zeros({N, C, Xs, Ys, Zs}, input.options()); 
  } else { // PROLONG
    int64_t Xt =           static_cast<int64_t>(std::floor(static_cast<double>(X) * fx));
    int64_t Yt = dim > 1 ? static_cast<int64_t>(std::floor(static_cast<double>(Y) * fy)) : 1L;
    int64_t Zt = dim > 2 ? static_cast<int64_t>(std::floor(static_cast<double>(Z) * fz)) : 1L;
    output = at::zeros({N, C, Xt, Yt, Zt}, input.options());
  }

  return output;
}

NI_HOST NI_INLINE std::vector<double>
prepare_scales(Tensor input, Tensor output, bool do_adjoint)
{
  int64_t dim = output.dim() - 2;
  int64_t Xo  = output.size(2);
  int64_t Yo  = dim > 1 ? output.size(3) : 1L;
  int64_t Zo  = dim > 2 ? output.size(4) : 1L;
  int64_t Xi  = input.size(2);
  int64_t Yi  = dim > 1 ? input.size(3) : 1L;
  int64_t Zi  = dim > 2 ? input.size(4) : 1L;

  return std::vector<double>({static_cast<double>(Xi) / static_cast<double>(Xo),
                              static_cast<double>(Yi) / static_cast<double>(Yo),
                              static_cast<double>(Zi) / static_cast<double>(Zo)});
}

// The restriction kernel has a fixed 4-tap stencil per dimension, which
// covers the support of the stretched kernel only if fine/coarse <= 2.
NI_HOST NI_INLINE void
check_restrict_scales(const std::vector<double> & scales, int64_t dim)
{
  for (int64_t d = 0; d < dim; ++d)
    TORCH_CHECK(scales[d] <= 2. + 1e-6,
                "fmg_restriction only supports downsampling by a factor of "
                "(at most) 2, but the input/output size ratio along spatial "
                "dimension ", d, " is ", scales[d], ". Use an output of "
                "size ceil(input size / 2).");
}

} // namespace

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
NI_HOST
Tensor multires_impl(
  Tensor input, Tensor output, ArrayRef<double> factor,
  BoundVectorRef bound, bool do_adjoint)
{
  output = prepare_tensor(input, output, factor, do_adjoint);
  auto scales = prepare_scales(input, output, do_adjoint);
  if (do_adjoint)
    check_restrict_scales(scales, input.dim() - 2);

  MultiResAllocator info(input.dim()-2, bound, scales, do_adjoint);
  info.ioset(input, output);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "multires_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      MultiResImpl<scalar_t, int32_t, scalar_t> algo(info);
      auto palgo = alloc_and_copy_to_device(&algo, stream);
      resize_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>(palgo);
      cudaFree(palgo);
    }
    else
    {
      MultiResImpl<scalar_t, int64_t, scalar_t> algo(info);
      auto palgo = alloc_and_copy_to_device(&algo, stream);
      resize_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>(palgo);
      cudaFree(palgo);
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

NI_HOST
Tensor multires_impl(
  Tensor input, Tensor output, ArrayRef<double> factor,
  BoundVectorRef bound, bool do_adjoint)
{
  output = prepare_tensor(input, output, factor, do_adjoint);
  auto scales = prepare_scales(input, output, do_adjoint);
  if (do_adjoint)
    check_restrict_scales(scales, input.dim() - 2);

  MultiResAllocator info(input.dim()-2, bound, scales, do_adjoint);
  info.ioset(input, output);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "multires_impl", [&] {
    MultiResImpl<scalar_t, int64_t, scalar_t> algo(info);
    algo.loop();
  });

  return output;
}

#endif // __CUDACC__


} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST
Tensor multires_impl(
  Tensor input, Tensor output, ArrayRef<double> factor,
  BoundVectorRef bound, bool do_adjoint)
{
  throw std::logic_error("Function not implemented for this device.");
}

} // namespace notimplemented

} // namespace ni

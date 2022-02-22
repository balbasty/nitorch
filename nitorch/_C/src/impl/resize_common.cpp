#include "common.h"                // write C++/CUDA compatible code
#include "../defines.h"            // useful macros
#include "../grid_align.h"         // enum type for align mode
#include "bounds_common.h"         // boundary conditions + enum
#include "interpolation_common.h"  // interpolation weights + enum
#include "allocator.h"             // base class handling offset sizes
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        INDEXING UTILS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResizeAllocator: public Allocator {
public:

  // ~~~ CONSTRUCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST
  ResizeAllocator(int dim, BoundVectorRef bound,
                    InterpolationVectorRef interpolation,
                    ArrayRef<double> shift,
                    ArrayRef<double> scale,
                    bool do_adjoint):
    dim(dim),
    VEC_UNFOLD(bound,         bound,         BoundType::Replicate),
    VEC_UNFOLD(interpolation, interpolation, InterpolationType::Linear),
    VEC_UNFOLD(shift,         shift,         0.),
    VEC_UNFOLD(scale,         scale,         1.),
    do_adjoint(do_adjoint)
  {
    iso = interpolation0 == interpolation1 &&
          interpolation0 == interpolation2;
  }

  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  NI_HOST void ioset
  (const Tensor& source, const Tensor& target)
  {
    init_all();
    init_source(source);
    init_target(target);
  }

  // We just check that all tensors that we own are compatible with 32b math
  bool canUse32BitIndexMath(int64_t max_elem=max_int32) const
  {
    return src_32b_ok && tgt_32b_ok;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void init_all();
  NI_HOST void init_source(const Tensor& source);
  NI_HOST void init_target(const Tensor& target);

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  InterpolationType interpolation0; // interpolation order // x|W
  InterpolationType interpolation1; // interpolation order // y|H
  InterpolationType interpolation2; // interpolation order // z|D
  double            shift0;         // shift               // x|W
  double            shift1;         // shift               // y|H
  double            shift2;         // shift               // z|D
  double            scale0;         // scale               // x|W
  double            scale1;         // scale               // y|H
  double            scale2;         // scale               // z|D
  bool              iso;            // isotropic interpolation?
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
  DECLARE_ALLOC_INFO_5D(src)
  DECLARE_ALLOC_INFO_5D(tgt)

  // Allow ResizeImpl's constructor to access ResizeAllocator's
  // private members.
  template <typename scalar_t, typename offset_t>
  friend class ResizeImpl;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          INITIALISATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


NI_HOST
void ResizeAllocator::init_all()
{
  N = C = 1L;
  src_X = src_Y = src_Z = 1L;
  tgt_X = tgt_Y = tgt_Z = 1L;
  src_ptr = tgt_ptr = static_cast<float*>(0);
  src_32b_ok = tgt_32b_ok = true;
}

NI_HOST
void ResizeAllocator::init_source(const Tensor& input)
{
  N       = input.size(0);
  C       = input.size(1);
  src_X   = input.size(2);
  src_Y   = dim < 2 ? 1L : input.size(3);
  src_Z   = dim < 3 ? 1L : input.size(4);
  src_sN  = input.stride(0);
  src_sC  = input.stride(1);
  src_sX  = input.stride(2);
  src_sY  = dim < 2 ? 0L : input.stride(3);
  src_sZ  = dim < 3 ? 0L : input.stride(4);
  src_ptr = input.data_ptr();
  src_32b_ok = tensorCanUse32BitIndexMath(input);
}

NI_HOST
void ResizeAllocator::init_target(const Tensor& input)
{
  tgt_X   = input.size(2);
  tgt_Y   = dim < 2 ? 1L : input.size(3);
  tgt_Z   = dim < 3 ? 1L : input.size(4);
  tgt_sN  = input.stride(0);
  tgt_sC  = input.stride(1);
  tgt_sX  = input.stride(2);
  tgt_sY  = dim < 2 ? 0L : input.stride(3);
  tgt_sZ  = dim < 3 ? 0L : input.stride(4);
  tgt_ptr = input.data_ptr();
  tgt_32b_ok = tensorCanUse32BitIndexMath(input);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            IMPLEMENTATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t>
class ResizeImpl {
  typedef ResizeImpl Self;
  typedef void (Self::*ResizeFn)(offset_t x, offset_t y, offset_t z, offset_t n) const;
public:

  // ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ResizeImpl(const ResizeAllocator & info):

#define COPY_FROM_INFO(name) name(info.name)
#define COPY_FROM_INFO3(name) \
    name##0(info.name##0), name##1(info.name##1), name##2(info.name##2) 

    COPY_FROM_INFO(dim),
    COPY_FROM_INFO3(bound),
    COPY_FROM_INFO3(interpolation),
    COPY_FROM_INFO3(shift),
    COPY_FROM_INFO3(scale),
    COPY_FROM_INFO(iso),
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

    INIT_ALLOC_INFO_5D(src),
    INIT_ALLOC_INFO_5D(tgt)
  {
#ifndef __CUDACC__
      set_resize();
#endif
  }

#ifndef __CUDACC__
  NI_HOST NI_INLINE void set_resize() 
  {
#   define ADJ 4
#   define ISO 8
#   define NN  InterpolationType::Nearest
#   define LIN InterpolationType::Linear
#   define QUD InterpolationType::Quadratic
    uint8_t mode = dim + 4 * do_adjoint + 8 * iso;
    switch (mode) {
      case 1:
        resize_ = &Self::resize1d; break;
      case 2:
        resize_ = &Self::resize2d; break;
      case 3:
        resize_ = &Self::resize3d; break;
      case 1+ADJ:
        resize_ = &Self::restrict1d; break; 
      case 2+ADJ:
        resize_ = &Self::restrict2d; break;
      case 3+ADJ:
        resize_ = &Self::restrict3d; break;
      case 1+ISO:
        switch (interpolation0) {
          case NN:
            resize_ = &Self::resize1d_nearest; break;
          case LIN:
            resize_ = &Self::resize1d_linear; break;
          case QUD:
            resize_ = &Self::resize1d_quadratic; break;
          default:
            resize_ = &Self::resize1d; break;
        } break;
      case 2+ISO:
        switch (interpolation0) {
          case NN:
            resize_ = &Self::resize2d_nearest; break;
          case LIN:
            resize_ = &Self::resize2d_linear; break;
          case QUD:
            resize_ = &Self::resize2d_quadratic; break;
          default:
            resize_ = &Self::resize2d; break;
        } break;
      case 3+ISO:
        switch (interpolation0) {
          case NN:
            resize_ = &Self::resize3d_nearest; break;
          case LIN:
            resize_ = &Self::resize3d_linear; break;
          case QUD:
            resize_ = &Self::resize3d_quadratic; break;
          default:
            resize_ = &Self::resize3d; break;
        } break;
      case 1+ADJ+ISO:
        switch (interpolation0) {
          case NN:
            resize_ = &Self::restrict1d_nearest; break;
          case LIN:
            resize_ = &Self::restrict1d_linear; break;
          case QUD:
            resize_ = &Self::restrict1d_quadratic; break;
          default:
            resize_ = &Self::restrict1d; break;
        } break;
      case 2+ADJ+ISO:
        switch (interpolation0) {
          case NN:
            resize_ = &Self::restrict2d_nearest; break;
          case LIN:
            resize_ = &Self::restrict2d_linear; break;
          case QUD:
            resize_ = &Self::restrict2d_quadratic; break;
          default:
            resize_ = &Self::restrict2d; break;
        } break;
      case 3+ADJ+ISO: 
        switch (interpolation0) {
          case NN:
            resize_ = &Self::restrict3d_nearest; break;
          case LIN:
            resize_ = &Self::restrict3d_linear; break;
          case QUD:
            resize_ = &Self::restrict3d_quadratic; break;
          default:
            resize_ = &Self::restrict3d; break;
        } break;
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
    return N * tgt_X * tgt_Y * tgt_Z;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_DEVICE NI_INLINE void resize(
    offset_t w, offset_t h, offset_t d, offset_t n) const;

#define DECLARE_RESIZE(name) \
  NI_DEVICE void name( \
    offset_t w, offset_t h, offset_t d, offset_t n) const;

  DECLARE_RESIZE(resize1d)
  DECLARE_RESIZE(resize1d_nearest)
  DECLARE_RESIZE(resize1d_linear)
  DECLARE_RESIZE(resize1d_quadratic)

  DECLARE_RESIZE(resize2d)
  DECLARE_RESIZE(resize2d_nearest)
  DECLARE_RESIZE(resize2d_linear)
  DECLARE_RESIZE(resize2d_quadratic)

  DECLARE_RESIZE(resize3d)
  DECLARE_RESIZE(resize3d_nearest)
  DECLARE_RESIZE(resize3d_linear)
  DECLARE_RESIZE(resize3d_quadratic)

  DECLARE_RESIZE(restrict1d)
  DECLARE_RESIZE(restrict1d_nearest)
  DECLARE_RESIZE(restrict1d_linear)
  DECLARE_RESIZE(restrict1d_quadratic)

  DECLARE_RESIZE(restrict2d)
  DECLARE_RESIZE(restrict2d_nearest)
  DECLARE_RESIZE(restrict2d_linear)
  DECLARE_RESIZE(restrict2d_quadratic)

  DECLARE_RESIZE(restrict3d)
  DECLARE_RESIZE(restrict3d_nearest)
  DECLARE_RESIZE(restrict3d_linear)
  DECLARE_RESIZE(restrict3d_quadratic)

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (1 or 2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  InterpolationType interpolation0; // interpolation order // x|W
  InterpolationType interpolation1; // interpolation order // y|H
  InterpolationType interpolation2; // interpolation order // z|D
  double            shift0;         // shift               // x|W
  double            shift1;         // shift               // y|H
  double            shift2;         // shift               // z|D
  double            scale0;         // scale               // x|W
  double            scale1;         // scale               // y|H
  double            scale2;         // scale               // z|D
  bool              iso;            // isotropic interpolation?
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
  DECLARE_STRIDE_INFO_5D(src)
  DECLARE_STRIDE_INFO_5D(tgt)
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize(
    offset_t x, offset_t y, offset_t z, offset_t n) const {
#ifdef __CUDACC__
    // dispatch
#   define ADJ 4
#   define ISO 8
#   define NN  InterpolationType::Nearest
#   define LIN InterpolationType::Linear
#   define QUD InterpolationType::Quadratic
    uint8_t mode = dim + 4 * do_adjoint + 8 * iso;
    switch (mode) {
      case 1:
        return resize1d(x, y, z, n);
      case 2:
        return resize2d(x, y, z, n);
      case 3:
        return resize3d(x, y, z, n);
      case 1+ADJ:
        return restrict1d(x, y, z, n); 
      case 2+ADJ:
        return restrict2d(x, y, z, n);
      case 3+ADJ:
        return restrict3d(x, y, z, n);
      case 1+ISO:
        switch (interpolation0) {
          case NN:
            return resize1d_nearest(x, y, z, n);
          case LIN:
            return resize1d_linear(x, y, z, n);
          case QUD:
            return resize1d_quadratic(x, y, z, n);
          default:
            return resize1d(x, y, z, n);
        }
      case 2+ISO:
        switch (interpolation0) {
          case NN:
            return resize2d_nearest(x, y, z, n);
          case LIN:
            return resize2d_linear(x, y, z, n);
          case QUD:
            return resize2d_quadratic(x, y, z, n);
          default:
            return resize2d(x, y, z, n);
        }
      case 3+ISO:
        switch (interpolation0) {
          case NN:
            return resize3d_nearest(x, y, z, n);
          case LIN:
            return resize3d_linear(x, y, z, n);
          case QUD:
            return resize3d_quadratic(x, y, z, n);
          default:
            return resize3d(x, y, z, n);
        }
      case 1+ADJ+ISO:
        switch (interpolation0) {
          case NN:
            return restrict1d_nearest(x, y, z, n);
          case LIN:
            return restrict1d_linear(x, y, z, n);
          case QUD:
            return restrict1d_quadratic(x, y, z, n);
          default:
            return restrict1d(x, y, z, n);
        }
      case 2+ADJ+ISO:
        switch (interpolation0) {
          case NN:
            return restrict2d_nearest(x, y, z, n);
          case LIN:
            return restrict2d_linear(x, y, z, n);
          case QUD:
            return restrict2d_quadratic(x, y, z, n);
          default:
            return restrict2d(x, y, z, n);
        }
      case 3+ADJ+ISO: 
        switch (interpolation0) {
          case NN:
            return restrict3d_nearest(x, y, z, n);
          case LIN:
            return restrict3d_linear(x, y, z, n);
          case QUD:
            return restrict3d_quadratic(x, y, z, n);
          default:
            return restrict3d(x, y, z, n);
        }
      default:
        return resize3d(x, y, z, n);
    }
#else
    CALL_MEMBER_FN(*this, resize_)(x, y, z, n);
#endif
}

#ifdef __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  offset_t index    = blockIdx * blockDim + threadIdx;
  offset_t tgt_YZ   =   tgt_Z * tgt_Y;
  offset_t tgt_XYZ  =  tgt_YZ * tgt_X;
  offset_t tgt_NXYZ = tgt_XYZ * N;
  offset_t n, w, h, d;
  for (offset_t i=index; index < tgt_NXYZ; index += blockDim*gridDim, i=index)
  {
      // Convert index: linear to sub
      n  = (i/tgt_XYZ);
      w  = (i/tgt_YZ) % tgt_X;
      h  = (i/tgt_Z)  % tgt_Y;
      d  = i % tgt_Z;

      resize(w, h, d, n);
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
void ResizeImpl<scalar_t,offset_t>::loop() const
{
  if (!has_atomic_add<scalar_t>::value && (do_adjoint))
  {
    // I do not have access to atomic operations so I cannot
    // parallelize across voxels.
    at::parallel_for(0, N, 0, [&](offset_t start, offset_t end) {
      for (offset_t n = start; n < end; ++n) {
        if (dim == 1) {
          for (offset_t w=0; w<tgt_X; ++w)
            resize(w, 0, 0, n);
        } else if (dim == 2) {
          for (offset_t h=0; h<tgt_Y; ++h)
          for (offset_t w=0; w<tgt_X; ++w)
            resize(w, h, 0, n);
        } else {
          for (offset_t d=0; d<tgt_Z; ++d)
          for (offset_t h=0; h<tgt_Y; ++h)
          for (offset_t w=0; w<tgt_X; ++w)
            resize(w, h, d, n);
        }
      }
    });
    return;
  }

  // Parallelize across voxels   
  offset_t tgt_YZ   =   tgt_Z * tgt_Y;
  offset_t tgt_XYZ  =  tgt_YZ * tgt_X;
  offset_t tgt_NXYZ = tgt_XYZ * N;
  at::parallel_for(0, tgt_NXYZ, GRAIN_SIZE,
                   [&](offset_t start, offset_t end) {
    offset_t n, w, h, d;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/tgt_XYZ);
      w  = (i/tgt_YZ) % tgt_X;
      h  = (i/tgt_Z)  % tgt_Y;
      d  = i % tgt_Z;

      resize(w, h, d, n);
    }
  }); 
}

#endif


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     GENERIC INTERPOLATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#undef  GET_INDEX
#define GET_INDEX \
  scalar_t x = scale0 * w + shift0;                               \
  scalar_t y = scale1 * h + shift1;                               \
  scalar_t z = scale2 * d + shift2;                               \
  offset_t bx0, bx1, by0, by1, bz0, bz1;                          \
  interpolation::bounds(interpolation0, x, bx0, bx1);             \
  interpolation::bounds(interpolation1, y, by0, by1);             \
  interpolation::bounds(interpolation2, z, bz0, bz1);             \
  offset_t dbx = bx1-bx0;                                         \
  offset_t dby = by1-by0;                                         \
  offset_t dbz = bz1-bz0;                                         \
  scalar_t  wx[8],  wy[8],  wz[8];                                \
  offset_t  ix[8],  iy[8],  iz[8];                                \
  uint8_t   sx[8],  sy[8],  sz[8];                                \
  {                                                               \
    scalar_t *owz = static_cast<scalar_t*>(wz);                   \
    offset_t *oiz = static_cast<offset_t*>(iz);                   \
    uint8_t  *osz = static_cast<uint8_t *>(sz);                   \
    for (offset_t bz = bz0; bz <= bz1; ++bz) {                    \
      scalar_t dz = z - bz;                                       \
      *(owz++)  = interpolation::fastweight(interpolation2, dz);  \
      *(osz++)  = bound::sign(bound2, bz, src_Z);                 \
      *(oiz++)  = bound::index(bound2, bz, src_Z);                \
    }                                                             \
  }                                                               \
  {                                                               \
    scalar_t *owy = static_cast<scalar_t*>(wy);                   \
    offset_t *oiy = static_cast<offset_t*>(iy);                   \
    uint8_t  *osy = static_cast<uint8_t *>(sy);                   \
    for (offset_t by = by0; by <= by1; ++by) {                    \
      scalar_t dy = y - by;                                       \
      *(owy++) = interpolation::fastweight(interpolation1, dy);   \
      *(osy++)  = bound::sign(bound1, by, src_Y);                 \
      *(oiy++)  = bound::index(bound1, by, src_Y);                \
    }                                                             \
  }                                                               \
  {                                                               \
    scalar_t *owx = static_cast<scalar_t*>(wx);                   \
    offset_t *oix = static_cast<offset_t*>(ix);                   \
    uint8_t  *osx = static_cast<uint8_t *>(sx);                   \
    for (offset_t bx = bx0; bx <= bx1; ++bx) {                    \
      scalar_t dx = x - bx;                                       \
      *(owx++)  = interpolation::fastweight(interpolation0, dx);  \
      *(osx++)  = bound::sign(bound0, bx, src_X);                 \
      *(oix++)  = bound::index(bound0, bx, src_X);                \
    }                                                             \
  }                                                               \
  scalar_t *src_ptr_NC0    = src_ptr  + n * src_sN;               \
  scalar_t *tgt_ptr_NCXYZ0 = tgt_ptr + n * tgt_sN + w * tgt_sX    \
                                     + h * tgt_sY + d * tgt_sZ;

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize3d(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX

  // Convolve coefficients with basis functions
  for (offset_t k = 0; k <= dbz; ++k) {
    offset_t osz = iz[k] * src_sZ;
    uint8_t  szz = sz[k];
    scalar_t wzz = wz[k];
    for (offset_t j = 0; j <= dby; ++j) {
      offset_t osyz = osz + iy[j] * src_sY;
      uint8_t  syz  = szz * sy[j];
      scalar_t wyz  = wzz * wy[j];
      for (offset_t i = 0; i <= dbx; ++i) {
        offset_t osxyz = osyz + ix[i] * src_sX;
        uint8_t  sxyz  = syz  * sx[i];
        scalar_t wxyz  = wyz  * wx[i];

        scalar_t * src_ptr_NC    = src_ptr_NC0;
        scalar_t * tgt_ptr_NCXYZ = tgt_ptr_NCXYZ0;
        for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC,
                                         src_ptr_NC    += src_sC)
          *tgt_ptr_NCXYZ += bound::get(src_ptr_NC, osxyz, sxyz) * wxyz;

        
      } // x
    } // y
  } // z
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict3d(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX

  scalar_t target[NI_MAX_NUM_CHANNELS];
  scalar_t * tgt_ptr_NCXYZ = tgt_ptr_NCXYZ0;
  for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC)
    target[c] = *tgt_ptr_NCXYZ;

  // Convolve coefficients with basis functions
  for (offset_t k = 0; k <= dbz; ++k) {
    offset_t osz = iz[k] * src_sZ;
    uint8_t  szz = sz[k];
    scalar_t wzz = wz[k];
    for (offset_t j = 0; j <= dby; ++j) {
      offset_t osyz = osz + iy[j] * src_sY;
      uint8_t  syz  = szz * sy[j];
      scalar_t wyz  = wzz * wy[j];
      for (offset_t i = 0; i <= dbx; ++i) {
        offset_t osxyz = osyz + ix[i] * src_sX;
        uint8_t  sxyz  = syz  * sx[i];
        scalar_t wxyz  = wyz  * wx[i];

        scalar_t * src_ptr_NC = src_ptr_NC0;
        for (offset_t c = 0; c < C; ++c, src_ptr_NC += src_sC)
          bound::add(src_ptr_NC, osxyz, wxyz * target[c], sxyz);
        
      } // x
    } // y
  } // z
}


template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize2d(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict2d(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize1d(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict1d(offset_t w, offset_t h, offset_t d, offset_t n) const {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     QUADRATIC INTERPOLATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#undef  GET_INDEX
#define GET_INDEX \
  scalar_t x = scale0 * w + shift0; \
  scalar_t y = scale1 * h + shift1; \
  scalar_t z = scale2 * d + shift2; \
  offset_t ix1 = static_cast<offset_t>(std::floor(x+0.5)); \
  offset_t iy1 = static_cast<offset_t>(std::floor(y+0.5)); \
  offset_t iz1 = static_cast<offset_t>(std::floor(z+0.5)); \
  scalar_t dx1 = interpolation::weight(interpolation0, x - ix1); \
  scalar_t dy1 = interpolation::weight(interpolation1, y - iy1); \
  scalar_t dz1 = interpolation::weight(interpolation2, z - iz1); \
  scalar_t dx0 = interpolation::fastweight(interpolation0, x - (ix1 - 1)); \
  scalar_t dy0 = interpolation::fastweight(interpolation1, y - (iy1 - 1)); \
  scalar_t dz0 = interpolation::fastweight(interpolation2, z - (iz1 - 1)); \
  scalar_t dx2 = interpolation::fastweight(interpolation0, (ix1 + 1) - x); \
  scalar_t dy2 = interpolation::fastweight(interpolation1, (iy1 + 1) - y); \
  scalar_t dz2 = interpolation::fastweight(interpolation2, (iz1 + 1) - z); \
  scalar_t w000 = dx0 * dy0; \
  scalar_t w001 = w000 * dz1; \
  scalar_t w002 = w000 * dz2; \
           w000 *= dz0; \
  scalar_t w010 = dx0 * dy1; \
  scalar_t w011 = w010 * dz1; \
  scalar_t w012 = w010 * dz2; \
           w010 *= dz0; \
  scalar_t w020 = dx0 * dy2; \
  scalar_t w021 = w020 * dz1; \
  scalar_t w022 = w020 * dz2; \
           w020 *= dz0; \
  scalar_t w100 = dx1 * dy0; \
  scalar_t w101 = w100 * dz1; \
  scalar_t w102 = w100 * dz2; \
           w100 *= dz0; \
  scalar_t w110 = dx1 * dy1; \
  scalar_t w111 = w110 * dz1; \
  scalar_t w112 = w110 * dz2; \
           w110 *= dz0; \
  scalar_t w120 = dx1 * dy2; \
  scalar_t w121 = w120 * dz1; \
  scalar_t w122 = w120 * dz2; \
           w120 *= dz0; \
  scalar_t w200 = dx2 * dy0; \
  scalar_t w201 = w200 * dz1; \
  scalar_t w202 = w200 * dz2; \
           w200 *= dz0; \
  scalar_t w210 = dx2 * dy1; \
  scalar_t w211 = w210 * dz1; \
  scalar_t w212 = w210 * dz2; \
           w210 *= dz0; \
  scalar_t w220 = dx2 * dy2; \
  scalar_t w221 = w220 * dz1; \
  scalar_t w222 = w220 * dz2; \
           w220 *= dz0; \
  int8_t  sx0 = bound::sign(bound0, ix1-1, src_X); \
  int8_t  sy0 = bound::sign(bound1, iy1-1, src_Y); \
  int8_t  sz0 = bound::sign(bound2, iz1-1, src_Z); \
  int8_t  sx2 = bound::sign(bound0, ix1+1, src_X); \
  int8_t  sy2 = bound::sign(bound1, iy1+1, src_Y); \
  int8_t  sz2 = bound::sign(bound2, iz1+1, src_Z); \
  int8_t  sx1 = bound::sign(bound0, ix1,   src_X); \
  int8_t  sy1 = bound::sign(bound1, iy1,   src_Y); \
  int8_t  sz1 = bound::sign(bound2, iz1,   src_Z); \
  int8_t s000 = sx0 * sy0; \
  int8_t s001 = s000 * sz1; \
  int8_t s002 = s000 * sz2; \
         s000 *= sz0; \
  int8_t s010 = sx0 * sy1; \
  int8_t s011 = s010 * sz1; \
  int8_t s012 = s010 * sz2; \
         s010 *= sz0; \
  int8_t s020 = sx0 * sy2; \
  int8_t s021 = s020 * sz1; \
  int8_t s022 = s020 * sz2; \
         s020 *= sz0; \
  int8_t s100 = sx1 * sy0; \
  int8_t s101 = s100 * sz1; \
  int8_t s102 = s100 * sz2; \
         s100 *= sz0; \
  int8_t s110 = sx1 * sy1; \
  int8_t s111 = s110 * sz1; \
  int8_t s112 = s110 * sz2; \
         s110 *= sz0; \
  int8_t s120 = sx1 * sy2; \
  int8_t s121 = s120 * sz1; \
  int8_t s122 = s120 * sz2; \
         s120 *= sz0; \
  int8_t s200 = sx2 * sy0; \
  int8_t s201 = s200 * sz1; \
  int8_t s202 = s200 * sz2; \
         s200 *= sz0; \
  int8_t s210 = sx2 * sy1; \
  int8_t s211 = s210 * sz1; \
  int8_t s212 = s210 * sz2; \
         s210 *= sz0; \
  int8_t s220 = sx2 * sy2; \
  int8_t s221 = s220 * sz1; \
  int8_t s222 = s220 * sz2; \
         s220 *= sz0; \
  offset_t ix0, iy0, iz0, ix2, iy2, iz2; \
  ix0 = bound::index(bound0, ix1-1, src_X) * src_sX; \
  iy0 = bound::index(bound1, iy1-1, src_Y) * src_sY; \
  iz0 = bound::index(bound2, iz1-1, src_Z) * src_sZ; \
  ix2 = bound::index(bound0, ix1+1, src_X) * src_sX; \
  iy2 = bound::index(bound1, iy1+1, src_Y) * src_sY; \
  iz2 = bound::index(bound2, iz1+1, src_Z) * src_sZ; \
  ix1 = bound::index(bound0, ix1,   src_X) * src_sX; \
  iy1 = bound::index(bound1, iy1,   src_Y) * src_sY; \
  iz1 = bound::index(bound2, iz1,   src_Z) * src_sZ; \
  offset_t o000 = ix0 + iy0 + iz0; \
  offset_t o001 = ix0 + iy0 + iz1; \
  offset_t o002 = ix0 + iy0 + iz2; \
  offset_t o010 = ix0 + iy1 + iz0; \
  offset_t o011 = ix0 + iy1 + iz1; \
  offset_t o012 = ix0 + iy1 + iz2; \
  offset_t o020 = ix0 + iy2 + iz0; \
  offset_t o021 = ix0 + iy2 + iz1; \
  offset_t o022 = ix0 + iy2 + iz2; \
  offset_t o100 = ix1 + iy0 + iz0; \
  offset_t o101 = ix1 + iy0 + iz1; \
  offset_t o102 = ix1 + iy0 + iz2; \
  offset_t o110 = ix1 + iy1 + iz0; \
  offset_t o111 = ix1 + iy1 + iz1; \
  offset_t o112 = ix1 + iy1 + iz2; \
  offset_t o120 = ix1 + iy2 + iz0; \
  offset_t o121 = ix1 + iy2 + iz1; \
  offset_t o122 = ix1 + iy2 + iz2; \
  offset_t o200 = ix2 + iy0 + iz0; \
  offset_t o201 = ix2 + iy0 + iz1; \
  offset_t o202 = ix2 + iy0 + iz2; \
  offset_t o210 = ix2 + iy1 + iz0; \
  offset_t o211 = ix2 + iy1 + iz1; \
  offset_t o212 = ix2 + iy1 + iz2; \
  offset_t o220 = ix2 + iy2 + iz0; \
  offset_t o221 = ix2 + iy2 + iz1; \
  offset_t o222 = ix2 + iy2 + iz2; \
  scalar_t *tgt_ptr_NCXYZ = tgt_ptr                   \
                          + n * tgt_sN + w * tgt_sX   \
                          + h * tgt_sY + d * tgt_sZ;  \
  scalar_t *src_ptr_NC = src_ptr + n * src_sN;

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize3d_quadratic(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX

  // NB: this function is fundamental to the FMG solver with bending energy
  // The bravkets in the sum matter a lot!
  // If they're removed, innacuracies creep in and the result of FMG is crap.
  // I've gathered terms by distance to the center voxel.

  for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC, 
                                   src_ptr_NC    += src_sC) {
    *tgt_ptr_NCXYZ = bound::get(src_ptr_NC, o111, s111) * w111
                   +(bound::get(src_ptr_NC, o011, s011) * w011
                   + bound::get(src_ptr_NC, o101, s101) * w101
                   + bound::get(src_ptr_NC, o110, s110) * w110
                   + bound::get(src_ptr_NC, o112, s112) * w112
                   + bound::get(src_ptr_NC, o121, s121) * w121
                   + bound::get(src_ptr_NC, o211, s211) * w211)
                   +(bound::get(src_ptr_NC, o001, s001) * w001
                   + bound::get(src_ptr_NC, o010, s010) * w010
                   + bound::get(src_ptr_NC, o100, s100) * w100
                   + bound::get(src_ptr_NC, o012, s012) * w012
                   + bound::get(src_ptr_NC, o021, s021) * w021
                   + bound::get(src_ptr_NC, o201, s201) * w201
                   + bound::get(src_ptr_NC, o210, s210) * w210
                   + bound::get(src_ptr_NC, o212, s212) * w212
                   + bound::get(src_ptr_NC, o221, s221) * w221
                   + bound::get(src_ptr_NC, o120, s120) * w120
                   + bound::get(src_ptr_NC, o122, s122) * w122
                   + bound::get(src_ptr_NC, o102, s102) * w102)
                   +(bound::get(src_ptr_NC, o000, s000) * w000
                   + bound::get(src_ptr_NC, o002, s002) * w002
                   + bound::get(src_ptr_NC, o020, s020) * w020
                   + bound::get(src_ptr_NC, o200, s200) * w200
                   + bound::get(src_ptr_NC, o022, s022) * w022
                   + bound::get(src_ptr_NC, o202, s202) * w202
                   + bound::get(src_ptr_NC, o220, s220) * w220
                   + bound::get(src_ptr_NC, o222, s222) * w222);
  }
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict3d_quadratic(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX

  for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC,
                                   src_ptr_NC    += src_sC) {
    scalar_t tgt = *tgt_ptr_NCXYZ;
    bound::add(src_ptr_NC, o000, w000 * tgt, s000);
    bound::add(src_ptr_NC, o001, w001 * tgt, s001);
    bound::add(src_ptr_NC, o002, w002 * tgt, s002);
    bound::add(src_ptr_NC, o010, w010 * tgt, s010);
    bound::add(src_ptr_NC, o011, w011 * tgt, s011);
    bound::add(src_ptr_NC, o012, w012 * tgt, s012);
    bound::add(src_ptr_NC, o020, w020 * tgt, s020);
    bound::add(src_ptr_NC, o021, w021 * tgt, s021);
    bound::add(src_ptr_NC, o022, w022 * tgt, s022);
    bound::add(src_ptr_NC, o100, w100 * tgt, s100);
    bound::add(src_ptr_NC, o101, w101 * tgt, s101);
    bound::add(src_ptr_NC, o102, w102 * tgt, s102);
    bound::add(src_ptr_NC, o110, w110 * tgt, s110);
    bound::add(src_ptr_NC, o111, w111 * tgt, s111);
    bound::add(src_ptr_NC, o112, w112 * tgt, s112);
    bound::add(src_ptr_NC, o120, w120 * tgt, s120);
    bound::add(src_ptr_NC, o121, w121 * tgt, s121);
    bound::add(src_ptr_NC, o122, w122 * tgt, s122);
    bound::add(src_ptr_NC, o200, w200 * tgt, s200);
    bound::add(src_ptr_NC, o201, w201 * tgt, s201);
    bound::add(src_ptr_NC, o202, w202 * tgt, s202);
    bound::add(src_ptr_NC, o210, w210 * tgt, s210);
    bound::add(src_ptr_NC, o211, w211 * tgt, s211);
    bound::add(src_ptr_NC, o212, w212 * tgt, s212);
    bound::add(src_ptr_NC, o220, w220 * tgt, s220);
    bound::add(src_ptr_NC, o221, w221 * tgt, s221);
    bound::add(src_ptr_NC, o222, w222 * tgt, s222);
  }
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize2d_quadratic(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict2d_quadratic(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize1d_quadratic(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict1d_quadratic(offset_t w, offset_t h, offset_t d, offset_t n) const {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     LINEAR INTERPOLATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#undef  GET_INDEX
#define GET_INDEX \
  scalar_t x = scale0 * w + shift0; \
  scalar_t y = scale1 * h + shift1; \
  scalar_t z = scale2 * d + shift2; \
  offset_t ix0 = static_cast<offset_t>(std::floor(x)); \
  offset_t iy0 = static_cast<offset_t>(std::floor(y)); \
  offset_t iz0 = static_cast<offset_t>(std::floor(z)); \
  scalar_t dx1 = x - ix0; \
  scalar_t dy1 = y - iy0; \
  scalar_t dz1 = z - iz0; \
  scalar_t dx0 = 1. - dx1; \
  scalar_t dy0 = 1. - dy1; \
  scalar_t dz0 = 1. - dz1; \
  scalar_t w000 = dx0 * dy0; \
  scalar_t w001 = w000 * dz1; \
           w000 *= dz0; \
  scalar_t w010 = dx0 * dy1; \
  scalar_t w011 = w010 * dz1; \
           w010 *= dz0; \
  scalar_t w100 = dx1 * dy0; \
  scalar_t w101 = w100 * dz1; \
           w100 *= dz0; \
  scalar_t w110 = dx1 * dy1; \
  scalar_t w111 = w110 * dz1; \
           w110 *= dz0; \
  int8_t  sx1 = bound::sign(bound0, ix0+1, src_X); \
  int8_t  sy1 = bound::sign(bound1, iy0+1, src_Y); \
  int8_t  sz1 = bound::sign(bound2, iz0+1, src_Z); \
  int8_t  sx0 = bound::sign(bound0, ix0,   src_X); \
  int8_t  sy0 = bound::sign(bound1, iy0,   src_Y); \
  int8_t  sz0 = bound::sign(bound2, iz0,   src_Z); \
  int8_t  s000 = sx0 * sy0; \
  int8_t  s001 = s000 * sz1; \
          s000 *= sz0; \
  int8_t  s010 = sx0 * sy1; \
  int8_t  s011 = s010 * sz1; \
          s010 *= sz0; \
  int8_t  s100 = sx1 * sy0; \
  int8_t  s101 = s100 * sz1; \
          s100 *= sz0; \
  int8_t  s110 = sx1 * sy1; \
  int8_t  s111 = s110 * sz1; \
          s110 *= sz0; \
  offset_t ix1, iy1, iz1; \
  ix1 = bound::index(bound0, ix0+1, src_X) * src_sX; \
  iy1 = bound::index(bound1, iy0+1, src_Y) * src_sY; \
  iz1 = bound::index(bound2, iz0+1, src_Z) * src_sZ; \
  ix0 = bound::index(bound0, ix0,   src_X) * src_sX; \
  iy0 = bound::index(bound1, iy0,   src_Y) * src_sY; \
  iz0 = bound::index(bound2, iz0,   src_Z) * src_sZ; \
  offset_t o000, o100, o010, o001, o110, o011, o101, o111; \
  o000 = ix0 + iy0; \
  o001 = o000 + iz1; \
  o000 += iz0; \
  o010 = ix0 + iy1; \
  o011 = o010 + iz1; \
  o010 += iz0; \
  o100 = ix1 + iy0; \
  o101 = o100 + iz1; \
  o100 += iz0; \
  o110 = ix1 + iy1; \
  o111 = o110 + iz1; \
  o110 += iz0; \
  scalar_t *tgt_ptr_NCXYZ = tgt_ptr                   \
                          + n * tgt_sN + w * tgt_sX   \
                          + h * tgt_sY + d * tgt_sZ;  \
  scalar_t *src_ptr_NC = src_ptr + n * src_sN;

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize3d_linear(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX

  for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC, 
                                   src_ptr_NC    += src_sC) {
    *tgt_ptr_NCXYZ = bound::get(src_ptr_NC, o000, s000) * w000
                   + bound::get(src_ptr_NC, o100, s100) * w100
                   + bound::get(src_ptr_NC, o010, s010) * w010
                   + bound::get(src_ptr_NC, o110, s110) * w110
                   + bound::get(src_ptr_NC, o001, s001) * w001
                   + bound::get(src_ptr_NC, o101, s101) * w101
                   + bound::get(src_ptr_NC, o011, s011) * w011
                   + bound::get(src_ptr_NC, o111, s111) * w111;
  }
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict3d_linear(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX

  for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC,
                                   src_ptr_NC    += src_sC) {
    scalar_t tgt = *tgt_ptr_NCXYZ;
    bound::add(src_ptr_NC, o000, w000 * tgt, s000);
    bound::add(src_ptr_NC, o100, w100 * tgt, s100);
    bound::add(src_ptr_NC, o010, w010 * tgt, s010);
    bound::add(src_ptr_NC, o110, w110 * tgt, s110);
    bound::add(src_ptr_NC, o001, w001 * tgt, s001);
    bound::add(src_ptr_NC, o101, w101 * tgt, s101);
    bound::add(src_ptr_NC, o011, w011 * tgt, s011);
    bound::add(src_ptr_NC, o111, w111 * tgt, s111);
  }
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize2d_linear(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict2d_linear(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize1d_linear(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict1d_linear(offset_t w, offset_t h, offset_t d, offset_t n) const {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  NEAREST NEIGHBOR INTERPOLATION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#undef  GET_INDEX
#define GET_INDEX \
  offset_t ix = static_cast<offset_t>(std::round(scale0 * w + shift0)); \
  offset_t iy = static_cast<offset_t>(std::round(scale1 * h + shift1)); \
  offset_t iz = static_cast<offset_t>(std::round(scale2 * d + shift1)); \
  int8_t   sx = bound::sign(bound0, ix, src_X);                         \
  int8_t   sy = bound::sign(bound1, iy, src_Y);                         \
  int8_t   sz = bound::sign(bound2, iz, src_Z);                         \
           ix = bound::index(bound0, ix,src_X);                         \
           iy = bound::index(bound1, iy,src_Y);                         \
           iz = bound::index(bound2, iz,src_Z);                         \
  int8_t    s = sz * sy * sx;                                           \
  offset_t  o = iz*src_sZ + iy*src_sY + ix*src_sX;                      \
  scalar_t *tgt_ptr_NCXYZ = tgt_ptr + n * tgt_sN + w * tgt_sX           \
                                    + h * tgt_sY + d * tgt_sZ;          \
  scalar_t *src_ptr_NC = src_ptr + n * src_sN;


template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize3d_nearest(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX
  for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC, 
                                   src_ptr_NC    += src_sC)
    bound::add(src_ptr_NC, o, *tgt_ptr_NCXYZ, s);
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict3d_nearest(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  GET_INDEX
  for (offset_t c = 0; c < C; ++c, tgt_ptr_NCXYZ += tgt_sC, 
                                   src_ptr_NC     += src_sC)
    *tgt_ptr_NCXYZ = bound::get(src_ptr_NC, o, s);
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize2d_nearest(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict2d_nearest(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::resize1d_nearest(offset_t w, offset_t h, offset_t d, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void ResizeImpl<scalar_t,offset_t>::restrict1d_nearest(offset_t w, offset_t h, offset_t d, offset_t n) const {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void resize_kernel(ResizeImpl<scalar_t,offset_t> * f) {
  f->loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST NI_INLINE void
check_same_nonspatial(Tensor source, Tensor target)
{
  bool same_nonspatial = (source.dim()   == target.dim())    &&
                         (source.size(0) == target.size(0))  &&
                         (source.size(1) == target.size(1));
  if (!same_nonspatial) {
    std::string const msg = "Source and target should have the same "
                            "batch and channel shapes but found dims "
                          + std::to_string(source.dim()) + " vs " 
                          + std::to_string(target.dim()) + " and shapes ["
                          + std::to_string(source.size(0)) + " " 
                          + std::to_string(source.size(1)) + "] vs ["
                          + std::to_string(target.size(0)) + " " 
                          + std::to_string(target.size(1)) + "].";
    throw std::invalid_argument(msg);
  }
}

NI_HOST NI_INLINE std::tuple<Tensor, Tensor>
prepare_tensors(Tensor source, Tensor target, ArrayRef<double> factor, bool do_adjoint)
{
  bool source_defined = (source.defined() && source.numel() > 0);
  bool target_defined = (target.defined() && target.numel() > 0);

  if (source_defined && target_defined) {
    check_same_nonspatial(source, target);
    if (do_adjoint)
      source.zero_();
    return std::tuple<Tensor, Tensor>(source, target);
  }

  double fx = factor.size() > 0 ? factor[0] : 1.;
  double fy = factor.size() > 1 ? factor[1] : fx;
  double fz = factor.size() > 2 ? factor[2] : fy;
  if (do_adjoint) {
    fx = 1./fx;
    fy = 1./fy;
    fz = 1./fz;
  }

  if (source_defined) { // PULL

    int64_t dim = source.dim() - 2;
    int64_t N = source.size(0);
    int64_t C = source.size(1);
    int64_t X = source.size(2);
    int64_t Y = dim > 1 ? source.size(3) : 1L;
    int64_t Z = dim > 2 ? source.size(4) : 1L;

    int64_t Xt =           static_cast<int64_t>(std::floor(static_cast<double>(X) * fx));
    int64_t Yt = dim > 1 ? static_cast<int64_t>(std::floor(static_cast<double>(Y) * fy)) : 1L;
    int64_t Zt = dim > 2 ? static_cast<int64_t>(std::floor(static_cast<double>(Z) * fz)) : 1L;

    target = at::zeros({N, C, Xt, Yt, Zt}, source.options());

  } else { // PUSH

    int64_t dim = target.dim() - 2;
    int64_t N = target.size(0);
    int64_t C = target.size(1);
    int64_t X = target.size(2);
    int64_t Y = dim > 1 ? target.size(3) : 1L;
    int64_t Z = dim > 2 ? target.size(4) : 1L;

    int64_t Xs =           static_cast<int64_t>(std::ceil(static_cast<double>(X) * fx));
    int64_t Ys = dim > 1 ? static_cast<int64_t>(std::ceil(static_cast<double>(Y) * fy)) : 1L;
    int64_t Zs = dim > 2 ? static_cast<int64_t>(std::ceil(static_cast<double>(Z) * fz)) : 1L;

    source = at::zeros({N, C, Xs, Ys, Zs}, target.options());

  }

  return std::tuple<Tensor, Tensor>(source, target);
}

NI_HOST NI_INLINE  std::pair<double, double>
prepare_affine(double factor, double Ds, double Dt, GridAlignType mode)
{
  double shift = 0., scale = 1./factor;
  switch (mode) {
    case GridAlignType::Edge:
      shift = (Ds / Dt - 1);
      scale = Ds / Dt;
      break;
    case GridAlignType::Center:
      shift = 0.;
      scale = (Ds - 1) / (Dt - 1);
      break;
    case GridAlignType::Last:
      shift = factor * (Ds - 1) / (Dt - 1);
      break;
    default:
      break;
  }
  return std::make_pair(shift, scale);
}

NI_HOST NI_INLINE std::pair<std::vector<double>,  std::vector<double> >
prepare_affines(Tensor source, Tensor target, ArrayRef<double> factor, GridAlignVectorRef mode, bool do_adjoint)
{

  double fx = factor.size() > 0 ? factor[0] : 1.;
  double fy = factor.size() > 1 ? factor[1] : fx;
  double fz = factor.size() > 2 ? factor[2] : fy;
  if (do_adjoint) {
    fx = 1./fx;
    fy = 1./fy;
    fz = 1./fz;
  }

  GridAlignType mx = mode.size() > 0 ? mode[0] : GridAlignType::Center;
  GridAlignType my = mode.size() > 1 ? mode[1] : mx;
  GridAlignType mz = mode.size() > 2 ? mode[2] : my;

  int64_t dim = target.dim() - 2;
  int64_t Xt  = target.size(2);
  int64_t Yt  = dim > 1 ? target.size(3) : 1L;
  int64_t Zt  = dim > 2 ? target.size(4) : 1L;
  int64_t Xs  = source.size(2);
  int64_t Ys  = dim > 1 ? source.size(3) : 1L;
  int64_t Zs  = dim > 2 ? source.size(4) : 1L;

  auto affinex = prepare_affine(fx, static_cast<double>(Xs), static_cast<double>(Xt), mx);
  auto affiney = prepare_affine(fy, static_cast<double>(Ys), static_cast<double>(Yt), my);
  auto affinez = prepare_affine(fz, static_cast<double>(Zs), static_cast<double>(Zt), mz);

  return make_pair(
      std::vector<double>({std::get<0>(affinex), std::get<0>(affiney), std::get<0>(affinez)}),
      std::vector<double>({std::get<1>(affinex), std::get<1>(affiney), std::get<1>(affinez)})
  );
}

} // namespace

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
NI_HOST
Tensor resize_impl(
  Tensor source, Tensor target, ArrayRef<double> factor,
  BoundVectorRef bound, InterpolationVectorRef interpolation, 
  GridAlignVectorRef mode, bool do_adjoint, bool normalize)
{
  auto tensors = prepare_tensors(source, target, factor, do_adjoint);
  source = std::get<0>(tensors);
  target = std::get<1>(tensors);
  auto affines = prepare_affines(source, target, factor, mode, do_adjoint);
  ArrayRef<double> shifts(std::get<0>(affines));
  ArrayRef<double> scales(std::get<1>(affines));

  ResizeAllocator info(source.dim()-2, bound, interpolation, shifts, scales, do_adjoint > 0);
  info.ioset(source, target);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(source.scalar_type(), "resize_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      ResizeImpl<scalar_t, int32_t> algo(info);
      auto palgo = alloc_and_copy_to_device(algo, stream);
      resize_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>(palgo);
      cudaFree(palgo);
    }
    else
    {
      ResizeImpl<scalar_t, int64_t> algo(info);
      auto palgo = alloc_and_copy_to_device(algo, stream);
      resize_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>(palgo);
      cudaFree(palgo);
    }
  });

  Tensor out = do_adjoint ? source : target;
  if (normalize)
    switch (out.dim() - 2) {
      case 1:  out *= scales[0];
      case 2:  out *= scales[0] * scales[1];
      default: out *= scales[0] * scales[1] * scales[2];
    }
  return out;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST
Tensor resize_impl(
  Tensor source, Tensor target, ArrayRef<double> factor,
  BoundVectorRef bound, InterpolationVectorRef interpolation,  
  GridAlignVectorRef mode, bool do_adjoint, bool normalize)
{
  auto tensors = prepare_tensors(source, target, factor, do_adjoint);
  source = std::get<0>(tensors);
  target = std::get<1>(tensors);
  auto affines = prepare_affines(source, target, factor, mode, do_adjoint);
  ArrayRef<double> shifts(std::get<0>(affines));
  ArrayRef<double> scales(std::get<1>(affines));

  ResizeAllocator info(source.dim()-2, bound, interpolation, shifts, scales, do_adjoint > 0);
  info.ioset(source, target);

  AT_DISPATCH_FLOATING_TYPES(source.scalar_type(), "resize_impl", [&] {
    ResizeImpl<scalar_t, int64_t> algo(info);
    algo.loop();
  });

  Tensor out = do_adjoint ? source : target;
  if (normalize)
    switch (out.dim() - 2) {
      case 1:  out *= scales[0];
      case 2:  out *= scales[0] * scales[1];
      default: out *= scales[0] * scales[1] * scales[2];
    }
  return out;
}

#endif // __CUDACC__


} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST
Tensor resize_impl(
  Tensor source, Tensor target, ArrayRef<double> factor,
  BoundVectorRef bound, InterpolationVectorRef interpolation, 
  GridAlignVectorRef mode, bool do_adjoint, bool normalize)
{
  throw std::logic_error("Function not implemented for this device.");
}

} // namespace notimplemented

} // namespace ni

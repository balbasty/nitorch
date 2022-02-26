#include "common.h"                // write C++/CUDA compatible code
#include "../defines.h"            // useful macros
#include "bounds_common.h"         // boundary conditions + enum
#include "allocator.h"             // base class handling offset sizes
#include "hessian.h"               // utility for handling Hessian matrices
#include "utils.h"                 // utility for dispatching
#include <ATen/ATen.h>             // tensors
#include <cmath>                   // fma (fused multiply add)

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
using c10::IntArrayRef;
using c10::ArrayRef;

// Required for stability. Value is currently about 1+8*eps
#define OnePlusTiny 1.000001

#define VEC_UNFOLD(ONAME, INAME, DEFAULT)             \
  ONAME##0(INAME.size() > 0 ? INAME[0] : DEFAULT),  \
  ONAME##1(INAME.size() > 1 ? INAME[1] :            \
           INAME.size() > 0 ? INAME[0] : DEFAULT),  \
  ONAME##2(INAME.size() > 2 ? INAME[2] :            \
           INAME.size() > 1 ? INAME[1] :            \
           INAME.size() > 0 ? INAME[0] : DEFAULT)

namespace ni {
NI_NAMESPACE_DEVICE { // cpu / cuda / ...

namespace { // anonymous namespace > everything inside has internal linkage


/* ========================================================================== */
/*                                                                            */
/*                                ALLOCATOR                                   */
/*                                                                            */
/* ========================================================================== */
class PrecondAllocator: public Allocator {
public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST
  PrecondAllocator(int dim, ArrayRef<double> absolute, 
                   ArrayRef<double> membrane, ArrayRef<double> bending,
                   ArrayRef<double> voxel_size, BoundVectorRef bound):
    dim(dim),
    VEC_UNFOLD(bound, bound,      BoundType::Replicate),
    VEC_UNFOLD(vx,    voxel_size, 1.),
    absolute(absolute),
    membrane(membrane),
    bending(bending)
  {
    vx0 = 1. / (vx0*vx0);
    vx1 = 1. / (vx1*vx1);
    vx2 = 1. / (vx2*vx2);
    if (dim < 3) vx2 = 0.;
    if (dim < 2) vx1 = 0.;
  }

  /* ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST void ioset
  (const Tensor& hess, const Tensor& grad, const Tensor& solution, const Tensor& weight)
  {
    init_all();
    init_gradient(grad);
    init_hessian(hess);
    init_solution(solution);
    init_weight(weight);
  }

  // We just check that all tensors that we own are compatible with 32b math
  bool canUse32BitIndexMath(int64_t max_elem=max_int32) const
  {
    return grd_32b_ok && wgt_32b_ok && hes_32b_ok && sol_32b_ok;
  }

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  NI_HOST void init_all();
  NI_HOST void init_gradient(const Tensor&);
  NI_HOST void init_hessian(const Tensor&);
  NI_HOST void init_solution(const Tensor&);
  NI_HOST void init_weight(const Tensor&);

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  int               dim;            // dimensionality (1 or 2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  double            vx0;            // voxel size          // x|W
  double            vx1;            // voxel size          // y|H
  double            vx2;            // voxel size          // z|D
  ArrayRef<double>  absolute;       // penalty on absolute values
  ArrayRef<double>  membrane;       // penalty on first derivatives
  ArrayRef<double>  bending;        // penalty on second derivatives

  /* ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#define DEFINE_ALLOC_INFO_5D(NAME)  \
  int64_t NAME##_sN;                \
  int64_t NAME##_sC;                \
  int64_t NAME##_sX;                \
  int64_t NAME##_sY;                \
  int64_t NAME##_sZ;                \
  bool NAME##_32b_ok;               \
  void * NAME##_ptr;

  int64_t N;
  int64_t C;
  int64_t CC;
  int64_t X;
  int64_t Y;
  int64_t Z;
  DEFINE_ALLOC_INFO_5D(grd)
  DEFINE_ALLOC_INFO_5D(hes)
  DEFINE_ALLOC_INFO_5D(sol)
  DEFINE_ALLOC_INFO_5D(wgt)

  // Allow PrecondImpl's constructor to access PrecondAllocator's
  // private members.
  template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t>
  friend class PrecondImpl;
};


NI_HOST
void PrecondAllocator::init_all()
{
  N = C = CC = X = Y = Z = 1L;
  grd_sN  = grd_sC   = grd_sX   = grd_sY  = grd_sZ   = 0L;
  hes_sN  = hes_sC   = hes_sX   = hes_sY  = hes_sZ   = 0L;
  sol_sN  = sol_sC   = sol_sX   = sol_sY  = sol_sZ   = 0L;
  wgt_sN  = wgt_sC   = wgt_sX   = wgt_sY  = wgt_sZ   = 0L;
  grd_ptr = hes_ptr  = sol_ptr  = wgt_ptr = static_cast<void*>(0);
  grd_32b_ok = hes_32b_ok = sol_32b_ok = wgt_32b_ok = true;
}

NI_HOST
void PrecondAllocator::init_gradient(const Tensor& input)
{
  N       = input.size(0);
  C       = input.size(1);
  X       = input.size(2);
  Y       = dim < 2 ? 1L : input.size(3);
  Z       = dim < 3 ? 1L : input.size(4);
  grd_sN  = input.stride(0);
  grd_sC  = input.stride(1);
  grd_sX  = input.stride(2);
  grd_sY  = dim < 2 ? 0L : input.stride(3);
  grd_sZ  = dim < 3 ? 0L : input.stride(4);
  grd_ptr = input.data_ptr();
  grd_32b_ok = tensorCanUse32BitIndexMath(input);
}

NI_HOST
void PrecondAllocator::init_hessian(const Tensor& input)
{
  if (!input.defined() || input.numel() == 0)
    return;
  CC      = input.size(1);
  hes_sN  = input.stride(0);
  hes_sC  = input.stride(1);
  hes_sX  = input.stride(2);
  hes_sY  = dim < 2 ? 0L : input.stride(3);
  hes_sZ  = dim < 3 ? 0L : input.stride(4);
  hes_ptr = input.data_ptr();
  hes_32b_ok = tensorCanUse32BitIndexMath(input);
}

NI_HOST
void PrecondAllocator::init_solution(const Tensor& input)
{
  sol_sN  = input.stride(0);
  sol_sC  = input.stride(1);
  sol_sX  = input.stride(2);
  sol_sY  = dim < 2 ? 0L : input.stride(3);
  sol_sZ  = dim < 3 ? 0L : input.stride(4);
  sol_ptr = input.data_ptr();
  sol_32b_ok = tensorCanUse32BitIndexMath(input);
}

NI_HOST
void PrecondAllocator::init_weight(const Tensor& weight)
{
  if (!weight.defined() || weight.numel() == 0)
    return;
  wgt_sN  = weight.stride(0);
  wgt_sC  = weight.stride(1);
  wgt_sX  = weight.stride(2);
  wgt_sY  = dim < 2 ? 0L : weight.stride(3);
  wgt_sZ  = dim < 3 ? 0L : weight.stride(4);
  wgt_ptr = weight.data_ptr();
  wgt_32b_ok = tensorCanUse32BitIndexMath(weight);
}

/* ========================================================================== */
/*                                                                            */
/*                                ALGORITHM                                   */
/*                                                                            */
/* ========================================================================== */

template <typename reduce_t, typename offset_t>
NI_HOST NI_INLINE bool any(const reduce_t * v, offset_t C) {
  for (offset_t c = 0; c < C; ++c, ++v) {
    if (*v) return true;
  }
  return false;
}

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t>
class PrecondImpl {

  using Self       = PrecondImpl;
  using PrecondFn  = void (Self::*)(offset_t, offset_t, offset_t, offset_t) const;
  static const int32_t MaxC  = hessian_t::max_length;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  PrecondImpl(const PrecondAllocator & info):
    dim(info.dim),
    bound0(info.bound0), bound1(info.bound1), bound2(info.bound2),
    N(static_cast<offset_t>(info.N)),
    C(static_cast<offset_t>(info.C)),
    CC(static_cast<offset_t>(info.CC)),
    X(static_cast<offset_t>(info.X)),
    Y(static_cast<offset_t>(info.Y)),
    Z(static_cast<offset_t>(info.Z)),

#define INIT_ALLOC_INFO_5D(NAME) \
    NAME##_sN(static_cast<offset_t>(info.NAME##_sN)), \
    NAME##_sC(static_cast<offset_t>(info.NAME##_sC)), \
    NAME##_sX(static_cast<offset_t>(info.NAME##_sX)), \
    NAME##_sY(static_cast<offset_t>(info.NAME##_sY)), \
    NAME##_sZ(static_cast<offset_t>(info.NAME##_sZ)), \
    NAME##_ptr(static_cast<scalar_t*>(info.NAME##_ptr))

    INIT_ALLOC_INFO_5D(grd),
    INIT_ALLOC_INFO_5D(hes),
    INIT_ALLOC_INFO_5D(sol),
    INIT_ALLOC_INFO_5D(wgt)
  {
    if (C > MaxC) throw std::logic_error("C > MaxC. This should not happen.");

    set_factors(info.absolute, info.membrane, info.bending);
    set_kernel(info.vx0, info.vx1, info.vx2);
  #ifndef __CUDACC__
    set_precond();
  #endif
  }

  NI_HOST void set_factors(ArrayRef<double> a, ArrayRef<double> m, ArrayRef<double> b)
  {
    offset_t na = static_cast<int32_t>(a.size());
    offset_t nm = static_cast<int32_t>(m.size());
    offset_t nb = static_cast<int32_t>(b.size());

    absolute[0] = static_cast<reduce_t>(na == 0 ? 0.   : a[0]);
    membrane[0] = static_cast<reduce_t>(nm == 0 ? 0.   : m[0]);
    bending[0]  = static_cast<reduce_t>(nb == 0 ? 0.   : b[0]);
    for (offset_t c = 1; c < C; ++c)
    {
      absolute[c] = static_cast<reduce_t>(na > c ? a[c] : absolute[c-1]);
      membrane[c] = static_cast<reduce_t>(nm > c ? m[c] : membrane[c-1]);
      bending[c]  = static_cast<reduce_t>(nb > c ? b[c] : bending[c-1]);
    }

    // `mode` encodes all options in a single byte (actually 5 bits)
    // [1*RLS 2*ENERGY 2*DIM]
    // We can use this byte to switch between implementations efficiently.
    mode = dim 
         + (any(bending, C) ? 12 : any(membrane, C) ? 8 : any(absolute, C) ? 4 : 0)
         + (wgt_ptr ? 16 : 0);
  } 

  NI_HOST void set_kernel(double vx0, double vx1, double vx2) 
  {
    for (offset_t c = 0; c < C; ++c)
      w000[c] = static_cast<reduce_t>((
                    bending[c]  * (6.0*(vx0*vx0+vx1*vx1+vx2*vx2) + 
                                   8.0*(vx0*vx1+vx0*vx2+vx1*vx2))
                  + membrane[c] * (2.0*(vx0+vx1+vx2))
                  + absolute[c]));

    m100 = static_cast<reduce_t>(-vx0);
    m010 = static_cast<reduce_t>(-vx1);
    m001 = static_cast<reduce_t>(-vx2);
    b100 = static_cast<reduce_t>(-4.0*vx0*(vx0+vx1+vx2));
    b010 = static_cast<reduce_t>(-4.0*vx1*(vx0+vx1+vx2));
    b001 = static_cast<reduce_t>(-4.0*vx2*(vx0+vx1+vx2));
    b200 = static_cast<reduce_t>(vx0*vx0);
    b020 = static_cast<reduce_t>(vx1*vx1);
    b002 = static_cast<reduce_t>(vx2*vx2);
    b110 = static_cast<reduce_t>(2.0*vx0*vx1);
    b101 = static_cast<reduce_t>(2.0*vx0*vx2);
    b011 = static_cast<reduce_t>(2.0*vx1*vx2);
  }

#ifndef __CUDACC__
  NI_HOST void set_precond() 
  {
#   define ABSOLUTE 4
#   define MEMBRANE 8
#   define BENDING  12
#   define RLS      16
    switch (mode) {
      case 1 + MEMBRANE + RLS:
        precond_ = &Self::precond1d_rls_membrane; break;
      case 2 + MEMBRANE + RLS:
        precond_ = &Self::precond2d_rls_membrane; break;
      case 3 + MEMBRANE + RLS:
        precond_ = &Self::precond3d_rls_membrane; break;
      case 1 + ABSOLUTE + RLS:
        precond_ = &Self::precond1d_rls_absolute; break;
      case 2 + ABSOLUTE + RLS:
        precond_ = &Self::precond2d_rls_absolute; break;
      case 3 + ABSOLUTE + RLS:
        precond_ = &Self::precond3d_rls_absolute; break;
      default:
        switch (dim) {
          case 1:
            precond_ = &Self::precond1d; break;
          case 2: 
            precond_ = &Self::precond2d; break;
          default:
            precond_ = &Self::precond3d; break;
        } break;
    }
  }
#endif

  /* ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

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
    return N * X * Y * Z;
  }
 

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  NI_DEVICE void precond(
    offset_t x, offset_t y, offset_t z, offset_t n) const;

#define DEFINE_PRECOND(SUFFIX) \
  NI_DEVICE void precond##SUFFIX( \
    offset_t x, offset_t y, offset_t z, offset_t n) const;
#define DEFINE_PRECOND_DIM(DIM)        \
  DEFINE_PRECOND(DIM##d)               \
  DEFINE_PRECOND(DIM##d_rls_absolute)  \
  DEFINE_PRECOND(DIM##d_rls_membrane)

  DEFINE_PRECOND_DIM(1)
  DEFINE_PRECOND_DIM(2)
  DEFINE_PRECOND_DIM(3)

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  offset_t          dim;
  uint8_t           mode;
  BoundType         bound0;            // boundary condition  // x|W
  BoundType         bound1;            // boundary condition  // y|H
  BoundType         bound2;            // boundary condition  // z|D
  reduce_t          absolute[MaxC];    // penalty on absolute values
  reduce_t          membrane[MaxC];    // penalty on first derivatives
  reduce_t          bending[MaxC];     // penalty on second derivatives

#ifndef __CUDACC__
  PrecondFn         precond_;          // Pointer to Precond function
#endif

  reduce_t w000[MaxC];
  reduce_t m100;
  reduce_t m010;
  reduce_t m001;
  reduce_t b100;
  reduce_t b010;
  reduce_t b001;
  reduce_t b200;
  reduce_t b020;
  reduce_t b002;
  reduce_t b110;
  reduce_t b101;
  reduce_t b011;

  /* ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#define DECLARE_STRIDE_INFO_5D(NAME)   \
  offset_t NAME##_sN;               \
  offset_t NAME##_sC;               \
  offset_t NAME##_sX;               \
  offset_t NAME##_sY;               \
  offset_t NAME##_sZ;               \
  scalar_t * NAME##_ptr;

  offset_t N;
  offset_t C;
  offset_t CC;
  offset_t X;
  offset_t Y;
  offset_t Z;
  DECLARE_STRIDE_INFO_5D(grd)
  DECLARE_STRIDE_INFO_5D(hes)
  DECLARE_STRIDE_INFO_5D(sol)
  DECLARE_STRIDE_INFO_5D(wgt)
};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond(
    offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  #ifdef __CUDACC__
#   define ABSOLUTE 4
#   define MEMBRANE 8
#   define BENDING  12
#   define RLS      16
  switch (mode) {
    case 1 + MEMBRANE + RLS:
      return precond1d_rls_membrane(x, y, z, n);
    case 2 + MEMBRANE + RLS:
      return precond2d_rls_membrane(x, y, z, n);
    case 3 + MEMBRANE + RLS:
      return precond3d_rls_membrane(x, y, z, n);
    case 1 + ABSOLUTE + RLS:
      return precond1d_rls_absolute(x, y, z, n);
    case 2 + ABSOLUTE + RLS:
      return precond2d_rls_absolute(x, y, z, n);
    case 3 + ABSOLUTE + RLS:
      return precond3d_rls_absolute(x, y, z, n);
    default:
      switch (dim) {
        case 1:
          return precond1d(x, y, z, n);
        case 2: 
          return precond2d(x, y, z, n);
        default:
          return precond3d(x, y, z, n);
      }
  }
#else
  CALL_MEMBER_FN(*this, precond_)(x, y, z, n);
#endif 
}

#ifdef __CUDACC__

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  offset_t index = static_cast<offset_t>(blockIdx * blockDim + threadIdx);
  offset_t YZ   = Y * Z;
  offset_t XYZ  = X * YZ;
  offset_t NXYZ = N * XYZ;
  offset_t n, x, y, z;
  for (offset_t i=index; index < NXYZ; index += blockDim*gridDim, i=index)
  {
      // Convert index: linear to sub
      n  = (i/XYZ);
      x  = (i/YZ) % X;
      y  = (i/Z)  % Y;
      z  =  i     % Z;
      precond(x, y, z, n);
  }
}

#else

// This bit loops over all target voxels. We therefore need to
// convert linear indices to multivariate indices. The way I do it
// might not be optimal.
template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_HOST
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop() const
{
  // Parallelize across voxels
  offset_t YZ   = Y * Z;
  offset_t XYZ  = X * YZ;
  offset_t NXYZ = N * XYZ;
  at::parallel_for(0, NXYZ, GRAIN_SIZE, [&](offset_t start, offset_t end) {
    offset_t n, x, y, z;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/XYZ);
      x  = (i/YZ) % X;
      y  = (i/Z)  % Y;
      z  =  i     % Z;
      precond(x, y, z, n);
    }
  });
}

#endif


/* ========================================================================== */
/*                               MACRO HELPERS                                */
/* ========================================================================== */
#define GET_COORD1_(x) offset_t x##0  = x - 1, x##1  = x + 1;
#define GET_COORD2_(x) offset_t x##00 = x - 2, x##11 = x + 2;
#define GET_SIGN1_(x, X, i)  \
  int8_t   s##x##0 = bound::sign(bound##i, x##0,  X); \
  int8_t   s##x##1 = bound::sign(bound##i, x##1,  X);
#define GET_SIGN2_(x, X, i)  \
  int8_t   s##x##00 = bound::sign(bound##i, x##00,  X); \
  int8_t   s##x##11 = bound::sign(bound##i, x##11,  X);
#define GET_WARP1_(x, X, i)  \
  x##0  = (bound::index(bound##i, x##0,  X) - x) * wgt_s##X; \
  x##1  = (bound::index(bound##i, x##1,  X) - x) * wgt_s##X;
#define GET_WARP2_(x, X, i)  \
  x##00  = (bound::index(bound##i, x##00,  X) - x) * wgt_s##X; \
  x##11  = (bound::index(bound##i, x##11,  X) - x) * wgt_s##X;


/* ========================================================================== */
/*                                     3D                                     */
/* ========================================================================== */

#define GET_COORD1 \
  GET_COORD1_(x) \
  GET_COORD1_(y) \
  GET_COORD1_(z)
#define GET_COORD2 \
  GET_COORD2_(x) \
  GET_COORD2_(y) \
  GET_COORD2_(z)
#define GET_SIGN1 \
  GET_SIGN1_(x, X, 0) \
  GET_SIGN1_(y, Y, 1) \
  GET_SIGN1_(z, Z, 2)
#define GET_SIGN2 \
  GET_SIGN2_(x, X, 0) \
  GET_SIGN2_(y, Y, 1) \
  GET_SIGN2_(z, Z, 2)
#define GET_WARP1 \
  GET_WARP1_(x, X, 0) \
  GET_WARP1_(y, Y, 1) \
  GET_WARP1_(z, Z, 2)
#define GET_WARP2 \
  GET_WARP2_(x, X, 0) \
  GET_WARP2_(y, Y, 1) \
  GET_WARP2_(z, Z, 2)

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond3d(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  reduce_t val[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN);
    for (int32_t c = 0; c < C; ++c, grd += grd_sC)
      val[c] = *grd;
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  reduce_t val[MaxC], wval[MaxC];
  {
    GET_COORD1
    GET_SIGN1
    GET_WARP1
    const reduce_t *a = absolute, *m = membrane;
    const scalar_t *wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ + n*grd_sN);
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN);

    for (int32_t c = 0; c < C; ++c, grd += grd_sC, wgt += wgt_sC)
    {
      scalar_t wcenter = *wgt;
      reduce_t wm = m100 * (wcenter + bound::get(wgt, x0, sx0))
                  + m100 * (wcenter + bound::get(wgt, x1, sx1))
                  + m010 * (wcenter + bound::get(wgt, y0, sy0))
                  + m010 * (wcenter + bound::get(wgt, y1, sy1))
                  + m001 * (wcenter + bound::get(wgt, z0, sz0))
                  + m001 * (wcenter + bound::get(wgt, z1, sz1));
      val[c] = *grd;
      wval[c] = ( (*(a++)) * wcenter - 0.5 * (*(m++)) * wm );
    }
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, wval);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
   reduce_t val[MaxC], wval[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN);
    const scalar_t *wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ + n*grd_sN);

    for (int32_t c = 0; c < C; ++c, grd += grd_sC, wgt += wgt_sC) {
      scalar_t wcenter = *wgt;
      val[c]  = *grd;
      wval[c] = absolute[c] * wcenter;
    }
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, wval);
}



/* ========================================================================== */
/*                                     2D                                     */
/* ========================================================================== */

#undef  GET_COORD1
#define GET_COORD1 \
  GET_COORD1_(x) \
  GET_COORD1_(y)
#undef  GET_COORD2
#define GET_COORD2 \
  GET_COORD2_(x) \
  GET_COORD2_(y)
#undef  GET_SIGN1
#define GET_SIGN1 \
  GET_SIGN1_(x, X, 0) \
  GET_SIGN1_(y, Y, 1)
#undef  GET_SIGN2
#define GET_SIGN2 \
  GET_SIGN2_(x, X, 0) \
  GET_SIGN2_(y, Y, 1)
#undef  GET_WARP1
#define GET_WARP1 \
  GET_WARP1_(x, X, 0) \
  GET_WARP1_(y, Y, 1)
#undef  GET_WARP2
#define GET_WARP2 \
  GET_WARP2_(x, X, 0) \
  GET_WARP2_(y, Y, 1)

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond2d(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
   reduce_t val[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN);
    for (int32_t c = 0; c < C; ++c, grd += grd_sC)
      val[c] = *grd;
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond2d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
   reduce_t val[MaxC], wval[MaxC];
  {
    GET_COORD1
    GET_SIGN1
    GET_WARP1
    const reduce_t *a = absolute, *m = membrane;
    const scalar_t *wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + n*grd_sN);
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + n*grd_sN);

    for (int32_t c = 0; c < C; ++c, grd += grd_sC, wgt += wgt_sC)
    {
      scalar_t wcenter = *wgt;
      reduce_t wm = m100 * (wcenter + bound::get(wgt, x0, sx0))
                  + m100 * (wcenter + bound::get(wgt, x1, sx1))
                  + m010 * (wcenter + bound::get(wgt, y0, sy0))
                  + m010 * (wcenter + bound::get(wgt, y1, sy1));
      val[c] = *grd;
      wval[c] = ( (*(a++)) * wcenter - 0.5 * (*(m++)) * wm );
    }
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, wval);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond2d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
   reduce_t val[MaxC], wval[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + n*grd_sN);
    const scalar_t *wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + n*grd_sN);

    for (int32_t c = 0; c < C; ++c, grd += grd_sC, wgt += wgt_sC) {
      scalar_t wcenter = *wgt;
      val[c]  = *grd;
      wval[c] = absolute[c] * wcenter;
    }
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, wval);
}

/* ========================================================================== */
/*                                     1D                                     */
/* ========================================================================== */

#undef  GET_COORD1
#define GET_COORD1 GET_COORD1_(x) 
#undef  GET_COORD2
#define GET_COORD2 GET_COORD2_(x)
#undef  GET_SIGN1
#define GET_SIGN1 GET_SIGN1_(x, X, 0)
#undef  GET_SIGN2
#define GET_SIGN2 GET_SIGN2_(x, X, 0)
#undef  GET_WARP1
#define GET_WARP1 GET_WARP1_(x, X, 0)
#undef  GET_WARP2
#define GET_WARP2 GET_WARP2_(x, X, 0)

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond1d(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
   reduce_t val[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + n*grd_sN);
    for (int32_t c = 0; c < C; ++c, grd += grd_sC)
      val[c] = *grd;
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond1d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
   reduce_t val[MaxC], wval[MaxC];
  {
    GET_COORD1
    GET_SIGN1
    GET_WARP1
    const reduce_t *a = absolute, *m = membrane;
    const scalar_t *wgt = wgt_ptr + (x*wgt_sX + n*grd_sN);
    const scalar_t *grd = grd_ptr + (x*grd_sX + n*grd_sN);

    for (int32_t c = 0; c < C; ++c, grd += grd_sC, wgt += wgt_sC)
    {
      scalar_t wcenter = *wgt;
      reduce_t wm = m100 * (wcenter + bound::get(wgt, x0, sx0))
                  + m100 * (wcenter + bound::get(wgt, x1, sx1));
      val[c] = *grd;
      wval[c] = ( (*(a++)) * wcenter - 0.5 * (*(m++)) * wm );
    }
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, wval);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t>::precond1d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
   reduce_t val[MaxC], wval[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + n*grd_sN);
    const scalar_t *wgt = wgt_ptr + (x*wgt_sX + n*grd_sN);

    for (int32_t c = 0; c < C; ++c, grd += grd_sC, wgt += wgt_sC) {
      scalar_t wcenter = *wgt;
      val[c]  = *grd;
      wval[c] = absolute[c] * wcenter;
    }
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + n*sol_sN);

  hessian_t::invert(C, sol, sol_sC, hes, hes_sC, val, wval);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void precond_kernel(const PrecondImpl<scalar_t,offset_t,reduce_t,hessian_t> * f) {
  f->loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif

NI_HOST std::tuple<Tensor, Tensor, Tensor>
prepare_tensors(const Tensor & gradient,
                Tensor hessian, Tensor solution, Tensor weight)
{

  if (!(solution.defined() && solution.numel() > 0))
    solution = at::zeros_like(gradient);
  if (!solution.is_same_size(gradient))
    throw std::invalid_argument("Initial solution must have the same shape as the gradient");

  if (hessian.defined() && hessian.numel() > 0)
  {

    int64_t dim = gradient.dim() - 2;
    int64_t N   = gradient.size(0);
    int64_t CC  = hessian.size(1);
    int64_t X   = gradient.size(2);
    int64_t Y   = dim > 1 ? gradient.size(3) : 1L;
    int64_t Z   = dim > 2 ? gradient.size(4) : 1L;
    if (dim == 1)
      hessian = hessian.expand({N, CC, X});
    if (dim == 2)
      hessian = hessian.expand({N, CC, X, Y});
    else
      hessian = hessian.expand({N, CC, X, Y, Z});
  }

  if (weight.defined() && weight.numel() > 0)
    weight = weight.expand_as(gradient);

  return std::tuple<Tensor, Tensor, Tensor>(hessian, solution, weight);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          DISPATCH HELPERS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  template <int32_t LogMaxC>
  struct DoAlgo {
    static const int32_t MaxC = power_of_two<LogMaxC>::value;

    template <typename A, typename H, typename S>
    static NI_HOST void 
    f(const A & alloc, const H & hessian_type, const S & scalar_type) 
    {
      NI_DISPATCH_HESSIAN_TYPE(hessian_type, [&] {
        AT_DISPATCH_FLOATING_TYPES(scalar_type, "precond_impl", [&] {
          using utils_t = HessianUtils<hessian_t, MaxC>;
#ifdef __CUDACC__
          auto stream = at::cuda::getCurrentCUDAStream();
          if (alloc.canUse32BitIndexMath())
          {
              PrecondImpl<scalar_t, int32_t, double, utils_t> algo(alloc);
              auto palgo = alloc_and_copy_to_device(algo, stream);
              precond_kernel
                  <<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>
                  (palgo);
              cudaFree(palgo);
          }
          else
          {
            PrecondImpl<scalar_t, int64_t, double, utils_t> algo(alloc);
            auto palgo = alloc_and_copy_to_device(algo, stream);
            precond_kernel
                <<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>
                (palgo);
            cudaFree(palgo);
          }
          /*
          Our implementation uses more stack per thread than the available local 
          memory. CUDA probably needs to use some of the global memory to 
          compensate, but there is a bug and this memory is never freed.
          The official solution is to call cudaDeviceSetLimit to reset the 
          stack size and free that memory:
          https://forums.developer.nvidia.com/t/61314/2
          */
          cudaDeviceSetLimit(cudaLimitStackSize, 0);
#else
          PrecondImpl<scalar_t, int64_t, double, utils_t> algo(alloc);
          algo.loop();
#endif
        });
      });
    }
  };

  template<typename A, typename H, typename S, int32_t... Indices>
  void
  dispatch(int32_t i, const A & a, const H & h, const S & s,
           indices<0, Indices...>)
  {
      static void (*lookup[])(const A &, const H &, const S &) 
        = { &DoAlgo<Indices>::template f<A,H,S>... };

      int32_t logi = log2_ceil(i);
      static const int32_t mx_channels = sizeof(lookup)/sizeof(void *);
      if (logi >= mx_channels) {
          const std::string msg = "Too many channels (" + 
                                  std::to_string(pow_int(2, logi)) + "). "
                                  "Maximum number of channels is " + 
                                  std::to_string(pow_int(2, mx_channels)) + ".";
          throw std::out_of_range(msg);
      }
      lookup[logi](a, h, s);
  }

  template<int N, typename A, typename H, typename S>
  void dispatch(int i, const A & a, const H & h, const S & s)
  {
    dispatch(i, a, h, s, typename build_indices<0, N>::type()); 
  }

} // anonymous namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Note that dispatch<0, 10> means that we support channels in the 
// range [2**0 -> 2**9] (included) == [1 -> 512]

NI_HOST Tensor precond_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending, 
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(gradient, hessian, solution, weight);
  hessian  = std::get<0>(tensors);
  solution = std::get<1>(tensors);
  weight   = std::get<2>(tensors);

  PrecondAllocator info(gradient.dim()-2, absolute, membrane, bending,
                      voxel_size, bound);
  info.ioset(hessian, gradient, solution, weight);

  const auto & T  = gradient.scalar_type();
  const auto & C  = gradient.size(1);
  const auto & CC = (hessian.defined() && hessian.numel() > 0 ? hessian.size(1) : 0);
  const auto & H  = guess_hessian_type(C, CC);

  dispatch<10>(C, info, H, T); // MAX CHANNELS = 512 = 2**9
  return solution;
}


} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST Tensor precond_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  throw std::logic_error("Function not implemented for this device.");
}


} // namespace notimplemented

} // namespace ni

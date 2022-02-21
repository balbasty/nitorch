#include "common.h"                // write C++/CUDA compatible code
#include "../defines.h"            // useful macros
#include "bounds_common.h"         // boundary conditions + enum
#include "allocator.h"             // base class handling offset sizes
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
class PrecondGridAllocator: public Allocator {
public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST
  PrecondGridAllocator(int dim, double absolute, double membrane, double bending,
                   double lame_shear, double lame_div,
                   ArrayRef<double> voxel_size, BoundVectorRef bound):
    dim(dim),
    VEC_UNFOLD(bound, bound,      BoundType::Replicate),
    VEC_UNFOLD(vx,    voxel_size, 1.),
    absolute(absolute),
    membrane(membrane),
    bending(bending),
    lame_shear(lame_shear),
    lame_div(lame_div)
  {
    vx0 = 1. / (vx0*vx0);
    vx1 = 1. / (vx1*vx1);
    vx2 = 1. / (vx2*vx2);
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
  double            absolute;       // penalty on absolute values
  double            membrane;       // penalty on first derivatives
  double            bending;        // penalty on second derivatives
  double            lame_shear;  
  double            lame_div;  

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

  // Allow PrecondGridImpl's constructor to access PrecondGridAllocator's
  // private members.
  template <typename scalar_t, typename offset_t, typename reduce_t>
  friend class PrecondGridImpl;
};


NI_HOST
void PrecondGridAllocator::init_all()
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
void PrecondGridAllocator::init_gradient(const Tensor& input)
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
void PrecondGridAllocator::init_hessian(const Tensor& input)
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
void PrecondGridAllocator::init_solution(const Tensor& input)
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
void PrecondGridAllocator::init_weight(const Tensor& weight)
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

template <typename scalar_t, typename offset_t, typename reduce_t>
class PrecondGridImpl {

  using Self       = PrecondGridImpl;
  using PrecondFn  = void (Self::*)(offset_t, offset_t, offset_t, offset_t) const;
  typedef void (Self::*InvertFn)(const scalar_t *, scalar_t *, const scalar_t *v,
                                 reduce_t, reduce_t, reduce_t) const;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  PrecondGridImpl(const PrecondGridAllocator & info):
    dim(info.dim),
    bound0(info.bound0), bound1(info.bound1), bound2(info.bound2),
    vx0(info.vx0), vx1(info.vx1), vx2(info.vx2),
    absolute(info.absolute), membrane(info.membrane), bending(info.bending), 
    lame_shear(info.lame_shear), lame_div(info.lame_div), 
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
    set_kernel();
  #ifndef __CUDACC__
    set_precond();
    set_invert();
  #endif
  }

  NI_HOST NI_INLINE void set_kernel() 
  {
    mode = dim 
         + (lame_shear || lame_div ? 16 : bending ? 12 : membrane ? 8 : absolute ? 4 : 0)
         + (wgt_ptr ? 32 : 0);

    double lam0 = absolute, lam1 = membrane, lam2 = bending, 
           mu = lame_shear, lam = lame_div;

    w000 = lam2*(6.0*(vx0*vx0+vx1*vx1+vx2*vx2) + 8*(vx0*vx1+vx0*vx2+vx1*vx2)) 
         + lam1*2*(vx0+vx1+vx2) + lam0;

    wx000 =  2.0*mu*(2.0*vx0+vx1+vx2)/vx0+2.0*lam + w000/vx0;
    wy000 =  2.0*mu*(vx0+2.0*vx1+vx2)/vx1+2.0*lam + w000/vx1;
    wz000 =  2.0*mu*(vx0+vx1+2.0*vx2)/vx2+2.0*lam + w000/vx2;

    m100 = -vx0;
    m010 = -vx1;
    m001 = -vx2;
    b100 = -4.0*vx0*(vx0+vx1+vx2);
    b010 = -4.0*vx1*(vx0+vx1+vx2);
    b001 = -4.0*vx2*(vx0+vx1+vx2);
    b200 = vx0*vx0;
    b020 = vx1*vx1;
    b002 = vx2*vx2;
    b110 = 2.0*vx0*vx1;
    b101 = 2.0*vx0*vx2;
    b011 = 2.0*vx1*vx2;

    w000  *= OnePlusTiny;
    wx000 *= OnePlusTiny;
    wy000 *= OnePlusTiny;
    wz000 *= OnePlusTiny;
  }

#ifndef __CUDACC__
  NI_HOST NI_INLINE void set_precond() 
  {
#   define ABSOLUTE 4
#   define MEMBRANE 8
#   define BENDING  12
#   define LAME     16
#   define RLS      32
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

  NI_HOST NI_INLINE void set_invert() 
  {
    if (hes_ptr) {
      if (CC == 1) {
        if (dim == 1)
          invert_ = &Self::invert1d;
        else if (dim == 2)
          invert_ = &Self::invert2d_eye;
        else
          invert_ = &Self::invert3d_eye;
      } else if (CC == C) {
        if (dim == 1)
          invert_ = &Self::invert1d;
        else if (dim == 2)
          invert_ = &Self::invert2d_diag;
        else
          invert_ = &Self::invert3d_diag;
      } else {
        if (dim == 1)
          invert_ = &Self::invert1d;
        else if (dim == 2)
          invert_ = &Self::invert2d_sym;
        else
          invert_ = &Self::invert3d_sym;
      }
    } else {
      if (dim == 1)
        invert_ = &Self::invert1d_none;
      else if (dim == 2)
        invert_ = &Self::invert2d_none;
      else
        invert_ = &Self::invert3d_none;
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
  NI_DEVICE NI_INLINE void precond(
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

#define DEFINE_INVERT(SUFFIX) \
  NI_DEVICE void invert##SUFFIX(  \
    const scalar_t *, scalar_t *, const scalar_t *v, \
    reduce_t, reduce_t, reduce_t) const;
#define DEFINE_INVERT_DIM(DIM) \
  DEFINE_INVERT(DIM##d_sym)    \
  DEFINE_INVERT(DIM##d_diag)   \
  DEFINE_INVERT(DIM##d_eye)    \
  DEFINE_INVERT(DIM##d_none)

  DEFINE_INVERT()
  DEFINE_INVERT(1d)
  DEFINE_INVERT(1d_none)
  DEFINE_INVERT_DIM(2)
  DEFINE_INVERT_DIM(3)

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  offset_t          dim;
  uint8_t           mode;
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  reduce_t          vx0;            // voxel size // x|W
  reduce_t          vx1;            // voxel size // y|H
  reduce_t          vx2;            // voxel size // z|D
  reduce_t          absolute;       // penalty on absolute values
  reduce_t          membrane;       // penalty on first derivatives
  reduce_t          bending;        // penalty on second derivatives
  reduce_t          lame_shear;
  reduce_t          lame_div;

#ifndef __CUDACC__
  PrecondFn         precond_;       // Pointer to Precond function
  InvertFn          invert_;        // Pointer to inversion function
#endif

  reduce_t w000;
  reduce_t wx000;
  reduce_t wy000;
  reduce_t wz000;

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

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond(
    offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  #ifdef __CUDACC__
#   define ABSOLUTE 4
#   define MEMBRANE 8
#   define BENDING  12
#   define LAME     16
#   define RLS      32
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

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::loop(
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
template <typename scalar_t, typename offset_t, typename reduce_t> NI_HOST
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::loop() const
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
/*                                   INVERT                                   */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
#ifdef __CUDACC__
  if (hes_ptr == 0)
    switch (dim) {
      case 3:  return invert3d_none(h, s, v, w0, w1, w2);
      case 2:  return invert2d_none(h, s, v, w0, w1, w2);
      case 1:  return invert1d_none(h, s, v, w0, w1, w2);
      default: return invert3d_none(h, s, v, w0, w1, w2);
    }
  else if (CC == 1)
    switch (dim) {
      case 3:  return invert3d_eye(h, s, v, w0, w1, w2);
      case 2:  return invert2d_eye(h, s, v, w0, w1, w2);
      case 1:  return invert1d(h, s, v, w0, w1, w2);
      default: return invert3d_eye(h, s, v, w0, w1, w2);
    }
  else if (CC == C)
    switch (dim) {
      case 3:  return invert3d_diag(h, s, v, w0, w1, w2);
      case 2:  return invert2d_diag(h, s, v, w0, w1, w2);
      case 1:  return invert1d(h, s, v, w0, w1, w2);
      default: return invert3d_diag(h, s, v, w0, w1, w2);
    }
  else
    switch (dim) {
      case 3:  return invert3d_sym(h, s, v, w0, w1, w2);
      case 2:  return invert2d_sym(h, s, v, w0, w1, w2);
      case 1:  return invert1d(h, s, v, w0, w1, w2);
      default: return invert3d_sym(h, s, v, w0, w1, w2);;
    }
#else
  CALL_MEMBER_FN(*this, invert_)(h, s, v, w0, w1, w2);
#endif
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert3d_sym(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  reduce_t h00 = h[0],        h11 = h[  hes_sC], h22 = h[2*hes_sC],
           h01 = h[3*hes_sC], h02 = h[4*hes_sC], h12 = h[5*hes_sC],
           v0  = v[0],        v1  = v[  grd_sC], v2  = v[2*grd_sC],
           idt;

  // solve
  h00  = h00 * OnePlusTiny + w0;
  h11  = h11 * OnePlusTiny + w1;
  h22  = h22 * OnePlusTiny + w2;
  idt  = 1.0/(h00*h11*h22 - h00*h12*h12 - h11*h02*h02 - h22*h01*h01 + 2*h01*h02*h12);
  s[       0] = idt*(v0*(h11*h22-h12*h12) + v1*(h02*h12-h01*h22) + v2*(h01*h12-h02*h11));
  s[  sol_sC] = idt*(v0*(h02*h12-h01*h22) + v1*(h00*h22-h02*h02) + v2*(h01*h02-h00*h12));
  s[2*sol_sC] = idt*(v0*(h01*h12-h02*h11) + v1*(h01*h02-h00*h12) + v2*(h00*h11-h01*h01));
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert2d_sym(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t /*unused*/) const 
{
  reduce_t h00 = h[0], h11 = h[  hes_sC], h01 = h[2*hes_sC],
           v0  = v[0], v1  = v[  grd_sC], idt;

  // solve
  h00  = h00 * OnePlusTiny + w0;
  h11  = h11 * OnePlusTiny + w1;
  idt  = 1.0/(h00*h11 - h01*h01);
  s[     0] = idt*(v0*h11 - v1*h01);
  s[sol_sC] = idt*(v1*h00 - v0*h01);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert3d_diag(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  s[       0] = v[       0] / (h[       0] * OnePlusTiny + w0);
  s[  sol_sC] = v[  grd_sC] / (h[  hes_sC] * OnePlusTiny + w1);
  s[2*sol_sC] = v[2*grd_sC] / (h[2*hes_sC] * OnePlusTiny + w2);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert2d_diag(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t  /*unused*/) const 
{
  s[     0] = v[     0] / (h[     0] * OnePlusTiny + w0);
  s[sol_sC] = v[grd_sC] / (h[hes_sC] * OnePlusTiny + w1);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert3d_eye(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  reduce_t h00 = *h;
  s[       0] = v[       0] / (h00 * OnePlusTiny + w0);
  s[  sol_sC] = v[  grd_sC] / (h00 * OnePlusTiny + w1);
  s[2*sol_sC] = v[2*grd_sC] / (h00 * OnePlusTiny + w2);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert2d_eye(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t  /*unused*/) const 
{
  reduce_t h00 = *h;

  // solve
  s[     0] = v[     0] / (h00 * OnePlusTiny + w0);
  s[sol_sC] = v[grd_sC] / (h00 * OnePlusTiny + w1);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert1d(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t  /*unused*/, reduce_t  /*unused*/) const 
{
  (*s) = (*v) / ((*h) * OnePlusTiny + w0);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert3d_none(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  s[       0] = v[       0] / w0;
  s[  sol_sC] = v[  grd_sC] / w1;
  s[2*sol_sC] = v[2*grd_sC] / w2;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert2d_none(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t  /*unused*/) const 
{
  s[     0] = v[     0] / w0;
  s[sol_sC] = v[grd_sC] / w1;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert1d_none(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t  /*unused*/, reduce_t  /*unused*/) const 
{
  (*s) = (*v) / w0;
}

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
  x##0  = (bound::index(bound##i, x##0,  X) - x) * sol_s##X; \
  x##1  = (bound::index(bound##i, x##1,  X) - x) * sol_s##X;
#define GET_WARP2_(x, X, i)  \
  x##00  = (bound::index(bound##i, x##00,  X) - x) * sol_s##X; \
  x##11  = (bound::index(bound##i, x##11,  X) - x) * sol_s##X;
#define GET_WARP1_RLS_(x, X, i) \
  x##0  = (bound::index(bound##i, x##0,  X) - x); \
  x##1  = (bound::index(bound##i, x##1,  X) - x); \
  offset_t w##x##0 = x##0 * wgt_s##X; \
  offset_t w##x##1 = x##1 * wgt_s##X; \
  x##0 *= sol_s##X; \
  x##1 *= sol_s##X;


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
#define GET_WARP1_RLS \
  GET_WARP1_RLS_(x, X, 0) \
  GET_WARP1_RLS_(y, Y, 1) \
  GET_WARP1_RLS_(z, Z, 2)

#define GET_POINTERS \
  const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN); \
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN); \
  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond3d(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS

  invert(hes, sol, grd, wx000, wy000, wz000);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  scalar_t wcenter = *wgt;
  reduce_t w1m00 = m100 * (wcenter + bound::get(wgt, wx0, sx0));
  reduce_t w1p00 = m100 * (wcenter + bound::get(wgt, wx1, sx1));
  reduce_t w01m0 = m010 * (wcenter + bound::get(wgt, wy0, sy0));
  reduce_t w01p0 = m010 * (wcenter + bound::get(wgt, wy1, sy1));
  reduce_t w001m = m001 * (wcenter + bound::get(wgt, wz0, sz0));
  reduce_t w001p = m001 * (wcenter + bound::get(wgt, wz1, sz1));

  reduce_t w = absolute * wcenter 
             + membrane * (w1m00 + w1p00 + w01m0 + w01p0 + w001m + w001p);
  invert(hes, sol, grd, w/vx0, w/vx1, w/vx2);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  reduce_t w = absolute * (*wgt);
  invert(hes, sol, grd, w/vx0, w/vx1, w/vx2);
}



/* ========================================================================== */
/*                                     2D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond2d(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond2d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond2d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}

/* ========================================================================== */
/*                                     1D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond1d(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond1d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond1d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t, typename reduce_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void precond_kernel(PrecondGridImpl<scalar_t,offset_t,reduce_t> * f) {
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

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST Tensor precond_grid_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div, 
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(gradient, hessian, solution, weight);
  hessian  = std::get<0>(tensors);
  solution = std::get<1>(tensors);
  weight   = std::get<2>(tensors);

  PrecondGridAllocator info(gradient.dim()-2, absolute, membrane, bending, lame_shear, lame_div,
                      voxel_size, bound);
  info.ioset(hessian, gradient, solution, weight);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradient.scalar_type(), "precond_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      PrecondGridImpl<scalar_t, int32_t, double> algo(info);
      auto palgo = alloc_and_copy_to_device(algo, stream);
      precond_kernel
          <<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>
          (palgo);
      cudaFree(palgo);
    }
    else
    {
      PrecondGridImpl<scalar_t, int64_t, double> algo(info);
      auto palgo = alloc_and_copy_to_device(algo, stream);
      precond_kernel
          <<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>
          (palgo);
      cudaFree(palgo);
    }
  });
  return solution;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST Tensor precond_grid_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div, 
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(gradient, hessian, solution, weight);
  hessian  = std::get<0>(tensors);
  solution = std::get<1>(tensors);
  weight   = std::get<2>(tensors);

  PrecondGridAllocator info(gradient.dim()-2, absolute, membrane, bending, lame_shear, lame_div,
                      voxel_size, bound);
  info.ioset(hessian, gradient, solution, weight);

  AT_DISPATCH_FLOATING_TYPES(gradient.scalar_type(), "precond_impl", [&] {
    PrecondGridImpl<scalar_t, int64_t, double> algo(info);
    algo.loop();
  });
  return solution;
}

#endif // __CUDACC__

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST Tensor precond_grid_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  throw std::logic_error("Function not implemented for this device.");
}


} // namespace notimplemented

} // namespace ni

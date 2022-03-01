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
class RelaxGridAllocator: public Allocator {
public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST
  RelaxGridAllocator(int dim, 
                     double absolute, double membrane, double bending,
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
  double            absolute;       // penalty on absolute values
  double            membrane;       // penalty on first derivatives
  double            bending;        // penalty on second derivatives
  double            lame_shear;     // penalty on symmetric part of Jacobian
  double            lame_div;       // penalty on trace of Jacobian

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

  // Allow RelaxGridImpl's constructor to access RelaxGridAllocator's
  // private members.
  template <typename scalar_t, typename offset_t, typename reduce_t>
  friend class RelaxGridImpl;
};


NI_HOST
void RelaxGridAllocator::init_all()
{
  N = C = CC = X = Y = Z = 1L;
  grd_sN  = grd_sC   = grd_sX   = grd_sY  = grd_sZ   = 0L;
  hes_sN  = hes_sC   = hes_sX   = hes_sY  = hes_sZ   = 0L;
  sol_sN  = sol_sC   = sol_sX   = sol_sY  = sol_sZ   = 0L;
  wgt_sN  = wgt_sC   = wgt_sX   = wgt_sY  = wgt_sZ   = 0L;
  grd_ptr = hes_ptr = sol_ptr = wgt_ptr = static_cast<float*>(0);
  grd_32b_ok = hes_32b_ok = sol_32b_ok = wgt_32b_ok = true;
}

NI_HOST
void RelaxGridAllocator::init_gradient(const Tensor& input)
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
void RelaxGridAllocator::init_hessian(const Tensor& input)
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
void RelaxGridAllocator::init_solution(const Tensor& input)
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
void RelaxGridAllocator::init_weight(const Tensor& weight)
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
class RelaxGridImpl {

  typedef RelaxGridImpl Self;
  typedef void (Self::*RelaxFn)(offset_t x, offset_t y, offset_t z, offset_t n) const;
  typedef void (Self::*InvertFn)(const scalar_t *, scalar_t *, 
                                 reduce_t, reduce_t, reduce_t, 
                                 reduce_t, reduce_t, reduce_t) const;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  RelaxGridImpl(const RelaxGridAllocator & info):
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
    set_bandwidth();
#ifndef __CUDACC__
    set_relax();
    set_invert();
#endif
  }

  NI_HOST NI_INLINE void set_kernel() 
  {
    mode = dim 
         + (bending ? 12 : membrane ? 8 : absolute ? 4 : 0)
         + (lame_shear || lame_div ? 16 : 0)
         + (wgt_ptr ? 32 : 0);

    double lam0 = absolute, lam1 = membrane, lam2 = bending, mu = lame_shear, lam = lame_div;

    w000 = lam2*(6.0*(vx0*vx0+vx1*vx1+vx2*vx2) + 8.0*(vx0*vx1+vx0*vx2+vx1*vx2)) 
         + lam1*2.0*(vx0+vx1+vx2) + lam0;
    w100 = lam2*(-4.0*vx0*(vx0+vx1+vx2)) -lam1*vx0;
    w010 = lam2*(-4.0*vx1*(vx0+vx1+vx2)) -lam1*vx1;
    w001 = lam2*(-4.0*vx2*(vx0+vx1+vx2)) -lam1*vx2;
    w200 = lam2*vx0*vx0;
    w020 = lam2*vx1*vx1;
    w002 = lam2*vx2*vx2;
    w110 = lam2*2.0*vx0*vx1;
    w101 = lam2*2.0*vx0*vx2;
    w011 = lam2*2.0*vx1*vx2;

    wx000 =  2.0*mu*(2.0*vx0+vx1+vx2)/vx0+2.0*lam + w000/vx0;
    wx100 = -2.0*mu-lam + w100/vx0;
    wx010 = -mu*vx1/vx0 + w010/vx0;
    wx001 = -mu*vx2/vx0 + w001/vx0;
    wy000 =  2.0*mu*(vx0+2.0*vx1+vx2)/vx1+2.0*lam + w000/vx1;
    wy100 = -mu*vx0/vx1 + w100/vx1;
    wy010 = -2.0*mu-lam + w010/vx1;
    wy001 = -mu*vx2/vx1 + w001/vx1;
    wz000 =  2.0*mu*(vx0+vx1+2.0*vx2)/vx2+2.0*lam + w000/vx2;
    wz100 = -mu*vx0/vx2 + w100/vx2;
    wz010 = -mu*vx1/vx2 + w010/vx2;
    wz001 = -2.0*mu-lam + w001/vx2;
    w2    = 0.25*mu+0.25*lam;

    w000  *= OnePlusTiny;
    wx000 *= OnePlusTiny;
    wy000 *= OnePlusTiny;
    wz000 *= OnePlusTiny;
  }

  NI_HOST NI_INLINE void set_bandwidth() 
  { 
    if (bending)
      bandwidth = 3;
    else if (lame_shear || lame_div)
      bandwidth = 2;
    else if (membrane)
      bandwidth = 0; // checkerboard
    else
      bandwidth = 1;

    if (bandwidth)
    {
      // Size of the band in each direction
      Fx = MIN(X, bandwidth);
      Fy = MIN(Y, bandwidth);
      Fz = MIN(Z, bandwidth);

      // size of the fold
      Xf = (X + Fx - 1) / Fx;
      Yf = (Y + Fy - 1) / Fy;
      Zf = (Z + Fz - 1) / Fz;
    }
  }

#ifndef __CUDACC__
  NI_HOST NI_INLINE void set_relax() 
  {
#   define ABSOLUTE 4
#   define MEMBRANE 8
#   define BENDING  12
#   define LAME     16
#   define RLS      32
    switch (mode) {
      case 1 + MEMBRANE + RLS:
        relax_ = &Self::relax1d_rls_membrane; break;
      case 2 + MEMBRANE + RLS:
        relax_ = &Self::relax2d_rls_membrane; break;
      case 3 + MEMBRANE + RLS:
        relax_ = &Self::relax3d_rls_membrane; break;
      case 1 + ABSOLUTE + RLS:
        relax_ = &Self::relax1d_rls_absolute; break;
      case 2 + ABSOLUTE + RLS:
        relax_ = &Self::relax2d_rls_absolute; break;
      case 3 + ABSOLUTE + RLS:
        relax_ = &Self::relax3d_rls_absolute; break;
      case 1 + BENDING + LAME:
        relax_ = &Self::relax1d_all; break;
      case 2 + BENDING + LAME:
        relax_ = &Self::relax2d_all; break;
      case 3 + BENDING + LAME:
        relax_ = &Self::relax3d_all; break;
      case 1 + BENDING:
        relax_ = &Self::relax1d_bending; break;
      case 2 + BENDING:
        relax_ = &Self::relax2d_bending; break;
      case 3 + BENDING:
        relax_ = &Self::relax3d_bending; break;
      case 1 + LAME: case 1 + LAME + MEMBRANE: case 1 + LAME + ABSOLUTE:
        relax_ = &Self::relax1d_lame; break;
      case 2 + LAME: case 2 + LAME + MEMBRANE: case 2 + LAME + ABSOLUTE:
        relax_ = &Self::relax2d_lame; break;
      case 3 + LAME: case 3 + LAME + MEMBRANE: case 3 + LAME + ABSOLUTE:
        relax_ = &Self::relax3d_lame; break;
      case 1 + MEMBRANE:
        relax_ = &Self::relax1d_membrane; break;
      case 2 + MEMBRANE:
        relax_ = &Self::relax2d_membrane; break;
      case 3 + MEMBRANE:
        relax_ = &Self::relax3d_membrane; break;
      case 1 + ABSOLUTE:
        relax_ = &Self::relax1d_absolute; break;
      case 2 + ABSOLUTE:
        relax_ = &Self::relax2d_absolute; break;
      case 3 + ABSOLUTE:
        relax_ = &Self::relax3d_absolute; break;
      case 1: case 1 + RLS:
        relax_ = &Self::solve1d; break;
      case 2: case 2 + RLS:
        relax_ = &Self::solve2d; break;
      case 3: case 3 + RLS:
        relax_ = &Self::solve3d; break;
      default:
        relax_ = &Self::solve3d; break;
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

#if __CUDACC__
  // Loop over voxels that belong to one CUDA block
  // This function is called by the CUDA kernel
  NI_DEVICE void loop(int threadIdx, int blockIdx,
                      int blockDim, int gridDim) const;
  NI_DEVICE void loop_band(int threadIdx, int blockIdx,
                           int blockDim, int gridDim) const;
  NI_DEVICE void loop_redblack(int threadIdx, int blockIdx,
                               int blockDim, int gridDim) const;
#else
  // Loop over all voxels
  void loop();
  void loop_band();
  void loop_redblack();
#endif

  NI_HOST NI_DEVICE int64_t voxcount() const {
    return N * X * Y * Z;
  }

  NI_HOST NI_DEVICE int64_t voxcountfold() const {
    return bandwidth == 0 ? voxcount() : N * Xf * Yf * Zf;
  }

  NI_HOST NI_DEVICE int64_t foldcount() const {
    return bandwidth == 0 ? 2 : Fx * Fy * Fz;
  }

  NI_HOST void set_fold(offset_t i) {
    if (bandwidth == 0)
      // checkerboard
      redblack = i;
    else {
      // index of the fold (lin2sub)
      fx = i/(Fy*Fz);
      fy = (i/Fz)  % Y;
      fz = i % Z;

      Xf = 1 + (X - fx - 1) / Fx;
      Yf = 1 + (Y - fy - 1) / Fy;
      Zf = 1 + (Z - fz - 1) / Fz;
    }
  }
 

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
#define DEFINE_RELAX(SUFFIX) \
  NI_DEVICE void relax##SUFFIX( \
    offset_t x, offset_t y, offset_t z, offset_t n) const;
#define DEFINE_RELAX_DIM(DIM)        \
  DEFINE_RELAX(DIM##d_absolute)      \
  DEFINE_RELAX(DIM##d_membrane)      \
  DEFINE_RELAX(DIM##d_bending)       \
  DEFINE_RELAX(DIM##d_lame)          \
  DEFINE_RELAX(DIM##d_all)           \
  DEFINE_RELAX(DIM##d_rls_absolute)  \
  DEFINE_RELAX(DIM##d_rls_membrane)  \
  NI_DEVICE void solve##DIM##d(      \
    offset_t x, offset_t y, offset_t z, offset_t n) const;

  DEFINE_RELAX()
  DEFINE_RELAX_DIM(1)
  DEFINE_RELAX_DIM(2)
  DEFINE_RELAX_DIM(3)

#define DEFINE_INVERT(SUFFIX) \
  NI_DEVICE void invert##SUFFIX(  \
    const scalar_t *, scalar_t *, \
    reduce_t, reduce_t, reduce_t, \
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

  /* ~~~ FOLD NAVIGATORS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  offset_t fx; // Index of the fold
  offset_t fy;
  offset_t fz;
  offset_t Xf; // Size of the fold
  offset_t Yf;
  offset_t Zf;
  offset_t redblack;  // Index of the fold for checkerboard scheme
  offset_t bandwidth;
  offset_t Fx; // Fold window
  offset_t Fy;
  offset_t Fz;

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  offset_t          dim;            // dimensionality (2 or 3)
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
  reduce_t          lame_shear;     // penalty on symmetric part of Jacobian
  reduce_t          lame_div;       // penalty on trace of Jacobian

#ifndef __CUDACC__
  RelaxFn           relax_;         // Pointer to relax function
  InvertFn          invert_;        // Pointer to inversion function
#endif

  reduce_t  w000;
  reduce_t  w100;
  reduce_t  w010;
  reduce_t  w001;
  reduce_t  w200;
  reduce_t  w020;
  reduce_t  w002;
  reduce_t  w110;
  reduce_t  w101;
  reduce_t  w011;

  reduce_t  wx000;
  reduce_t  wx100;
  reduce_t  wx010;
  reduce_t  wx001;
  reduce_t  wy000;
  reduce_t  wy100;
  reduce_t  wy010;
  reduce_t  wy001;
  reduce_t  wz000;
  reduce_t  wz100;
  reduce_t  wz010;
  reduce_t  wz001;
  reduce_t  w2;

  /* ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#define DEFINE_STRIDE_INFO_5D(NAME)   \
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
  DEFINE_STRIDE_INFO_5D(grd)
  DEFINE_STRIDE_INFO_5D(hes)
  DEFINE_STRIDE_INFO_5D(sol)
  DEFINE_STRIDE_INFO_5D(wgt)
};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax(
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
      return relax1d_rls_membrane(x, y, z, n);
    case 2 + MEMBRANE + RLS:
      return relax2d_rls_membrane(x, y, z, n);
    case 3 + MEMBRANE + RLS:
      return relax3d_rls_membrane(x, y, z, n);
    case 1 + ABSOLUTE + RLS:
      return relax1d_rls_absolute(x, y, z, n);
    case 2 + ABSOLUTE + RLS:
      return relax2d_rls_absolute(x, y, z, n);
    case 3 + ABSOLUTE + RLS:
      return relax3d_rls_absolute(x, y, z, n);
    case 1 + BENDING + LAME:
      return relax1d_all(x, y, z, n);
    case 2 + BENDING + LAME:
      return relax2d_all(x, y, z, n);
    case 3 + BENDING + LAME:
      return relax3d_all(x, y, z, n);
    case 1 + BENDING:
      return relax1d_bending(x, y, z, n);
    case 2 + BENDING:
      return relax2d_bending(x, y, z, n);
    case 3 + BENDING:
      return relax3d_bending(x, y, z, n);
    case 1 + LAME: case 1 + LAME + MEMBRANE: case 1 + LAME + ABSOLUTE:
      return relax1d_lame(x, y, z, n);
    case 2 + LAME: case 2 + LAME + MEMBRANE: case 2 + LAME + ABSOLUTE:
      return relax2d_lame(x, y, z, n);
    case 3 + LAME: case 3 + LAME + MEMBRANE: case 3 + LAME + ABSOLUTE:
      return relax3d_lame(x, y, z, n);
    case 1 + MEMBRANE:
      return relax1d_membrane(x, y, z, n);
    case 2 + MEMBRANE:
      return relax2d_membrane(x, y, z, n);
    case 3 + MEMBRANE:
      return relax3d_membrane(x, y, z, n);
    case 1 + ABSOLUTE:
      return relax1d_absolute(x, y, z, n);
    case 2 + ABSOLUTE:
      return relax2d_absolute(x, y, z, n);
    case 3 + ABSOLUTE:
      return relax3d_absolute(x, y, z, n);
    case 1: case 1 + RLS:
      return solve1d(x, y, z, n);
    case 2: case 2 + RLS:
      return solve2d(x, y, z, n);
    case 3: case 3 + RLS:
      return solve3d(x, y, z, n);
    default:
      return solve3d(x, y, z, n);
  }
#else
  CALL_MEMBER_FN(*this, relax_)(x, y, z, n);
#endif
}

#if __CUDACC__

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  if (bandwidth == 0)
    return loop_redblack(threadIdx, blockIdx, blockDim, gridDim);
  else
    return loop_band(threadIdx, blockIdx, blockDim, gridDim);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::loop_band(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  int64_t index = blockIdx * blockDim + threadIdx;
  int64_t nthreads = N * Xf * Yf * Zf;
  offset_t YZf   = Yf * Zf;
  offset_t XYZf  = Xf * YZf;
  offset_t n, x, y, z;
  for (offset_t i=index; index < nthreads; index += blockDim*gridDim, i=index)
  {
    // Convert index: linear to sub
    n  = (i/XYZf);
    x  = ((i/YZf) % Xf) * Fx + fx;
    y  = ((i/Zf)  % Yf) * Fy + fy;
    z  = (i       % Zf) * Fz + fz;
    relax(x, y, z, n);
  }
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::loop_redblack(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  int64_t index = blockIdx * blockDim + threadIdx;
  int64_t nthreads = N * X * Y * Z;
  offset_t YZ   = Y * Z;
  offset_t XYZ  = X * YZ;
  offset_t n, x, y, z;
  for (offset_t i=index; index < nthreads; index += blockDim*gridDim, i=index)
  {
    // Convert index: linear to sub
    n  = (i/XYZ);
    x  = ((i/YZ) % X);
    y  = ((i/Z)  % Y);
    z  = (i      % Z);
    if ((x+y+z) % 2 == redblack)
      relax(x, y, z, n);
  }
}

#else

template <typename scalar_t, typename offset_t, typename reduce_t> NI_HOST
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::loop()
{
  if (bandwidth == 0)
    return loop_redblack();
  else
    return loop_band();
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_HOST
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::loop_redblack()
{
  // Parallelize across voxels
  offset_t NXYZ = Z * Y * X * N;
  offset_t XYZ  = Z * Y * X;
  offset_t YZ   = Z * Y;

  for (offset_t redblack = 0; redblack < 2; ++redblack) {
    set_fold(redblack);
    at::parallel_for(0, NXYZ, GRAIN_SIZE, [&](offset_t start, offset_t end) {
      offset_t n, x, y, z;
      for (offset_t i = start; i < end; ++i) {
        // Convert index: linear to sub
        n  = (i/XYZ);
        x  = (i/YZ) % X;
        y  = (i/Z)  % Y;
        z  = i % Z;
        if ((x+y+z) % 2 == redblack)
          relax(x, y, z, n);
      }
    });
  }
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_HOST
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::loop_band()
{
  for (offset_t fold = 0; fold < Fx*Fy*Fz; ++fold) {
    // Index of the fold
    set_fold(fold);
    offset_t YZf   =   Zf * Yf;
    offset_t XYZf  =  YZf * Xf;
    offset_t NXYZf = XYZf * N;

    at::parallel_for(0, NXYZf, GRAIN_SIZE, [&](offset_t start, offset_t end) {
      offset_t n, x, y, z;
      for (offset_t i = start; i < end; ++i) {
        // Convert index: linear to sub
        n  = (i/XYZf);
        x  = ((i/YZf) % Xf) * Fx + fx;
        y  = ((i/Zf)  % Yf) * Fy + fy;
        z  = (i       % Zf) * Fz + fz;
        relax(x, y, z, n);
      }
    });
  }
}

#endif

/* ========================================================================== */
/*                                   INVERT                                   */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t v2, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
#ifdef __CUDACC__
  if (hes_ptr == 0)
    switch (dim) {
      case 3:  return invert3d_none(h, s, v0, v1, v2, w0, w1, w2);
      case 2:  return invert2d_none(h, s, v0, v1, v2, w0, w1, w2);
      case 1:  return invert1d_none(h, s, v0, v1, v2, w0, w1, w2);
      default: return invert3d_none(h, s, v0, v1, v2, w0, w1, w2);
    }
  else if (CC == 1)
    switch (dim) {
      case 3:  return invert3d_eye(h, s, v0, v1, v2, w0, w1, w2);
      case 2:  return invert2d_eye(h, s, v0, v1, v2, w0, w1, w2);
      case 1:  return invert1d(h, s, v0, v1, v2, w0, w1, w2);
      default: return invert3d_eye(h, s, v0, v1, v2, w0, w1, w2);
    }
  else if (CC == C)
    switch (dim) {
      case 3:  return invert3d_diag(h, s, v0, v1, v2, w0, w1, w2);
      case 2:  return invert2d_diag(h, s, v0, v1, v2, w0, w1, w2);
      case 1:  return invert1d(h, s, v0, v1, v2, w0, w1, w2);
      default: return invert3d_diag(h, s, v0, v1, v2, w0, w1, w2);
    }
  else
    switch (dim) {
      case 3:  return invert3d_sym(h, s, v0, v1, v2, w0, w1, w2);
      case 2:  return invert2d_sym(h, s, v0, v1, v2, w0, w1, w2);
      case 1:  return invert1d(h, s, v0, v1, v2, w0, w1, w2);
      default: return invert3d_sym(h, s, v0, v1, v2, w0, w1, w2);;
    }
#else
  CALL_MEMBER_FN(*this, invert_)(h, s, v0, v1, v2, w0, w1, w2);
#endif
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert3d_sym(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t v2, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  reduce_t h00 = h[0],        h11 = h[  hes_sC], h22 = h[2*hes_sC],
           h01 = h[3*hes_sC], h02 = h[4*hes_sC], h12 = h[5*hes_sC],
           s0  = s[0],        s1  = s[  sol_sC], s2  = s[2*sol_sC],
           idt;

  // matvec
  v0 -= h00*s0 + h01*s1 + h02*s2;
  v1 -= h01*s0 + h11*s1 + h12*s2;
  v2 -= h02*s0 + h12*s1 + h22*s2;

  // solve
  h00  = h00 * OnePlusTiny + w0;
  h11  = h11 * OnePlusTiny + w1;
  h22  = h22 * OnePlusTiny + w2;
  idt  = 1.0/(h00*h11*h22 - h00*h12*h12 - h11*h02*h02 - h22*h01*h01 + 2*h01*h02*h12);
  s[       0] += idt*(v0*(h11*h22-h12*h12) + v1*(h02*h12-h01*h22) + v2*(h01*h12-h02*h11));
  s[  sol_sC] += idt*(v0*(h02*h12-h01*h22) + v1*(h00*h22-h02*h02) + v2*(h01*h02-h00*h12));
  s[2*sol_sC] += idt*(v0*(h01*h12-h02*h11) + v1*(h01*h02-h00*h12) + v2*(h00*h11-h01*h01));
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert2d_sym(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t /*unused*/, 
    reduce_t w0, reduce_t w1, reduce_t /*unused*/) const 
{
  reduce_t h00 = h[0], h11 = h[  hes_sC], h01 = h[2*hes_sC],
           s0  = s[0], s1  = s[  sol_sC], idt;

  // matvec
  v0 -= h00*s0 + h01*s1;
  v1 -= h01*s0 + h11*s1;

  // solve
  h00  = h00 * OnePlusTiny + w0;
  h11  = h11 * OnePlusTiny + w1;
  idt  = 1.0/(h00*h11 - h01*h01);
  s[     0] += idt*(v0*h11 - v1*h01);
  s[sol_sC] += idt*(v1*h00 - v0*h01);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert3d_diag(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t v2, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  reduce_t h00 = h[0], h11 = h[hes_sC], h22 = h[2*hes_sC],
           s0  = s[0], s1  = s[sol_sC], s2  = s[2*sol_sC];

  // matvec
  v0 -= h00 * s0;
  v1 -= h11 * s1;
  v2 -= h22 * s2;

  // solve
  s[       0] += v0 / (h00 * OnePlusTiny + w0);
  s[  sol_sC] += v1 / (h11 * OnePlusTiny + w1);
  s[2*sol_sC] += v2 / (h22 * OnePlusTiny + w2);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert2d_diag(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t /*unused*/, 
    reduce_t w0, reduce_t w1, reduce_t  /*unused*/) const 
{
  reduce_t h00 = h[0], h11 = h[hes_sC],
           s0  = s[0], s1  = s[sol_sC];

  // matvec
  v0 -= h00 * s0;
  v1 -= h11 * s1;

  // sve
  s[     0] += v0 / (h00 * OnePlusTiny + w0);
  s[sol_sC] += v1 / (h11 * OnePlusTiny + w1);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert3d_eye(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t v2, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  reduce_t h00 = *h, s0  = s[0], s1  = s[sol_sC], s2  = s[2*sol_sC];

  // matvec
  v0 -= h00 * s0;
  v1 -= h00 * s1;
  v2 -= h00 * s2;

  // solve
  h00 *= OnePlusTiny;
  s[       0] += v0 / (h00 + w0);
  s[  sol_sC] += v1 / (h00 + w1);
  s[2*sol_sC] += v2 / (h00 + w2);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert2d_eye(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t /*unused*/, 
    reduce_t w0, reduce_t w1, reduce_t  /*unused*/) const 
{
  reduce_t h00 = *h, s0  = s[0], s1  = s[sol_sC];

  // matvec
  v0 -= h00 * s0;
  v1 -= h00 * s1;

  // solve
  h00 *= OnePlusTiny;
  s[     0] += v0 / (h00 + w0);
  s[sol_sC] += v1 / (h00 + w1);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert1d(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t  /*unused*/, reduce_t  /*unused*/, 
    reduce_t w0, reduce_t  /*unused*/, reduce_t  /*unused*/) const 
{
  reduce_t h00 = *h, s0 = *s;

  // matvec
  v0 -= h00 * s0;

  // solve
  (*s) += v0 / (h00 * OnePlusTiny + w0);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert3d_none(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t v2, 
    reduce_t w0, reduce_t w1, reduce_t w2) const 
{
  s[       0] += v0 / w0;
  s[  sol_sC] += v1 / w1;
  s[2*sol_sC] += v2 / w2;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert2d_none(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t v1, reduce_t  /*unused*/, 
    reduce_t w0, reduce_t w1, reduce_t  /*unused*/) const 
{
  s[     0] += v0 / w0;
  s[sol_sC] += v1 / w1;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::invert1d_none(
    const scalar_t * h, scalar_t * s, 
    reduce_t v0, reduce_t  /*unused*/, reduce_t  /*unused*/, 
    reduce_t w0, reduce_t  /*unused*/, reduce_t  /*unused*/) const 
{
  (*s) += v0 / w0;
}

/* ========================================================================== */
/*                               MACRO HELPERS                                */
/* ========================================================================== */
#define GET_COORD1_(x) offset_t x##0  = x - 1, x##1 = x + 1;
#define GET_COORD2_(x) offset_t x##00  = x - 2, x##11 = x + 2;
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
  const scalar_t *grd0 = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN), \
                 *grd1 = grd0 + grd_sC, *grd2 = grd1 + grd_sC,                  \
                 *hes  = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN); \
        scalar_t *sol0 = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN), \
                 *sol1 = sol0 + sol_sC, *sol2 = sol1 + sol_sC;

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax3d_all(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1   // Sign (/!\ compute sign before warping indices)
  GET_SIGN2
  GET_WARP1   // Warp indices
  GET_WARP2
  GET_POINTERS

  reduce_t val0, val1, val2;

  // For numerical stability, we subtract the center value before convolving.
  // We define a lambda function for ease.

  {
    scalar_t c = *sol0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
       (wx100*(get(sol0, x0, sx0) + get(sol0, x1, sx1))
      + wx010*(get(sol0, y0, sy0) + get(sol0, y1, sy1))
      + wx001*(get(sol0, z0, sz0) + get(sol0, z1, sz1)))
      + w2   *( bound::get(sol1, x1+y0, sx1*sy0) - bound::get(sol1, x1+y1, sx1*sy1)
              + bound::get(sol1, x0+y1, sx0*sy1) - bound::get(sol1, x0+y0, sx0*sy0)
              + bound::get(sol2, x1+z0, sx1*sz0) - bound::get(sol2, x1+z1, sx1*sz1)
              + bound::get(sol2, x0+z1, sx0*sz1) - bound::get(sol2, x0+z0, sx0*sz0) )
      + ( absolute*c
        +(w110*(get(sol0, x0+y0, sx0*sy0) + get(sol0, x1+y0, sx1*sy0) +
                get(sol0, x0+y1, sx0*sy1) + get(sol0, x1+y1, sx1*sy1))
        + w101*(get(sol0, x0+z0, sx0*sz0) + get(sol0, x1+z0, sx1*sz0) +
                get(sol0, x0+z1, sx0*sz1) + get(sol0, x1+z1, sx1*sz1))
        + w011*(get(sol0, y0+z0, sy0*sz0) + get(sol0, y1+z0, sy1*sz0) +
                get(sol0, y0+z1, sy0*sz1) + get(sol0, y1+z1, sy1*sz1)))
        +(w200*(get(sol0, x00,   sx00)    + get(sol0, x11,   sx11))
        + w020*(get(sol0, y00,   sy00)    + get(sol0, y11,   sy11))
        + w002*(get(sol0, z00,   sz00)    + get(sol0, z11,   sz11))) ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
       (wy100*(get(sol1, x0, sx0) + get(sol1, x1, sx1))
      + wy010*(get(sol1, y0, sy0) + get(sol1, y1, sy1))
      + wy001*(get(sol1, z0, sz0) + get(sol1, z1, sz1)))
      + w2   *( bound::get(sol0, y1+x0, sy1*sx0) - bound::get(sol0, y1+x1, sy1*sx1)
              + bound::get(sol0, y0+x1, sy0*sx1) - bound::get(sol0, y0+x0, sy0*sx0)
              + bound::get(sol2, y1+z0, sy1*sz0) - bound::get(sol2, y1+z1, sy1*sz1)
              + bound::get(sol2, y0+z1, sy0*sz1) - bound::get(sol2, y0+z0, sy0*sz0) )
      + ( absolute*c
        +(w110*(get(sol1, x0+y0, sx0*sy0) + get(sol1, x1+y0, sx1*sy0) +
                get(sol1, x0+y1, sx0*sy1) + get(sol1, x1+y1, sx1*sy1))
        + w101*(get(sol1, x0+z0, sx0*sz0) + get(sol1, x1+z0, sx1*sz0) +
                get(sol1, x0+z1, sx0*sz1) + get(sol1, x1+z1, sx1*sz1))
        + w011*(get(sol1, y0+z0, sy0*sz0) + get(sol1, y1+z0, sy1*sz0) +
                get(sol1, y0+z1, sy0*sz1) + get(sol1, y1+z1, sy1*sz1)))
        +(w200*(get(sol1, x00,   sx00)    + get(sol1, x11,   sx11))
        + w020*(get(sol1, y00,   sy00)    + get(sol1, y11,   sy11))
        + w002*(get(sol1, z00,   sz00)    + get(sol1, z11,   sz11))) ) / vx1
    );
  }

  {
    scalar_t c = *sol2;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val2 = (*grd2) - (
       (wz100*(get(sol2, x0, sx0) + get(sol2, x1, sx1))
      + wz010*(get(sol2, y0, sy0) + get(sol2, y1, sy1))
      + wz001*(get(sol2, z0, sz0) + get(sol2, z1, sz1)))
      + w2   *( bound::get(sol0, z1+x0, sz1*sx0) - bound::get(sol0, z1+x1, sz1*sx1)
              + bound::get(sol0, z0+x1, sz0*sx1) - bound::get(sol0, z0+x0, sz0*sx0)
              + bound::get(sol1, z1+y0, sz1*sy0) - bound::get(sol1, z1+y1, sz1*sy1)
              + bound::get(sol1, z0+y1, sz0*sy1) - bound::get(sol1, z0+y0, sz0*sy0) )
      + ( absolute*c
        +(w110*(get(sol2, x0+y0, sx0*sy0) + get(sol2, x1+y0, sx1*sy0) +
                get(sol2, x0+y1, sx1*sy1) + get(sol2, x1+y1, sx1*sy1))
        + w101*(get(sol2, x0+z0, sx0*sz0) + get(sol2, x1+z0, sx1*sz0) +
                get(sol2, x0+z1, sx1*sz1) + get(sol2, x1+z1, sx1*sz1))
        + w011*(get(sol2, y0+z0, sy0*sz0) + get(sol2, y1+z0, sy1*sz0) +
                get(sol2, y0+z1, sy1*sz1) + get(sol2, y1+z1, sy1*sz1)))
        +(w200*(get(sol2, x00,   sx00)    + get(sol2, x11,   sx11))
        + w020*(get(sol2, y00,   sy00)    + get(sol2, y11,   sy11))
        + w002*(get(sol2, z00,   sz00)    + get(sol2, z11,   sz11))) ) / vx2 
      );
  }

  invert(hes, sol0, val0, val1, val2, wx000, wy000, wz000);
}

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax3d_lame(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1
  GET_POINTERS
  reduce_t val0, val1, val2;

  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
       (wx100*(get(sol0, x0, sx0) + get(sol0, x1, sx1))
      + wx010*(get(sol0, y0, sy0) + get(sol0, y1, sy1))
      + wx001*(get(sol0, z0, sz0) + get(sol0, z1, sz1)))
      + w2   *( bound::get(sol1, x1+y0, sx1*sy0) - bound::get(sol1, x1+y1, sx1*sy1)
              + bound::get(sol1, x0+y1, sx0*sy1) - bound::get(sol1, x0+y0, sx0*sy0)
              + bound::get(sol2, x1+z0, sx1*sz0) - bound::get(sol2, x1+z1, sx1*sz1)
              + bound::get(sol2, x0+z1, sx0*sz1) - bound::get(sol2, x0+z0, sx0*sz0) )
      + absolute * c / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
       (wy100*(get(sol1, x0, sx0) + get(sol1, x1, sx1))
      + wy010*(get(sol1, y0, sy0) + get(sol1, y1, sy1))
      + wy001*(get(sol1, z0, sz0) + get(sol1, z1, sz1)))
      + w2   *( bound::get(sol0, y1+x0, sy1*sx0) - bound::get(sol0, y1+x1, sy1*sx1)
              + bound::get(sol0, y0+x1, sy0*sx1) - bound::get(sol0, y0+x0, sy0*sx0)
              + bound::get(sol2, y1+z0, sy1*sz0) - bound::get(sol2, y1+z1, sy1*sz1)
              + bound::get(sol2, y0+z1, sy0*sz1) - bound::get(sol2, y0+z0, sy0*sz0) )
      + absolute * c / vx1
    );
  }

  {
    scalar_t c = *sol2;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val2 = (*grd2) - (
       (wz100*(get(sol2, x0, sx0) + get(sol2, x1, sx1))
      + wz010*(get(sol2, y0, sy0) + get(sol2, y1, sy1))
      + wz001*(get(sol2, z0, sz0) + get(sol2, z1, sz1)))
      + w2   *( bound::get(sol0, z1+x0, sz1*sx0) - bound::get(sol0, z1+x1, sz1*sx1)
              + bound::get(sol0, z0+x1, sz0*sx1) - bound::get(sol0, z0+x0, sz0*sx0)
              + bound::get(sol1, z1+y0, sz1*sy0) - bound::get(sol1, z1+y1, sz1*sy1)
              + bound::get(sol1, z0+y1, sz0*sy1) - bound::get(sol1, z0+y0, sz0*sy0) )
      + absolute * c / vx2 
      );
  }

  invert(hes, sol0, val0, val1, val2, wx000, wy000, wz000);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax3d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1 
  GET_SIGN2
  GET_WARP1 
  GET_WARP2
  GET_POINTERS
  reduce_t val0, val1, val2;


  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
        ( absolute*c
        + (w100*(get(sol0, x0,    sx0)     + get(sol0, x1,    sx1))
        +  w010*(get(sol0, y0,    sy0)     + get(sol0, y1,    sy1))
        +  w001*(get(sol0, z0,    sz0)     + get(sol0, z1,    sz1)))
        + (w200*(get(sol0, x00,   sx00)    + get(sol0, x11,   sx11))
        +  w020*(get(sol0, y00,   sy00)    + get(sol0, y11,   sy11))
        +  w002*(get(sol0, z00,   sz00)    + get(sol0, z11,   sz11)))
        + (w110*(get(sol0, x0+y0, sx0*sy0) + get(sol0, x1+y0, sx1*sy0) +
                 get(sol0, x0+y1, sx0*sy1) + get(sol0, x1+y1, sx1*sy1))
        +  w101*(get(sol0, x0+z0, sx0*sz0) + get(sol0, x1+z0, sx1*sz0) +
                 get(sol0, x0+z1, sx0*sz1) + get(sol0, x1+z1, sx1*sz1))
        +  w011*(get(sol0, y0+z0, sy0*sz0) + get(sol0, y1+z0, sy1*sz0) +
                 get(sol0, y0+z1, sy0*sz1) + get(sol0, y1+z1, sy1*sz1))) ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
        ( absolute*c
        + (w100*(get(sol1, x0,    sx0)     + get(sol1, x1,    sx1))
        +  w010*(get(sol1, y0,    sy0)     + get(sol1, y1,    sy1))
        +  w001*(get(sol1, z0,    sz0)     + get(sol1, z1,    sz1)))
        + (w200*(get(sol1, x00,   sx00)    + get(sol1, x11,   sx11))
        +  w020*(get(sol1, y00,   sy00)    + get(sol1, y11,   sy11))
        +  w002*(get(sol1, z00,   sz00)    + get(sol1, z11,   sz11)))
        + (w110*(get(sol1, x0+y0, sx0*sy0) + get(sol1, x1+y0, sx1*sy0) +
                 get(sol1, x0+y1, sx0*sy1) + get(sol1, x1+y1, sx1*sy1))
        +  w101*(get(sol1, x0+z0, sx0*sz0) + get(sol1, x1+z0, sx1*sz0) +
                 get(sol1, x0+z1, sx0*sz1) + get(sol1, x1+z1, sx1*sz1))
        +  w011*(get(sol1, y0+z0, sy0*sz0) + get(sol1, y1+z0, sy1*sz0) +
                 get(sol1, y0+z1, sy0*sz1) + get(sol1, y1+z1, sy1*sz1))) ) / vx1
    );
  }

  {
    scalar_t c = *sol2;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val2 = (*grd2) - (
        ( absolute*c
        + (w100*(get(sol2, x0,    sx0)     + get(sol2, x1,    sx1))
        +  w010*(get(sol2, y0,    sy0)     + get(sol2, y1,    sy1))
        +  w001*(get(sol2, z0,    sz0)     + get(sol2, z1,    sz1)))
        + (w200*(get(sol2, x00,   sx00)    + get(sol2, x11,   sx11))
        +  w020*(get(sol2, y00,   sy00)    + get(sol2, y11,   sy11))
        +  w002*(get(sol2, z00,   sz00)    + get(sol2, z11,   sz11)))
        + (w110*(get(sol2, x0+y0, sx0*sy0) + get(sol2, x1+y0, sx1*sy0) +
                 get(sol2, x0+y1, sx1*sy1) + get(sol2, x1+y1, sx1*sy1))
        +  w101*(get(sol2, x0+z0, sx0*sz0) + get(sol2, x1+z0, sx1*sz0) +
                 get(sol2, x0+z1, sx1*sz1) + get(sol2, x1+z1, sx1*sz1))
        +  w011*(get(sol2, y0+z0, sy0*sz0) + get(sol2, y1+z0, sy1*sz0) +
                 get(sol2, y0+z1, sy1*sz1) + get(sol2, y1+z1, sy1*sz1))) ) / vx2 
      );
  }

  invert(hes, sol0, val0, val1, val2, w000/vx0, w000/vx1, w000/vx2);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax3d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS
  reduce_t val0, val1, val2;

  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
        ( absolute*c
        +(w100*(get(sol0, x0, sx0) + get(sol0, x1, sx1))
        + w010*(get(sol0, y0, sy0) + get(sol0, y1, sy1))
        + w001*(get(sol0, z0, sz0) + get(sol0, z1, sz1))) ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
        ( absolute*c
        +(w100*(get(sol1, x0, sx0) + get(sol1, x1, sx1))
        + w010*(get(sol1, y0, sy0) + get(sol1, y1, sy1))
        + w001*(get(sol1, z0, sz0) + get(sol1, z1, sz1))) ) / vx1
    );
  }

  {
    scalar_t c = *sol2;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val2 = (*grd2) - (
        ( absolute*c
        +(w100*(get(sol2, x0, sx0) + get(sol2, x1, sx1))
        + w010*(get(sol2, y0, sy0) + get(sol2, y1, sy1))
        + w001*(get(sol2, z0, sz0) + get(sol2, z1, sz1))) ) / vx2 
      );
  }

  invert(hes, sol0, val0, val1, val2, w000/vx0, w000/vx1, w000/vx2);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax3d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  reduce_t val0, val1, val2;

  {
    scalar_t c = *sol0;
    val0 = (*grd0) - ( absolute * c / vx0 );
  }
  {
    scalar_t c = *sol1;
    val1 = (*grd1) - ( absolute * c / vx1 );
  }
  {
    scalar_t c = *sol2;
    val2 = (*grd2) - ( absolute * c / vx2 );
  }

  invert(hes, sol0, val0, val1, val2, w000/vx0, w000/vx1, w000/vx2);
}



template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS
  reduce_t val0, val1, val2;

  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ + n*wgt_sN);

  // In `grid` mode, the weight map is single channel
  scalar_t wcenter = *wgt;
  reduce_t w1m00 = w100 * (wcenter + bound::get(wgt, wx0, sx0));
  reduce_t w1p00 = w100 * (wcenter + bound::get(wgt, wx1, sx1));
  reduce_t w01m0 = w010 * (wcenter + bound::get(wgt, wy0, sy0));
  reduce_t w01p0 = w010 * (wcenter + bound::get(wgt, wy1, sy1));
  reduce_t w001m = w001 * (wcenter + bound::get(wgt, wz0, sz0));
  reduce_t w001p = w001 * (wcenter + bound::get(wgt, wz1, sz1));

  {
    scalar_t c = *sol0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };


    val0 = (*grd0) - (
      ( absolute * wcenter * c
      +(w1m00 * get(sol0, x0, sx0)
      + w1p00 * get(sol0, x1, sx1)
      + w01m0 * get(sol0, y0, sy0)
      + w01p0 * get(sol0, y1, sy1)
      + w001m * get(sol0, z0, sz0)
      + w001p * get(sol0, z1, sz1)) * 0.5 ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
      ( absolute * wcenter * c
      +(w1m00 * get(sol1, x0, sx0)
      + w1p00 * get(sol1, x1, sx1)
      + w01m0 * get(sol1, y0, sy0)
      + w01p0 * get(sol1, y1, sy1)
      + w001m * get(sol1, z0, sz0)
      + w001p * get(sol1, z1, sz1)) * 0.5 ) / vx1
    );
  }

  {
    scalar_t c = *sol2;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val2 = (*grd2) - (
      ( absolute * wcenter * c
      +(w1m00 * get(sol2, x0, sx0)
      + w1p00 * get(sol2, x1, sx1)
      + w01m0 * get(sol2, y0, sy0)
      + w01p0 * get(sol2, y1, sy1)
      + w001m * get(sol2, z0, sz0)
      + w001p * get(sol2, z1, sz1)) * 0.5 ) / vx2
    );
  }

  reduce_t w = (absolute * wcenter -  
                0.5 * (w1m00 + w1p00 + w01m0 + w01p0 + w001m + w001p));
  invert(hes, sol0, val0, val1, val2, w/vx0, w/vx1, w/vx2);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  reduce_t val0, val1, val2;
  scalar_t w = *(wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ + n*wgt_sN));
  w *= absolute;
  {
    scalar_t c = *sol0;
    val0 = (*grd0) - ( c * w / vx0 );
  }
  {
    scalar_t c = *sol1;
    val1 = (*grd1) - ( c * w  / vx1 );
  }
  {
    scalar_t c = *sol2;
    val2 = (*grd2) - ( c * w  / vx2 );
  }

  reduce_t wd = absolute * w;
  invert(hes, sol0, val0, val1, val2, wd/vx0, wd/vx1, wd/vx2);
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
#undef  GET_WARP1_RLS
#define GET_WARP1_RLS \
  GET_WARP1_RLS_(x, X, 0) \
  GET_WARP1_RLS_(y, Y, 1)

#undef  GET_POINTERS
#define GET_POINTERS \
  const scalar_t *grd0 = grd_ptr + (x*grd_sX + y*grd_sY + n*grd_sN), \
                 *grd1 = grd0 + grd_sC,                              \
                 *hes  = hes_ptr + (x*hes_sX + y*hes_sY + n*hes_sN); \
        scalar_t *sol0 = sol_ptr + (x*sol_sX + y*sol_sY + n*sol_sN), \
                 *sol1 = sol0 + sol_sC;

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax2d_all(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1   // Sign (/!\ compute sign before warping indices)
  GET_SIGN2
  GET_WARP1   // Warp indices
  GET_WARP2
  GET_POINTERS

  reduce_t val0, val1;

  // For numerical stability, we subtract the center value before convolving.
  // We define a lambda function for ease.

  {
    scalar_t c = *sol0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
       (wx100*(get(sol0, x0, sx0) + get(sol0, x1, sx1))
      + wx010*(get(sol0, y0, sy0) + get(sol0, y1, sy1)))
      + w2   *( bound::get(sol1, x1+y0, sx1*sy0) - bound::get(sol1, x1+y1, sx1*sy1)
              + bound::get(sol1, x0+y1, sx0*sy1) - bound::get(sol1, x0+y0, sx0*sy0) )
      + ( absolute*c
        +(w110*(get(sol0, x0+y0, sx0*sy0) + get(sol0, x1+y0, sx1*sy0) +
                get(sol0, x0+y1, sx0*sy1) + get(sol0, x1+y1, sx1*sy1)))
        +(w200*(get(sol0, x00,   sx00)    + get(sol0, x11,   sx11))
        + w020*(get(sol0, y00,   sy00)    + get(sol0, y11,   sy11))) ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
       (wy100*(get(sol1, x0, sx0) + get(sol1, x1, sx1))
      + wy010*(get(sol1, y0, sy0) + get(sol1, y1, sy1)))
      + w2   *( bound::get(sol0, y1+x0, sy1*sx0) - bound::get(sol0, y1+x1, sy1*sx1)
              + bound::get(sol0, y0+x1, sy0*sx1) - bound::get(sol0, y0+x0, sy0*sx0) )
      + ( absolute*c
        +(w110*(get(sol1, x0+y0, sx0*sy0) + get(sol1, x1+y0, sx1*sy0) +
                get(sol1, x0+y1, sx0*sy1) + get(sol1, x1+y1, sx1*sy1)))
        +(w200*(get(sol1, x00,   sx00)    + get(sol1, x11,   sx11))
        + w020*(get(sol1, y00,   sy00)    + get(sol1, y11,   sy11))) ) / vx1
    );
  }

  invert(hes, sol0, val0, val1, 0, wx000, wy000, 0);
}

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax2d_lame(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1
  GET_POINTERS
  reduce_t val0, val1;

  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
       (wx100*(get(sol0, x0, sx0) + get(sol0, x1, sx1))
      + wx010*(get(sol0, y0, sy0) + get(sol0, y1, sy1)))
      + w2   *( bound::get(sol1, x1+y0, sx1*sy0) - bound::get(sol1, x1+y1, sx1*sy1)
              + bound::get(sol1, x0+y1, sx0*sy1) - bound::get(sol1, x0+y0, sx0*sy0) )
      + absolute * c / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
       (wy100*(get(sol1, x0, sx0) + get(sol1, x1, sx1))
      + wy010*(get(sol1, y0, sy0) + get(sol1, y1, sy1)))
      + w2   *( bound::get(sol0, y1+x0, sy1*sx0) - bound::get(sol0, y1+x1, sy1*sx1)
              + bound::get(sol0, y0+x1, sy0*sx1) - bound::get(sol0, y0+x0, sy0*sx0) )
      + absolute * c / vx1
    );
  }

  invert(hes, sol0, val0, val1, 0, wx000, wy000, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax2d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1 
  GET_SIGN2
  GET_WARP1 
  GET_WARP2
  GET_POINTERS
  reduce_t val0, val1;


  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
        ( absolute*c
        + (w100*(get(sol0, x0,    sx0)     + get(sol0, x1,    sx1))
        +  w010*(get(sol0, y0,    sy0)     + get(sol0, y1,    sy1)))
        + (w200*(get(sol0, x00,   sx00)    + get(sol0, x11,   sx11))
        +  w020*(get(sol0, y00,   sy00)    + get(sol0, y11,   sy11)))
        + (w110*(get(sol0, x0+y0, sx0*sy0) + get(sol0, x1+y0, sx1*sy0) +
                 get(sol0, x0+y1, sx0*sy1) + get(sol0, x1+y1, sx1*sy1))) ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
        ( absolute*c
        + (w100*(get(sol1, x0,    sx0)     + get(sol1, x1,    sx1))
        +  w010*(get(sol1, y0,    sy0)     + get(sol1, y1,    sy1)))
        + (w200*(get(sol1, x00,   sx00)    + get(sol1, x11,   sx11))
        +  w020*(get(sol1, y00,   sy00)    + get(sol1, y11,   sy11)))
        + (w110*(get(sol1, x0+y0, sx0*sy0) + get(sol1, x1+y0, sx1*sy0) +
                 get(sol1, x0+y1, sx0*sy1) + get(sol1, x1+y1, sx1*sy1))) ) / vx1
    );
  }

  invert(hes, sol0, val0, val1, 0, w000/vx0, w000/vx1, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax2d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS
  reduce_t val0, val1;

  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
        ( absolute*c
        +(w100*(get(sol0, x0, sx0) + get(sol0, x1, sx1))
        + w010*(get(sol0, y0, sy0) + get(sol0, y1, sy1))) ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
        ( absolute*c
        +(w100*(get(sol1, x0, sx0) + get(sol1, x1, sx1))
        + w010*(get(sol1, y0, sy0) + get(sol1, y1, sy1))) ) / vx1
    );
  }

  invert(hes, sol0, val0, val1, 0, w000/vx0, w000/vx1, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax2d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  reduce_t val0, val1;

  {
    scalar_t c = *sol0;
    val0 = (*grd0) - ( absolute * c / vx0 );
  }
  {
    scalar_t c = *sol1;
    val1 = (*grd1) - ( absolute * c / vx1 );
  }

  invert(hes, sol0, val0, val1, 0, w000/vx0, w000/vx1, 0);
}



template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax2d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS
  reduce_t val0, val1;

  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + n*wgt_sN);

  // In `grid` mode, the weight map is single channel
  scalar_t wcenter = *wgt;
  reduce_t w1m00 = w100 * (wcenter + bound::get(wgt, wx0, sx0));
  reduce_t w1p00 = w100 * (wcenter + bound::get(wgt, wx1, sx1));
  reduce_t w01m0 = w010 * (wcenter + bound::get(wgt, wy0, sy0));
  reduce_t w01p0 = w010 * (wcenter + bound::get(wgt, wy1, sy1));

  {
    scalar_t c = *sol0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };


    val0 = (*grd0) - (
      ( absolute * wcenter * c
      +(w1m00 * get(sol0, x0, sx0)
      + w1p00 * get(sol0, x1, sx1)
      + w01m0 * get(sol0, y0, sy0)
      + w01p0 * get(sol0, y1, sy1)) * 0.5 ) / vx0
    );
  }

  {
    scalar_t c = *sol1;
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val1 = (*grd1) - (
      ( absolute * wcenter * c
      +(w1m00 * get(sol1, x0, sx0)
      + w1p00 * get(sol1, x1, sx1)
      + w01m0 * get(sol1, y0, sy0)
      + w01p0 * get(sol1, y1, sy1)) * 0.5 ) / vx1
    );
  }

  reduce_t w = (absolute * wcenter - 0.5 * (w1m00 + w1p00 + w01m0 + w01p0));
  invert(hes, sol0, val0, val1, 0, w/vx0, w/vx1, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax2d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  reduce_t val0, val1;
  scalar_t w = *(wgt_ptr + (x*wgt_sX + y*wgt_sY + n*wgt_sN));
  w *= absolute;
  {
    scalar_t c = *sol0;
    val0 = (*grd0) - ( c * w / vx0 );
  }
  {
    scalar_t c = *sol1;
    val1 = (*grd1) - ( c * w  / vx1 );
  }

  reduce_t wd = absolute * w;
  invert(hes, sol0, val0, val1, 0, wd/vx0, wd/vx1, 0);
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
#undef  GET_WARP1_RLS
#define GET_WARP1_RLS GET_WARP1_RLS_(x, X, 0)

#undef  GET_POINTERS
#define GET_POINTERS \
  const scalar_t *grd0 = grd_ptr + (x*grd_sX + n*grd_sN), \
                 *hes  = hes_ptr + (x*hes_sX + n*hes_sN); \
        scalar_t *sol0 = sol_ptr + (x*sol_sX + n*sol_sN);

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax1d_all(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1   // Sign (/!\ compute sign before warping indices)
  GET_SIGN2
  GET_WARP1   // Warp indices
  GET_WARP2
  GET_POINTERS

  reduce_t val0;

  // For numerical stability, we subtract the center value before convolving.
  // We define a lambda function for ease.

  {
    scalar_t c = *sol0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
       (wx100*(get(sol0, x0, sx0) + get(sol0, x1, sx1)))
      + ( absolute*c
        +(w200*(get(sol0, x00,   sx00)    + get(sol0, x11,   sx11))) ) / vx0
    );
  }

  invert(hes, sol0, val0, 0, 0, wx000, 0, 0);
}

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax1d_lame(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1
  GET_POINTERS
  reduce_t val0;

  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
       (wx100*(get(sol0, x0, sx0) + get(sol0, x1, sx1)))
      + absolute * c / vx0
    );
  }

  invert(hes, sol0, val0, 0, 0, wx000, 0, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax1d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1 
  GET_SIGN2
  GET_WARP1 
  GET_WARP2
  GET_POINTERS
  reduce_t val0;


  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
        ( absolute*c
        + (w100*(get(sol0, x0,    sx0)     + get(sol0, x1,    sx1)))
        + (w200*(get(sol0, x00,   sx00)    + get(sol0, x11,   sx11))) ) / vx0
    );
  }

  invert(hes, sol0, val0, 0, 0, w000/vx0, 0, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax1d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS
  reduce_t val0;

  {
    scalar_t c = *sol0; 
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    val0 = (*grd0) - (
        ( absolute*c
        +(w100*(get(sol0, x0, sx0) + get(sol0, x1, sx1))) ) / vx0
    );
  }

  invert(hes, sol0, val0, 0, 0, w000/vx0, 0, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax1d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  reduce_t val0;

  {
    scalar_t c = *sol0;
    val0 = (*grd0) - ( absolute * c / vx0 );
  }

  invert(hes, sol0, val0, 0, 0, w000/vx0, 0, 0);
}



template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax1d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS
  reduce_t val0;

  scalar_t * wgt = wgt_ptr + (x*wgt_sX + n*wgt_sN);

  // In `grid` mode, the weight map is single channel
  scalar_t wcenter = *wgt;
  reduce_t w1m00 = w100 * (wcenter + bound::get(wgt, wx0, sx0));
  reduce_t w1p00 = w100 * (wcenter + bound::get(wgt, wx1, sx1));

  {
    scalar_t c = *sol0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };


    val0 = (*grd0) - (
      ( absolute * wcenter * c
      +(w1m00 * get(sol0, x0, sx0)
      + w1p00 * get(sol0, x1, sx1)) * 0.5 ) / vx0
    );
  }

  reduce_t w = (absolute * wcenter - 0.5 * (w1m00 + w1p00));
  invert(hes, sol0, val0, 0, 0, w/vx0, 0, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::relax1d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  reduce_t val0;
  scalar_t w = *(wgt_ptr + (x*wgt_sX + n*wgt_sN));
  w *= absolute;
  {
    scalar_t c = *sol0;
    val0 = (*grd0) - ( c * w / vx0 );
  }

  reduce_t wd = absolute * w;
  invert(hes, sol0, val0, 0, 0, wd/vx0, 0, 0);
}


/* ========================================================================== */
/*                                     SOLVE                                  */
/* ========================================================================== */


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::solve1d(offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN),          
                 *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN);

  invert(hes, sol, grd[0], grd[grd_sC], grd[2*grd_sC], 0, 0, 0);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::solve2d(offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + n*grd_sN),
                 *hes = hes_ptr + (x*hes_sX + y*hes_sY + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + n*sol_sN);

  invert(hes, sol, grd[0], grd[grd_sC], 0, 0, 0, 0);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RelaxGridImpl<scalar_t,offset_t,reduce_t>::solve3d(offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  const scalar_t *grd = grd_ptr + (x*grd_sX + n*grd_sN),                 
                 *hes = hes_ptr + (x*hes_sX + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + n*sol_sN);

  invert(hes, sol, grd[0], 0, 0, 0, 0, 0);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t, typename reduce_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void relax_kernel(RelaxGridImpl<scalar_t,offset_t,reduce_t> f) {
  f.loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
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
    else if (dim == 2)
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
//                    FUNCTIONAL FORM WITH relax
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST Tensor relax_grid_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  ArrayRef<double> voxel_size, BoundVectorRef bound, int64_t nb_iter)
{
  auto tensors = prepare_tensors(gradient, hessian, solution, weight);
  hessian  = std::get<0>(tensors);
  solution = std::get<1>(tensors);
  weight   = std::get<2>(tensors);

  RelaxGridAllocator info(gradient.dim()-2, absolute, membrane, bending,
                      lame_shear, lame_div, voxel_size, bound);
  info.ioset(hessian, gradient, solution, weight);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradient.scalar_type(), "relax_grid_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      RelaxGridImpl<scalar_t, int32_t, double> algo(info);
      for (int32_t i=0; i < nb_iter; ++i)
        for (int32_t fold = 0; fold < algo.foldcount(); ++fold) {
            algo.set_fold(fold);
            relax_kernel<<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
        }
    }
    else
    {
      RelaxGridImpl<scalar_t, int64_t, double> algo(info);
      for (int64_t i=0; i < nb_iter; ++i)
        for (int64_t fold = 0; fold < algo.foldcount(); ++fold) {
            algo.set_fold(fold);
            relax_kernel<<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
        }
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

  return solution;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST Tensor relax_grid_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  ArrayRef<double> voxel_size, BoundVectorRef bound, int64_t nb_iter)
{
  auto tensors = prepare_tensors(gradient, hessian, solution, weight);
  hessian  = std::get<0>(tensors);
  solution = std::get<1>(tensors);
  weight   = std::get<2>(tensors);

  RelaxGridAllocator info(gradient.dim()-2, absolute, membrane, bending,
                      lame_shear, lame_div, voxel_size, bound);
  info.ioset(hessian, gradient, solution, weight);

  AT_DISPATCH_FLOATING_TYPES(gradient.scalar_type(), "relax_grid_impl", [&] {
    RelaxGridImpl<scalar_t, int64_t, double> algo(info);
    for (int64_t i=0; i < nb_iter; ++i)
      algo.loop();
  });
  return solution;
}

#endif // __CUDACC__

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST Tensor relax_grid_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  ArrayRef<double> voxel_size, BoundVectorRef bound, int64_t nb_iter)
{
  throw std::logic_error("Function not implemented for this device.");
}


} // namespace notimplemented

} // namespace ni

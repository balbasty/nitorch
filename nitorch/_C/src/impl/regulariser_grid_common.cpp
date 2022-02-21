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

// Macro to cleanly invoke a pointer to member function
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
#define MIN(a,b) (a < b ? a : b)

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
class RegulariserGridAllocator: public Allocator {
public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST
  RegulariserGridAllocator(int dim, 
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
  }

  /* ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST void ioset
  (const Tensor& input, const Tensor& output, const Tensor& weight, const Tensor& hessian)
  {
    init_all();
    init_input(input);
    init_weight(weight);
    init_output(output);
    init_hessian(hessian);
  }

  // We just check that all tensors that we own are compatible with 32b math
  bool canUse32BitIndexMath(int64_t max_elem=max_int32) const
  {
    return inp_32b_ok && wgt_32b_ok && out_32b_ok;
  }

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  NI_HOST void init_all();
  NI_HOST void init_input(const Tensor&);
  NI_HOST void init_weight(const Tensor&);
  NI_HOST void init_output(const Tensor&);
  NI_HOST void init_hessian(const Tensor&);

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
  DEFINE_ALLOC_INFO_5D(inp)
  DEFINE_ALLOC_INFO_5D(wgt)
  DEFINE_ALLOC_INFO_5D(out)
  DEFINE_ALLOC_INFO_5D(hes)

  // Allow RegulariserGridImpl's constructor to access RegulariserGridAllocator's
  // private members.
  template <typename scalar_t, typename offset_t, typename reduce_t>
  friend class RegulariserGridImpl;
};


NI_HOST
void RegulariserGridAllocator::init_all()
{
  N = C = CC = X = Y = Z = 1L;
  inp_sN  = inp_sC  = inp_sX  = inp_sY  = inp_sZ   = 0L;
  wgt_sN  = wgt_sC  = wgt_sX  = wgt_sY  = wgt_sZ   = 0L;
  out_sN  = out_sC  = out_sX  = out_sY  = out_sZ   = 0L;
  hes_sN  = hes_sC  = hes_sX  = hes_sY  = hes_sZ   = 0L;
  inp_ptr = wgt_ptr = out_ptr = hes_ptr = static_cast<float*>(0);
  inp_32b_ok = wgt_32b_ok = out_32b_ok = hes_32b_ok = true;
}

NI_HOST
void RegulariserGridAllocator::init_input(const Tensor& input)
{
  N       = input.size(0);
  C       = input.size(1);
  X       = input.size(2);
  Y       = dim < 2 ? 1L : input.size(3);
  Z       = dim < 3 ? 1L : input.size(4);
  inp_sN  = input.stride(0);
  inp_sC  = input.stride(1);
  inp_sX  = input.stride(2);
  inp_sY  = dim < 2 ? 0L : input.stride(3);
  inp_sZ  = dim < 3 ? 0L : input.stride(4);
  inp_ptr = input.data_ptr();
  inp_32b_ok = tensorCanUse32BitIndexMath(input);
}

NI_HOST
void RegulariserGridAllocator::init_weight(const Tensor& weight)
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

NI_HOST
void RegulariserGridAllocator::init_output(const Tensor& output)
{
  out_sN   = output.stride(0);
  out_sC   = output.stride(1);
  out_sX   = output.stride(2);
  out_sY   = dim < 2 ? 0L : output.stride(3);
  out_sZ   = dim < 3 ? 0L : output.stride(4);
  out_ptr  = output.data_ptr();
  out_32b_ok = tensorCanUse32BitIndexMath(output);
}

NI_HOST
void RegulariserGridAllocator::init_hessian(const Tensor& hessian)
{
  if (!hessian.defined() || hessian.numel() == 0)
    return;
  CC      = hessian.size(1);
  hes_sN  = hessian.stride(0);
  hes_sC  = hessian.stride(1);
  hes_sX  = hessian.stride(2);
  hes_sY  = dim < 2 ? 0L : hessian.stride(3);
  hes_sZ  = dim < 3 ? 0L : hessian.stride(4);
  hes_ptr = hessian.data_ptr();
  hes_32b_ok = tensorCanUse32BitIndexMath(hessian);
}

/* ========================================================================== */
/*                                                                            */
/*                                ALGORITHM                                   */
/*                                                                            */
/* ========================================================================== */
template <typename scalar_t, typename offset_t, typename reduce_t>
class RegulariserGridImpl {

  typedef RegulariserGridImpl Self;
  typedef void (Self::*Vel2MomFn)(offset_t x, offset_t y, offset_t z, offset_t n) const;
  typedef void (Self::*MatVecFn)(const scalar_t *, const scalar_t *, scalar_t *) const;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  RegulariserGridImpl(const RegulariserGridAllocator & info):
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

    INIT_ALLOC_INFO_5D(inp),
    INIT_ALLOC_INFO_5D(wgt),
    INIT_ALLOC_INFO_5D(out),
    INIT_ALLOC_INFO_5D(hes)
  {
    set_kernel();
#ifndef __CUDACC__
    set_vel2mom();
    set_matvec();
#endif
  }

  NI_HOST NI_INLINE void set_kernel() 
  {
    mode = dim 
         + (bending ? 12 : membrane ? 8 : absolute ? 4 : 0)
         + (lame_shear || lame_div ? 16 : 0)
         + (wgt_ptr ? 32 : 0);

    double lam0 = absolute, lam1 = membrane, lam2 = bending, mu = lame_shear, lam = lame_div;

    w000 = lam2*(6.0*(vx0*vx0+vx1*vx1+vx2*vx2) + 8*(vx0*vx1+vx0*vx2+vx1*vx2)) + lam1*2*(vx0+vx1+vx2) + lam0;
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

    // TODO: correct for 1d/2d cases

    wx000 *= OnePlusTiny;
    wy000 *= OnePlusTiny;
    wz000 *= OnePlusTiny;
  }

#ifndef __CUDACC__
  NI_HOST NI_INLINE void set_vel2mom() 
  {
    if (wgt_ptr)
    {
      if (bending || lame_div || lame_shear)
        throw std::logic_error("RLS only implemented for absolute/membrane.");
      else if (dim == 1) {
        if (membrane)
            vel2mom = &Self::vel2mom1d_rls_membrane;
        else if (absolute)
            vel2mom = &Self::vel2mom1d_rls_absolute;
        else
            vel2mom = &Self::zeros;
      } else if (dim == 2) {
        if (membrane)
            vel2mom = &Self::vel2mom2d_rls_membrane;
        else if (absolute)
            vel2mom = &Self::vel2mom2d_rls_absolute;
        else
            vel2mom = &Self::zeros;
      } else if (dim == 3) {
        if (membrane)
            vel2mom = &Self::vel2mom3d_rls_membrane;
        else if (absolute)
            vel2mom = &Self::vel2mom3d_rls_absolute;
        else
            vel2mom = &Self::zeros;
      }
    }
    else if (dim == 1) {
        if ((lame_shear or lame_div) and bending)
            vel2mom = &Self::vel2mom1d_all;
        else if (lame_shear or lame_div)
            vel2mom = &Self::vel2mom1d_lame;
        else if (bending)
            vel2mom = &Self::vel2mom1d_bending;
        else if (membrane)
            vel2mom = &Self::vel2mom1d_membrane;
        else if (absolute)
            vel2mom = &Self::vel2mom1d_absolute;
        else
            vel2mom = &Self::zeros;
    } else if (dim == 2) {
        if ((lame_shear or lame_div) and bending)
            vel2mom = &Self::vel2mom2d_all;
        else if (lame_shear or lame_div)
            vel2mom = &Self::vel2mom2d_lame;
        else if (bending)
            vel2mom = &Self::vel2mom2d_bending;
        else if (membrane)
            vel2mom = &Self::vel2mom2d_membrane;
        else if (absolute)
            vel2mom = &Self::vel2mom2d_absolute;
        else
            vel2mom = &Self::zeros;
    } else if (dim == 3) {
        if ((lame_shear or lame_div) and bending)
            vel2mom = &Self::vel2mom3d_all;
        else if (lame_shear or lame_div)
            vel2mom = &Self::vel2mom3d_lame;
        else if (bending)
            vel2mom = &Self::vel2mom3d_bending;
        else if (membrane)
            vel2mom = &Self::vel2mom3d_membrane;
        else if (absolute)
            vel2mom = &Self::vel2mom3d_absolute;
        else
            vel2mom = &Self::zeros;
    } else
        throw std::logic_error("RLS only implemented for dimension 1/2/3.");
  }

  NI_HOST NI_INLINE void set_matvec() 
  {
    if (hes_ptr) {
      if (CC == 1) {
        if (dim == 1)
          matvec_ = &Self::matvec1d;
        else if (dim == 2)
          matvec_ = &Self::matvec2d_eye;
        else
          matvec_ = &Self::matvec3d_eye;
      } else if (CC == C) {
        if (dim == 1)
          matvec_ = &Self::matvec1d;
        else if (dim == 2)
          matvec_ = &Self::matvec2d_diag;
        else
          matvec_ = &Self::matvec3d_diag;
      } else {
        if (dim == 1)
          matvec_ = &Self::matvec1d;
        else if (dim == 2)
          matvec_ = &Self::matvec2d_sym;
        else
          matvec_ = &Self::matvec3d_sym;
      }
    }
    else
      matvec_ = &Self::matvec_none;
  }
#endif

  /* ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#if __CUDACC__
  // Loop over voxels that belong to one CUDA block
  // This function is called by the CUDA kernel
  NI_DEVICE void loop(int threadIdx, int blockIdx,
                      int blockDim, int gridDim) const;
#else
  // Loop over all voxels
  void loop() const;
#endif

  NI_HOST NI_INLINE int64_t voxcount() const { return N * X * Y * Z; }
 

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  NI_DEVICE NI_INLINE void dispatch(
    offset_t x, offset_t y, offset_t z, offset_t n) const;
  NI_DEVICE NI_INLINE void zeros(
    offset_t x, offset_t y, offset_t z, offset_t n) const;

#define DEFINE_VEL2MOM(SUFFIX) \
  NI_DEVICE void vel2mom##SUFFIX( \
    offset_t x, offset_t y, offset_t z, offset_t n) const;
#define DEFINE_VEL2MOM_DIM(DIM)      \
  DEFINE_VEL2MOM(DIM##d_absolute)  \
  DEFINE_VEL2MOM(DIM##d_membrane)  \
  DEFINE_VEL2MOM(DIM##d_bending)   \
  DEFINE_VEL2MOM(DIM##d_lame)      \
  DEFINE_VEL2MOM(DIM##d_all)       \
  DEFINE_VEL2MOM(DIM##d_rls_absolute)  \
  DEFINE_VEL2MOM(DIM##d_rls_membrane)

  DEFINE_VEL2MOM_DIM(1)
  DEFINE_VEL2MOM_DIM(2)
  DEFINE_VEL2MOM_DIM(3)


NI_DEVICE void matvec(const scalar_t *, const scalar_t *, scalar_t *) const;

#define DEFINE_MATVEC(SUFFIX) \
  NI_DEVICE void matvec##SUFFIX(const scalar_t *, const scalar_t *, scalar_t *) const;
#define DEFINE_MATVEC_DIM(DIM)        \
  DEFINE_MATVEC(DIM##d_sym)           \
  DEFINE_MATVEC(DIM##d_diag)          \
  DEFINE_MATVEC(DIM##d_eye)

  DEFINE_MATVEC(1d)
  DEFINE_MATVEC(_none)
  DEFINE_MATVEC_DIM(2)
  DEFINE_MATVEC_DIM(3)

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
  Vel2MomFn         vel2mom;        // Pointer to vel2mom function
  MatVecFn          matvec_;        // Pointer to matvec function

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
  DEFINE_STRIDE_INFO_5D(inp)
  DEFINE_STRIDE_INFO_5D(wgt)
  DEFINE_STRIDE_INFO_5D(out)
  DEFINE_STRIDE_INFO_5D(hes)
};


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                    LOOP                                    */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::dispatch(
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
      return vel2mom1d_rls_membrane(x, y, z, n);
    case 2 + MEMBRANE + RLS:
      return vel2mom2d_rls_membrane(x, y, z, n);
    case 3 + MEMBRANE + RLS:
      return vel2mom3d_rls_membrane(x, y, z, n);
    case 1 + ABSOLUTE + RLS:
      return vel2mom1d_rls_absolute(x, y, z, n);
    case 2 + ABSOLUTE + RLS:
      return vel2mom2d_rls_absolute(x, y, z, n);
    case 3 + ABSOLUTE + RLS:
      return vel2mom3d_rls_absolute(x, y, z, n);
    case 1 + BENDING + LAME:
      return vel2mom1d_all(x, y, z, n);
    case 2 + BENDING + LAME:
      return vel2mom2d_all(x, y, z, n);
    case 3 + BENDING + LAME:
      return vel2mom3d_all(x, y, z, n);
    case 1 + BENDING:
      return vel2mom1d_bending(x, y, z, n);
    case 2 + BENDING:
      return vel2mom2d_bending(x, y, z, n);
    case 3 + BENDING:
      return vel2mom3d_bending(x, y, z, n);
    case 1 + LAME:
      return vel2mom1d_lame(x, y, z, n);
    case 2 + LAME:
      return vel2mom2d_lame(x, y, z, n);
    case 3 + LAME:
      return vel2mom3d_lame(x, y, z, n);
    case 1 + MEMBRANE:
      return vel2mom1d_membrane(x, y, z, n);
    case 2 + MEMBRANE:
      return vel2mom2d_membrane(x, y, z, n);
    case 3 + MEMBRANE:
      return vel2mom3d_membrane(x, y, z, n);
    case 1 + ABSOLUTE:
      return vel2mom1d_absolute(x, y, z, n);
    case 2 + ABSOLUTE:
      return vel2mom2d_absolute(x, y, z, n);
    case 3 + ABSOLUTE:
      return vel2mom3d_absolute(x, y, z, n);
    default:
      return zeros(x, y, z, n);
  }
#else
    CALL_MEMBER_FN(*this, vel2mom)(x, y, z, n);
#endif
}


#if __CUDACC__

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::loop(
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
      dispatch(x, y, z, n);
  }
}

#else

// This bit loops over all target voxels. We therefore need to
// convert linear indices to multivariate indices. The way I do it
// might not be optimal.
template <typename scalar_t, typename offset_t, typename reduce_t> NI_HOST
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::loop() const
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
      dispatch(x, y, z, n);
    }
  });
}

#endif


/* ========================================================================== */
/*                                   MATVEC                                   */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
#ifdef __CUDACC__
  if (hes_ptr == 0) return matvec_none(h, x, s);
  if (CC == 1)
    switch (dim) {
      case 3:  return matvec3d_eye(h, x, s);
      case 2:  return matvec2d_eye(h, x, s);
      case 1:  return matvec1d(h, x, s);
      default: return matvec3d_eye(h, x, s);
    }
  else if (CC == C)
    switch (dim) {
      case 3:  return matvec3d_diag(h, x, s);
      case 2:  return matvec2d_diag(h, x, s);
      case 1:  return matvec1d(h, x, s);
      default: return matvec3d_diag(h, x, s);
    }
  else
    switch (dim) {
      case 3:  return matvec3d_sym(h, x, s);
      case 2:  return matvec2d_sym(h, x, s);
      case 1:  return matvec1d(h, x, s);
      default: return matvec3d_sym(h, x, s);
    }
#else
    CALL_MEMBER_FN(*this, matvec_)(h, x, s);
#endif
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec3d_sym(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
  reduce_t h00 = h[0],        h11 = h[  hes_sC], h22 = h[2*hes_sC],
           h01 = h[3*hes_sC], h02 = h[4*hes_sC], h12 = h[5*hes_sC],
           x0  = x[0],        x1  = x[  inp_sC], x2  = x[2*inp_sC],
           tmp;

  tmp = *s; tmp = std::fma(x0, h00, tmp); tmp = std::fma(x1, h01, tmp); tmp = std::fma(x2, h02, tmp); *s = tmp; s += out_sC;
  tmp = *s; tmp = std::fma(x0, h01, tmp); tmp = std::fma(x1, h11, tmp); tmp = std::fma(x2, h12, tmp); *s = tmp; s += out_sC;
  tmp = *s; tmp = std::fma(x0, h02, tmp); tmp = std::fma(x1, h11, tmp); tmp = std::fma(x2, h22, tmp); *s = tmp;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec2d_sym(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
  reduce_t h00 = h[0], h11 = h[hes_sC], h01 = h[2*hes_sC],
           x0  = x[0], x1  = x[inp_sC], tmp;

  tmp = *s; tmp = std::fma(x0, h00, tmp); tmp = std::fma(x1, h01, tmp); *s = tmp; s += out_sC;
  tmp = *s; tmp = std::fma(x0, h01, tmp); tmp = std::fma(x1, h11, tmp); *s = tmp;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec3d_diag(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
  reduce_t h00 = h[0], h11 = h[hes_sC], h22 = h[2*hes_sC],
           x0  = x[0], x1  = x[inp_sC], x2  = x[2*inp_sC];

  *s += x0*h00; s += out_sC;
  *s += x1*h11; s += out_sC;
  *s += x2*h22;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec2d_diag(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
  reduce_t h00 = h[0], h11 = h[hes_sC],
           x0  = x[0], x1  = x[inp_sC];

  s[       0] += x0*h00;
  s[  out_sC] += x1*h11;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec3d_eye(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
  reduce_t h00 = *h, x0  = x[0], x1  = x[inp_sC], x2  = x[2*inp_sC];

  *s += h00 * x0; s += out_sC;
  *s += h00 * x1; s += out_sC;
  *s += h00 * x2;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec2d_eye(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
  reduce_t h00 = *h, x0  = x[0], x1  = x[inp_sC];

  s[       0] += h00 * x0;
  s[  out_sC] += h00 * x1;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec1d(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{
  (*s) += (*h) * (*x);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::matvec_none(
    const scalar_t * h, const scalar_t * x, scalar_t * s) const 
{}

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
  x##0  = (bound::index(bound##i, x##0,  X) - x) * inp_s##X; \
  x##1  = (bound::index(bound##i, x##1,  X) - x) * inp_s##X;
#define GET_WARP2_(x, X, i)  \
  x##00  = (bound::index(bound##i, x##00,  X) - x) * inp_s##X; \
  x##11  = (bound::index(bound##i, x##11,  X) - x) * inp_s##X;
#define GET_WARP1_RLS_(x, X, i) \
  x##0  = (bound::index(bound##i, x##0,  X) - x); \
  x##1  = (bound::index(bound##i, x##1,  X) - x); \
  offset_t w##x##0 = x##0 * wgt_s##X; \
  offset_t w##x##1 = x##1 * wgt_s##X; \
  x##0 *= inp_s##X; \
  x##1 *= inp_s##X;


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
  scalar_t *out0 = out_ptr + (x*out_sX + y*out_sY + z*out_sZ + n*out_sN);  \
  scalar_t *out1 = out0 + out_sC, *out2 = out0 + 2 * out_sC;    \
  scalar_t *inp0 = inp_ptr + (x*inp_sX + y*inp_sY + z*inp_sZ);  \
  scalar_t *inp1 = inp0 + inp_sC, *inp2 = inp0 + 2 * inp_sC + n*inp_sN; \
  scalar_t *hes0 = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ);

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_all(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_COORD2
  GET_SIGN1   // Sign (/!\ compute sign before warping indices)
  GET_SIGN2
  GET_WARP1   // Warp indices
  GET_WARP2
  GET_POINTERS

  // For numerical stability, we subtract the center value before convolving.
  // We define a lambda function for ease.

  {
    scalar_t c = *inp0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out0 = static_cast<scalar_t>(
        (wx100*(get(inp0, x0, sx0) + get(inp0, x1, sx1))
      +  wx010*(get(inp0, y0, sy0) + get(inp0, y1, sy1))
      +  wx001*(get(inp0, z0, sz0) + get(inp0, z1, sz1)))
      + (w2   *( bound::get(inp1, x1+y0, sx1*sy0) - bound::get(inp1, x1+y1, sx1*sy1)
               + bound::get(inp1, x0+y1, sx0*sy1) - bound::get(inp1, x0+y0, sx0*sy0)
               + bound::get(inp2, x1+z0, sx1*sz0) - bound::get(inp2, x1+z1, sx1*sz1)
               + bound::get(inp2, x0+z1, sx0*sz1) - bound::get(inp2, x0+z0, sx0*sz0) ))
      + ( absolute*c
        + (w110*(get(inp0, x0+y0, sx0*sy0) + get(inp0, x1+y0, sx1*sy0) +
                 get(inp0, x0+y1, sx1*sy1) + get(inp0, x1+y1, sx1*sy1))
        +  w101*(get(inp0, x0+z0, sx0*sz0) + get(inp0, x1+z0, sx1*sz0) +
                 get(inp0, x0+z1, sx1*sz1) + get(inp0, x1+z1, sx1*sz1))
        +  w011*(get(inp0, y0+z0, sy0*sz0) + get(inp0, y1+z0, sy1*sz0) +
                 get(inp0, y0+z1, sy1*sz1) + get(inp0, y1+z1, sy1*sz1)))
        + (w200*(get(inp0, x00, sx00) + get(inp0, x11, sx11))
        +  w020*(get(inp0, y00, sy00) + get(inp0, y11, sy11))
        +  w002*(get(inp0, z00, sz00) + get(inp0, z11, sz11))) ) / vx0
    );
  }

  {
    scalar_t c = *inp1;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out1 = static_cast<scalar_t>(
        (wy100*(get(inp1, x0, sx0) + get(inp1, x1, sx1))
      +  wy010*(get(inp1, y0, sy0) + get(inp1, y1, sy1))
      +  wy001*(get(inp1, z0, sz0) + get(inp1, z1, sz1)))
      + (w2   *( bound::get(inp0, y1+x0, sy1*sx0) - bound::get(inp0, y1+x1, sy1*sx1)
               + bound::get(inp0, y0+x1, sy0*sx1) - bound::get(inp0, y0+x0, sy0*sx0)
               + bound::get(inp2, y1+z0, sy1*sz0) - bound::get(inp2, y1+z1, sy1*sz1)
               + bound::get(inp2, y0+z1, sy0*sz1) - bound::get(inp2, y0+z0, sy0*sz0) ))
      + ( absolute*c
        + (w110*(get(inp1, x0+y0, sx0*sy0) + get(inp1, x1+y0, sx1*sy0) +
                 get(inp1, x0+y1, sx1*sy1) + get(inp1, x1+y1, sx1*sy1))
        +  w101*(get(inp1, x0+z0, sx0*sz0) + get(inp1, x1+z0, sx1*sz0) +
                 get(inp1, x0+z1, sx1*sz1) + get(inp1, x1+z1, sx1*sz1))
        +  w011*(get(inp1, y0+z0, sy0*sz0) + get(inp1, y1+z0, sy1*sz0) +
                 get(inp1, y0+z1, sy1*sz1) + get(inp1, y1+z1, sy1*sz1)))
        + (w200*(get(inp1, x00,   sx00)    + get(inp1, x11,   sx11))
        +  w020*(get(inp1, y00,   sy00)    + get(inp1, y11,   sy11))
        +  w002*(get(inp1, z00,   sz00)    + get(inp1, z11,   sz11))) ) / vx1
    );
  }

  {
    scalar_t c = *inp2;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out2 = static_cast<scalar_t>(
        (wz100*(get(inp2, x0, sx0) + get(inp2, x1, sx1))
      +  wz010*(get(inp2, y0, sy0) + get(inp2, y1, sy1))
      +  wz001*(get(inp2, z0, sz0) + get(inp2, z1, sz1)))
      + (w2   *( bound::get(inp0, z1+x0, sz1*sx0) - bound::get(inp0, z1+x1, sz1*sx1)
               + bound::get(inp0, z0+x1, sz0*sx1) - bound::get(inp0, z0+x0, sz0*sx0)
               + bound::get(inp1, z1+y0, sz1*sy0) - bound::get(inp1, z1+y1, sz1*sy1)
               + bound::get(inp1, z0+y1, sz0*sy1) - bound::get(inp1, z0+y0, sz0*sy0) ))
      + ( absolute*c
        + (w110*(get(inp2, x0+y0, sx0*sy0) + get(inp2, x1+y0, sx1*sy0) +
                 get(inp2, x0+y1, sx1*sy1) + get(inp2, x1+y1, sx1*sy1))
        +  w101*(get(inp2, x0+z0, sx0*sz0) + get(inp2, x1+z0, sx1*sz0) +
                 get(inp2, x0+z1, sx1*sz1) + get(inp2, x1+z1, sx1*sz1))
        +  w011*(get(inp2, y0+z0, sy0*sz0) + get(inp2, y1+z0, sy1*sz0) +
                 get(inp2, y0+z1, sy1*sz1) + get(inp2, y1+z1, sy1*sz1)))
        + (w200*(get(inp2, x00,   sx00)    + get(inp2, x11,   sx11))
        +  w020*(get(inp2, y00,   sy00)    + get(inp2, y11,   sy11))
        +  w002*(get(inp2, z00,   sz00)    + get(inp2, z11,   sz11))) ) / vx2 
      );
  }

  matvec(hes0, inp0, out0);
}

 
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_lame(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_SIGN1 
  GET_WARP1
  GET_POINTERS

  {
    scalar_t c = *inp0; 
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out0 = static_cast<scalar_t>(
        wx100*(get(inp0, x0, sx0) + get(inp0, x1, sx1))
      + wx010*(get(inp0, y0, sy0) + get(inp0, y1, sy1))
      + wx001*(get(inp0, z0, sz0) + get(inp0, z1, sz1))
      + w2   *( bound::get(inp1, x1+y0, sx1*sy0) - bound::get(inp1, x1+y1, sx1*sy1)
              + bound::get(inp1, x0+y1, sx0*sy1) - bound::get(inp1, x0+y0, sx0*sy0)
              + bound::get(inp2, x1+z0, sx1*sz0) - bound::get(inp2, x1+z1, sx1*sz1)
              + bound::get(inp2, x0+z1, sx0*sz1) - bound::get(inp2, x0+z0, sx0*sz0) )
      + absolute*c/vx0
    );
  }

  {
    scalar_t c = *inp1;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out1 = static_cast<scalar_t>(
        wy100*(get(inp1, x0, sx0) + get(inp1, x1, sx1))
      + wy010*(get(inp1, y0, sy0) + get(inp1, y1, sy1))
      + wy001*(get(inp1, z0, sz0) + get(inp1, z1, sz1))
      + w2   *( bound::get(inp0, y1+x0, sy1*sx0) - bound::get(inp0, y1+x1, sy1*sx1)
              + bound::get(inp0, y0+x1, sy0*sx1) - bound::get(inp0, y0+x0, sy0*sx0)
              + bound::get(inp2, y1+z0, sy1*sz0) - bound::get(inp2, y1+z1, sy1*sz1)
              + bound::get(inp2, y0+z1, sy0*sz1) - bound::get(inp2, y0+z0, sy0*sz0) )
      + absolute*c/vx1
    );
  }

  {
    scalar_t c = *inp2;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out2 = static_cast<scalar_t>(
        wz100*(get(inp2, x0, sx0) + get(inp2, x1, sx1))
      + wz010*(get(inp2, y0, sy0) + get(inp2, y1, sy1))
      + wz001*(get(inp2, z0, sz0) + get(inp2, z1, sz1))
      + w2   *( bound::get(inp0, z1+x0, sz1*sx0) - bound::get(inp0, z1+x1, sz1*sx1)
              + bound::get(inp0, z0+x1, sz0*sx1) - bound::get(inp0, z0+x0, sz0*sx0)
              + bound::get(inp1, z1+y0, sz1*sy0) - bound::get(inp1, z1+y1, sz1*sy1)
              + bound::get(inp1, z0+y1, sz0*sy1) - bound::get(inp1, z0+y0, sz0*sy0) )
      + absolute*c/vx2 
      );
  }

  matvec(hes0, inp0, out0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_COORD2
  GET_SIGN1 
  GET_SIGN2
  GET_WARP1 
  GET_WARP2
  GET_POINTERS


  {
    scalar_t c = *inp0; 
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out0 = static_cast<scalar_t>(
        ( absolute*c
        + (w100*(get(inp0, x0, sx0) + get(inp0, x1, sx1))
        +  w010*(get(inp0, y0, sy0) + get(inp0, y1, sy1))
        +  w001*(get(inp0, z0, sz0) + get(inp0, z1, sz1)))
        + (w110*(get(inp0, x0+y0, sx0*sy0) + get(inp0, x1+y0, sx1*sy0) +
                 get(inp0, x0+y1, sx1*sy1) + get(inp0, x1+y1, sx1*sy1))
        +  w101*(get(inp0, x0+z0, sx0*sz0) + get(inp0, x1+z0, sx1*sz0) +
                 get(inp0, x0+z1, sx1*sz1) + get(inp0, x1+z1, sx1*sz1))
        +  w011*(get(inp0, y0+z0, sy0*sz0) + get(inp0, y1+z0, sy1*sz0) +
                 get(inp0, y0+z1, sy1*sz1) + get(inp0, y1+z1, sy1*sz1)))
        + (w200*(get(inp0, x00, sx00) + get(inp0, x11, sx11))
        +  w020*(get(inp0, y00, sy00) + get(inp0, y11, sy11))
        +  w002*(get(inp0, z00, sz00) + get(inp0, z11, sz11))) ) / vx0
    );
  }

  {
    scalar_t c = *inp1;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out1 = static_cast<scalar_t>(
        ( absolute*c
        + (w100*(get(inp1, x0, sx0) + get(inp1, x1, sx1))
        +  w010*(get(inp1, y0, sy0) + get(inp1, y1, sy1))
        +  w001*(get(inp1, z0, sz0) + get(inp1, z1, sz1)))
        + (w110*(get(inp1, x0+y0, sx0*sy0) + get(inp1, x1+y0, sx1*sy0) +
                 get(inp1, x0+y1, sx1*sy1) + get(inp1, x1+y1, sx1*sy1))
        +  w101*(get(inp1, x0+z0, sx0*sz0) + get(inp1, x1+z0, sx1*sz0) +
                 get(inp1, x0+z1, sx1*sz1) + get(inp1, x1+z1, sx1*sz1))
        +  w011*(get(inp1, y0+z0, sy0*sz0) + get(inp1, y1+z0, sy1*sz0) +
                 get(inp1, y0+z1, sy1*sz1) + get(inp1, y1+z1, sy1*sz1)))
        + (w200*(get(inp1, x00,   sx00)    + get(inp1, x11,   sx11))
        + w020*(get(inp1, y00,   sy00)    + get(inp1, y11,   sy11))
        + w002*(get(inp1, z00,   sz00)    + get(inp1, z11,   sz11))) ) / vx1
    );
  }

  {
    scalar_t c = *inp2;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out2 = static_cast<scalar_t>(
        ( absolute*c
        + (w100*(get(inp2, x0, sx0) + get(inp2, x1, sx1))
        +  w010*(get(inp2, y0, sy0) + get(inp2, y1, sy1))
        +  w001*(get(inp2, z0, sz0) + get(inp2, z1, sz1)))
        + (w110*(get(inp2, x0+y0, sx0*sy0) + get(inp2, x1+y0, sx1*sy0) +
                 get(inp2, x0+y1, sx1*sy1) + get(inp2, x1+y1, sx1*sy1))
        +  w101*(get(inp2, x0+z0, sx0*sz0) + get(inp2, x1+z0, sx1*sz0) +
                 get(inp2, x0+z1, sx1*sz1) + get(inp2, x1+z1, sx1*sz1))
        +  w011*(get(inp2, y0+z0, sy0*sz0) + get(inp2, y1+z0, sy1*sz0) +
                 get(inp2, y0+z1, sy1*sz1) + get(inp2, y1+z1, sy1*sz1)))
        + (w200*(get(inp2, x00,   sx00)    + get(inp2, x11,   sx11))
        +  w020*(get(inp2, y00,   sy00)    + get(inp2, y11,   sy11))
        +  w002*(get(inp2, z00,   sz00)    + get(inp2, z11,   sz11))) ) / vx2 
      );
  }

  matvec(hes0, inp0, out0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS

  {
    scalar_t c = *inp0; 
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out0 = static_cast<scalar_t>(
        ( absolute*c
        + w100*(get(inp0, x0, sx0) + get(inp0, x1, sx1))
        + w010*(get(inp0, y0, sy0) + get(inp0, y1, sy1))
        + w001*(get(inp0, z0, sz0) + get(inp0, z1, sz1)) ) / vx0
    );
  }

  {
    scalar_t c = *inp1;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out1 = static_cast<scalar_t>(
        ( absolute*c
        + w100*(get(inp1, x0, sx0) + get(inp1, x1, sx1))
        + w010*(get(inp1, y0, sy0) + get(inp1, y1, sy1))
        + w001*(get(inp1, z0, sz0) + get(inp1, z1, sz1)) ) / vx1
    );
  }

  {
    scalar_t c = *inp2;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out2 = static_cast<scalar_t>(
        ( absolute*c
        + w100*(get(inp2, x0, sx0) + get(inp2, x1, sx1))
        + w010*(get(inp2, y0, sy0) + get(inp2, y1, sy1))
        + w001*(get(inp2, z0, sz0) + get(inp2, z1, sz1)) ) / vx2 
      );
  }

  matvec(hes0, inp0, out0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS

  {
    scalar_t c = *inp0;
    *out0 = static_cast<scalar_t>(  absolute * c / vx0 );
  }
  {
    scalar_t c = *inp1;
    *out1 = static_cast<scalar_t>(  absolute * c / vx1 );
  }
  {
    scalar_t c = *inp2;
    *out2 = static_cast<scalar_t>(  absolute * c / vx2 );
  }

  matvec(hes0, inp0, out0);
}



template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{

  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS

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
    scalar_t c = *inp0;  // no need to use `get` -> we know we are in the FOV
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };


    *out0 = static_cast<scalar_t>(
      ( absolute * wcenter * c
      + w1m00 * get(inp0, x0, sx0)
      + w1p00 * get(inp0, x1, sx1)
      + w01m0 * get(inp0, y0, sy0)
      + w01p0 * get(inp0, y1, sy1)
      + w001m * get(inp0, z0, sz0)
      + w001p * get(inp0, z1, sz1) ) / vx0
    );
  }

  {
    scalar_t c = *inp1;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out1 = static_cast<scalar_t>(
      ( absolute * wcenter * c
      + w1m00 * get(inp1, x0, sx0)
      + w1p00 * get(inp1, x1, sx1)
      + w01m0 * get(inp1, y0, sy0)
      + w01p0 * get(inp1, y1, sy1)
      + w001m * get(inp1, z0, sz0)
      + w001p * get(inp1, z1, sz1) ) / vx1
    );
  }

  {
    scalar_t c = *inp2;
    auto get = [c](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - c;
    };

    *out2 = static_cast<scalar_t>(
      ( absolute * wcenter * c
      + w1m00 * get(inp2, x0, sx0)
      + w1p00 * get(inp2, x1, sx1)
      + w01m0 * get(inp2, y0, sy0)
      + w01p0 * get(inp2, y1, sy1)
      + w001m * get(inp2, z0, sz0)
      + w001p * get(inp2, z1, sz1) ) / vx2
    );
  }

  matvec(hes0, inp0, out0);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  scalar_t w = *(wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ + n*wgt_sN));
  w *= absolute;
  {
    scalar_t c = *inp0;
    *out0 = static_cast<scalar_t>( c * w / vx0 );
  }
  {
    scalar_t c = *inp1;
    *out1 = static_cast<scalar_t>( c * w  / vx1 );
  }
  {
    scalar_t c = *inp2;
    *out2 = static_cast<scalar_t>( c * w  / vx2 );
  }

  matvec(hes0, inp0, out0);
}



/* ========================================================================== */
/*                                     2D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_all(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_lame(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}

/* ========================================================================== */
/*                                     1D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_all(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_lame(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}


/* ========================================================================== */
/*                                     ZEROS                                  */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserGridImpl<scalar_t,offset_t,reduce_t>::zeros(offset_t x, offset_t y, offset_t z, offset_t n) const {}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t, typename reduce_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void regulariser_kernel(RegulariserGridImpl<scalar_t,offset_t,reduce_t> f) {
  f.loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    ALLOCATE OUTPUT // RESHAPE WEIGHT
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST std::tuple<Tensor, Tensor, Tensor>
prepare_tensors(const Tensor & input, Tensor output, Tensor weight, Tensor hessian)
{
  if (!(output.defined() && output.numel() > 0))
    output = at::empty_like(input);
  if (!output.is_same_size(input))
    throw std::invalid_argument("Output tensor must have the same shape as the input tensor");

  if (weight.defined() && weight.numel() > 0)
    weight = weight.expand_as(input);

  if (hessian.defined() && hessian.numel() > 0)
  {
    int64_t dim = input.dim() - 2;
    int64_t N   = input.size(0);
    int64_t CC  = hessian.size(1);
    int64_t X   = input.size(2);
    int64_t Y   = dim > 1 ? input.size(3) : 1L;
    int64_t Z   = dim > 2 ? input.size(4) : 1L;
    if (dim == 1)
      hessian = hessian.expand({N, CC, X});
    else if (dim == 2)
      hessian = hessian.expand({N, CC, X, Y});
    else
      hessian = hessian.expand({N, CC, X, Y, Z});
  }

  return std::tuple<Tensor, Tensor, Tensor>(output, weight, hessian);
}

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (input, weight)
NI_HOST Tensor regulariser_grid_impl(
  const Tensor& input, Tensor output, Tensor weight, Tensor hessian,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(input, output, weight, hessian);
  output       = std::get<0>(tensors);
  weight       = std::get<1>(tensors);
  hessian      = std::get<2>(tensors);

  RegulariserGridAllocator info(input.dim()-2, absolute, membrane, bending,
                            lame_shear, lame_div, voxel_size, bound);
  info.ioset(input, output, weight, hessian);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "regulariser_grid_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      RegulariserGridImpl<scalar_t, int32_t, double> algo(info);
      regulariser_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
    }
    else
    {
      RegulariserGridImpl<scalar_t, int64_t, double> algo(info);
      regulariser_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
    }
  });
  return output;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (input, weight)
NI_HOST Tensor regulariser_grid_impl(
  const Tensor& input, Tensor output, Tensor weight, Tensor hessian,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(input, output, weight, hessian);
  output       = std::get<0>(tensors);
  weight       = std::get<1>(tensors);
  hessian      = std::get<2>(tensors);

  RegulariserGridAllocator info(input.dim()-2, absolute, membrane, bending,
                            lame_shear, lame_div, voxel_size, bound);
  info.ioset(input, output, weight, hessian);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "regulariser_grid_impl", [&] {
    RegulariserGridImpl<scalar_t, int64_t, double> algo(info);
    algo.loop();
  });
  return output;
}

#endif // __CUDACC__

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST Tensor regulariser_grid_impl(
  const Tensor& input, Tensor output, Tensor weight, Tensor hessian,
  double absolute, double membrane, double bending, double lame_shear, double lame_div,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  throw std::logic_error("Function not implemented for this device.");
}

} // namespace notimplemented

} // namespace ni

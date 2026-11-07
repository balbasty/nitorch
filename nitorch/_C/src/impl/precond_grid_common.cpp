#include "common.h"                // write C++/CUDA compatible code
#include "../defines.h"            // useful macros
#include "bounds_common.h"         // boundary conditions + enum
#include "allocator.h"             // base class handling offset sizes
#include "border_common.h"         // border pass (definition F)
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
    // (the border voxels are a thin set)
    int64_t BORDER_GRAIN_SIZE = GRAIN_SIZE / 32;
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
    wbound0(bound::even(info.bound0)), wbound1(bound::even(info.bound1)),
    wbound2(bound::even(info.bound2)),
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
    set_border();
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

  // Border pass (see border_common.h): the diagonal of the exact operator
  // differs from that of the stencil near the borders (B, and the taps
  // that fold back onto their own voxel). The voxels within
  // `border::radius` of the borders are solved again with their exact
  // diagonal (`precond<true>`, which overwrites them). Decided once.
  NI_HOST NI_INLINE void set_border()
  {
    BoundType b[3] = {bound0, bound1, bound2};
    double ivx[3] = {vx0, vx1, vx2};   // (1 / voxel_size**2)
    offset_t nn[3] = {X, Y, Z};
    bt.set(dim, b, nn, ivx, membrane, bending, lame_shear, lame_div);
    // stencil taps of component c along a (as in `set_kernel`)
    double lam1 = membrane, lam2 = bending, mu = lame_shear, lam = lame_div;
    double sv = ivx[0] + ivx[1] + ivx[2];
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a) {
      bool on = a < dim && c < dim;
      double w1 = lam2 * (-4.0 * ivx[a] * sv) - lam1 * ivx[a];
      t1[c][a] = static_cast<reduce_t>(!on ? 0. :
        (a == c ? -2.0 * mu - lam : -mu * ivx[a] / ivx[c]) + w1 / ivx[c]);
      t2[c][a] = static_cast<reduce_t>(on ? lam2 * ivx[a] * ivx[a] / ivx[c] : 0.);
      trls[c][a] = static_cast<reduce_t>(on ? -lam1 * ivx[a] / ivx[c] : 0.);
      for (int bb = 0; bb < 3; ++bb)
        t11[c][a][bb] = static_cast<reduce_t>(on && bb < dim ?
          lam2 * 2.0 * ivx[a] * ivx[bb] / ivx[c] : 0.);
    }
    bool taps = (membrane || bending || lame_shear || lame_div);
    offset_t r[3];
    needs_border = false;
    for (int a = 0; a < 3; ++a) {
      r[a] = (taps && a < dim) ? border::radius(b[a], nn[a]) : 0;
      needs_border = needs_border || r[a] > 0;
    }
    box.set(X, Y, Z, r[0], r[1], r[2]);
    region = 0;
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
        precond_ = &Self::template precond1d_rls_membrane<false>; precondb_ = &Self::template precond1d_rls_membrane<true>; break;
      case 2 + MEMBRANE + RLS:
        precond_ = &Self::template precond2d_rls_membrane<false>; precondb_ = &Self::template precond2d_rls_membrane<true>; break;
      case 3 + MEMBRANE + RLS:
        precond_ = &Self::template precond3d_rls_membrane<false>; precondb_ = &Self::template precond3d_rls_membrane<true>; break;
      case 1 + ABSOLUTE + RLS:
        precond_ = &Self::template precond1d_rls_absolute<false>; precondb_ = &Self::template precond1d_rls_absolute<true>; break;
      case 2 + ABSOLUTE + RLS:
        precond_ = &Self::template precond2d_rls_absolute<false>; precondb_ = &Self::template precond2d_rls_absolute<true>; break;
      case 3 + ABSOLUTE + RLS:
        precond_ = &Self::template precond3d_rls_absolute<false>; precondb_ = &Self::template precond3d_rls_absolute<true>; break;
      default:
        switch (dim) {
          case 1:
            precond_ = &Self::template precond1d<false>; precondb_ = &Self::template precond1d<true>; break;
          case 2: 
            precond_ = &Self::template precond2d<false>; precondb_ = &Self::template precond2d<true>; break;
          default:
            precond_ = &Self::template precond3d<false>; precondb_ = &Self::template precond3d<true>; break;
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

  // Voxels of each region (0: all voxels, 2: border voxels, solved
  // again). The CUDA host code mutates `region` and launches one kernel
  // per region.
  NI_HOST NI_INLINE bool split() const { return needs_border; }
  NI_HOST NI_INLINE void set_region(uint8_t r) { region = r; }
  NI_HOST NI_DEVICE int64_t voxcount(uint8_t r) const {
    return r == 2 ? N * box.border_count() : voxcount();
  }
 

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  template <bool BORDER> NI_DEVICE NI_INLINE void precond(
    offset_t x, offset_t y, offset_t z, offset_t n) const;

#define DEFINE_PRECOND(SUFFIX) \
  template <bool BORDER> NI_DEVICE void precond##SUFFIX( \
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

  template <bool BORDER>
  NI_DEVICE void finish(offset_t x, offset_t y, offset_t z, offset_t n,
    const scalar_t * hes, scalar_t * sol, const scalar_t * grd,
    reduce_t d0, reduce_t d1, reduce_t d2) const;
  NI_DEVICE void border(offset_t x, offset_t y, offset_t z, offset_t n,
    const scalar_t * sol, reduce_t & d0, reduce_t & d1, reduce_t & d2) const;

  /* ~~~ BORDER PASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  uint8_t           region;         // 0: all voxels, 2: border
  bool              needs_border;
  border::BorderBox<offset_t> box;  // interior box (voxels)
  border::GridTerms<reduce_t, offset_t> bt; // terms of B
  reduce_t          t1[3][3];       // stencil taps +/-1 [comp][axis]
  reduce_t          t2[3][3];       // stencil taps +/-2
  reduce_t          t11[3][3][3];   // stencil taps (+/-1, +/-1)
  reduce_t          trls[3][3];     // weighted membrane taps (RLS)

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  offset_t          dim;
  uint8_t           mode;
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  BoundType         wbound0;        // boundary of the weights (even) // x|W
  BoundType         wbound1;        // boundary of the weights (even) // y|H
  BoundType         wbound2;        // boundary of the weights (even) // z|D
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
  PrecondFn         precondb_;      // Pointer to Precond function (border voxels)
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

template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
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
      return precond1d_rls_membrane<BORDER>(x, y, z, n);
    case 2 + MEMBRANE + RLS:
      return precond2d_rls_membrane<BORDER>(x, y, z, n);
    case 3 + MEMBRANE + RLS:
      return precond3d_rls_membrane<BORDER>(x, y, z, n);
    case 1 + ABSOLUTE + RLS:
      return precond1d_rls_absolute<BORDER>(x, y, z, n);
    case 2 + ABSOLUTE + RLS:
      return precond2d_rls_absolute<BORDER>(x, y, z, n);
    case 3 + ABSOLUTE + RLS:
      return precond3d_rls_absolute<BORDER>(x, y, z, n);
    default:
      switch (dim) {
        case 1:
          return precond1d<BORDER>(x, y, z, n);
        case 2: 
          return precond2d<BORDER>(x, y, z, n);
        default:
          return precond3d<BORDER>(x, y, z, n);
      }
  }
#else
  CALL_MEMBER_FN(*this, BORDER ? precondb_ : precond_)(x, y, z, n);
#endif 
}

#ifdef __CUDACC__

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  offset_t index = static_cast<offset_t>(blockIdx * blockDim + threadIdx);
  offset_t n, x, y, z;
  if (region == 2) {
    offset_t M  = box.border_count();
    offset_t NM = N * M;
    for (offset_t i=index; index < NM; index += blockDim*gridDim, i=index)
    {
      n = i / M;
      box.decode(i % M, x, y, z);
      precond<true>(x, y, z, n);
    }
    return;
  }
  offset_t YZ   = Y * Z;
  offset_t XYZ  = X * YZ;
  offset_t NXYZ = N * XYZ;
  for (offset_t i=index; index < NXYZ; index += blockDim*gridDim, i=index)
  {
      // Convert index: linear to sub
      n  = (i/XYZ);
      x  = (i/YZ) % X;
      y  = (i/Z)  % Y;
      z  =  i     % Z;
      precond<false>(x, y, z, n);
  }
}

#else

// This bit loops over all target voxels. We therefore need to
// convert linear indices to multivariate indices. The way I do it
// might not be optimal. With a border pass, the border voxels are then
// solved again, with their exact diagonal.
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
      precond<false>(x, y, z, n);
    }
  });
  if (!needs_border) return;
  // the border voxels again, with their exact diagonal
  offset_t M = box.border_count();
  at::parallel_for(0, N * M, BORDER_GRAIN_SIZE, [&](offset_t start, offset_t end) {
    offset_t x, y, z;
    for (offset_t i = start; i < end; ++i) {
      box.decode(i % M, x, y, z);
      precond<true>(x, y, z, i / M);
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
  reduce_t h00 = (*h) * OnePlusTiny;
  s[       0] = v[       0] / (h00 + w0);
  s[  sol_sC] = v[  grd_sC] / (h00 + w1);
  s[2*sol_sC] = v[2*grd_sC] / (h00 + w2);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::invert2d_eye(
    const scalar_t * h, scalar_t * s, const scalar_t *v, 
    reduce_t w0, reduce_t w1, reduce_t  /*unused*/) const 
{
  reduce_t h00 = (*h) * OnePlusTiny;

  // solve
  s[     0] = v[     0] / (h00 + w0);
  s[sol_sC] = v[grd_sC] / (h00 + w1);
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
// ~~~ Border voxels (see border_common.h) ~~~
// `d` is the diagonal of the stencil, as if all its taps were other
// voxels. At the border voxels, it becomes the exact diagonal (that of
// S, with the taps that fold back onto the voxel, plus that of B).
template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::finish(
    offset_t x, offset_t y, offset_t z, offset_t n,
    const scalar_t * hes, scalar_t * sol, const scalar_t * grd,
    reduce_t d0, reduce_t d1, reduce_t d2) const
{
  if (BORDER)
    border(x, y, z, n, sol, d0, d1, d2);
  invert(hes, sol, grd, d0, d1, d2);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::border(
    offset_t x, offset_t y, offset_t z, offset_t n, const scalar_t * sol0,
    reduce_t & d0, reduce_t & d1, reduce_t & d2) const
{
  const offset_t q[3]  = {x, y, z};
  const offset_t s[3]  = {sol_sX, sol_sY, sol_sZ};
  const offset_t sw[3] = {wgt_sX, wgt_sY, wgt_sZ};
  const scalar_t * wgt = wgt_ptr
    ? wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ + n*wgt_sN)
    : static_cast<const scalar_t *>(0);
  reduce_t D[3] = {0, 0, 0}, T1[3][3];
  for (int a = 0; a < dim; ++a) {
    border::Plane<offset_t> P[3], PW;
    bool any = false;
    for (int c = 0; c < dim; ++c) {
      P[c].set(bt.tab[c][a], bt.n[a], q[a], s[a]);
      any = any || (bt.fany[c][a] && P[c].targeted());
    }
    if (wgt) PW.set(bt.wtab[a], bt.n[a], q[a], sw[a]);
    // self image of the stencil taps along a (and of the pairs (b, a))
    for (int c = 0; c < dim; ++c) {
      T1[c][a] = border::tapself(P[c], q[a], bt.n[a], -1)
               + border::tapself(P[c], q[a], bt.n[a],  1);
      if (wgt) {
        D[c] += 0.5 * trls[c][a] *
                border::rls_self<reduce_t>(P[c], PW, wgt, q[a], bt.n[a]);
        continue;
      }
      D[c] += t1[c][a] * T1[c][a]
            + t2[c][a] * (border::tapself(P[c], q[a], bt.n[a], -2)
                        + border::tapself(P[c], q[a], bt.n[a],  2));
      for (int bb = 0; bb < a; ++bb)
        D[c] += t11[c][bb][a] * T1[c][bb] * T1[c][a];
    }
    if (!any) continue;
    // diagonal of B along a
    for (int c = 0; c < dim; ++c)
      if (bt.fany[c][a])
        D[c] += bt.template axis<true>(a, c, q, s, sol0, sol_sC, wgt, PW, P);
  }
  d0 += D[0]; d1 += D[1]; d2 += D[2];
}

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
// Weights are sampled with their own (even) boundary condition
#define GET_WARP1_RLS_(x, X, i) \
  offset_t w##x##0 = (bound::index(wbound##i, x##0, X) - x) * wgt_s##X; \
  offset_t w##x##1 = (bound::index(wbound##i, x##1, X) - x) * wgt_s##X; \
  x##0  = (bound::index(bound##i, x##0,  X) - x) * sol_s##X; \
  x##1  = (bound::index(bound##i, x##1,  X) - x) * sol_s##X;


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


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond3d(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS

  finish<BORDER>(x, y, z, n, hes, sol, grd, wx000, wy000, wz000);
}


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_WARP1_RLS
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  scalar_t wcenter = *wgt;
  reduce_t w1m00 = m100 * (wcenter + wgt[wx0]);
  reduce_t w1p00 = m100 * (wcenter + wgt[wx1]);
  reduce_t w01m0 = m010 * (wcenter + wgt[wy0]);
  reduce_t w01p0 = m010 * (wcenter + wgt[wy1]);
  reduce_t w001m = m001 * (wcenter + wgt[wz0]);
  reduce_t w001p = m001 * (wcenter + wgt[wz1]);

  reduce_t w = absolute * wcenter 
             - 0.5 * membrane * (w1m00 + w1p00 + w01m0 + w01p0 + w001m + w001p);
  finish<BORDER>(x, y, z, n, hes, sol, grd, w/vx0, w/vx1, w/vx2);
}


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  reduce_t w = absolute * (*wgt);
  finish<BORDER>(x, y, z, n, hes, sol, grd, w/vx0, w/vx1, w/vx2);
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
  const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + n*grd_sN); \
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + n*sol_sN); \
  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + n*hes_sN);


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond2d(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS

  finish<BORDER>(x, y, z, n, hes, sol, grd, wx000, wy000, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond2d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_WARP1_RLS
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY);

  scalar_t wcenter = *wgt;
  reduce_t w1m00 = m100 * (wcenter + wgt[wx0]);
  reduce_t w1p00 = m100 * (wcenter + wgt[wx1]);
  reduce_t w01m0 = m010 * (wcenter + wgt[wy0]);
  reduce_t w01p0 = m010 * (wcenter + wgt[wy1]);

  reduce_t w = absolute * wcenter 
             - 0.5 * membrane * (w1m00 + w1p00 + w01m0 + w01p0);
  finish<BORDER>(x, y, z, n, hes, sol, grd, w/vx0, w/vx1, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond2d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY);

  reduce_t w = absolute * (*wgt);
  finish<BORDER>(x, y, z, n, hes, sol, grd, w/vx0, w/vx1, 0);
}

/* ========================================================================== */
/*                                     1D                                     */
/* ========================================================================== */

#undef  GET_COORD1
#define GET_COORD1 \
  GET_COORD1_(x)
#undef  GET_COORD2
#define GET_COORD2 \
  GET_COORD2_(x) 
#undef  GET_SIGN1
#define GET_SIGN1 \
  GET_SIGN1_(x, X, 0)
#undef  GET_SIGN2
#define GET_SIGN2 \
  GET_SIGN2_(x, X, 0) 
#undef  GET_WARP1
#define GET_WARP1 \
  GET_WARP1_(x, X, 0) 
#undef  GET_WARP2
#define GET_WARP2 \
  GET_WARP2_(x, X, 0) 
#undef  GET_WARP1_RLS
#define GET_WARP1_RLS \
  GET_WARP1_RLS_(x, X, 0)

#undef  GET_POINTERS
#define GET_POINTERS \
  const scalar_t *grd = grd_ptr + (x*grd_sX + n*grd_sN); \
        scalar_t *sol = sol_ptr + (x*sol_sX + n*sol_sN); \
  const scalar_t *hes = hes_ptr + (x*hes_sX + n*hes_sN);


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond1d(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS

  finish<BORDER>(x, y, z, n, hes, sol, grd, wx000, 0, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond1d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_WARP1_RLS
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY);

  scalar_t wcenter = *wgt;
  reduce_t w1m00 = m100 * (wcenter + wgt[wx0]);
  reduce_t w1p00 = m100 * (wcenter + wgt[wx1]);

  reduce_t w = absolute * wcenter 
             - 0.5 * membrane * (w1m00 + w1p00);
  finish<BORDER>(x, y, z, n, hes, sol, grd, w/vx0, 0, 0);
}


template <typename scalar_t, typename offset_t, typename reduce_t>
template <bool BORDER> NI_DEVICE
void PrecondGridImpl<scalar_t,offset_t,reduce_t>::precond1d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX);

  reduce_t w = absolute * (*wgt);
  finish<BORDER>(x, y, z, n, hes, sol, grd, w/vx0, 0, 0);
}

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
      // (all voxels, then the border voxels again, if they need it)
      for (uint8_t r = 0; r < (algo.split() ? 3 : 1); r += 2) {
        algo.set_region(r);
        if (!algo.voxcount(r)) continue;
        auto palgo = alloc_and_copy_to_device(&algo, stream);
        precond_kernel
            <<<GET_BLOCKS(algo.voxcount(r)), CUDA_NUM_THREADS, 0, stream>>>
            (palgo);
        cudaFree(palgo);
      }
    }
    else
    {
      PrecondGridImpl<scalar_t, int64_t, double> algo(info);
      // (all voxels, then the border voxels again, if they need it)
      for (uint8_t r = 0; r < (algo.split() ? 3 : 1); r += 2) {
        algo.set_region(r);
        if (!algo.voxcount(r)) continue;
        auto palgo = alloc_and_copy_to_device(&algo, stream);
        precond_kernel
            <<<GET_BLOCKS(algo.voxcount(r)), CUDA_NUM_THREADS, 0, stream>>>
            (palgo);
        cudaFree(palgo);
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

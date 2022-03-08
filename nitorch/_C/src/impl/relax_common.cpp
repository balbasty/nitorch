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
#ifdef OnePlusTiny
#undef OnePlusTiny
#endif
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
class RelaxAllocator: public Allocator {
public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST
  RelaxAllocator(int dim, ArrayRef<double> absolute, 
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

  // Allow RelaxImpl's constructor to access RelaxAllocator's
  // private members.
  template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t>
  friend class RelaxImpl;
};


NI_HOST
void RelaxAllocator::init_all()
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
void RelaxAllocator::init_gradient(const Tensor& input)
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
void RelaxAllocator::init_hessian(const Tensor& input)
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
void RelaxAllocator::init_solution(const Tensor& input)
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
void RelaxAllocator::init_weight(const Tensor& weight)
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
class RelaxImpl {

  using Self     = RelaxImpl;
  using RelaxFn  = void (Self::*)(offset_t, offset_t, offset_t, offset_t) const;
  static const int32_t MaxC  = hessian_t::max_length;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  RelaxImpl(const RelaxAllocator & info):
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
    set_bandwidth();
  #ifndef __CUDACC__
    set_relax();
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

    has_absolute = any(absolute, C);
    has_membrane = any(membrane, C);
    has_bending  = any(bending, C);

    // `mode` encodes all options in a single byte (actually 5 bits)
    // [1*RLS 2*ENERGY 2*DIM]
    // We can use this byte to switch between implementations efficiently.
    mode = dim 
         + (has_bending ? 12 : has_membrane ? 8 : has_absolute ? 4 : 0)
         + (wgt_ptr ? 16 : 0);
  } 

  NI_HOST void set_kernel(double vx0, double vx1, double vx2) 
  {
    for (offset_t c = 0; c < C; ++c)
      w000[c] = static_cast<reduce_t>((
                    bending[c]  * (6.0*(vx0*vx0+vx1*vx1+vx2*vx2) + 
                                   8.0*(vx0*vx1+vx0*vx2+vx1*vx2))
                  + membrane[c] * (2.0*(vx0+vx1+vx2))
                  + absolute[c]) * OnePlusTiny);

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

  NI_HOST void set_bandwidth() 
  { 
    if (has_bending)
      bandwidth = 3;
    else if (has_membrane)
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
      Xf = 1 + (X - 1) / Fx;
      Yf = 1 + (Y - 1) / Fy;
      Zf = 1 + (Z - 1) / Fz;
    }
  }

#ifndef __CUDACC__
  NI_HOST void set_relax() 
  {

    if (wgt_ptr)
    {
      if (has_bending)
        throw std::logic_error("RLS only implemented for absolute/membrane.");
      else if (dim == 1) {
        if (has_membrane)
            relax_ = &Self::relax1d_rls_membrane;
        else if (has_absolute)
            relax_ = &Self::relax1d_rls_absolute;
        else
            relax_ = &Self::solve1d;
      } else if (dim == 2) {
        if (has_membrane)
            relax_ = &Self::relax2d_rls_membrane;
        else if (has_absolute)
            relax_ = &Self::relax2d_rls_absolute;
        else
            relax_ = &Self::solve2d;
      } else if (dim == 3) {
        if (has_membrane)
            relax_ = &Self::relax3d_rls_membrane;
        else if (has_absolute)
            relax_ = &Self::relax3d_rls_absolute;
        else
            relax_ = &Self::solve3d;
      }
    }
    else if (dim == 1) {
        if (has_bending)
            relax_ = &Self::relax1d_bending;
        else if (has_membrane)
            relax_ = &Self::relax1d_membrane;
        else if (has_absolute)
            relax_ = &Self::relax1d_absolute;
        else
            relax_ = &Self::solve1d;
    } else if (dim == 2) {
        if (has_bending)
            relax_ = &Self::relax2d_bending;
        else if (has_membrane)
            relax_ = &Self::relax2d_membrane;
        else if (has_absolute)
            relax_ = &Self::relax2d_absolute;
        else
            relax_ = &Self::solve2d;
    } else if (dim == 3) {
        if (has_bending)
            relax_ = &Self::relax3d_bending;
        else if (has_membrane)
            relax_ = &Self::relax3d_membrane;
        else if (has_absolute)
            relax_ = &Self::relax3d_absolute;
        else
            relax_ = &Self::solve3d;
    } else
        throw std::logic_error("RLS only implemented for dimension 1/2/3.");
  }
#endif

  /* ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#ifdef __CUDACC__
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
      fy = (i/Fz)  % Fy;
      fz = i % Fz;

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
  DEFINE_RELAX(DIM##d_rls_absolute)  \
  DEFINE_RELAX(DIM##d_rls_membrane)  \
  NI_DEVICE void solve##DIM##d(      \
    offset_t x, offset_t y, offset_t z, offset_t n) const;

  DEFINE_RELAX()
  DEFINE_RELAX_DIM(1)
  DEFINE_RELAX_DIM(2)
  DEFINE_RELAX_DIM(3)

  /* ~~~ FOLD NAVIGATORS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  // This is super hacky!!!
  // I make sure that [fx fy fz Xf Yf Zf redblack] are the first fields in 
  // the object so tht I can easily access them from the adress of the 
  // englobing object. This is so I can mutate these fields when we change fold
  // without having to move the whole object again from host to device.

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
  offset_t          dim;
  uint8_t           mode;
  BoundType         bound0;          // boundary condition  // x|W
  BoundType         bound1;          // boundary condition  // y|H
  BoundType         bound2;          // boundary condition  // z|D
  reduce_t          absolute[MaxC];  // penalty on absolute values
  reduce_t          membrane[MaxC];  // penalty on first derivatives
  reduce_t          bending[MaxC];   // penalty on second derivatives

#ifndef __CUDACC__
  RelaxFn           relax_;          // Pointer to relax function
#endif

  bool has_absolute;
  bool has_membrane;
  bool has_bending;

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
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax(
    offset_t x, offset_t y, offset_t z, offset_t n) const 
{
#ifdef __CUDACC__
#   define ABSOLUTE 4
#   define MEMBRANE 8
#   define BENDING  12
#   define RLS      16
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
    case 1 + BENDING:
      return relax1d_bending(x, y, z, n);
    case 2 + BENDING:
      return relax2d_bending(x, y, z, n);
    case 3 + BENDING:
      return relax3d_bending(x, y, z, n);
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

#ifdef __CUDACC__

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  if (bandwidth == 0)
    return loop_redblack(threadIdx, blockIdx, blockDim, gridDim);
  else
    return loop_band(threadIdx, blockIdx, blockDim, gridDim);
}

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop_band(
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

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop_redblack(
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

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_HOST
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop()
{
  if (bandwidth == 0)
    return loop_redblack();
  else
    return loop_band();
}

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_HOST
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop_redblack()
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

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_HOST
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::loop_band()
{
  for (offset_t fold = 0; fold < Fx*Fy*Fz; ++fold) {
    // Index of the fold
    set_fold(fold);
    offset_t   YZf =   Zf * Yf;
    offset_t  XYZf =  YZf * Xf;
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

#define GET_GRD_POINTER \
  const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN);
#define GET_SOL_POINTER \
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN);
#define GET_HES_POINTER \
  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);
#define GET_WGT_POINTER \
  const scalar_t *wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ + n*wgt_sN);

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax3d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  reduce_t val[MaxC];
  GET_SOL_POINTER
  {
    GET_COORD1
    GET_COORD2
    GET_SIGN1 
    GET_SIGN2
    GET_WARP1 
    GET_WARP2
    GET_GRD_POINTER

    const reduce_t *a = absolute, *m = membrane, *b = bending;
    reduce_t aa, mm, bb;

    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
    {
      scalar_t center = *sol; 
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      aa = *(a++);
      mm = *(m++);
      bb = *(b++);

      reduce_t w100 = mm * m100 + bb * b100;
      reduce_t w010 = mm * m010 + bb * b010;
      reduce_t w001 = mm * m001 + bb * b001;

      val[c] = (*grd) - (
            aa * center
          +(w100*(get(sol, x0,    sx0)     + get(sol, x1,    sx1))
          + w010*(get(sol, y0,    sy0)     + get(sol, y1,    sy1))
          + w001*(get(sol, z0,    sz0)     + get(sol, z1,    sz1)))
          + bb * (
             (b110*(get(sol, x0+y0, sx0*sy0) + get(sol, x1+y0, sx1*sy0) +
                    get(sol, x0+y1, sx0*sy1) + get(sol, x1+y1, sx1*sy1))
            + b101*(get(sol, x0+z0, sx0*sz0) + get(sol, x1+z0, sx1*sz0) +
                    get(sol, x0+z1, sx0*sz1) + get(sol, x1+z1, sx1*sz1))
            + b011*(get(sol, y0+z0, sy0*sz0) + get(sol, y1+z0, sy1*sz0) +
                    get(sol, y0+z1, sy0*sz1) + get(sol, y1+z1, sy1*sz1)))
            +(b200*(get(sol, x00,   sx00)    + get(sol, x11,   sx11))
            + b020*(get(sol, y00,   sy00)    + get(sol, y11,   sy11))
            + b002*(get(sol, z00,   sz00)    + get(sol, z11,   sz11))) )
      );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax3d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  reduce_t val[MaxC];
  GET_SOL_POINTER
  {
    GET_COORD1
    GET_SIGN1 
    GET_WARP1 
    GET_GRD_POINTER

    const reduce_t *a = absolute, *m = membrane;

    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
    {
      scalar_t center = *sol; 
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      val[c] = (*grd) - (
            (*(a++)) * center
          + (*(m++)) * (
              m100*(get(sol, x0, sx0) + get(sol, x1, sx1))
            + m010*(get(sol, y0, sy0) + get(sol, y1, sy1))
            + m001*(get(sol, z0, sz0) + get(sol, z1, sz1)) )
      );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax3d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[NI_MAX_NUM_CHANNELS];
  {
    GET_GRD_POINTER
    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
      val[c] = (*grd) - ( absolute[c] * (*sol) );
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


/*
Reweighted least squares
========================

The spatial regulariser (for a single channel) is L = K'*W*K, 
where K*x returns the forward and backward gradients of x 
(weighted by 1/sqrt(2)) and W = diag(w) contains the weight map. 
If w = ones, we recover L = K'*K which corresponds to the (L2) membrane
regulariser.

In the forward pass, we can still implement L as a convolution, except that 
the convolution weights are themselves obtained by convolving the weight map.
Let us write the membrane convolution kernel K'*K (in 2d) as:
[     m01     ]
[ m10 m00 m10 ]
[     m01     ]
(Note that in all our kernels, weights sum to zero, such that the center weight 
is equal to the negative of the sum of the off-diagonal weights. 
Here: m00 = -2*(m01 + m10).)
Then the convolution kernel at voxel n in the reweighted case is
[                         |  m01 * (w[n,-] + w[n,n]) |                         ]
[ m10 * (w[-,n] + w[n,n]) |  w00                     | m10 * (w[+,n] + w[n,n]) ]
[                         |  m01 * (w[n,+] + w[n,n]) |                         ]
Again weights sum to zero so the center weight is not explicitely given.

To relax, we need the diagonal of the regulariser diag(K'*W*K). This can again
be expressed as a convolution of the weight map, this time by the absolute value 
of the L2 kernel:
          [       |m01|       ]
diag(L) = [ |m10| |m00| |m10| ] * w
          [       |m01|       ]
*/


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC], wval[MaxC];
  {
    GET_COORD1
    GET_SIGN1
    GET_WARP1_RLS
    GET_GRD_POINTER
    GET_WGT_POINTER

    const reduce_t *a = absolute, *m = membrane;
    reduce_t aa, mm;

    for (offset_t c = 0; c < C; 
         ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC)
    {
      scalar_t wcenter = *wgt;
      reduce_t w1m00 = m100 * (wcenter + bound::get(wgt, wx0, sx0));
      reduce_t w1p00 = m100 * (wcenter + bound::get(wgt, wx1, sx1));
      reduce_t w01m0 = m010 * (wcenter + bound::get(wgt, wy0, sy0));
      reduce_t w01p0 = m010 * (wcenter + bound::get(wgt, wy1, sy1));
      reduce_t w001m = m001 * (wcenter + bound::get(wgt, wz0, sz0));
      reduce_t w001p = m001 * (wcenter + bound::get(wgt, wz1, sz1));

      scalar_t center = *sol;  // no need to use `get` -> we know we are in the FOV
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      aa = *(a++);
      mm = *(m++) * 0.5;

      val[c] = (*grd) - (
          aa * wcenter * center
        + mm * (
            w1m00 * get(sol, x0, sx0)
          + w1p00 * get(sol, x1, sx1)
          + w01m0 * get(sol, y0, sy0)
          + w01p0 * get(sol, y1, sy1)
          + w001m * get(sol, z0, sz0)
          + w001p * get(sol, z1, sz1) )
      );

      wval[c] = ( aa * wcenter
                - mm * (w1m00 + w1p00 + w01m0 + w01p0 + w001m + w001p) );
    }
  }
  
  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, wval);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC], wval[MaxC];
  {
    GET_GRD_POINTER
    GET_WGT_POINTER

    for (offset_t c = 0; c < C; 
         ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC) {
      scalar_t wcenter = *wgt;
      val[c]  = (*grd) - ( absolute[c] * wcenter * (*sol) );
      wval[c] = absolute[c] * wcenter;
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, wval);
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

#undef  GET_GRD_POINTER
#define GET_GRD_POINTER \
  const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + n*grd_sN);
#undef  GET_SOL_POINTER
#define GET_SOL_POINTER \
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + n*sol_sN);
#undef  GET_HES_POINTER
#define GET_HES_POINTER \
  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + n*hes_sN);
#undef  GET_WGT_POINTER
#define GET_WGT_POINTER \
  const scalar_t *wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + n*wgt_sN);

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax2d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC];
  {
    GET_COORD1
    GET_COORD2
    GET_SIGN1 
    GET_SIGN2
    GET_WARP1 
    GET_WARP2
    GET_GRD_POINTER

    const reduce_t *a = absolute, *m = membrane, *b = bending;
    reduce_t aa, mm, bb;

    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
    {
      scalar_t center = *sol; 
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      aa = *(a++);
      mm = *(m++);
      bb = *(b++);

      reduce_t w100 = mm * m100 + bb * b100;
      reduce_t w010 = mm * m010 + bb * b010;

      val[c] = (*grd) - (
            aa * center
          +(w100*(get(sol, x0,    sx0)     + get(sol, x1,    sx1))
          + w010*(get(sol, y0,    sy0)     + get(sol, y1,    sy1)))
          + bb * (
             (b110*(get(sol, x0+y0, sx0*sy0) + get(sol, x1+y0, sx1*sy0) +
                    get(sol, x0+y1, sx0*sy1) + get(sol, x1+y1, sx1*sy1)))
            +(b200*(get(sol, x00,   sx00)    + get(sol, x11,   sx11))
            + b020*(get(sol, y00,   sy00)    + get(sol, y11,   sy11))) )
      );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax2d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC];
  {
    GET_COORD1
    GET_SIGN1 
    GET_WARP1 
    GET_GRD_POINTER

    const reduce_t *a = absolute, *m = membrane;

    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
    {
      scalar_t center = *sol; 
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      val[c] = (*grd) - (
            (*(a++)) * center
          + (*(m++)) * (
              m100*(get(sol, x0, sx0) + get(sol, x1, sx1))
            + m010*(get(sol, y0, sy0) + get(sol, y1, sy1)) )
      );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);

}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax2d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC];
  {
    GET_GRD_POINTER
    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
      val[c] = (*grd) - ( absolute[c] * (*sol) );
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);
}



template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax2d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC], wval[MaxC];
  {
    GET_COORD1
    GET_SIGN1
    GET_WARP1_RLS
    GET_GRD_POINTER
    GET_WGT_POINTER

    const reduce_t *a = absolute, *m = membrane;
    reduce_t aa, mm;

    for (offset_t c = 0; c < C; 
         ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC)
    {
      scalar_t wcenter = *wgt;
      reduce_t w1m00 = m100 * (wcenter + bound::get(wgt, wx0, sx0));
      reduce_t w1p00 = m100 * (wcenter + bound::get(wgt, wx1, sx1));
      reduce_t w01m0 = m010 * (wcenter + bound::get(wgt, wy0, sy0));
      reduce_t w01p0 = m010 * (wcenter + bound::get(wgt, wy1, sy1));

      scalar_t center = *sol;  // no need to use `get` -> we know we are in the FOV
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      aa = *(a++);
      mm = *(m++) * 0.5;

      val[c] = (*grd) - (
          aa * wcenter * center
        + mm * (
            w1m00 * get(sol, x0, sx0)
          + w1p00 * get(sol, x1, sx1)
          + w01m0 * get(sol, y0, sy0)
          + w01p0 * get(sol, y1, sy1) )
      );

      wval[c] = ( aa * wcenter
                - mm * (w1m00 + w1p00 + w01m0 + w01p0) );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, wval);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax2d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC], wval[MaxC];
  {
    GET_GRD_POINTER
    GET_WGT_POINTER

    for (offset_t c = 0; c < C; 
         ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC) {
      scalar_t wcenter = *wgt;
      val[c]  = (*grd) - ( absolute[c] * wcenter * (*sol) );
      wval[c] = absolute[c] * wcenter;
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, wval);
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

#undef  GET_GRD_POINTER
#define GET_GRD_POINTER \
  const scalar_t *grd = grd_ptr + (x*grd_sX + n*grd_sN);
#undef  GET_SOL_POINTER
#define GET_SOL_POINTER \
        scalar_t *sol = sol_ptr + (x*sol_sX + n*sol_sN);
#undef  GET_HES_POINTER
#define GET_HES_POINTER \
  const scalar_t *hes = hes_ptr + (x*hes_sX + n*hes_sN);
#undef  GET_WGT_POINTER
#define GET_WGT_POINTER \
  const scalar_t *wgt = wgt_ptr + (x*wgt_sX + n*wgt_sN);


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax1d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC];
  {
    GET_COORD1
    GET_COORD2
    GET_SIGN1 
    GET_SIGN2
    GET_WARP1 
    GET_WARP2
    GET_GRD_POINTER

    const reduce_t *a = absolute, *m = membrane, *b = bending;
    reduce_t aa, mm, bb;

    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
    {
      scalar_t center = *sol; 
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      aa = *(a++);
      mm = *(m++);
      bb = *(b++);

      reduce_t w100 = mm * m100 + bb * b100;

      val[c] = (*grd) - (
            aa * center
          +(w100*(get(sol, x0,    sx0)     + get(sol, x1,    sx1)))
          + bb * (
            (b200*(get(sol, x00,   sx00)    + get(sol, x11,   sx11))) )
      );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax1d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC];
  {
    GET_COORD1
    GET_SIGN1 
    GET_WARP1 
    GET_GRD_POINTER

    const reduce_t *a = absolute, *m = membrane;

    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
    {
      scalar_t center = *sol; 
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      val[c] = (*grd) - (
            (*(a++)) * center
          + (*(m++)) * (
              m100*(get(sol, x0, sx0) + get(sol, x1, sx1)) )
      );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);

}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax1d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC];
  {
    GET_GRD_POINTER
    for (offset_t c = 0; c < C; ++c, sol += sol_sC, grd += grd_sC)
      val[c] = (*grd) - ( absolute[c] * (*sol) );
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, w000);
}



template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax1d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC], wval[MaxC];
  {
    GET_COORD1
    GET_SIGN1
    GET_WARP1_RLS
    GET_GRD_POINTER
    GET_WGT_POINTER

    const reduce_t *a = absolute, *m = membrane;
    reduce_t aa, mm;

    for (offset_t c = 0; c < C; 
         ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC)
    {
      scalar_t wcenter = *wgt;
      reduce_t w1m00 = m100 * (wcenter + bound::get(wgt, wx0, sx0));
      reduce_t w1p00 = m100 * (wcenter + bound::get(wgt, wx1, sx1));

      scalar_t center = *sol;  // no need to use `get` -> we know we are in the FOV
      auto get = [center](const scalar_t * x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      aa = *(a++);
      mm = *(m++) * 0.5;

      val[c] = (*grd) - (
          aa * wcenter * center
        + mm * (
            w1m00 * get(sol, x0, sx0)
          + w1p00 * get(sol, x1, sx1) )
      );

      wval[c] = ( aa * wcenter
                - mm * (w1m00 + w1p00) );
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, wval);
}


template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::relax1d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_SOL_POINTER
  reduce_t val[MaxC], wval[MaxC];
  {
    GET_GRD_POINTER
    GET_WGT_POINTER

    for (offset_t c = 0; c < C; 
         ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC) {
      scalar_t wcenter = *wgt;
      val[c]  = (*grd) - ( absolute[c] * wcenter * (*sol) );
      wval[c] = absolute[c] * wcenter;
    }
  }

  sol -= C*sol_sC;
  GET_HES_POINTER
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, wval);
}

/* ========================================================================== */
/*                                     SOLVE                                  */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::solve1d(
  offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  reduce_t val[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + z*grd_sZ + n*grd_sN);
    for (offset_t c = 0; c < C; ++c,  grd += grd_sC) 
      val[c] = *grd;
  }
       
  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + z*sol_sZ + n*sol_sN);
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, static_cast<reduce_t*>(0));
}

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::solve2d(
  offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  reduce_t val[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + y*grd_sY + n*grd_sN);
    for (offset_t c = 0; c < C; ++c,  grd += grd_sC) 
      val[c] = *grd;
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + y*sol_sY + n*sol_sN);
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, static_cast<reduce_t*>(0));
}

template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t>::solve3d(
  offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  reduce_t val[MaxC];
  {
    const scalar_t *grd = grd_ptr + (x*grd_sX + n*grd_sN);
    for (offset_t c = 0; c < C; ++c,  grd += grd_sC) 
      val[c] = *grd;
  }

  const scalar_t *hes = hes_ptr + (x*hes_sX + n*hes_sN);
        scalar_t *sol = sol_ptr + (x*sol_sX + n*sol_sN);
  hessian_t::addinvert(C, sol, sol_sC, hes, hes_sC, val, static_cast<reduce_t*>(0));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t, typename reduce_t, typename hessian_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void relax_kernel(const RelaxImpl<scalar_t,offset_t,reduce_t,hessian_t> * f) {
  f->loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}

template <typename offset_t, typename T, typename Stream>
NI_HOST void copy_fold_info_to_device(const T & obj, T * ptr, Stream stream)
{
  // Super hacky !!!
  // we copy the first 7 fields of the object to device
  // [fx fy fz Xf Yf Zf redblack]
  auto field_ptr_in  = reinterpret_cast<const offset_t *>(&obj);
  auto field_ptr_out = reinterpret_cast<offset_t *>(ptr);
  
  cudaMemcpyAsync(field_ptr_out, field_ptr_in, 7*sizeof(offset_t), 
                  cudaMemcpyHostToDevice, stream);
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
  f(const A & alloc, const H & hessian_type, const S & scalar_type, int64_t nb_iter) 
  {
    NI_DISPATCH_HESSIAN_TYPE(hessian_type, [&] {
      AT_DISPATCH_FLOATING_TYPES(scalar_type, "precond_impl", [&] {
        using utils_t = HessianUtils<hessian_t, MaxC>;
#ifdef __CUDACC__
        auto stream = at::cuda::getCurrentCUDAStream();
        if (alloc.canUse32BitIndexMath())
        {
          using Impl = RelaxImpl<scalar_t, int32_t, double, utils_t>;
          Impl   algo(alloc);
          Impl * palgo = alloc_and_copy_to_device(&algo, stream);
          for (int32_t i=0; i < nb_iter; ++i)
            for (int32_t fold = 0; fold < algo.foldcount(); ++fold) {
                algo.set_fold(fold);
                copy_fold_info_to_device<int32_t>(algo, palgo, stream);
                relax_kernel
                  <<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0, stream>>>
                  (palgo);
            }
          cudaFree(palgo);
        }
        else
        {
          using Impl = RelaxImpl<scalar_t, int64_t, double, utils_t>;
          Impl   algo(alloc);
          Impl * palgo = alloc_and_copy_to_device(&algo, stream);
          for (int64_t i=0; i < nb_iter; ++i)
            for (int64_t fold = 0; fold < algo.foldcount(); ++fold) {
                algo.set_fold(fold);
                copy_fold_info_to_device<int64_t>(algo, palgo, stream);
                relax_kernel
                  <<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0, stream>>>
                  (palgo);
            }
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
        RelaxImpl<scalar_t, int64_t, double, utils_t> algo(alloc);
        for (int64_t i=0; i < nb_iter; ++i)
          algo.loop();
#endif
      });
    });
  }
};


template <>
struct DoAlgo<0> {
  static const int32_t MaxC = power_of_two<0>::value;

  template <typename A, typename H, typename S>
  static NI_HOST void 
  f(const A & alloc, const H & hessian_type, const S & scalar_type, int64_t nb_iter) 
  {
    NI_DISPATCH_HESSIAN_TYPE0(hessian_type, [&] {
      AT_DISPATCH_FLOATING_TYPES(scalar_type, "precond_impl", [&] {
        using utils_t = HessianUtils<hessian_t, MaxC>;
#ifdef __CUDACC__
        auto stream = at::cuda::getCurrentCUDAStream();
        if (alloc.canUse32BitIndexMath())
        {
          using Impl = RelaxImpl<scalar_t, int32_t, double, utils_t>;
          Impl   algo(alloc);
          Impl * palgo = alloc_and_copy_to_device(&algo, stream);
          for (int32_t i=0; i < nb_iter; ++i)
            for (int32_t fold = 0; fold < algo.foldcount(); ++fold) {
                algo.set_fold(fold);
                copy_fold_info_to_device<int32_t>(algo, palgo, stream);
                relax_kernel
                  <<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0, stream>>>
                  (palgo);
            }
          cudaFree(palgo);
        }
        else
        {
          using Impl = RelaxImpl<scalar_t, int64_t, double, utils_t>;
          Impl   algo(alloc);
          Impl * palgo = alloc_and_copy_to_device(&algo, stream);
          for (int64_t i=0; i < nb_iter; ++i)
            for (int64_t fold = 0; fold < algo.foldcount(); ++fold) {
                algo.set_fold(fold);
                copy_fold_info_to_device<int64_t>(algo, palgo, stream);
                relax_kernel
                  <<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0, stream>>>
                  (palgo);
            }
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
        RelaxImpl<scalar_t, int64_t, double, utils_t> algo(alloc);
        for (int64_t i=0; i < nb_iter; ++i)
          algo.loop();
#endif
      });
    });
  }
};


template <>
struct DoAlgo<1> {
  static const int32_t MaxC = power_of_two<1>::value;

  template <typename A, typename H, typename S>
  static NI_HOST void 
  f(const A & alloc, const H & hessian_type, const S & scalar_type, int64_t nb_iter) 
  {
    NI_DISPATCH_HESSIAN_TYPE1(hessian_type, [&] {
      AT_DISPATCH_FLOATING_TYPES(scalar_type, "precond_impl", [&] {
        using utils_t = HessianUtils<hessian_t, MaxC>;
#ifdef __CUDACC__
        auto stream = at::cuda::getCurrentCUDAStream();
        if (alloc.canUse32BitIndexMath())
        {
          using Impl = RelaxImpl<scalar_t, int32_t, double, utils_t>;
          Impl   algo(alloc);
          Impl * palgo = alloc_and_copy_to_device(&algo, stream);
          for (int32_t i=0; i < nb_iter; ++i)
            for (int32_t fold = 0; fold < algo.foldcount(); ++fold) {
                algo.set_fold(fold);
                copy_fold_info_to_device<int32_t>(algo, palgo, stream);
                relax_kernel
                  <<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0, stream>>>
                  (palgo);
            }
          cudaFree(palgo);
        }
        else
        {
          using Impl = RelaxImpl<scalar_t, int64_t, double, utils_t>;
          Impl   algo(alloc);
          Impl * palgo = alloc_and_copy_to_device(&algo, stream);
          for (int64_t i=0; i < nb_iter; ++i)
            for (int64_t fold = 0; fold < algo.foldcount(); ++fold) {
                algo.set_fold(fold);
                copy_fold_info_to_device<int64_t>(algo, palgo, stream);
                relax_kernel
                  <<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0, stream>>>
                  (palgo);
            }
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
        RelaxImpl<scalar_t, int64_t, double, utils_t> algo(alloc);
        for (int64_t i=0; i < nb_iter; ++i)
          algo.loop();
#endif
      });
    });
  }
};


template<typename A, typename H, typename S, int32_t... Indices>
void
dispatch(int32_t i, const A & a, const H & h, const S & s, int64_t nb_iter,
         indices<0, Indices...>)
{
    static void (*lookup[])(const A &, const H &, const S &, int64_t) 
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
    lookup[logi](a, h, s, nb_iter);
}

template<int N, typename A, typename H, typename S>
void dispatch(int i, const A & a, const H & h, const S & s, int64_t nb_iter)
{
  dispatch(i, a, h, s, nb_iter, typename build_indices<0, N>::type()); 
}

} // anonymous namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST Tensor relax_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending, 
  ArrayRef<double> voxel_size, BoundVectorRef bound, int64_t nb_iter)
{
  auto tensors = prepare_tensors(gradient, hessian, solution, weight);
  hessian  = std::get<0>(tensors);
  solution = std::get<1>(tensors);
  weight   = std::get<2>(tensors);

  RelaxAllocator info(gradient.dim()-2, absolute, membrane, bending,
                      voxel_size, bound);
  info.ioset(hessian, gradient, solution, weight);

  const auto & T  = gradient.scalar_type();
  const auto & C  = gradient.size(1);
  const auto & CC = (hessian.defined() && hessian.numel() > 0 ? hessian.size(1) : 0);
  const auto & H  = guess_hessian_type(C, CC);

  if (C <= 4) 
    dispatch<3>(C, info, H, T, nb_iter); // {2**0 2**1, 2**2}
  else
    DoAlgo<6>::f(info, H, T, nb_iter);      // MAX CHANNELS = 64 = 2**5
  return solution;
}

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST Tensor relax_impl(
  Tensor hessian, const Tensor& gradient, Tensor solution, Tensor weight,
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound, int64_t nb_iter)
{
  throw std::logic_error("Function not implemented for this device.");
}


} // namespace notimplemented

} // namespace ni

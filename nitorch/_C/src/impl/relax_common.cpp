#include "common.h"
#include "bounds_common.h"
#include "allocator.h"
#include <ATen/ATen.h>
#include <limits>
#include <vector>

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
using std::vector;

// Required for stability. Value is currently about 1+8*eps
#define OnePlusTiny 1.000001

// Macro to cleanly invoke a pointer to member function
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)

#define VEC_UNFOLD(ONAME, INAME, DEFAULT)             \
  ONAME##0(INAME.size() > 0 ? INAME[0] : DEFAULT),  \
  ONAME##1(INAME.size() > 1 ? INAME[1] :            \
           INAME.size() > 0 ? INAME[0] : DEFAULT),  \
  ONAME##2(INAME.size() > 2 ? INAME[2] :            \
           INAME.size() > 1 ? INAME[1] :            \
           INAME.size() > 0 ? INAME[0] : DEFAULT)

#ifndef NI_MAX_NUM_CHANNELS
# define NI_MAX_NUM_CHANNELS 1024
#endif

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

  static constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();

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
  template <typename scalar_t, typename offset_t>
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
  grd_ptr = hes_ptr = sol_ptr = wgt_ptr = static_cast<float*>(0);
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

template <typename offset_t>
NI_HOST NI_INLINE bool any(const double * v, offset_t C) {
  for (offset_t c = 0; c < C; ++c, ++v) {
    if (*v) return true;
  }
  return false;
}

template <typename scalar_t, typename offset_t>
class RelaxImpl {

  typedef RelaxImpl Self;
  typedef void (Self::*RelaxFn)(offset_t x, offset_t y, offset_t z, offset_t n) const;
  typedef void (Self::*InvertFn)(scalar_t *, double *, double *, const double *) const;
  typedef void (Self::*GetHFn)(const scalar_t *, double *) const;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  RelaxImpl(const RelaxAllocator & info):
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
    set_factors(info.absolute, info.membrane, info.bending);
    set_kernel(info.dim, info.vx0, info.vx1, info.vx2);
    set_relax(info.dim);
    set_bandwidth();
    set_invert();
  }

  NI_HOST void set_factors(ArrayRef<double> a, ArrayRef<double> m, ArrayRef<double> b)
  {
    for (offset_t c = 0; c < C; ++c)
    {
      absolute[c] = (a.size() == 0                       ? 0. : 
                     static_cast<offset_t>(a.size()) > c ? a[c] 
                                                         : absolute[c-1]);
      membrane[c] = (m.size() == 0                       ? 0. : 
                     static_cast<offset_t>(m.size()) > c ? m[c] 
                                                         : membrane[c-1]);
      bending[c]  = (b.size()  == 0                      ? 0. : 
                     static_cast<offset_t>(b.size()) > c ? b[c]  
                                                         : bending[c-1]);
    }
  } 

  NI_HOST NI_INLINE void set_kernel(int64_t dim, double vx0, double vx1, double vx2) 
  {
    for (offset_t c=0; c < C; ++c)
      w000[c] = bending[c]  * (6.0*(vx0*vx0+vx1*vx1+vx2*vx2) + 8*(vx0*vx1+vx0*vx2+vx1*vx2)) + 
                membrane[c] * 2*(vx0+vx1+vx2) + 
                absolute[c];
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
  }

  NI_HOST NI_INLINE void set_relax(int64_t dim) 
  {
    bool has_membrane = any(membrane, C);
    bool has_absolute = any(absolute, C);
    bool has_bending  = any(bending, C);

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

  NI_HOST NI_INLINE void set_bandwidth() 
  { 
    if (any(bending, C))
      bandwidth = 3;
    else if (any(membrane, C))
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

  NI_HOST NI_INLINE void set_invert() 
  {
    if (hes_ptr) {
      if (CC == 1) {
        invert_ = &Self::invert_eye;
        get_h_  = &Self::get_h_eye;
      } else if (CC == C) {
        invert_ = &Self::invert_diag;
        get_h_  = &Self::get_h_diag;
      } else {
        invert_ = &Self::invert_sym;
        get_h_  = &Self::get_h_sym;
      }
    } else {
      invert_ = &Self::invert_none;
      get_h_  = &Self::get_h_none;
    }
  }

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
      fy = (i/Fz)  % Fy;
      fz = i % Fz;

      Xf = 1 + (X - fx - 1) / Fx;
      Yf = 1 + (Y - fy - 1) / Fy;
      Zf = 1 + (Z - fz - 1) / Fz;
    }
  }
 

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  NI_DEVICE NI_INLINE void relax(
    offset_t x, offset_t y, offset_t z, offset_t n) const;

#define DEFINE_relax(SUFFIX) \
  NI_DEVICE void relax##SUFFIX( \
    offset_t x, offset_t y, offset_t z, offset_t n) const;
#define DEFINE_relax_DIM(DIM)        \
  DEFINE_relax(DIM##d_absolute)      \
  DEFINE_relax(DIM##d_membrane)      \
  DEFINE_relax(DIM##d_bending)       \
  DEFINE_relax(DIM##d_rls_absolute)  \
  DEFINE_relax(DIM##d_rls_membrane)  \
  NI_DEVICE void solve##DIM##d(      \
    offset_t x, offset_t y, offset_t z, offset_t n) const;

  DEFINE_relax_DIM(1)
  DEFINE_relax_DIM(2)
  DEFINE_relax_DIM(3)

  NI_DEVICE void get_h(const scalar_t * , double *) const;
  NI_DEVICE void get_h_sym(const scalar_t * , double *) const;
  NI_DEVICE void get_h_diag(const scalar_t * , double *) const;
  NI_DEVICE void get_h_eye(const scalar_t * , double *) const;
  NI_DEVICE void get_h_none(const scalar_t * , double *) const;

  NI_DEVICE void invert(
    scalar_t *, const scalar_t *, double *, const double *) const;
  // NI_DEVICE void invert3(
  //   scalar_t *, const scalar_t *, double *, const double *) const;
  NI_DEVICE void invert_sym(
    scalar_t *, double *, double *, const double *) const;
  NI_DEVICE void invert_diag(
    scalar_t *, double *, double *, const double *) const;
  NI_DEVICE void invert_eye(
    scalar_t *, double *, double *, const double *) const;
  NI_DEVICE void invert_none(
    scalar_t *, double *, double *, const double *) const;

  NI_DEVICE void cholesky(double a[], double p[]) const;
  NI_DEVICE void cholesky_solve(const double a[], const double p[], double x[]) const;

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  double            absolute[NI_MAX_NUM_CHANNELS];       // penalty on absolute values
  double            membrane[NI_MAX_NUM_CHANNELS];       // penalty on first derivatives
  double            bending[NI_MAX_NUM_CHANNELS];        // penalty on second derivatives
  RelaxFn           relax_;         // Pointer to relax function
  InvertFn          invert_;        // Pointer to inversion function
  GetHFn            get_h_;         // Pointer to inversion function

  double w000[NI_MAX_NUM_CHANNELS];
  double m100;
  double m010;
  double m001;
  double b100;
  double b010;
  double b001;
  double b200;
  double b020;
  double b002;
  double b110;
  double b101;
  double b011;


  // ~~~ FOLD NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  offset_t bandwidth;
  offset_t Fx; // Fold window
  offset_t Fy;
  offset_t Fz;
  offset_t fx; // Index of the fold
  offset_t fy;
  offset_t fz;
  offset_t Xf; // Size of the fold
  offset_t Yf;
  offset_t Zf;
  offset_t redblack;  // Index of the fold for checkerboard scheme

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

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax(
    offset_t x, offset_t y, offset_t z, offset_t n) const {
    CALL_MEMBER_FN(*this, relax_)(x, y, z, n);
}

#if __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  if (bandwidth == 0)
    return loop_redblack(threadIdx, blockIdx, blockDim, gridDim);
  else
    return loop_band(threadIdx, blockIdx, blockDim, gridDim);
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::loop_band(
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

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::loop_redblack(
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

template <typename scalar_t, typename offset_t> NI_HOST
void RelaxImpl<scalar_t,offset_t>::loop()
{
  if (bandwidth == 0)
    return loop_redblack();
  else
    return loop_band();
}

template <typename scalar_t, typename offset_t> NI_HOST
void RelaxImpl<scalar_t,offset_t>::loop_redblack()
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

template <typename scalar_t, typename offset_t> NI_HOST
void RelaxImpl<scalar_t,offset_t>::loop_band()
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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             Cholesky
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Cholesky decomposition (actually, LDL decomposition)
// @param[inout]  a: CxC matrix
// @param[out]    p: C vector
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::cholesky(double a[], double p[]) const
{
  double sm, sm0;

  
  sm0  = 1e-40;
  for(offset_t c = 0; c < C; ++c) sm0 += a[c*C+c];
  sm0 *= 1e-7;
  sm0 *= sm0;
  // sm0 = MAX(sm0, 1e-16);

  for (offset_t c = 0; c < C; ++c)
  {
    for (offset_t b = c; b < C; ++b)
    {
      sm = a[c*C+b];
      for(offset_t d = c-1; d >= 0; --d)
        sm -= a[c*C+d] * a[b*C+d];
      if (c == b)
      {
        sm = MAX(sm, sm0);
        p[c] = std::sqrt(sm);
      }
      else 
        a[b*C+c] = sm / p[c];
    }
  }
}

// Cholesky solver (inplace)
// @param[in]    a: CxC matrix
// @param[in]    p: C vector
// @param[inout] x: C vector
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::cholesky_solve(
    const double a[], const double p[], double x[]) const
{
  double sm;

  for (offset_t c = 0; c < C; ++c)
  {
    sm = x[c];
    for (offset_t cc = c-1; cc >= 0; --cc)
      sm -= a[c*C+cc] * x[cc];
    x[c] = sm / p[c];
  }
  for(offset_t c = C-1; c >= 0; --c)
  {
    sm = x[c];
    for(offset_t cc = c+1; cc < C; ++cc)
      sm -= a[cc*C+c] * x[cc];
    x[c] = sm / p[c];
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             Invert
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Our relaxation routines perform
//      x += (H + diag(w)) \ ( g - (H + L) * x )
// Often, g is the gradient, (H+L) is the Hessian where H is easy
// to invert, x is the previous estimate and w is a stabilizing
// constant (in our case, the diagonal of L)
// This subroutines expects
//      v = g - L * x
// and performs
//      v -= H * x
//      x += (H + diag(w)) \ v
// (k is a placeholder to store cholesky coefficients)


template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::invert(
    scalar_t * x, const scalar_t * h, double * v, const double * w) const {
  double m[NI_MAX_NUM_CHANNELS*NI_MAX_NUM_CHANNELS];
  get_h(h, m);
  CALL_MEMBER_FN(*this, invert_)(x, m, v, w);
}

#if 0
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::invert3(
    scalar_t * x, const scalar_t * h, double * v, const double * w) const 
{

  double h00 = h[0],        h11 = h[  hes_sC], h22 = h[2*hes_sC],
         h01 = h[3*hes_sC], h02 = h[4*hes_sC], h12 = h[5*hes_sC],
         x0  = x[0],        x1  = x[  sol_sC], x2  = x[2*sol_sC],
         idt;

  // matvec
  v[0] -= (h00*x0 + h01*x1 + h02*x2);
  v[1] -= (h01*x0 + h11*x1 + h12*x2);
  v[2] -= (h02*x0 + h12*x1 + h22*x2);

  // solve
  h00  = h00 * OnePlusTiny + w[0];
  h11  = h11 * OnePlusTiny + w[1];
  h22  = h22 * OnePlusTiny + w[2];
  idt  = 1.0/(h00*h11*h22 - h00*h12*h12 - h11*h02*h02 - h22*h01*h01 + 2*h01*h02*h12);
  x[       0] += idt*(v[0]*(h11*h22-h12*h12) + v[1]*(h02*h12-h01*h22) + v[2]*(h01*h12-h02*h11));
  x[  sol_sC] += idt*(v[0]*(h02*h12-h01*h22) + v[1]*(h00*h22-h02*h02) + v[2]*(h01*h02-h00*h12));
  x[2*sol_sC] += idt*(v[0]*(h01*h12-h02*h11) + v[1]*(h01*h02-h00*h12) + v[2]*(h00*h11-h01*h01));
}
#endif

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::invert_sym(
    scalar_t * x, double * h, double * v, const double * w) const {

  for (offset_t c = 0; c < C; ++c) {
    // matvec (part of the forward pass)
    v[c] -= h[c*C+c] * x[c*sol_sC];
    for (offset_t cc = c+1; cc < C; ++cc)
      v[c] -= h[c*C+cc] * x[cc*sol_sC];
    // load diagonal
    h[c+C*c] += w[c];
  }
  double k[NI_MAX_NUM_CHANNELS];
  cholesky(h, k);            // cholesky decomposition
  cholesky_solve(h, k, v);   // solve linear system inplace
  for (offset_t c = 0; c < C; ++c)
    x[c*sol_sC] += v[c];
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::invert_diag(
    scalar_t * x, double * h, double * v, const double * w) const {

  for (offset_t c = 0; c < C; ++c) {
    v[c] -= h[c] * x[c*sol_sC];
    x[c*sol_sC] += v[c] / (h[c] + w[c]);
  }
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::invert_eye(
    scalar_t * x, double * h, double * v, const double * w) const {
  double hh = h[0];
  for (offset_t c = 0; c < C; ++c) {
    v[c] -= hh * x[c*sol_sC];
    x[c*sol_sC] += v[c] / (hh + w[c]);
  }
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::invert_none(
    scalar_t * x, double * h, double * v, const double * w) const {
  for (offset_t c = 0; c < C; ++c)
    x[c*sol_sC] += *(v++) / *(w++);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             GetH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::get_h(
    const scalar_t * h, double * m) const {
  CALL_MEMBER_FN(*this, get_h_)(h, m);
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::get_h_sym(
    const scalar_t * hessian, double * mat) const {
  for (offset_t c = 0; c < C; ++c, hessian += hes_sC)
    mat[c+C*c] = *hessian;
  for (offset_t c = 0; c < C; ++c)
    for (offset_t cc = c+1; cc < C; ++cc, hessian += hes_sC)
      mat[c+C*cc] = mat[cc+C*c] = *hessian;
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::get_h_diag(
    const scalar_t * hessian, double * mat) const {
  for (offset_t c = 0; c < C; ++c, hessian += hes_sC)
    mat[c] = *hessian;
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::get_h_eye(
    const scalar_t * hessian, double * mat) const {
  *mat = *hessian;
}

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::get_h_none(
    const scalar_t * hessian, double * mat) const 
{}


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


template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax3d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1 
  GET_SIGN2
  GET_WARP1 
  GET_WARP2
  GET_POINTERS

  double val[NI_MAX_NUM_CHANNELS];
  const double *a = absolute, *m = membrane, *b = bending;
  double aa, mm, bb;

  for (offset_t c = 0; c < C; 
       ++c, sol += sol_sC, grd += grd_sC)
  {
    scalar_t center = *sol; 
    auto get = [center](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    aa = *(a++);
    mm = *(m++);
    bb = *(b++);

    double w100 = mm * m100 + bb * b100;
    double w010 = mm * m010 + bb * b010;
    double w001 = mm * m001 + bb * b001;

    val[c] = (*grd) - (
          aa * center
        + w100*(get(sol, x0,    sx0)     + get(sol, x1,    sx1))
        + w010*(get(sol, y0,    sy0)     + get(sol, y1,    sy1))
        + w001*(get(sol, z0,    sz0)     + get(sol, z1,    sz1))
        + bb * (
            b110*(get(sol, x0+y0, sx0*sy0) + get(sol, x1+y0, sx1*sy0) +
                  get(sol, x0+y1, sx1*sy1) + get(sol, x1+y1, sx1*sy1))
          + b101*(get(sol, x0+z0, sx0*sz0) + get(sol, x1+z0, sx1*sz0) +
                  get(sol, x0+z1, sx1*sz1) + get(sol, x1+z1, sx1*sz1))
          + b011*(get(sol, y0+z0, sy0*sz0) + get(sol, y1+z0, sy1*sz0) +
                  get(sol, y0+z1, sy1*sz1) + get(sol, y1+z1, sy1*sz1))
          + b200*(get(sol, x00,   sx00)    + get(sol, x11,   sx11))
          + b020*(get(sol, y00,   sy00)    + get(sol, y11,   sy11))
          + b002*(get(sol, z00,   sz00)    + get(sol, z11,   sz11)) )
    );
  }

  sol -= C*sol_sC;
  invert(sol, hes, val, w000);
}


template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax3d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS

  double val[NI_MAX_NUM_CHANNELS];
  const double *a = absolute, *m = membrane;

  for (offset_t c = 0; c < C; 
       ++c, sol += sol_sC, grd += grd_sC)
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

  sol -= C*sol_sC;
  invert(sol, hes, val, w000);

}


template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax3d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  double val[NI_MAX_NUM_CHANNELS];
  for (offset_t c = 0; c < C; 
       ++c, sol += sol_sC, grd += grd_sC)
    val[c] = (*grd) - ( absolute[c] * (*sol) );

  sol -= C*sol_sC;
  invert(sol, hes, val, w000);
}



template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS

  double val[NI_MAX_NUM_CHANNELS], wval[NI_MAX_NUM_CHANNELS];
  const double *a = absolute, *m = membrane;
  double aa, mm;

  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  for (offset_t c = 0; c < C; 
       ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC)
  {
    scalar_t wcenter = *wgt;
    double w1m00 = m100 * (wcenter + bound::get(wgt, wx0, sx0));
    double w1p00 = m100 * (wcenter + bound::get(wgt, wx1, sx1));
    double w01m0 = m010 * (wcenter + bound::get(wgt, wy0, sy0));
    double w01p0 = m010 * (wcenter + bound::get(wgt, wy1, sy1));
    double w001m = m001 * (wcenter + bound::get(wgt, wz0, sz0));
    double w001p = m001 * (wcenter + bound::get(wgt, wz1, sz1));

    scalar_t center = *sol;  // no need to use `get` -> we know we are in the FOV
    auto get = [center](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    aa = *(a++);
    mm = *(m++);

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
              + mm * (w1m00 + w1p00 + w01m0 + w01p0 + w001m + w001p) );
  }

  sol -= C*sol_sC;
  invert(sol, hes, val, wval);
}


template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  double val[NI_MAX_NUM_CHANNELS], wval[NI_MAX_NUM_CHANNELS];
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  for (offset_t c = 0; c < C; 
       ++c, sol += sol_sC, grd += grd_sC, wgt += wgt_sC) {
    scalar_t wcenter = *wgt;
    val[c]  = (*grd) - ( absolute[c] * wcenter * (*sol) );
    wval[c] = absolute[c] * wcenter;
  }

  sol -= C*sol_sC;
  invert(sol, hes, val, wval);
}



/* ========================================================================== */
/*                                     2D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax2d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax2d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax2d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax2d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax2d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}

/* ========================================================================== */
/*                                     1D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax1d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax1d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax1d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax1d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::relax1d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}


/* ========================================================================== */
/*                                     SOLVE                                  */
/* ========================================================================== */

template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::solve1d(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::solve2d(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RelaxImpl<scalar_t,offset_t>::solve3d(offset_t x, offset_t y, offset_t z, offset_t n) const {}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void regulariser_kernel(RelaxImpl<scalar_t,offset_t> f) {
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
//                    FUNCTIONAL FORM WITH relax
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradient.scalar_type(), "relax_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      RelaxImpl<scalar_t, int32_t> algo(info);
      for (int64_t i=0; i < nb_iter; ++i)
        for (offset_t fold = 0; fold < algo.foldcount(); ++fold) {
            algo.set_fold(fold);
            relax_kernel<<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
        }
    }
    else
    {
      RelaxImpl<scalar_t, int64_t> algo(info);
      for (int64_t i=0; i < nb_iter; ++i)
        for (offset_t fold = 0; fold < algo.foldcount(); ++fold) {
            algo.set_fold(fold);
            relax_kernel<<<GET_BLOCKS(algo.voxcountfold()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
        }
    }
  });
  return solution;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

  AT_DISPATCH_FLOATING_TYPES(gradient.scalar_type(), "relax_impl", [&] {
    RelaxImpl<scalar_t, int64_t> algo(info);
    for (int64_t i=0; i < nb_iter; ++i)
      algo.loop();
  });
  return solution;
}

#endif // __CUDACC__

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

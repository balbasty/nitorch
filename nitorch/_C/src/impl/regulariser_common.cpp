#include "common.h"                // write C++/CUDA compatible code
#include "../defines.h"            // useful macros
#include "bounds_common.h"         // boundary conditions + enum
#include "allocator.h"             // base class handling offset sizes
// #include "utils.h"                 // unrolled for loop.h"
#include <ATen/ATen.h>             // tensors

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
class RegulariserAllocator: public Allocator {
public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST
  RegulariserAllocator(int dim, ArrayRef<double> absolute, 
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
    return inp_32b_ok && wgt_32b_ok && out_32b_ok && hes_32b_ok;
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
  ArrayRef<double>  absolute;       // penalty on absolute values
  ArrayRef<double>  membrane;       // penalty on first derivatives
  ArrayRef<double>  bending;        // penalty on second derivatives

  /* ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#define DECLARE_ALLOC_INFO_5D(NAME)  \
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
  DECLARE_ALLOC_INFO_5D(inp)
  DECLARE_ALLOC_INFO_5D(wgt)
  DECLARE_ALLOC_INFO_5D(out)
  DECLARE_ALLOC_INFO_5D(hes)

  // Allow RegulariserImpl's constructor to access RegulariserAllocator's
  // private members.
  template <typename scalar_t, typename offset_t, typename reduce_t>
  friend class RegulariserImpl;
};


NI_HOST
void RegulariserAllocator::init_all()
{
  N = C = CC = X = Y = Z = 1L;
  inp_sN  = inp_sC   = inp_sX   = inp_sY  = inp_sZ   = 0L;
  wgt_sN  = wgt_sC   = wgt_sX   = wgt_sY  = wgt_sZ   = 0L;
  out_sN  = out_sC   = out_sX   = out_sY  = out_sZ   = 0L;
  hes_sN  = hes_sC   = hes_sX   = hes_sY  = hes_sZ   = 0L;
  inp_ptr = wgt_ptr = out_ptr = hes_ptr = static_cast<float*>(0);
  inp_32b_ok = wgt_32b_ok = out_32b_ok = hes_32b_ok = true;
}

NI_HOST
void RegulariserAllocator::init_input(const Tensor& input)
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
void RegulariserAllocator::init_weight(const Tensor& weight)
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
void RegulariserAllocator::init_output(const Tensor& output)
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
void RegulariserAllocator::init_hessian(const Tensor& hessian)
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

template <typename reduce_t, typename offset_t>
NI_HOST NI_INLINE bool any(const reduce_t * v, offset_t C) {
  for (offset_t c = 0; c < C; ++c, ++v) {
    if (*v) return true;
  }
  return false;
}

template <typename scalar_t, typename offset_t, typename reduce_t>
class RegulariserImpl {

  typedef RegulariserImpl Self;
  typedef void (Self::*Vel2MomFn)(offset_t x, offset_t y, offset_t z, offset_t n) const;
  typedef void (Self::*MatVecFn)(scalar_t *, const scalar_t *, const scalar_t *) const;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  RegulariserImpl(const RegulariserAllocator & info):
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

    INIT_ALLOC_INFO_5D(inp),
    INIT_ALLOC_INFO_5D(wgt),
    INIT_ALLOC_INFO_5D(out),
    INIT_ALLOC_INFO_5D(hes)
  {
    set_factors(info.absolute, info.membrane, info.bending);
    set_kernel(info.vx0, info.vx1, info.vx2);
#ifndef __CUDACC__
    set_vel2mom();
    set_matvec();
#endif
  }

  NI_HOST NI_INLINE void set_factors(ArrayRef<double> a, ArrayRef<double> m, ArrayRef<double> b)
  {
    using size_type = ArrayRef<double>::size_type;
    for (size_type c = 0; c < static_cast<size_type>(C); ++c)
    {
      absolute[c] = static_cast<reduce_t>(a.size() == 0 ? 0.   : 
                                          a.size() > c  ? a[c] : absolute[c-1]);
      membrane[c] = static_cast<reduce_t>(m.size() == 0 ? 0.   : 
                                          m.size() > c  ? m[c] : membrane[c-1]);
      bending[c]  = static_cast<reduce_t>(b.size() == 0 ? 0.   : 
                                          b.size() > c  ? b[c] : bending[c-1]);
    }
    has_absolute = any(absolute, C);
    has_membrane = any(membrane, C);
    has_bending  = any(bending, C);

    mode = dim 
         + (has_bending ? 12 : has_membrane ? 8 : has_absolute ? 4 : 0)
         + (wgt_ptr ? 16 : 0);
  } 

  NI_HOST NI_INLINE void set_kernel(double vx0, double vx1, double vx2) 
  {
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

#ifdef __CUDACC__
#else
  NI_HOST NI_INLINE void set_vel2mom() 
  {
    if (wgt_ptr)
    {
      if (has_bending)
        throw std::logic_error("RLS only implemented for absolute/membrane.");
      else if (dim == 1) {
        if (has_membrane)
            vel2mom = &Self::vel2mom1d_rls_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom1d_rls_absolute;
        else
            vel2mom = &Self::zeros;
      } else if (dim == 2) {
        if (has_membrane)
            vel2mom = &Self::vel2mom2d_rls_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom2d_rls_absolute;
        else
            vel2mom = &Self::zeros;
      } else if (dim == 3) {
        if (has_membrane)
            vel2mom = &Self::vel2mom3d_rls_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom3d_rls_absolute;
        else
            vel2mom = &Self::zeros;
      }
    }
    else if (dim == 1) {
        if (has_bending)
            vel2mom = &Self::vel2mom1d_bending;
        else if (has_membrane)
            vel2mom = &Self::vel2mom1d_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom1d_absolute;
        else
            vel2mom = &Self::zeros;
    } else if (dim == 2) {
        if (has_bending)
            vel2mom = &Self::vel2mom2d_bending;
        else if (has_membrane)
            vel2mom = &Self::vel2mom2d_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom2d_absolute;
        else
            vel2mom = &Self::zeros;
    } else if (dim == 3) {
        if (has_bending)
            vel2mom = &Self::vel2mom3d_bending;
        else if (has_membrane)
            vel2mom = &Self::vel2mom3d_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom3d_absolute;
        else
            vel2mom = &Self::zeros;
    } else
        throw std::logic_error("RLS only implemented for dimension 1/2/3.");
  }

  NI_HOST NI_INLINE void set_matvec() 
  {
    if (hes_ptr) 
    {
      if (CC == 1)
        matvec_ = &Self::matvec_eye;
      else if (CC == C)
        matvec_ = &Self::matvec_diag;
      else
        matvec_ = &Self::matvec_sym;
    } 
    else
      matvec_ = &Self::matvec_none;
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

  NI_HOST NI_INLINE int64_t voxcount() const { return N * X * Y * Z; }
 

private:

  /* ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  NI_DEVICE NI_INLINE void dispatch(
    offset_t x, offset_t y, offset_t z, offset_t n) const;
  NI_DEVICE NI_INLINE void zeros(
    offset_t x, offset_t y, offset_t z, offset_t n) const;

#define DECLARE_ALLOC_INFO_5D_VEL2MOM(SUFFIX) \
  NI_DEVICE void vel2mom##SUFFIX( \
    offset_t x, offset_t y, offset_t z, offset_t n) const;
#define DECLARE_ALLOC_INFO_5D_VEL2MOM_DIM(DIM)      \
  DECLARE_ALLOC_INFO_5D_VEL2MOM(DIM##d_absolute)  \
  DECLARE_ALLOC_INFO_5D_VEL2MOM(DIM##d_membrane)  \
  DECLARE_ALLOC_INFO_5D_VEL2MOM(DIM##d_bending)   \
  DECLARE_ALLOC_INFO_5D_VEL2MOM(DIM##d_rls_absolute)  \
  DECLARE_ALLOC_INFO_5D_VEL2MOM(DIM##d_rls_membrane)

  DECLARE_ALLOC_INFO_5D_VEL2MOM_DIM(1)
  DECLARE_ALLOC_INFO_5D_VEL2MOM_DIM(2)
  DECLARE_ALLOC_INFO_5D_VEL2MOM_DIM(3)

  NI_DEVICE void matvec(
    scalar_t *, const scalar_t *, const scalar_t *) const;
  NI_DEVICE void matvec_sym(
    scalar_t *, const scalar_t *, const scalar_t *) const;
  NI_DEVICE void matvec_diag(
    scalar_t *, const scalar_t *, const scalar_t *) const;
  NI_DEVICE void matvec_eye(
    scalar_t *, const scalar_t *, const scalar_t *) const;
  NI_DEVICE void matvec_none(
    scalar_t *, const scalar_t *, const scalar_t *) const;

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  offset_t          dim;            // dimensionality (1 or 2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  reduce_t          absolute[NI_MAX_NUM_CHANNELS]; // penalty on absolute values
  reduce_t          membrane[NI_MAX_NUM_CHANNELS]; // penalty on first derivatives
  reduce_t          bending[NI_MAX_NUM_CHANNELS];  // penalty on second derivatives

#ifndef __CUDACC__ // We cannot work with member function pointers in cuda
  Vel2MomFn         vel2mom;        // Pointer to vel2mom function
  MatVecFn          matvec_;        // Pointer to matvec function
#endif

  uint8_t   mode;
  bool      has_absolute;
  bool      has_membrane;
  bool      has_bending;
  reduce_t  m100;
  reduce_t  m010;
  reduce_t  m001;
  reduce_t  b100;
  reduce_t  b010;
  reduce_t  b001;
  reduce_t  b200;
  reduce_t  b020;
  reduce_t  b002;
  reduce_t  b110;
  reduce_t  b101;
  reduce_t  b011;

  /* ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#define DECLARE_STRIDE_INFO_5D(NAME)   \
  offset_t    NAME##_sN;               \
  offset_t    NAME##_sC;               \
  offset_t    NAME##_sX;               \
  offset_t    NAME##_sY;               \
  offset_t    NAME##_sZ;               \
  scalar_t *  NAME##_ptr;

  offset_t  N;
  offset_t  C;
  offset_t  CC;
  offset_t  X;
  offset_t  Y;
  offset_t  Z;
  DECLARE_STRIDE_INFO_5D(inp)
  DECLARE_STRIDE_INFO_5D(wgt)
  DECLARE_STRIDE_INFO_5D(out)
  DECLARE_STRIDE_INFO_5D(hes)
};


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                    LOOP                                    */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::dispatch(
    offset_t x, offset_t y, offset_t z, offset_t n) const {
#ifdef __CUDACC__
    // dispatch
#   define ABSOLUTE 4
#   define MEMBRANE 8
#   define BENDING  12
#   define RLS      16
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
      case 1 + BENDING:
        return vel2mom1d_bending(x, y, z, n);
      case 2 + BENDING:
        return vel2mom2d_bending(x, y, z, n);
      case 3 + BENDING:
        return vel2mom3d_bending(x, y, z, n);
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


#ifdef __CUDACC__

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::loop(
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
void RegulariserImpl<scalar_t,offset_t,reduce_t>::loop() const
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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             MatVec
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if 0
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec(
    scalar_t * x, const scalar_t * h, const scalar_t * y) const {
  if (!hes_ptr) return;
  double m[NI_MAX_NUM_CHANNELS*NI_MAX_NUM_CHANNELS];
  get_h(h, m);
  double v[NI_MAX_NUM_CHANNELS];
  double * vv = v;
  for (offset_t c = 0; c < C; ++c, ++vv, y += inp_sC)
    *vv = *y;
  CALL_MEMBER_FN(*this, matvec_)(x, m, v);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_sym(
    scalar_t * x, const double * h, const double * v) const 
{
  double placeholder[NI_MAX_NUM_CHANNELS];
  double * o = placeholder;
  for (offset_t c = 0; c < C; ++c, ++o, ++v, h += C+1)
    (*o) = (*h) * (*v);
  v -= C;
  h -= C*(C+1);
  o = placeholder;
  for (offset_t c = 0; c < C; ++c, ++o, ++v, h += C+1)
  {
    double v_ = (*v);
    double * oo = o + 1;
    const double * vv = v + 1;
    const double * hh = h + 1;
    for (offset_t cc = c+1; cc < C; ++cc, ++oo, ++vv, ++hh) 
    {
      double h_ = (*hh);
      (*oo) += h_ * v_;
      (*o)  += h_ * (*vv);
    }
  }
  o = placeholder;
  for (offset_t c = 0; c < C; ++c, ++o, x += out_sC)
    (*x) += (*o);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_diag(
    scalar_t * x, const double * h, const double * v) const 
{
  for (offset_t c = 0; c < C; ++c, ++v, ++h, x += out_sC)
    (*x) += (*v) * (*h);
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_eye(
    scalar_t * x, const double * h, const double * v) const 
{
  double hh = h[0];
  for (offset_t c = 0; c < C; ++c, ++v, x += out_sC)
    (*x) += (*v) * hh;
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_none(
    scalar_t * x, const double * h, const double * v) const {}

#else

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec(
    scalar_t * x, const scalar_t * h, const scalar_t * y) const {
  if (!hes_ptr) return;
#ifdef __CUDACC__
  if (CC == 1)
    return matvec_eye(x, h, y);
  else if (CC == C)
    return matvec_diag(x, h, y);
  else
    return matvec_sym(x, h, y);
#else
  CALL_MEMBER_FN(*this, matvec_)(x, h, y);
#endif
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_sym(
    scalar_t * x, const scalar_t * h, const scalar_t * v) const 
{
  for (offset_t c = 0; c < C; ++c)
    x[c*out_sC] += h[c*hes_sC] * v[c*inp_sC];

  h += C * hes_sC;
  for (offset_t c = 0; c < C; ++c)
  {
    reduce_t v_ = v[c * inp_sC];
    for (offset_t cc = c+1; cc < C; ++cc, h += hes_sC) 
    {
      reduce_t h_ = (*h);
      x[cc * out_sC] += h_ * v_;
      x[c  * out_sC] += h_ * v[cc * inp_sC];
    }
  }
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_diag(
    scalar_t * x, const scalar_t * h, const scalar_t * v) const 
{
  for (offset_t c = 0; c < C; ++c)
    x[c*out_sC] += h[c*hes_sC] * v[c*inp_sC];
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_eye(
    scalar_t * x, const scalar_t * h, const scalar_t * v) const 
{
  scalar_t h_ = h[0];
  for (offset_t c = 0; c < C; ++c)
    x[c*out_sC] += h_ * v[c*inp_sC];
}

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::matvec_none(
    scalar_t * x, const scalar_t * h, const scalar_t * v) const {}

#endif

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
        scalar_t *out = out_ptr + (x*out_sX + y*out_sY + z*out_sZ + n*out_sN); \
  const scalar_t *inp = inp_ptr + (x*inp_sX + y*inp_sY + z*inp_sZ + n*inp_sN); \
  const scalar_t *hes = hes_ptr + (x*hes_sX + y*hes_sY + z*hes_sZ + n*hes_sN);


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1 
  GET_SIGN2
  GET_WARP1 
  GET_WARP2
  GET_POINTERS

  reduce_t w100, w010, w001;
  const reduce_t *a = absolute, *m = membrane, *b = bending;
  reduce_t aa, mm, bb;

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC)
  {
    scalar_t center = *inp; 
    auto get = [center](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    aa = *(a++);
    mm = *(m++);
    bb = *(b++);

    w100 = mm * m100 + bb * b100;
    w010 = mm * m010 + bb * b010;
    w001 = mm * m001 + bb * b001;

    *out = static_cast<scalar_t>(
          aa * center
        + w100*(get(inp, x0,    sx0)     + get(inp, x1,    sx1))
        + w010*(get(inp, y0,    sy0)     + get(inp, y1,    sy1))
        + w001*(get(inp, z0,    sz0)     + get(inp, z1,    sz1))
        + bb * (
            b110*(get(inp, x0+y0, sx0*sy0) + get(inp, x1+y0, sx1*sy0) +
                  get(inp, x0+y1, sx1*sy1) + get(inp, x1+y1, sx1*sy1))
          + b101*(get(inp, x0+z0, sx0*sz0) + get(inp, x1+z0, sx1*sz0) +
                  get(inp, x0+z1, sx1*sz1) + get(inp, x1+z1, sx1*sz1))
          + b011*(get(inp, y0+z0, sy0*sz0) + get(inp, y1+z0, sy1*sz0) +
                  get(inp, y0+z1, sy1*sz1) + get(inp, y1+z1, sy1*sz1))
          + b200*(get(inp, x00,   sx00)    + get(inp, x11,   sx11))
          + b020*(get(inp, y00,   sy00)    + get(inp, y11,   sy11))
          + b002*(get(inp, z00,   sz00)    + get(inp, z11,   sz11)) )
    );
  }

  inp -= C*inp_sC;
  out -= C*out_sC;
  matvec(out, hes, inp);
}

#if 0
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS

  for_unroll(C, [&](offset_t c) {

    const scalar_t * inp_c = inp + c * inp_sC;
    scalar_t center = (*inp_c); 
    auto get = [center](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    out[c * out_sC] = static_cast<scalar_t>(
          absolute[c] * center
        + membrane[c] * (
            m100*(get(inp_c, x0, sx0) + get(inp_c, x1, sx1))
          + m010*(get(inp_c, y0, sy0) + get(inp_c, y1, sy1))
          + m001*(get(inp_c, z0, sz0) + get(inp_c, z1, sz1)) )
    );
  });

  matvec(out, hes, inp);
}
#endif

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS

  const reduce_t *a = absolute, *m = membrane;

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC)
  {
    scalar_t center = *inp; 
    auto get = [center](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    *out = static_cast<scalar_t>(
          (*(a++)) * center
        + (*(m++)) * (
            m100*(get(inp, x0, sx0) + get(inp, x1, sx1))
          + m010*(get(inp, y0, sy0) + get(inp, y1, sy1))
          + m001*(get(inp, z0, sz0) + get(inp, z1, sz1)) )
    );
  }

  inp -= C*inp_sC;
  out -= C*out_sC;
  matvec(out, hes, inp);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  const reduce_t *a = absolute;
  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC)
    *out = static_cast<scalar_t>( (*(a++)) * (*inp) );

  inp -= C*inp_sC;
  out -= C*out_sC;
  matvec(out, hes, inp);
}



template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS

  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);
  const reduce_t *a = absolute, *m = membrane;

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC, wgt += wgt_sC)
  {
    scalar_t wcenter = *wgt;
    reduce_t w1m00 = m100 * (wcenter + bound::get(wgt, wx0, sx0));
    reduce_t w1p00 = m100 * (wcenter + bound::get(wgt, wx1, sx1));
    reduce_t w01m0 = m010 * (wcenter + bound::get(wgt, wy0, sy0));
    reduce_t w01p0 = m010 * (wcenter + bound::get(wgt, wy1, sy1));
    reduce_t w001m = m001 * (wcenter + bound::get(wgt, wz0, sz0));
    reduce_t w001p = m001 * (wcenter + bound::get(wgt, wz1, sz1));

    scalar_t center = *inp;  // no need to use `get` -> we know we are in the FOV
    auto get = [center](const scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    *out = static_cast<scalar_t>(
        (*(a++)) * wcenter * center
      + (*(m++)) * (
          w1m00 * get(inp, x0, sx0)
        + w1p00 * get(inp, x1, sx1)
        + w01m0 * get(inp, y0, sy0)
        + w01p0 * get(inp, y1, sy1)
        + w001m * get(inp, z0, sz0)
        + w001p * get(inp, z1, sz1) )
    );
  }

  inp -= C*inp_sC;
  out -= C*out_sC;
  matvec(out, hes, inp);
}


template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);
  const reduce_t *a = absolute;

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC, wgt += wgt_sC)
    *out = static_cast<scalar_t>( (*(a++)) * (*wgt) * (*inp) );

  inp -= C*inp_sC;
  out -= C*out_sC;
  matvec(out, hes, inp);
}



/* ========================================================================== */
/*                                     2D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom2d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}

/* ========================================================================== */
/*                                     1D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::vel2mom1d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}

/* ========================================================================== */
/*                                     COPY                                   */
/* ========================================================================== */

template <typename scalar_t, typename offset_t, typename reduce_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t,reduce_t>::zeros(offset_t x, offset_t y, offset_t z, offset_t n) const 
{
  GET_POINTERS

  for (offset_t c = 0; c < C; out += out_sC)
    *out = static_cast<scalar_t>(0);

  out -= C*out_sC;
  matvec(out, hes, inp);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__

// CUDA Kernel

template <typename scalar_t, typename offset_t, typename reduce_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void regulariser_kernel(RegulariserImpl<scalar_t, offset_t, reduce_t> * f)
{
  f->loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
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
    if (dim == 2)
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

NI_HOST Tensor regulariser_impl(
  const Tensor& input, Tensor output, Tensor weight, Tensor hessian, 
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(input, output, weight, hessian);
  output       = std::get<0>(tensors);
  weight       = std::get<1>(tensors);
  hessian      = std::get<2>(tensors);

  RegulariserAllocator info(input.dim()-2, absolute, membrane, bending, voxel_size, bound);
  info.ioset(input, output, weight, hessian);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "regulariser_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      RegulariserImpl<scalar_t, int32_t, double> algo(info);
      auto palgo = copy_to_device(algo, stream);
      regulariser_kernel
          <<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>
          (palgo);
      cudaFree(palgo);
    }
    else
    {
      RegulariserImpl<scalar_t, int64_t, double> algo(info);
      auto palgo = copy_to_device(algo, stream);
      regulariser_kernel
          <<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0, stream>>>
          (palgo);
      cudaFree(palgo);
    }
  });
  return output;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST Tensor regulariser_impl(
  const Tensor& input, Tensor output, Tensor weight, Tensor hessian, 
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(input, output, weight, hessian);
  output       = std::get<0>(tensors);
  weight       = std::get<1>(tensors);
  hessian      = std::get<2>(tensors);

  RegulariserAllocator info(input.dim()-2, absolute, membrane, bending, voxel_size, bound);
  info.ioset(input, output, weight, hessian);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "regulariser_impl", [&] {
    RegulariserImpl<scalar_t, int64_t, double> algo(info);
    algo.loop();
  });
  return output;
}

#endif // __CUDACC__

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST Tensor regulariser_impl(
  const Tensor& input, Tensor output, Tensor weight, Tensor hessian, 
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  throw std::logic_error("Function not implemented for this device.");
}

} // namespace notimplemented

} // namespace ni

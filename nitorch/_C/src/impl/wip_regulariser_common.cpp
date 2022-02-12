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

  static constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();

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
    vx0 *= vx0;
    vx1 *= vx1;
    vx2 *= vx2;
    vx0 = 1. / vx0;
    vx1 = 1. / vx1;
    vx2 = 1. / vx2;
  }

  /* ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  NI_HOST void ioset
  (const Tensor& input, const Tensor& output, const Tensor& weight)
  {
    init_all();
    init_input(input);
    init_weight(weight);
    init_output(output);
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
  int64_t X;
  int64_t Y;
  int64_t Z;
  DEFINE_ALLOC_INFO_5D(inp)
  DEFINE_ALLOC_INFO_5D(wgt)
  DEFINE_ALLOC_INFO_5D(out)

  // Allow RegulariserImpl's constructor to access RegulariserAllocator's
  // private members.
  template <typename scalar_t, typename offset_t>
  friend class RegulariserImpl;
};


NI_HOST
void RegulariserAllocator::init_all()
{
  N = C = X = Y = Z = 1L;
  inp_sN  = inp_sC   = inp_sX   = inp_sY  = inp_sZ   = 0L;
  wgt_sN  = wgt_sC   = wgt_sX   = wgt_sY  = wgt_sZ   = 0L;
  out_sN  = out_sC   = out_sX   = out_sY  = out_sZ   = 0L;
  inp_ptr = wgt_ptr = out_ptr = static_cast<float*>(0);
  inp_32b_ok = wgt_32b_ok = out_32b_ok = true;
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

/* ========================================================================== */
/*                                                                            */
/*                                ALGORITHM                                   */
/*                                                                            */
/* ========================================================================== */

NI_HOST NI_INLINE bool any(const ArrayRef<double> & v) {
  for (auto it = v.cbegin(); it < v.cend(); ++it) {
    if (*it) return true;
  }
  return false;
}

template <typename scalar_t, typename offset_t>
class RegulariserImpl {

  typedef RegulariserImpl Self;
  typedef void (Self::*Vel2MomFn)(offset_t x, offset_t y, offset_t z, offset_t n) const;

public:

  /* ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  RegulariserImpl(const RegulariserAllocator & info):
    dim(info.dim),
    bound0(info.bound0), bound1(info.bound1), bound2(info.bound2),
    vx0(info.vx0), vx1(info.vx1), vx2(info.vx2),
    absolute(info.C), membrane(info.C), bending(info.C),
    w000(info.C), w100(info.C), w010(info.C), w001(info.C), w200(info.C), 
    w020(info.C), w002(info.C), w110(info.C), w101(info.C), w011(info.C), 
    N(static_cast<offset_t>(info.N)),
    C(static_cast<offset_t>(info.C)),
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
    INIT_ALLOC_INFO_5D(out)
  {
    set_factors(info.absolute, info.membrane, info.bending);
    set_kernel();
    set_vel2mom();
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
      bending[c]  = (bending.size()  == 0                ? 0. : 
                     static_cast<offset_t>(b.size()) > c ? b[c]  
                                                         : bending[c-1]);
    }
  } 

  NI_HOST void set_kernel() 
  {
    double lam0, lam1, lam2;
    for (offset_t c = 0; c < C; ++c)
    {
      lam0     = absolute[c];
      lam1     = membrane[c];
      lam2     = bending[c]; 
      w000[c] = lam2*(6.0*(vx0*vx0+vx1*vx1+vx2*vx2) + 8*(vx0*vx1+vx0*vx2+vx1*vx2)) + lam1*2*(vx0+vx1+vx2) + lam0;
      w100[c] = lam2*(-4.0*vx0*(vx0+vx1+vx2)) -lam1*vx0;
      w010[c] = lam2*(-4.0*vx1*(vx0+vx1+vx2)) -lam1*vx1;
      w001[c] = lam2*(-4.0*vx2*(vx0+vx1+vx2)) -lam1*vx2;
      w200[c] = lam2*vx0*vx0;
      w020[c] = lam2*vx1*vx1;
      w002[c] = lam2*vx2*vx2;
      w110[c] = lam2*2.0*vx0*vx1;
      w101[c] = lam2*2.0*vx0*vx2;
      w011[c] = lam2*2.0*vx1*vx2;
    }
    // TODO: correct for 1d/2d cases
  }

  NI_HOST NI_INLINE void set_vel2mom() 
  {
    bool has_bending  = any(bending);
    bool has_membrane = any(membrane);
    bool has_absolute = any(absolute);
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
            vel2mom = &Self::copy1d;
      } else if (dim == 2) {
        if (has_membrane)
            vel2mom = &Self::vel2mom2d_rls_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom2d_rls_absolute;
        else
            vel2mom = &Self::copy2d;
      } else if (dim == 3) {
        if (has_membrane)
            vel2mom = &Self::vel2mom3d_rls_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom3d_rls_absolute;
        else
            vel2mom = &Self::copy3d;
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
            vel2mom = &Self::copy1d;
    } else if (dim == 2) {
        if (has_bending)
            vel2mom = &Self::vel2mom2d_bending;
        else if (has_membrane)
            vel2mom = &Self::vel2mom2d_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom2d_absolute;
        else
            vel2mom = &Self::copy2d;
    } else if (dim == 3) {
        if (has_bending)
            vel2mom = &Self::vel2mom3d_bending;
        else if (has_membrane)
            vel2mom = &Self::vel2mom3d_membrane;
        else if (has_absolute)
            vel2mom = &Self::vel2mom3d_absolute;
        else
            vel2mom = &Self::copy3d;
    } else
        throw std::logic_error("RLS only implemented for dimension 1/2/3.");
  }

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

#define DEFINE_VEL2MOM(SUFFIX) \
  NI_DEVICE void vel2mom##SUFFIX( \
    offset_t x, offset_t y, offset_t z, offset_t n) const;
#define DEFINE_VEL2MOM_DIM(DIM)      \
  DEFINE_VEL2MOM(DIM##d_absolute)  \
  DEFINE_VEL2MOM(DIM##d_membrane)  \
  DEFINE_VEL2MOM(DIM##d_bending)   \
  DEFINE_VEL2MOM(DIM##d_rls_absolute)  \
  DEFINE_VEL2MOM(DIM##d_rls_membrane)  \
  NI_DEVICE void copy##DIM##d(             \
    offset_t x, offset_t y, offset_t z, offset_t n) const;

  DEFINE_VEL2MOM_DIM(1)
  DEFINE_VEL2MOM_DIM(2)
  DEFINE_VEL2MOM_DIM(3)

  /* ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  double            vx0;            // voxel size // x|W
  double            vx1;            // voxel size // y|H
  double            vx2;            // voxel size // z|D
  vector<double>    absolute;       // penalty on absolute values
  vector<double>    membrane;       // penalty on first derivatives
  vector<double>    bending;        // penalty on second derivatives
  Vel2MomFn         vel2mom;        // Pointer to vel2mom function

  vector<double>  w000;
  vector<double>  w100;
  vector<double>  w010;
  vector<double>  w001;
  vector<double>  w200;
  vector<double>  w020;
  vector<double>  w002;
  vector<double>  w110;
  vector<double>  w101;
  vector<double>  w011;

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
  offset_t X;
  offset_t Y;
  offset_t Z;
  DEFINE_STRIDE_INFO_5D(inp)
  DEFINE_STRIDE_INFO_5D(wgt)
  DEFINE_STRIDE_INFO_5D(out)
};


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                    LOOP                                    */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::dispatch(
    offset_t x, offset_t y, offset_t z, offset_t n) const {
    CALL_MEMBER_FN(*this, vel2mom)(x, y, z, n);
}


#if __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::loop(
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
template <typename scalar_t, typename offset_t> NI_HOST
void RegulariserImpl<scalar_t,offset_t>::loop() const
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
  x##0  = (bound::index(bound##i, x##0,  X) - x) * inp_s##X; \
  x##1  = (bound::index(bound##i, x##1,  X) - x) * inp_s##X; \
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
  scalar_t *out = out_ptr + (x*out_sX + y*out_sY + z*out_sZ); \
  scalar_t *inp = inp_ptr + (x*inp_sX + y*inp_sY + z*inp_sZ);


template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom3d_bending(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_COORD2
  GET_SIGN1 
  GET_SIGN2
  GET_WARP1 
  GET_WARP2
  GET_POINTERS

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC)
  {
    scalar_t center = *inp; 
    auto get = [center](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    *out = static_cast<scalar_t>(
          absolute[c]*center
        + w100[c]*(get(inp, x0,    sx0)     + get(inp, x1,    sx1))
        + w010[c]*(get(inp, y0,    sy0)     + get(inp, y1,    sy1))
        + w001[c]*(get(inp, z0,    sz0)     + get(inp, z1,    sz1))
        + w110[c]*(get(inp, x0+y0, sx0*sy0) + get(inp, x1+y0, sx1*sy0) +
                   get(inp, x0+y1, sx1*sy1) + get(inp, x1+y1, sx1*sy1))
        + w101[c]*(get(inp, x0+z0, sx0*sz0) + get(inp, x1+z0, sx1*sz0) +
                   get(inp, x0+z1, sx1*sz1) + get(inp, x1+z1, sx1*sz1))
        + w011[c]*(get(inp, y0+z0, sy0*sz0) + get(inp, y1+z0, sy1*sz0) +
                   get(inp, y0+z1, sy1*sz1) + get(inp, y1+z1, sy1*sz1))
        + w200[c]*(get(inp, x00,   sx00)    + get(inp, x11,   sx11))
        + w020[c]*(get(inp, y00,   sy00)    + get(inp, y11,   sy11))
        + w002[c]*(get(inp, z00,   sz00)    + get(inp, z11,   sz11))
    );
  }
}


template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom3d_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1 
  GET_WARP1 
  GET_POINTERS

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC)
  {
    scalar_t center = *inp; 
    auto get = [center](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    *out = static_cast<scalar_t>(
          absolute[c]*center
        + w100[c]*(get(inp, x0, sx0) + get(inp, x1, sx1))
        + w010[c]*(get(inp, y0, sy0) + get(inp, y1, sy1))
        + w001[c]*(get(inp, z0, sz0) + get(inp, z1, sz1)) 
    );
  }
}


template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom3d_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC)
    *out = static_cast<scalar_t>( absolute[c] * (*inp) );
}



template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom3d_rls_membrane(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_COORD1
  GET_SIGN1
  GET_WARP1_RLS
  GET_POINTERS

  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC, wgt += wgt_sC)
  {
    scalar_t wcenter = *wgt;
    double w1m00 = w100[c] * (wcenter + bound::get(wgt, wx0, sx0));
    double w1p00 = w100[c] * (wcenter + bound::get(wgt, wx1, sx1));
    double w01m0 = w010[c] * (wcenter + bound::get(wgt, wy0, sy0));
    double w01p0 = w010[c] * (wcenter + bound::get(wgt, wy1, sy1));
    double w001m = w001[c] * (wcenter + bound::get(wgt, wz0, sz0));
    double w001p = w001[c] * (wcenter + bound::get(wgt, wz1, sz1));

    scalar_t center = *inp;  // no need to use `get` -> we know we are in the FOV
    auto get = [center](scalar_t * x, offset_t o, int8_t s)
    {
      return bound::get(x, o, s) - center;
    };

    *out = static_cast<scalar_t>(
        absolute[c] * wcenter * center
      + w1m00 * get(inp, x0, sx0)
      + w1p00 * get(inp, x1, sx1)
      + w01m0 * get(inp, y0, sy0)
      + w01p0 * get(inp, y1, sy1)
      + w001m * get(inp, z0, sz0)
      + w001p * get(inp, z1, sz1)
    );
  }
}


template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom3d_rls_absolute(
  offset_t x, offset_t y, offset_t z, offset_t n) const
{
  GET_POINTERS
  scalar_t * wgt = wgt_ptr + (x*wgt_sX + y*wgt_sY + z*wgt_sZ);

  for (offset_t c = 0; c < C; 
       ++c, inp += inp_sC, out += out_sC, wgt += wgt_sC)
    *out = static_cast<scalar_t>( absolute[c] * (*wgt) * (*inp) );
}



/* ========================================================================== */
/*                                     2D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom2d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom2d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom2d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom2d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom2d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}

/* ========================================================================== */
/*                                     1D                                     */
/* ========================================================================== */

template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom1d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom1d_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom1d_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom1d_rls_membrane(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::vel2mom1d_rls_absolute(offset_t x, offset_t y, offset_t z, offset_t n) const {}

/* ========================================================================== */
/*                                     COPY                                   */
/* ========================================================================== */

template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::copy1d(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::copy2d(offset_t x, offset_t y, offset_t z, offset_t n) const {}
template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::copy3d(offset_t x, offset_t y, offset_t z, offset_t n) const {}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void regulariser_kernel(RegulariserImpl<scalar_t,offset_t> f) {
  f.loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    ALLOCATE OUTPUT // RESHAPE WEIGHT
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NI_HOST std::tuple<Tensor, Tensor>
prepare_tensors(const Tensor & input, Tensor output, Tensor weight)
{
  if (!(output.defined() && output.numel() > 0))
    output = at::empty_like(input);
  if (!output.is_same_size(input))
    throw std::invalid_argument("Output tensor must have the same shape as the input tensor");

  if (weight.defined() && weight.numel() > 0)
    weight = weight.expand_as(input);

  return std::tuple<Tensor, Tensor>(output, weight);
}

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (input, weight)
NI_HOST Tensor regulariser_impl(
  const Tensor& input, Tensor output, Tensor weight,
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(input, output, weight);
  output       = std::get<0>(tensors);
  weight       = std::get<1>(tensors);

  RegulariserAllocator info(input.dim()-2, absolute, membrane, bending, voxel_size, bound);
  info.ioset(input, output, weight);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "regulariser_impl", [&] {
    if (info.canUse32BitIndexMath())
    {
      RegulariserImpl<scalar_t, int32_t> algo(info);
      regulariser_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
    }
    else
    {
      RegulariserImpl<scalar_t, int64_t> algo(info);
      regulariser_kernel<<<GET_BLOCKS(algo.voxcount()), CUDA_NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(algo);
    }
  });
  return output;
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (input, weight)
NI_HOST Tensor regulariser_impl(
  const Tensor& input, Tensor output, Tensor weight, 
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  auto tensors = prepare_tensors(input, output, weight);
  output       = std::get<0>(tensors);
  weight       = std::get<1>(tensors);

  RegulariserAllocator info(input.dim()-2, absolute, membrane, bending, voxel_size, bound);
  info.ioset(input, output, weight);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "regulariser_impl", [&] {
    RegulariserImpl<scalar_t, int64_t> algo(info);
    algo.loop();
  });
  return output;
}

#endif // __CUDACC__

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST Tensor regulariser_impl(
  const Tensor& input, Tensor output, Tensor weight, 
  ArrayRef<double> absolute, ArrayRef<double> membrane, ArrayRef<double> bending,
  ArrayRef<double> voxel_size, BoundVectorRef bound)
{
  throw std::logic_error("Function not implemented for this device.");
}

} // namespace notimplemented

} // namespace ni

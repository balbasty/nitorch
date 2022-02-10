#include "common.h"
#include "bounds_common.h"
#include "allocator.h"
#include <ATen/ATen.h>
#include <limits>

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
using at::TensorOptions;
using c10::IntArrayRef;
using c10::ArrayRef;

// Required for stability. Value is currently about 1+8*eps
#define OnePlusTiny 1.000001

namespace ni {
NI_NAMESPACE_DEVICE { // cpu / cuda / ...

namespace { // anonymous namespace > everything inside has internal linkage


class RegulariserAllocator: public Allocator {
public:

  static constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();

  // ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST
  RegulariserAllocator(int dim, BoundVectorRef bound,
                       double absolute, double membrane, double bending,
                       double lame_shear, double lame_div, bool grid,
                       ArrayRef<double> factor, ArrayRef<double> voxel_size):
    dim(dim),
    bound0(bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound1(bound.size() > 1 ? bound[1] :
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound2(bound.size() > 2 ? bound[2] :
           bound.size() > 1 ? bound[1] :
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    absolute(absolute),
    membrane(membrane),
    bending(bending),
    lame_shear(lame_shear),
    lame_div(lame_div),
    factor(factor),
    grid(grid),
    vx0(voxel_size.size() > 0 ? voxel_size[0] : 1.),
    vx1(voxel_size.size() > 1 ? voxel_size[1] :
        voxel_size.size() > 0 ? voxel_size[0] : 1.),
    vx2(voxel_size.size() > 2 ? voxel_size[2] :
        voxel_size.size() > 1 ? voxel_size[1] :
        voxel_size.size() > 0 ? voxel_size[0] : 1.),
    f0(factor.size() > 0 ? factor[0] : 1.),
    f1(factor.size() > 1 ? factor[1] :
       factor.size() > 0 ? factor[0] : 1.),
    f2(factor.size() > 2 ? factor[2] :
       factor.size() > 1 ? factor[1] :
       factor.size() > 0 ? factor[0] : 1.)
  {
    vx0 = vx0 * vx0;
    vx1 = vx1 * vx1;
    vx2 = vx2 * vx2;
    if (grid)
    {
        f0 *= vx0;
        f1 *= vx1;
        f2 *= vx2;
    }
    vx0 = 1. / vx0;
    vx1 = 1. / vx1;
    vx2 = 1. / vx2;

    w000 = lam2*(6.0*(v0*v0+v1*v1+v2*v2) + 8*(v0*v1+v0*v2+v1*v2)) + lam1*2*(v0+v1+v2) + lam0;
    w100 = lam2*(-4.0*v0*(v0+v1+v2)) -lam1*v0;
    w010 = lam2*(-4.0*v1*(v0+v1+v2)) -lam1*v1;
    w001 = lam2*(-4.0*v2*(v0+v1+v2)) -lam1*v2;
    w200 = lam2*v0*v0;
    w020 = lam2*v1*v1;
    w002 = lam2*v2*v2;
    w110 = lam2*2.0*v0*v1;
    w101 = lam2*2.0*v0*v2;
    w011 = lam2*2.0*v1*v2;

    wx000 =  2.0*mu*(2.0*v0+v1+v2)/v0+2.0*lam + w000/v0;
    wx100 = -2.0*mu-lam + w100/v0;
    wx010 = -mu*v1/v0 + w010/v0;
    wx001 = -mu*v2/v0 + w001/v0;
    wy000 =  2.0*mu*(v0+2.0*v1+v2)/v1+2.0*lam + w000/v1;
    wy100 = -mu*v0/v1 + w100/v1;
    wy010 = -2.0*mu-lam + w010/v1;
    wy001 = -mu*v2/v1 + w001/v1;
    wz000 =  2.0*mu*(v0+v1+2.0*v2)/v2+2.0*lam + w000/v2;
    wz100 = -mu*v0/v2 + w100/v2;
    wz010 = -mu*v1/v2 + w010/v2;
    wz001 = -2.0*mu-lam + w001/v2;
    w2    = 0.25*mu+0.25*lam;

    // TODO: correct for 1d/2d cases

    wx000 *= OnePlusTiny;
    wy000 *= OnePlusTiny;
    wz000 *= OnePlusTiny;
  }


  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST void ioset
  (const Tensor& input)
  {
    init_all();
    init_input(input);
    init_output();
  }

  NI_HOST void ioset
  (const Tensor& input, const Tensor& weight)
  {
    init_all();
    init_input(input);
    init_weight(weight);
    init_output();
  }

  // We just check that all tensors that we own are compatible with 32b math
  bool canUse32BitIndexMath(int64_t max_elem=max_int32) const
  {
    return inp_32b_ok && wgt_32b_ok && out_32b_ok;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void init_all();
  NI_HOST void init_input(const Tensor& input);
  NI_HOST void init_output();

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  double            vx0;            // voxel size // x|W
  double            vx1;            // voxel size // y|H
  double            vx2;            // voxel size // z|D
  double            f0;             // factor (grid only) // x|W
  double            f1;             // factor (grid only) // y|H
  double            f2;             // factor (grid only) // z|D
  double            absolute;       // penalty on absolute values
  double            membrane;       // penalty on first derivatives
  double            bending;        // penalty on second derivatives
  double            lame_shear;     // penalty on symmetric part of Jacobian
  double            lame_div;       // penalty on trace of Jacobian
  ArrayRef<double>  factor;         // Modulating factor per channel
  bool              grid;           // Displacement field mode

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Tensor output;
  TensorOptions inp_opt;
  int64_t N;
  int64_t C;
  int64_t X;
  int64_t Y;
  int64_t Z;
  int64_t inp_sN;
  int64_t inp_sC;
  int64_t inp_sX;
  int64_t inp_sY;
  int64_t inp_sZ;
  bool inp_32b_ok;
  void *inp_ptr;
  int64_t wgt_sN;
  int64_t wgt_sC;
  int64_t wgt_sX;
  int64_t wgt_sY;
  int64_t wgt_sZ;
  bool wgt_32b_ok;
  void *wgt_ptr;
  int64_t out_sN;
  int64_t out_sC;
  int64_t out_sX;
  int64_t out_sY;
  int64_t out_sZ;
  bool out_32b_ok;
  void *out_ptr;

  // Allow RegulariserImpl's constructor to access RegulariserAllocator's
  // private members.
  template <typename scalar_t, typename offset_t>
  friend class RegulariserImpl;
};


NI_HOST
void RegulariserAllocator::init_all()
{
  inp_opt = TensorOptions();
  N = C = X = Y = Z = 1L;
  inp_sN  = inp_sC   = inp_sX   = inp_sY  = inp_sZ   = 0L;
  wgt_sN  = wgt_sC   = wgt_sX   = wgt_sY  = wgt_sZ   = 0L;
  out_sN  = out_sC   = out_sX   = out_sY  = out_sZ   = 0L;
  inp_ptr = wgt_ptr = out_ptr = static_cast<float*>(0);
  inp_32b_ok = wgt_32b_ok = out_32b_ok = true;
}

NI_HOST
void PushPullAllocator::init_input(const Tensor& input)
{
  N       = input.size(0);
  C       = input.size(1);
  inp_X   = input.size(2);
  inp_Y   = dim < 2 ? 1L : input.size(3);
  inp_Z   = dim < 3 ? 1L : input.size(4);
  inp_sN  = input.stride(0);
  inp_sC  = input.stride(1);
  inp_sX  = input.stride(2);
  inp_sY  = dim < 2 ? 0L : input.stride(3);
  inp_sZ  = dim < 3 ? 0L : input.stride(4);
  inp_ptr = input.data_ptr();
  inp_opt = input.options();
  inp_32b_ok = tensorCanUse32BitIndexMath(input);
}

NI_HOST
void PushPullAllocator::init_weight(const Tensor& weight)
{
  wgt_sN  = weight.stride(0);
  wgt_sC  = weight.stride(1);
  wgt_sX  = weight.stride(2);
  wgt_sY  = dim < 2 ? 0L : weight.stride(3);
  wgt_sZ  = dim < 3 ? 0L : weight.stride(4);
  wgt_ptr = weight.data_ptr();
  wgt_opt = weight.options();
  wgt_32b_ok = tensorCanUse32BitIndexMath(weight);
}

NI_HOST
void PushPullAllocator::init_output()
{
    if (dim == 1)
      output = at::empty({N, C, X}, inp_opt);
    else if (dim == 2)
      output = at::empty({N, C, X, Y}, inp_opt);
    else
      output = at::empty({N, C, X, Y, Z}, inp_opt);
    out_sN   = output.stride(0);
    out_sC   = output.stride(1);
    out_sX   = output.stride(2);
    out_sY   = dim < 2 ? 0L : output.stride(3);
    out_sZ   = dim < 3 ? 0L : output.stride(4);
    out_ptr  = output.data_ptr();
    out_32b_ok = tensorCanUse32BitIndexMath(output);
}


template <typename scalar_t, typename offset_t>
class RegulariserImpl {
public:

  // ~~~ CONSTRUCTOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  RegulariserImpl(const RegulariserAllocator & info):
    output(info.output),
    dim(info.dim),
    bound0(info.bound0), bound1(info.bound1), bound2(info.bound2),
    vx0(info.vx0), vx1(info.vx1), vx2(info.vx2), grid(grid),
    f0(info.f0), f1(info.f1), f2(info.f2), grid(grid),
    absolute(info.absolute), membrane(info.membrane), bending(info.bending),
    lame_shear(info.lame_shear), lame_div(info.lame_div), factor(factor),
    N(static_cast<offset_t>(info.N)),
    C(static_cast<offset_t>(info.C)),
    X(static_cast<offset_t>(info.X)),
    Y(static_cast<offset_t>(info.Y)),
    Z(static_cast<offset_t>(info.Z)),
    inp_sN(static_cast<offset_t>(info.inp_sN)),
    inp_sC(static_cast<offset_t>(info.inp_sC)),
    inp_sX(static_cast<offset_t>(info.inp_sX)),
    inp_sY(static_cast<offset_t>(info.inp_sY)),
    inp_sZ(static_cast<offset_t>(info.inp_sZ)),
    inp_ptr(static_cast<scalar_t*>(info.inp_ptr)),
    wgt_sN(static_cast<offset_t>(info.wgt_sN)),
    wgt_sC(static_cast<offset_t>(info.wgt_sC)),
    wgt_sX(static_cast<offset_t>(info.wgt_sX)),
    wgt_sY(static_cast<offset_t>(info.wgt_sY)),
    wgt_sZ(static_cast<offset_t>(info.wgt_sZ)),
    wgt_ptr(static_cast<scalar_t*>(info.wgt_ptr)),
    out_sN(static_cast<offset_t>(info.out_sN)),
    out_sC(static_cast<offset_t>(info.out_sC)),
    out_sX(static_cast<offset_t>(info.out_sX)),
    out_sY(static_cast<offset_t>(info.out_sY)),
    out_sZ(static_cast<offset_t>(info.out_sZ)),
    out_ptr(static_cast<scalar_t*>(info.out_ptr))
  {}

    // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if __CUDACC__
  // Loop over voxels that belong to one CUDA block
  // This function is called by the CUDA kernel
  NI_DEVICE void loop(int threadIdx, int blockIdx,
                      int blockDim, int gridDim) const;
#else
  // Loop over all voxels
  void loop() const;
#endif

  NI_HOST NI_DEVICE int64_t voxcount() const {
    return N * trgt_X * trgt_Y * trgt_Z;
  }


private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_DEVICE void vel2mom1d_absolute(
    scalar_t x, offset_t n) const;
  NI_DEVICE void vel2mom1d_membrane(
    scalar_t x, offset_t n) const;
  NI_DEVICE void vel2mom1d_bending(
    scalar_t x, offset_t n) const;
  NI_DEVICE void vel2mom1d_lame(
    scalar_t x,  offset_t n) const;
  NI_DEVICE void vel2mom2d_absolute(
    scalar_t x, scalar_t y, offset_t n) const;
  NI_DEVICE void vel2mom2d_membrane(
    scalar_t x, scalar_t y, offset_t n) const;
  NI_DEVICE void vel2mom2d_bending(
    scalar_t x, scalar_t y, offset_t n) const;
  NI_DEVICE void vel2mom2d_lame(
    scalar_t x, scalar_t y, offset_t n) const;
  NI_DEVICE void vel2mom3d_absolute(
    scalar_t x, scalar_t y, scalar_t z, offset_t n) const;
  NI_DEVICE void vel2mom3d_membrane(
    scalar_t x, scalar_t y, scalar_t z, offset_t n) const;
  NI_DEVICE void vel2mom3_bending(
    scalar_t x, scalar_t y, scalar_t z, offset_t n) const;
  NI_DEVICE void vel2mom3d_lame(
    scalar_t x, scalar_t y, scalar_t z, offset_t n) const;

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  double            vx0;            // voxel size // x|W
  double            vx1;            // voxel size // y|H
  double            vx2;            // voxel size // z|D
  double            f0;             // factor (grid only) // x|W
  double            f1;             // factor (grid only) // y|H
  double            f2;             // factor (grid only) // z|D
  double            absolute;       // penalty on absolute values
  double            membrane;       // penalty on first derivatives
  double            bending;        // penalty on second derivatives
  double            lame_shear;     // penalty on symmetric part of Jacobian
  double            lame_div;       // penalty on trace of Jacobian
  ArrayRef<double>  factor;         // Modulating factor per channel
  bool              grid;           // Displacement field mode

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Tensor output;
  TensorOptions inp_opt;
  int64_t N;
  int64_t C;
  int64_t X;
  int64_t Y;
  int64_t Z;
  int64_t inp_sN;
  int64_t inp_sC;
  int64_t inp_sX;
  int64_t inp_sY;
  int64_t inp_sZ;
  bool inp_32b_ok;
  void *inp_ptr;
  int64_t wgt_sN;
  int64_t wgt_sC;
  int64_t wgt_sX;
  int64_t wgt_sY;
  int64_t wgt_sZ;
  bool wgt_32b_ok;
  void *wgt_ptr;
  int64_t out_sN;
  int64_t out_sC;
  int64_t out_sX;
  int64_t out_sY;
  int64_t out_sZ;
  bool out_32b_ok;
  void *out_ptr;
};



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::dispatch(
    offset_t x, offset_t y, offset_t z, offset_t n) const {

    if (dim == 1)
        if (lame_shear or lame_div)
            vel2mom1d_lame(x, n);
        else if (bending)
            vel2mom1d_bending(x, n);
        else if (membrane)
            vel2mom1d_bending(x, n);
        else if (absolute)
            vel2mom1d_bending(x, n);
        else
            copy(x, w, n);
    else if (dim == 2)
        if (lame_shear or lame_div)
            vel2mom2d_lame(x, y, n);
        else if (bending)
            vel2mom2d_bending(x, y, n);
        else if (membrane)
            vel2mom2d_bending(x, y, n);
        else if (absolute)
            vel2mom2d_bending(x, y, n);
        else
            copy(x, y, n);
    else
        if (lame_shear or lame_div)
            vel2mom3d_lame(x, y, z, n);
        else if (bending)
            vel2mom3d_bending(x, y, z, n);
        else if (membrane)
            vel2mom3d_bending(x, y, z, n);
        else if (absolute)
            vel2mom3d_bending(x, y, z, n);
        else
            copy(x, y, z, n);
}


#if __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void RegulariserImpl<scalar_t,offset_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  int64_t index = blockIdx * blockDim + threadIdx;
  int64_t nthreads = voxcount();
  offset_t XYZ  = Z * Y * X;
  offset_t YZ   = Z * Y;
  offset_t n, x, y, z;
  for (offset_t i=index; index < nthreads; index += blockDim*gridDim, i=index)
  {
      // Convert index: linear to sub
      n  = (i/XYZ);
      x  = (i/YZ) % X;
      y  = (i/Z)  % Y;
      z  = i % Z;
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
  offset_t NXYZ = Z * Y * X * N;
  offset_t XYZ  = Z * Y * X;
  offset_t YZ   = Z * Y;
  at::parallel_for(0, NXYZ, GRAIN_SIZE, [&](offset_t start, offset_t end) {
    offset_t n, x, y, z;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/XYZ);
      x  = (i/YZ) % X;
      y  = (i/Z)  % Y;
      z  = i % Z;
      dispatch(x, y, z, n);
    }
  });
}

#endif

template <typename scalar_t, typename offset_t> NI_DEVICE
void vel2mom3d_lame(offset_t x, offset_t y, offset_t z, offset_t n) const {

  // WE KNOW WE ARE IN GRID MODE -> 3 channels //

  offset_t x0  = x-1, y0  = y-1, z0  = z-1, x1  = x+1, y1  = y+1, z1  = z+1;
  offset_t x00 = x-2, y00 = y-2, z00 = z-2, x11 = x+2, y11 = y+2, z11 = z+2;

  // Sign (/!\ compute sign before warping indices)
  int8_t  sx0  = bound::sign(bound0, x0,  X);
  int8_t  sy0  = bound::sign(bound1, y0,  Y);
  int8_t  sz0  = bound::sign(bound2, z0,  Z);
  int8_t  sx1  = bound::sign(bound0, x1,  X);
  int8_t  sy1  = bound::sign(bound1, y1,  Y);
  int8_t  sz1  = bound::sign(bound2, z1,  Z);
  int8_t  sx00 = bound::sign(bound0, x00, X);
  int8_t  sy00 = bound::sign(bound1, y00, Y);
  int8_t  sz00 = bound::sign(bound2, z00, Z);
  int8_t  sx11 = bound::sign(bound0, x11, X);
  int8_t  sy11 = bound::sign(bound1, y11, Y);
  int8_t  sz11 = bound::sign(bound2, z11, Z);

  // Warp indices
  x0  = bound::index(bound0, x0,  inp_X) * src_X;
  y0  = bound::index(bound1, y0,  inp_Y) * src_Y;
  z0  = bound::index(bound2, z0,  inp_Z) * src_Z;
  x1  = bound::index(bound0, x1,  inp_X) * src_X;
  y1  = bound::index(bound1, y1,  inp_Y) * src_Y;
  z1  = bound::index(bound2, z1,  inp_Z) * src_Z;
  x00 = bound::index(bound0, x00, inp_X) * src_X;
  y00 = bound::index(bound1, y00, inp_Y) * src_Y;
  z00 = bound::index(bound2, z00, inp_Z) * src_Z;
  x11 = bound::index(bound0, x11, inp_X) * src_X;
  y11 = bound::index(bound1, y11, inp_Y) * src_Y;
  z11 = bound::index(bound2, z11, inp_Z) * src_Z;

  offset_t out0 = x*out_sX + y*out_sY + z*out_sZ;
  offset_t out1 = out0 + out_sC, out2 = out0 + 2 * out_sC;
  offset_t inp0 = x*inp_sX + y*inp_sY + z*inp_sZ;
  offset_t inp1 = inp0 + inp_sC, inp2 = inp0 + 2 * inp_sC;

  // For numerical stability, we subtract the center value before convolving.
  // We define a lambda function for ease.
  scalar_t c = *inp0;  // no need to use `get` -> we know we are in the FOV
  scalar_t get2 = [c, bound](offset_t x, offset_t o, int8_t s)
  {
    return bound::get(x, o, s) - c;
  };

  *out0 = (float)(wx100*(get2(inp0, x0, sx0) + get2(inp0, x1, sx1))
                + wx010*(get2(inp0, y0, sy0) + get2(inp0, y1, sy1))
                + wx001*(get2(inp0, z0, sz0) + get2(inp0, z1, sz1))
                + w2   *( bound::get(inp1, x1+y0, sx1*sy0) - bound::get(inp1, x1+y1, sx1*sy1)
                        + bound::get(inp1, x0+y1, sx0*sy1) - bound::get(inp1, x0+y0, sx0*sy0)
                        + bound::get(inp2, x1+z0, sx1*sz0) - bound::get(inp2, x1+z1, sx1*sz1)
                        + bound::get(inp2, x0+z1, sx0*sz1) - bound::get(inp2, x0+z0, sx0*sz0))
                 + (lam0*c
                 +  w110*(get2(inp0, x0+y0, sx0*sy0) + get2(inp0, x1+y0, sx1*sy0) +
                          get2(inp0, x0+y1, sx1*sy1) + get2(inp0, x1+y1, sx1*sy1))
                 +  w101*(get2(inp0, x0+z0, sx0*sz0) + get2(inp0, x1+z0, sx1*sz0) +
                          get2(inp0, x0+z1, sx1*sz1) + get2(inp0, x1+z1, sx1*sz1))
                 +  w011*(get2(inp0, y0+z0, sy0*sz0) + get2(inp0, y1+z0, sy1*sz0) +
                          get2(inp0, y0+z1, sy1*sz1) + get2(inp0, y1+z1, sy1*sz1))
                 +  w200*(get2(inp0, x00, sx00) + get2(inp0, x11, sx11))
                 +  w020*(get2(inp0, y00, sy00) + get2(inp0, y11, sy11))
                 +  w002*(get2(inp0, z00, sz00) + get2(inp0, z11, sz11)))*f0);

  *out1 = (float)(wy100*(get2(inp1, x0, sx0) + get2(inp1, x1, sx1))
                + wy010*(get2(inp1, y0, sy0) + get2(inp1, y1, sy1))
                + wy001*(get2(inp1, z0, sz0) + get2(inp1, z1, sz1))
                + w2   *( bound::get(inp0, y1+x0, sy1*sx0) - bound::get(inp0, y1+x1, sy1*sx1)
                        + bound::get(inp0, y0+x1, sy0*sx1) - bound::get(inp0, y0+x0, sy0*sx0)
                        + bound::get(inp2, y1+z0, sy1*sz0) - bound::get(inp2, y1+z1, sy1*sz1)
                        + bound::get(inp2, y0+z1, sy0*sz1) - bound::get(inp2, y0+z0, sy0*sz0))
                 + (lam0*c
                 +  w110*(get2(inp1, x0+y0, sx0*sy0) + get2(inp1, x1+y0, sx1*sy0) +
                          get2(inp1, x0+y1, sx1*sy1) + get2(inp1, x1+y1, sx1*sy1))
                 +  w101*(get2(inp1, x0+z0, sx0*sz0) + get2(inp1, x1+z0, sx1*sz0) +
                          get2(inp1, x0+z1, sx1*sz1) + get2(inp1, x1+z1, sx1*sz1))
                 +  w011*(get2(inp1, y0+z0, sy0*sz0) + get2(inp1, y1+z0, sy1*sz0) +
                          get2(inp1, y0+z1, sy1*sz1) + get2(inp1, y1+z1, sy1*sz1))
                 +  w200*(get2(inp1, x00, sx00) + get2(inp1, x11, sx11))
                 +  w020*(get2(inp1, y00, sy00) + get2(inp1, y11, sy11))
                 +  w002*(get2(inp1, z00, sz00) + get2(inp1, z11, sz11)))*f1);

  *out2 = (float)(wy100*(get2(inp2, x0, sx0) + get2(inp2, x1, sx1))
                + wy010*(get2(inp2, y0, sy0) + get2(inp2, y1, sy1))
                + wy001*(get2(inp2, z0, sz0) + get2(inp2, z1, sz1))
                + w2   *( bound::get(inp0, z1+x0, sz1*sx0) - bound::get(inp0, z1+x1, sz1*sx1)
                        + bound::get(inp0, z0+x1, sz0*sx1) - bound::get(inp0, z0+x0, sz0*sx0)
                        + bound::get(inp1, z1+y0, sz1*sy0) - bound::get(inp1, z1+y1, sz1*sy1)
                        + bound::get(inp1, z0+y1, sz0*sy1) - bound::get(inp1, z0+y0, sz0*sy0))
                 + (lam0*c
                 +  w110*(get2(inp2, x0+y0, sx0*sy0) + get2(inp2, x1+y0, sx1*sy0) +
                          get2(inp2, x0+y1, sx1*sy1) + get2(inp2, x1+y1, sx1*sy1))
                 +  w101*(get2(inp2, x0+z0, sx0*sz0) + get2(inp2, x1+z0, sx1*sz0) +
                          get2(inp2, x0+z1, sx1*sz1) + get2(inp2, x1+z1, sx1*sz1))
                 +  w011*(get2(inp2, y0+z0, sy0*sz0) + get2(inp2, y1+z0, sy1*sz0) +
                          get2(inp2, y0+z1, sy1*sz1) + get2(inp2, y1+z1, sy1*sz1))
                 +  w200*(get2(inp2, x00, sx00) + get2(inp2, x11, sx11))
                 +  w020*(get2(inp2, y00, sy00) + get2(inp2, y11, sy11))
                 +  w002*(get2(inp2, z00, sz00) + get2(inp2, z11, sz11)))*f2);
}


template <typename scalar_t, typename offset_t> NI_DEVICE
void vel2mom3d_bending(offset_t x, offset_t y, offset_t z, offset_t n) const {

  offset_t x0  = x-1, y0  = y-1, z0  = z-1, x1  = x+1, y1  = y+1, z1  = z+1;
  offset_t x00 = x-2, y00 = y-2, z00 = z-2, x11 = x+2, y11 = y+2, z11 = z+2;

  // Sign (/!\ compute sign before warping indices)
  int8_t  sx0  = bound::sign(bound0, x0,  X);
  int8_t  sy0  = bound::sign(bound1, y0,  Y);
  int8_t  sz0  = bound::sign(bound2, z0,  Z);
  int8_t  sx1  = bound::sign(bound0, x1,  X);
  int8_t  sy1  = bound::sign(bound1, y1,  Y);
  int8_t  sz1  = bound::sign(bound2, z1,  Z);
  int8_t  sx00 = bound::sign(bound0, x00, X);
  int8_t  sy00 = bound::sign(bound1, y00, Y);
  int8_t  sz00 = bound::sign(bound2, z00, Z);
  int8_t  sx11 = bound::sign(bound0, x11, X);
  int8_t  sy11 = bound::sign(bound1, y11, Y);
  int8_t  sz11 = bound::sign(bound2, z11, Z);

  // Warp indices
  x0  = bound::index(bound0, x0,  inp_X) * src_X;
  y0  = bound::index(bound1, y0,  inp_Y) * src_Y;
  z0  = bound::index(bound2, z0,  inp_Z) * src_Z;
  x1  = bound::index(bound0, x1,  inp_X) * src_X;
  y1  = bound::index(bound1, y1,  inp_Y) * src_Y;
  z1  = bound::index(bound2, z1,  inp_Z) * src_Z;
  x00 = bound::index(bound0, x00, inp_X) * src_X;
  y00 = bound::index(bound1, y00, inp_Y) * src_Y;
  z00 = bound::index(bound2, z00, inp_Z) * src_Z;
  x11 = bound::index(bound0, x11, inp_X) * src_X;
  y11 = bound::index(bound1, y11, inp_Y) * src_Y;
  z11 = bound::index(bound2, z11, inp_Z) * src_Z;

  offset_t out_ = x*out_sX + y*out_sY + z*out_sZ;
  offset_t inp_ = x*inp_sX + y*inp_sY + z*inp_sZ;

  for (offset_t c = 0; c < C; ++c, out_ += out_sC, inp_ += inp_sC)
  {
      double f = factor[c];

      // For numerical stability, we subtract the center value before convolving.
      // We define a lambda function for ease.
      scalar_t center = *inp_;  // no need to use `get` -> we know we are in the FOV
      scalar_t get2 = [center, bound](offset_t x, offset_t o, int8_t s)
      {
        return bound::get(x, o, s) - center;
      };

      *out_ = (float)((lam0*center
                     +  w100*(get2(inp_, x0, sx0) + get2(inp_, x1, sx1))
                     +  w010*(get2(inp_, y0, sy0) + get2(inp_, y1, sy1))
                     +  w001*(get2(inp_, z0, sz0) + get2(inp_, z1, sz1))
                     +  w110*(get2(inp_, x0+y0, sx0*sy0) + get2(inp_, x1+y0, sx1*sy0) +
                              get2(inp_, x0+y1, sx1*sy1) + get2(inp_, x1+y1, sx1*sy1))
                     +  w101*(get2(inp_, x0+z0, sx0*sz0) + get2(inp_, x1+z0, sx1*sz0) +
                              get2(inp_, x0+z1, sx1*sz1) + get2(inp_, x1+z1, sx1*sz1))
                     +  w011*(get2(inp_, y0+z0, sy0*sz0) + get2(inp_, y1+z0, sy1*sz0) +
                              get2(inp_, y0+z1, sy1*sz1) + get2(inp_, y1+z1, sy1*sz1))
                     +  w200*(get2(inp_, x00, sx00) + get2(inp_, x11, sx11))
                     +  w020*(get2(inp_, y00, sy00) + get2(inp_, y11, sy11))
                     +  w002*(get2(inp_, z00, sz00) + get2(inp_, z11, sz11)))*f);
  }

}
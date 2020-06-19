// This file implements a convolution whose output domain is the same as 
// the input domains. This is in contrasts with pytorch's native conv
// which returns the common domain of the input and weight (resulting in 
// an implicit 'crop').
// Also, more boundary conditions are implemented (see bounds.h).

// TODO:
// . generic 3d
// . generic 2d
// . generic 1d
// . specialized 3d 3x3x3
// . specialized 3d 5x5x5
// . specialized 2d 3x3
// . specialized 2d 5x5
// . specialized 1d 3
// . specialized 1d 5
// . specialized    1x1x...
// ? spearable kernels
// ? sliding

#include "common.h"
#include "bounds_common.h"
#include <ATen/ATen.h>
#include <tuple>
//#include <cstdio>

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

// maximum number of channels
// > not used in mode isotropic nearest/linear
#ifndef NI_MAX_NUM_CHANNELS
# define NI_MAX_NUM_CHANNELS 1024
#endif

using at::Tensor;
using at::TensorOptions;
using c10::IntArrayRef;

namespace ni {
NI_NAMESPACE_DEVICE { // cpu / cuda / ...

namespace { // anonymous namespace > everything inside has internal linkage

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        GENERIC CONV CLASS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This class implements the bulk of the code.
// /!\ No type and shape checking is performed here.

template <typename scalar_t, typename offset_t>
class ConvImpl {
public:

  // ~~~ USEFUL CONST VALUES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  static constexpr scalar_t   sONE  = static_cast<scalar_t>(1);
  static constexpr scalar_t   sZERO = static_cast<scalar_t>(0);
  static constexpr offset_t   oONE  = static_cast<offset_t>(1);
  static constexpr offset_t   oZERO = static_cast<offset_t>(0);
  static constexpr scalar_t * pZERO = static_cast<scalar_t*>(0);

  // ~~~ CONSTRUCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST
  ConvImpl(int dim, int groups, BoundVectorRef bound, 
           IntArrayRef stride, IntArrayRef dilation,
           IntArrayRef offsetlow, IntArrayRef offsetup, 
           bool do_conv, bool do_deconv, bool do_grad):
    dim(dim),
    bound0(bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound1(bound.size() > 1 ? bound[1] : 
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound2(bound.size() > 2 ? bound[2] : 
           bound.size() > 1 ? bound[1] : 
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    stride0(stride.size() > 0 ? stride[0] : oONE),
    stride1(stride.size() > 1 ? stride[1] : 
            stride.size() > 0 ? stride[0] : oONE),
    stride2(stride.size() > 2 ? stride[2] : 
            stride.size() > 1 ? stride[1] : 
            stride.size() > 0 ? stride[0] : oONE),
    dilation0(dilation.size() > 0 ? dilation[0] : oONE),
    dilation1(dilation.size() > 1 ? dilation[1] : 
              dilation.size() > 0 ? dilation[0] : oONE),
    dilation2(dilation.size() > 2 ? dilation[2] : 
              dilation.size() > 1 ? dilation[1] : 
              dilation.size() > 0 ? dilation[0] : oONE),
    offsetlow0(offsetlow.size() > 0 ? offsetlow[0] : oZERO),
    offsetlow1(offsetlow.size() > 1 ? offsetlow[1] : 
               offsetlow.size() > 0 ? offsetlow[0] : oZERO),
    offsetlow2(offsetlow.size() > 2 ? offsetlow[2] : 
               offsetlow.size() > 1 ? offsetlow[1] : 
               offsetlow.size() > 0 ? offsetlow[0] : oZERO),
    offsetup0(offsetup.size() > 0 ? offsetup[0] : oZERO),
    offsetup1(offsetup.size() > 1 ? offsetup[1] : 
              offsetup.size() > 0 ? offsetup[0] : oZERO),
    offsetup2(offsetup.size() > 2 ? offsetup[2] : 
              offsetup.size() > 1 ? offsetup[1] : 
              offsetup.size() > 0 ? offsetup[0] : oZERO),
    do_conv(do_conv),
    do_deconv(do_deconv),
    do_grad(do_grad),
    G(static_cast<offset_t>(groups))
  {}

  // ~~~ PUBLIC VALUE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::deque<Tensor> output;

   NI_HOST NI_DEVICE void printInfo() const {
     printf("dim:        %d\n", dim);
     printf("do_conv:    %d\n", do_conv);
     printf("do_deconv:  %d\n", do_deconv);
     printf("do_grad:    %d\n", do_grad);
     printf("bound:      [%d %d %d]\n", static_cast<int>(bound0),
       static_cast<int>(bound1), static_cast<int>(bound2));
     printf("stride:     [%d %d %d]\n", stride0, stride1, stride2);
     printf("dilation:   [%d %d %d]\n", dilation0, dilation1, dilation2);
     printf("offset-:    [%d %d %d]\n", offsetlow0, offsetlow1, offsetlow2);
     printf("offset+:    [%d %d %d]\n", offsetup0, offsetup1, offsetup2);
     printf("center:     [%d %d %d]\n", center0, center1, center2);
     printf("src:        [%d %d %d]\n", src_X, src_Y, src_Z);
     printf("trgt:       [%d %d %d]\n", trgt_X, trgt_Y, trgt_Z);
     printf("wght:       [%d %d %d]\n", wght_X, wght_Y, wght_Z);
     printf("N:          %d\n", N);
     printf("C (target): %d\n", trgt_C);
     printf("C (source): %d\n", src_C);
     printf("groups:     %d\n", G);
     printf("src      -> %lu\n", reinterpret_cast<std::uintptr_t>(src_ptr));
     printf("trgt     -> %lu\n", reinterpret_cast<std::uintptr_t>(trgt_ptr));
     printf("wght     -> %lu\n", reinterpret_cast<std::uintptr_t>(wght_ptr));
     printf("bias     -> %lu\n", reinterpret_cast<std::uintptr_t>(bias_ptr));
     printf("out      -> %lu\n", reinterpret_cast<std::uintptr_t>(out_ptr));
     printf("grad     -> %lu\n", reinterpret_cast<std::uintptr_t>(grad_ptr));
   }

  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void ioset // Conv
  (const Tensor& input, const Tensor& weight, const Tensor& bias,
   IntArrayRef center, bool deconv)
  {
    if (!deconv)
    {
      init_all();
      init_source(input);
      init_weight(weight);
      init_bias(bias);
      init_center(center);
      init_target();
      init_output();
    }
    else
    {
      init_all();
      init_weight(weight);
      init_bias(bias);
      init_target(input);
      init_center(center);
      init_source();
      init_output();
    }
  }

  template <typename SourceType>
  NI_HOST void ioset // Backward
  (const SourceType& source, const Tensor& weight, const Tensor& bias,
   const Tensor& target, IntArrayRef center)
  {
    init_all();
    init_source(source);
    init_weight(weight);
    init_bias(bias);
    init_target(target);
    init_center(center);
    init_output();
  }

#if __CUDACC__
  NI_DEVICE void loop(int threadIdx, int blockIdx, 
                      int blockDim, int gridDim) const;
#else
  void loop() const;
#endif

  NI_HOST NI_DEVICE int64_t voxcount() const { 
    return N * trgt_X * trgt_Y * trgt_Z;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void init_all();
  NI_HOST void init_source(IntArrayRef source);
  NI_HOST void init_source(const Tensor& source);
  NI_HOST void init_source();
  NI_HOST void init_weight(const Tensor& weight); 
  NI_HOST void init_bias(const Tensor& bias); 
  NI_HOST void init_target(const Tensor& target);
  NI_HOST void init_target();
  NI_HOST void init_center(IntArrayRef source_size);
  NI_HOST void init_output();
  NI_DEVICE void conv1d(
    offset_t w, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv1d_3(
    offset_t w, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv1d_5(
    offset_t w, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv2d(
    offset_t w, offset_t h, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv2d_3x3(
    offset_t w, offset_t h, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv2d_5x5(
    offset_t w, offset_t h, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv3d(
    offset_t w, offset_t h, offset_t d, offset_t n) const;
  NI_DEVICE void conv3d_3x3x3(
    offset_t w, offset_t h, offset_t d, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv3d_5x5x5(
    offset_t w, offset_t h, offset_t d, offset_t n) const { /* TODO */ };

  // ~~~ OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  int               dim;            // dimensionality (2 or 3)
  BoundType         bound0;         // boundary condition  // x|W
  BoundType         bound1;         // boundary condition  // y|H
  BoundType         bound2;         // boundary condition  // z|D
  int               stride0;        // stride // x|W
  int               stride1;        // stride // y|H
  int               stride2;        // stride // z|D
  int               dilation0;      // dilation // x|W
  int               dilation1;      // dilation // y|H
  int               dilation2;      // dilation // z|D
  int               center0;        // weight center // x|W
  int               center1;        // weight center // y|H
  int               center2;        // weight center // z|D
  int               offsetlow0;     // offset / left side // x|W
  int               offsetlow1;     // offset / left side // y|H
  int               offsetlow2;     // offset / left side // z|D
  int               offsetup0;      // offset / right side // x|W
  int               offsetup1;      // offset / right side // y|H
  int               offsetup2;      // offset / right side // z|D
  bool              iso;            // kernel is a cube
  bool              do_conv;        // convolve
  bool              do_deconv;      // deconvole
  bool              do_grad;        // backprop: gradient of weights

  // ~~~ NAVIGATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  TensorOptions src_opt;
  TensorOptions wght_opt;
  TensorOptions trgt_opt;
  offset_t N;
  offset_t G;
  offset_t src_C;
  offset_t src_X;
  offset_t src_Y;
  offset_t src_Z;
  offset_t trgt_C;
  offset_t trgt_X;
  offset_t trgt_Y;
  offset_t trgt_Z;
  offset_t wght_X;
  offset_t wght_Y;
  offset_t wght_Z;
  offset_t src_sN;
  offset_t src_sC;
  offset_t src_sX;
  offset_t src_sY;
  offset_t src_sZ;
  scalar_t *src_ptr;
  offset_t trgt_sN;
  offset_t trgt_sC;
  offset_t trgt_sX;
  offset_t trgt_sY;
  offset_t trgt_sZ;
  scalar_t *trgt_ptr;
  offset_t wght_sCt;
  offset_t wght_sCs;
  offset_t wght_sX;
  offset_t wght_sY;
  offset_t wght_sZ;
  scalar_t *wght_ptr;
  offset_t bias_sC;
  scalar_t *bias_ptr;
  offset_t out_sN;
  offset_t out_sC;
  offset_t out_sX;
  offset_t out_sY;
  offset_t out_sZ;
  scalar_t *out_ptr;
  offset_t grad_sCt;
  offset_t grad_sCs;
  offset_t grad_sX;
  offset_t grad_sY;
  offset_t grad_sZ;
  scalar_t *grad_ptr;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          INITIALISATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t>
void ConvImpl<scalar_t,offset_t>::init_all()
{
  src_opt  = wght_opt = trgt_opt = TensorOptions();
  N        = G        = oONE;
  wght_X   = wght_Y   = wght_Z   = oONE;
  src_X    = src_Y    = src_Z    = src_C    = oONE;
  trgt_X   = trgt_Y   = trgt_Z   = trgt_C   = oONE;
  src_sX   = src_sY   = src_sZ   = src_sN   = src_sC   = oZERO;
  wght_sX  = wght_sY  = wght_sZ  = wght_sCs = wght_sCt = oZERO;
  grad_sX  = grad_sY  = grad_sZ  = grad_sCs = grad_sCt = oZERO;
  trgt_sX  = trgt_sY  = trgt_sZ  = trgt_sN  = trgt_sC  = oZERO;
  out_sX   = out_sY   = out_sZ   = out_sN   = out_sC   = oZERO;
  src_ptr  = trgt_ptr = wght_ptr = out_ptr  = grad_ptr = bias_ptr = pZERO;
  bias_sC  = oZERO;

}

template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_source(const Tensor& source)
{
  N       = source.size(0);
  src_C   = source.size(1);
  src_X   = source.size(2);
  src_Y   = dim >= 2 ? source.size(3) : oONE;
  src_Z   = dim >= 3 ? source.size(4) : oONE;
  src_sN  = source.stride(0);
  src_sC  = source.stride(1);
  src_sX  = source.stride(2);
  src_sY  = dim >= 2 ? source.stride(3) : oZERO;
  src_sZ  = dim >= 3 ? source.stride(4) : oZERO;
  src_ptr = source.data_ptr<scalar_t>();
  src_opt = source.options();
}
template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_source(IntArrayRef source)
{
  if (source.size() > 0) {
    N = source[0];
    if (source.size() > 1) {
      src_C = source[1];
      if (source.size() > 2) {
        src_X = source[2];
        if (source.size() > 3) {
          src_X = source[3];
          if (source.size() > 4) {
            src_X = source[4];
          }
        }
      }
    }
  }
}

template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_target(const Tensor& target)
{
  N        = target.size(0);
  trgt_C   = target.size(1);
  trgt_X   = target.size(2);
  trgt_Y   = dim >= 2 ? target.size(3) : oONE;
  trgt_Z   = dim >= 3 ? target.size(4) : oONE;
  trgt_sN  = target.stride(0);
  trgt_sC  = target.stride(1);
  trgt_sX  = target.stride(2);
  trgt_sY  = dim >= 2 ? target.stride(3) : oZERO;
  trgt_sZ  = dim >= 3 ? target.stride(4) : oZERO;;
  trgt_ptr = target.data_ptr<scalar_t>();
  trgt_opt = target.options();
}

template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_weight(const Tensor& weight)
{
  trgt_C   = weight.size(0);
  src_C    = weight.size(1) * G;
  wght_X   = weight.size(2);
  wght_Y   = dim >= 2 ? weight.size(3) : oONE;
  wght_Z   = dim >= 3 ? weight.size(4) : oONE;
  wght_sCt = weight.stride(0);
  wght_sCs = weight.stride(1);
  wght_sX  = weight.stride(2);
  wght_sY  = dim >= 2 ? weight.stride(3) : oZERO;
  wght_sZ  = dim >= 3 ? weight.stride(4) : oZERO;
  wght_ptr = weight.data_ptr<scalar_t>();
  wght_opt = weight.options();

  iso      = wght_X == wght_Y && wght_X == wght_Z;
  if (offsetlow0>=center0*dilation0 && offsetup0>=(wght_X-center0-1)*dilation0)
    bound0 = BoundType::NoCheck;
  if (offsetlow1>=center1*dilation1 && offsetup1>=(wght_Y-center1-1)*dilation1) 
    bound1 = BoundType::NoCheck;
  if (offsetlow2>=center2*dilation2 && offsetup2>=(wght_Z-center2-1)*dilation2)
    bound2 = BoundType::NoCheck;
}

template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_bias(const Tensor& bias)
{
  if (!bias.data_ptr<scalar_t>())
    return;
  bias_sC  = bias.stride(0);
  bias_ptr = bias.data_ptr<scalar_t>();
}

template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_center(IntArrayRef center)
{
  center0 = center.size() > 0 ? center[0] : static_cast<offset_t>(-1);
  center1 = center.size() > 1 ? center[1] : 
            center.size() > 0 ? center[0] : static_cast<offset_t>(-1);
  center2 = center.size() > 2 ? center[2] : 
            center.size() > 1 ? center[1] : 
            center.size() > 0 ? center[0] : static_cast<offset_t>(-1);
  if (center0 < 0)
    center0 = wght_X/2;
  if (center1 < 0)
    center1 = wght_Y/2;
  if (center2 < 0)
    center2 = wght_Z/2;
}

template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_source()
{
  // Deconvolution: we assume that target and weight are known
  src_X = stride0*trgt_Z + offsetlow0 - offsetup0;
  if (dim > 1) {
    src_Y = stride1*trgt_Y + offsetlow1 - offsetup1;
    if (dim > 2)
      src_Z = stride2*trgt_X + offsetlow2 - offsetup2;
  }
}

template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_target()
{
  // Convolution: we assume that source and weight are known
  trgt_X = (src_X - offsetlow0 + offsetup0) / stride0;
  if (dim > 1) {
    trgt_Y = (src_Y - offsetlow1 + offsetup1) / stride1;
    if (dim > 2)
      trgt_Z = (src_Z - offsetlow2 + offsetup2) / stride2;
  }
}


template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::init_output()
{
  output.clear();
  if (do_conv) {
    if (dim == 1)
      output.push_back(at::empty({N, trgt_C, trgt_X}, src_opt));
    else if (dim == 2)
      output.push_back(at::empty({N, trgt_C, trgt_X, trgt_Y}, src_opt));
    else
      output.push_back(at::empty({N, trgt_C, trgt_X, trgt_Y, trgt_Z}, src_opt));
    auto conv = output.back();
    out_sN   = conv.stride(0);
    out_sC   = conv.stride(1);
    out_sX   = conv.stride(2);
    out_sY   = dim >= 2 ? conv.stride(3) : oZERO;
    out_sZ   = dim >= 3 ? conv.stride(4) : oZERO;
    out_ptr  = conv.data_ptr<scalar_t>();

    if (!iso || !(wght_X == 3 || wght_X == 5))
      conv.zero_();
  }
  if (do_deconv) {
    if (dim == 1)
      output.push_back(at::empty({N, src_C, src_X}, trgt_opt));
    else if (dim == 2)
      output.push_back(at::empty({N, src_C, src_X, src_Y}, trgt_opt));
    else
      output.push_back(at::empty({N, src_C, src_X, src_Y, src_Z}, trgt_opt));
    auto deconv = output.back();
    out_sN   = deconv.stride(0);
    out_sC   = deconv.stride(1);
    out_sX   = deconv.stride(2);
    out_sY   = dim >= 2 ? deconv.stride(3) : oZERO;
    out_sZ   = dim >= 3 ? deconv.stride(4) : oZERO;
    out_ptr  = deconv.data_ptr<scalar_t>();

    if (!iso || !(wght_Z == 3 || wght_Z == 5))
      deconv.zero_();
  }
  if (do_grad) {
    if (dim == 1)
      output.push_back(at::zeros({trgt_C, src_C/G, wght_X}, wght_opt));
    else if (dim == 2)
      output.push_back(at::zeros({trgt_C, src_C/G, wght_X, wght_Y}, wght_opt));
    else
      output.push_back(at::zeros({trgt_C, src_C/G, wght_X, wght_Y, wght_Z}, wght_opt));
    auto grad = output.back();
    grad_sCt  = grad.stride(0);
    grad_sCs  = grad.stride(1);
    grad_sX   = grad.stride(2);
    grad_sY   = dim >= 2 ? grad.stride(3) : oZERO;
    grad_sZ   = dim >= 3 ? grad.stride(4) : oZERO;
    grad_ptr  = grad.data_ptr<scalar_t>();
  }

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void ConvImpl<scalar_t,offset_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  int64_t index = blockIdx * blockDim + threadIdx;
  int64_t nthreads = voxcount(do_conv);
  offset_t W, H, D;
  if (do_conv) {
    W   = trgt_X;
    H   = trgt_Y;
    D   = trgt_Z;
  } else {
    W   = src_X;
    H   = src_Y;
    D   = src_Z;
  }
  offset_t HW  = W  * H;
  offset_t DHW = HW * D;
  offset_t n, w, h, d;
  for (offset_t i=index; index < nthreads; index += blockDim*gridDim, i=index) {
    // Convert index: linear to sub
    n  = (i/DHW);
    d  = (i/HW) % D;
    h  = (i/W)  % H;
    w  = i % W;

    if (iso) {
      if (wght_Z == 3)
        if (dim == 1)
          return conv1d_3(w, n);
        else if (dim == 2)
          return conv2d_3x3(w, h, n);
        else
          return conv3d_3x3x3(w, h, d, n);
      else if (wght_Z == 5)
        if (dim == 1)
          return conv1d_5(w, n);
        else if (dim == 2)
          return conv2d_5x5(w, h, n);
        else
          return conv3d_5x5x5(w, h, d, n);
    }
    if (dim == 1)
      return conv1d(w, n);
    else if (dim == 2)
      return conv2d(w, h, n);
    else
      return conv3d(w, h, d, n);
  }
}

#else

// This bit loops over all target voxels. We therefore need to
// convert linear indices to multivariate indices. The way I do it
// might not be optimal.
// Note that I parallelize across all voxels (wheareas ATen's grid 
// sampler is only parallelized across batches).
//
// TODO: check that the default grain size is optimal. We do quite a lot 
// of compute per voxel, so a smaller value might be better suited.
template <typename scalar_t, typename offset_t> NI_HOST
void ConvImpl<scalar_t,offset_t>::loop() const
{

  printInfo();

  // Parallelize across voxels   
  offset_t trgt_NXYZ = trgt_Z * trgt_Y * trgt_X * N;
  offset_t trgt_XYZ  = trgt_Z * trgt_Y * trgt_X;
  offset_t trgt_YZ   = trgt_Z * trgt_Y;
  at::parallel_for(0, trgt_NXYZ, GRAIN_SIZE,
                   [&](offset_t start, offset_t end) {
    offset_t n, w, h, d;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/trgt_XYZ);
      w  = (i/trgt_YZ) % trgt_X;
      h  = (i/trgt_Z)  % trgt_Y;
      d  = i % trgt_Z;

      conv3d(w, h, d, n);

//      if (iso) {
//        if (wght_Z == 3)
//          if (dim == 1)
//            return conv1d_3(w, n);
//          else if (dim == 2)
//            return conv2d_3x3(w, h, n);
//          else
//            return conv3d_3x3x3(w, h, d, n);
//        else if (wght_Z == 5)
//          if (dim == 1)
//            return conv1d_5(w, n);
//          else if (dim == 2)
//            return conv2d_5x5(w, h, n);
//          else
//            return conv3d_5x5x5(w, h, d, n);
//      }
//      if (dim == 1)
//        return conv1d(w, n);
//      else if (dim == 2)
//        return conv2d(w, h, n);
//      else
//        return conv3d(w, h, d, n);

    }
  });

  printf("done\n");
}

#endif

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     GENERIC CONVOLUTION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void ConvImpl<scalar_t,offset_t>::conv3d(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  // Precompute pointers 
  scalar_t *trgt_ptr_NXYZ = trgt_ptr + n * trgt_sN  + w * trgt_sX
                                     + h * trgt_sY  + d * trgt_sZ;

  // Compute coordinate of centre voxel in source volume
  offset_t x0 = w * stride0 + offsetlow0;
  offset_t y0 = h * stride1 + offsetlow1;
  offset_t z0 = d * stride2 + offsetlow2;

  // printf("[%d %d %d] -> [%d %d %d]\n", w, h, d, x0, y0, z0);

  if (do_conv) {

    scalar_t *out_ptr_NXYZ  = out_ptr  + n * out_sN + w * out_sX
                                       + h * out_sY + d * out_sZ;

    // Convolve
    // Note that we can't pre-compute indices because we don't have 
    // an upper bound on the kernel length (or we'd need to dynamically
    // allocate memory and that's worse)

    for (offset_t ct = 0; ct < trgt_C; ++ct) {
      scalar_t target;
      if (do_grad) target = trgt_ptr_NXYZ[ct*trgt_sC];
      scalar_t * wght_ptr_T = wght_ptr + ct * wght_sCt;

      scalar_t val2 = sZERO;
      for (offset_t k = 0; k < wght_Z; ++k) {
        offset_t owz = k * wght_sZ;
        offset_t ogz = k * grad_sZ;
        offset_t z   = z0 + (k-center2)*dilation2;
        offset_t osz = bound::index(bound2, z, src_Z) * src_sZ;
        uint8_t  sz  = bound::sign(bound2, z, src_Z);

        scalar_t val1 = sZERO;
        for (offset_t j = 0; j < wght_Y; ++j) {
          offset_t owyz = owz + j * wght_sY;
          offset_t ogyz = ogz + j * grad_sY;
          offset_t y    = y0 + (j-center1)*dilation1;
          offset_t osyz = osz + bound::index(bound1, y, src_Y) * src_sY;
          uint8_t  syz  = sz * bound::sign(bound1, y, src_Y);

          scalar_t val0 = sZERO;
          for (offset_t i = 0; i < wght_X; ++i) {
            offset_t owxyz = owyz + i * wght_sX;
            offset_t ogxyz = ogyz + i * grad_sX;
            offset_t x     = x0 + (i-center0)*dilation0;
            uint8_t  sxyz  = syz * bound::sign(bound0, x, src_X);
            offset_t osxyz = osyz + bound::index(bound0, x, src_X) * src_sX;
            scalar_t * src_ptr_NXYZ = src_ptr + n * src_sN + osxyz;
            scalar_t * wght_ptr_XYZ = wght_ptr_T + owxyz;
            scalar_t * grad_ptr_XYZ = grad_ptr + ogxyz;

            scalar_t valc = sZERO;
            for (offset_t cs = 0; cs < src_C; ++cs) {
              offset_t src = bound::get(src_ptr_NXYZ, cs*src_sC, sxyz);
              valc += src * wght_ptr_XYZ[(cs%G)*wght_sCs];
              if (do_grad)
                bound::add(grad_ptr_XYZ, (cs%G)*grad_sCs, src * target);
                // ^ this is probably very bad in terms of floating point precision
            } // cs

            val0 += valc;
          } // x
          val1 += val0;
        } // y
        val2 += val1;
      } // z
      if (bias_ptr) val2 += bias_ptr[ct*bias_sC];
      out_ptr_NXYZ[ct*out_sC] = val2;
    } // ct


  } else if (do_deconv) {

    for (offset_t ct = 0; ct < trgt_C; ++ct) {
      scalar_t target;
      if (do_grad) target = trgt_ptr_NXYZ[ct*trgt_sC];

      for (offset_t k = 0; k < wght_Z; ++k) {
        offset_t owz = k * wght_sZ;
        offset_t ooz = k * out_sZ;
        offset_t ogz = k * grad_sZ;
        offset_t z  = z0 + (k-center2)*dilation2;
        offset_t osz = bound::index(bound2, z, src_Z) * src_sZ;
        uint8_t  sz  = bound::sign(bound2, z, src_Z);

        for (offset_t j = 0; j < wght_Y; ++j) {
          offset_t owyz = owz + j * wght_sY;
          offset_t ooyz = ooz + j * out_sY;
          offset_t ogyz = ogz + j * grad_sY;
          offset_t y   = y0 + (j-center1)*dilation1;
          offset_t osyz = osz + bound::index(bound1, y, src_Y) * src_sY;
          uint8_t  syz  = sz * bound::sign(bound1, y, src_Y);

          for (offset_t i = 0; i < wght_X; ++i) {
            offset_t owxyz = owyz + i * wght_sX;
            offset_t ooxyz = ooyz + i * out_sX;
            offset_t ogxyz = ogyz + i * grad_sX;
            offset_t x  = x0 + (i-center0)*dilation0;
            offset_t osxyz = osyz + bound::index(bound0, x, src_X)* src_sX;
            uint8_t  sxyz  = syz * bound::sign(bound0, x, src_X);
            scalar_t * src_ptr_NXYZ = src_ptr + n * src_sN + osxyz;
            scalar_t * out_ptr_NXYZ = out_ptr + n * out_sN + ooxyz;
            scalar_t * wght_ptr_XYZ = wght_ptr + owxyz;
            scalar_t * grad_ptr_XYZ = grad_ptr + ogxyz;

            for (offset_t cs = 0; cs < src_C; ++cs) {
              scalar_t val = target * wght_ptr_XYZ[(cs%G)*wght_sCs];
              bound::add(out_ptr_NXYZ, cs*out_sC, val, sxyz);
              // ^ this is probably very bad in terms of floating point precision
              if (do_grad) {
                offset_t src = bound::get(src_ptr_NXYZ, cs*src_sC, sxyz);
                bound::add(grad_ptr_XYZ, (cs%G)*grad_sCs, src * target);
                // ^ this is probably very bad in terms of floating point precision
              }
            } // cs

          } // x
        } // y
      } // z
    } // ct
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  CUDA KERNEL (MUST BE OUT OF CLASS)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __CUDACC__
// CUDA Kernel
template <typename scalar_t, typename offset_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void conv_kernel(ConvImpl<scalar_t,offset_t> f) {
  f.loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define CONV_INSTANTIATE1(SourceType) \
  template std::deque<Tensor> conv( \
    const SourceType&, const Tensor&, const Tensor&, const Tensor&, \
    int, BoundVectorRef, \
    IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, \
    bool, bool, bool)
#define CONV_INSTANTIATE() \
  CONV_INSTANTIATE1(Tensor); \
  CONV_INSTANTIATE1(IntArrayRef)

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Three arguments (source/target, weight, bias)
NI_HOST
std::deque<Tensor> conv(
  const Tensor& input, const Tensor& weight, const Tensor& bias,
  int groups, BoundVectorRef bound, IntArrayRef stride, IntArrayRef dilation,
  IntArrayRef offsetlow, IntArrayRef offsetup, IntArrayRef center,
  bool do_conv, bool do_deconv, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "conv", [&] {
    ConvImpl<scalar_t,int32_t>
    f(weight.dim()-2, groups, bound, stride, dilation, offsetlow, offsetup,
      do_conv, do_deconv, do_grad);
    f.ioset(input, weight, bias, center, do_deconv);
    CONV_kernel<<<GET_BLOCKS(f.voxcount()), CUDA_NUM_THREADS, 0, 
                      at::cuda::getCurrentCUDAStream()>>>(f);
    return f.output;
  });
}

// Four arguments (source, weight, bias, target)
template <typename SourceType>
NI_HOST
std::deque<Tensor> conv(
  const SourceType& source, const Tensor& weight, const Tensor& bias, const Tensor& target,
  int groups, BoundVectorRef bound, IntArrayRef stride, IntArrayRef dilation,
  IntArrayRef offsetlow, IntArrayRef offsetup, IntArrayRef center,
  bool do_conv, bool do_deconv, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "conv", [&] {
    ConvImpl<scalar_t,int32_t>
    f(weight.dim()-2, groups, bound, stride, dilation, offsetlow, offsetup,
      do_conv, do_deconv, do_grad);
    f.ioset(source, weight, bias, target, center);
    CONV_kernel<<<GET_BLOCKS(f.voxcount()), CUDA_NUM_THREADS, 0, 
                      at::cuda::getCurrentCUDAStream()>>>(f);
    return f.output;
  });
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Three arguments (source/target, weight, bias)
NI_HOST
std::deque<Tensor> conv(
  const Tensor& input, const Tensor& weight, const Tensor& bias,
  int groups, BoundVectorRef bound, IntArrayRef stride, IntArrayRef dilation,
  IntArrayRef offsetlow, IntArrayRef offsetup, IntArrayRef center,
  bool do_conv, bool do_deconv, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "conv", [&] {
    ConvImpl<scalar_t,int64_t>
    f(weight.dim()-2, groups, bound, stride, dilation, offsetlow, offsetup,
      do_conv, do_deconv, do_grad);
    f.ioset(input, weight, bias, center, do_deconv);
    f.loop();
    return f.output;
  });
}

// Three arguments (source, weight, bias, target)
template <typename SourceType>
NI_HOST
std::deque<Tensor> conv(
  const SourceType& source, const Tensor& weight, const Tensor& bias, const Tensor& target,
  int groups, BoundVectorRef bound, IntArrayRef stride, IntArrayRef dilation,
  IntArrayRef offsetlow, IntArrayRef offsetup, IntArrayRef center,
  bool do_conv, bool do_deconv, bool do_grad)
{
  return AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "conv", [&] {
    ConvImpl<scalar_t,int64_t>
    f(weight.dim()-2, groups, bound, stride, dilation, offsetlow, offsetup,
      do_conv, do_deconv, do_grad);
    f.ioset(source, weight, bias, target, center);
    f.loop();
    return f.output;
  });
}

#endif // __CUDACC__

CONV_INSTANTIATE();

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

NI_HOST
std::deque<Tensor> conv(
  const Tensor& input, const Tensor& weight, const Tensor& bias,
  int groups, BoundVectorRef bound, IntArrayRef stride, IntArrayRef dilation,
  IntArrayRef offsetlow, IntArrayRef offsetup, IntArrayRef center,
  bool do_conv, bool do_deconv, bool do_grad)
{
  throw std::logic_error("Function not implemented for this device.");
}

template <typename SourceType>
NI_HOST
std::deque<Tensor> conv(
  const SourceType& source, const Tensor& weight, const Tensor& bias, const Tensor& target,
  int groups, BoundVectorRef bound, IntArrayRef stride, IntArrayRef dilation,
  IntArrayRef offsetlow, IntArrayRef offsetup, IntArrayRef center,
  bool do_conv, bool do_deconv, bool do_grad)
{
  throw std::logic_error("Function not implemented for this device.");
}

CONV_INSTANTIATE();

} // namespace notimplemented

} // namespace ni

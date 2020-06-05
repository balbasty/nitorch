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

  // ~~~ CONSTRUCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  NI_HOST
  ConvImpl(int dim, int groups, BoundVectorRef bound, 
           IntArrayRef stride, IntArrayRef dilation,
           IntArrayRef offsetlow, IntArrayRef offsetup, 
           bool do_conv, bool do_deconv, bool do_grad):
    dim(dim),
    G(static_cast<offset_t>(groups)),
    bound0(bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound1(bound.size() > 1 ? bound[1] : 
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    bound2(bound.size() > 2 ? bound[2] : 
           bound.size() > 1 ? bound[1] : 
           bound.size() > 0 ? bound[0] : BoundType::Replicate),
    stride0(stride.size() > 0 ? stride[0] : 1),
    stride1(stride.size() > 1 ? stride[1] : 
            stride.size() > 0 ? stride[0] : 1),
    stride2(stride.size() > 2 ? stride[2] : 
            stride.size() > 1 ? stride[1] : 
            stride.size() > 0 ? stride[0] : 1),
    dilation0(dilation.size() > 0 ? dilation[0] : 1),
    dilation1(dilation.size() > 1 ? dilation[1] : 
              dilation.size() > 0 ? dilation[0] : 1),
    dilation2(dilation.size() > 2 ? dilation[2] : 
              dilation.size() > 1 ? dilation[1] : 
              dilation.size() > 0 ? dilation[0] : 1),
    offsetlow0(offsetlow.size() > 0 ? offsetlow[0] : 0),
    offsetlow1(offsetlow.size() > 1 ? offsetlow[1] : 
               offsetlow.size() > 0 ? offsetlow[0] : 0),
    offsetlow2(offsetlow.size() > 2 ? offsetlow[2] : 
               offsetlow.size() > 1 ? offsetlow[1] : 
               offsetlow.size() > 0 ? offsetlow[0] : 0),
    offsetup0(offsetup.size() > 0 ? offsetup[0] : 0),
    offsetup1(offsetup.size() > 1 ? offsetup[1] : 
              offsetup.size() > 0 ? offsetup[0] : 0),
    offsetup2(offsetup.size() > 2 ? offsetup[2] : 
              offsetup.size() > 1 ? offsetup[1] : 
              offsetup.size() > 0 ? offsetup[0] : 0),
    do_conv(do_conv),
    do_deconv(do_deconv),
    do_grad(do_grad)
  {}

  // ~~~ PUBLIC VALUE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::deque<Tensor> output;

  // NI_HOST NI_DEVICE void printInfo() const {
  //   printf("dim: %d\n", dim);
  //   printf("do_pull:  %d\n", do_pull);
  //   printf("do_push:  %d\n", do_push);
  //   printf("do_count: %d\n", do_count);
  //   printf("do_sgrad: %d\n", do_sgrad);
  //   printf("do_grad:  %d\n", do_grad);
  //   printf("bound:         [%d %d %d]\n", static_cast<int>(bound0), 
  //     static_cast<int>(bound1), static_cast<int>(bound2));
  //   printf("interpolation: [%d %d %d]\n", static_cast<int>(interpolation0), 
  //     static_cast<int>(interpolation1), static_cast<int>(interpolation2));
  //   printf("src:  [%d %d %d]\n", src_W, src_H, src_D);
  //   printf("trgt: [%d %d %d (%d)]\n", trgt_W, trgt_H, trgt_D, trgt_K);
  //   printf("N: %d\n", N);
  //   printf("C: %d\n", C);
  //   printf("src  -> %lu\n", reinterpret_cast<std::uintptr_t>(src_ptr));
  //   printf("trgt -> %lu\n", reinterpret_cast<std::uintptr_t>(trgt_ptr));
  //   printf("grid -> %lu\n", reinterpret_cast<std::uintptr_t>(grid_ptr));
  //   printf("out  -> %lu\n", reinterpret_cast<std::uintptr_t>(out_ptr));
  //   printf("grad -> %lu\n", reinterpret_cast<std::uintptr_t>(grad_ptr));
  // }

  // ~~~ FUNCTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void ioset // Conv
  (const Tensor& source, const Tensor& weight, const Tensor& bias, IntArrayRef center)
  {
    init_all();
    init_source(source);
    init_weight(weight);
    init_bias(bias);
    init_center(center);
    init_target();
    init_output();
  }

  NI_HOST void ioset // Backward
  (const Tensor& source, const Tensor& weight, const Tensor& bias, const Tensor& target, 
   IntArrayRef center)
  {
    init_all();
    init_source(source);
    init_weight(weight);
    init_bias(bias);
    init_target(target);
    init_center(center);
    init_output();
  }

  NI_HOST void ioset // Deconv
  (const Tensor& weight, const Tensor& bias, const Tensor& target, IntArrayRef center)
  {
    init_all();
    init_weight(weight);
    init_bias(bias);
    init_target(target);
    init_center(center);
    init_source();
    init_output();
  }

#if __CUDACC__
  NI_DEVICE void loop(int threadIdx, int blockIdx, 
                      int blockDim, int gridDim) const;
#else
  void loop() const;
#endif

  NI_HOST NI_DEVICE int64_t voxcount() const { 
    return N * trgt_D * trgt_H * trgt_W;
  }

private:

  // ~~~ COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NI_HOST void init_all();
  NI_HOST void init_source(const Tensor& source);
  NI_HOST void init_weight(const Tensor& weight); 
  NI_HOST void init_bias(const Tensor& bias); 
  NI_HOST void init_target(const Tensor& target); 
  NI_HOST void init_center(IntArrayRef source_size);
  NI_HOST void init_output();
  NI_DEVICE void conv1d(
    scalar_t x, offset_t w, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv1d_3(
    scalar_t x, offset_t w, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv1d_5(
    scalar_t x, offset_t w, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv2d(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv2d_3x3(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv2d_5x5(
    scalar_t x, scalar_t y,
    offset_t w, offset_t h, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv3d(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const;
  NI_DEVICE void conv3d_3x3x3(
    scalar_t x, scalar_t y, scalar_t z, 
    offset_t w, offset_t h, offset_t d, offset_t n) const { /* TODO */ };
  NI_DEVICE void conv3d_5x5x5(
    scalar_t x, scalar_t y, scalar_t z, 
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
  offset_t src_D;
  offset_t src_H;
  offset_t src_W;
  offset_t trgt_C;
  offset_t trgt_D;
  offset_t trgt_H;
  offset_t trgt_W;
  offset_t wght_D;
  offset_t wght_H;
  offset_t wght_W;
  offset_t src_sN;
  offset_t src_sC;
  offset_t src_sD;
  offset_t src_sH;
  offset_t src_sW;
  scalar_t *src_ptr;
  offset_t trgt_sN;
  offset_t trgt_sC;
  offset_t trgt_sD;
  offset_t trgt_sH;
  offset_t trgt_sW;
  scalar_t *trgt_ptr;
  offset_t wght_sCs;
  offset_t wght_sCt;
  offset_t wght_sD;
  offset_t wght_sH;
  offset_t wght_sW;
  scalar_t *wght_ptr;
  offset_t bias_sC;
  scalar_t *bias_ptr;
  offset_t out_sN;
  offset_t out_sC;
  offset_t out_sD;
  offset_t out_sH;
  offset_t out_sW;
  scalar_t *out_ptr;
  offset_t grad_sCo;
  offset_t grad_sCi;
  offset_t grad_sD;
  offset_t grad_sH;
  offset_t grad_sW;
  scalar_t *grad_ptr;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          INITIALISATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t>
void PushPullImpl<scalar_t,offset_t>::init_all()
{
  src_opt  = wght_opt = trgt_opt = TensorOptions();
  N = G    = static_cast<offset_t>(1);
  wght_D   = wght_H   = wght_W   = static_cast<offset_t>(1);
  src_D    = src_H    = src_W    = src_C   = static_cast<offset_t>(1);
  trgt_D   = trgt_H   = trgt_W   = trgt_C  = static_cast<offset_t>(1);
  src_sN   = src_sC   = src_sD   = src_sH  = src_sW   = static_cast<offset_t>(0);
  wght_sCs = wght_sCt = wght_sD  = wght_sH = wght_sW  = static_cast<offset_t>(0);
  grad_sN  = grad_sC  = grad_sD  = grad_sH = grad_sW  = static_cast<offset_t>(0);
  trgt_sN  = trgt_sC  = trgt_sD  = trgt_sH = trgt_sW  = static_cast<offset_t>(0);
  out_sN   = out_sC   = out_sD   = out_sH  = out_sW   = static_cast<offset_t>(0);
  src_ptr  = trgt_ptr = wght_ptr = out_ptr = grad_ptr = static_cast<scalar_t*>(0);
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_source(const Tensor& source)
{
  N       = source.size(0);
  src_C   = source.size(1);
  src_D   = dim == 3 ? source.size(2) : static_cast<offset_t>(1);
  src_H   = dim >= 2 ? source.size(dim == 2 ? 2 : 3) : static_cast<offset_t>(1);
  src_W   = source.size(dim == 3 ? 4 : dim == 2 ? 3 : 2);
  src_sN  = source.stride(0);
  src_sC  = source.stride(1);
  src_sD  = dim == 3 ? source.stride(2) : static_cast<offset_t>(0);
  src_sH  = dim >= 2 ? source.stride(dim == 2 ? 2 : 3) : static_cast<offset_t>(0);
  src_sW  = source.stride(dim == 3 ? 4 : dim == 2 ? 3 : 2);
  src_ptr = source.data_ptr<scalar_t>();
  src_opt = source.options();
}
template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_target(const Tensor& target)
{
  N        = target.size(0);
  trgt_C   = target.size(1); // What if target is a 'count'?
  trgt_D   = dim == 3 ? target.size(2) : static_cast<offset_t>(1);
  trgt_H   = dim >= 2 ? target.size(dim == 2 ? 2 : 3) : static_cast<offset_t>(1);
  trgt_W   = target.size(dim == 3 ? 4 : dim == 2 ? 3 : 2);
  trgt_sN  = target.stride(0);
  trgt_sC  = target.stride(1);
  trgt_sD  = dim == 3 ? target.stride(2) : static_cast<offset_t>(0);
  trgt_sH  = dim >= 2 ? target.stride(dim == 2 ? 2 : 3) : static_cast<offset_t>(0);
  trgt_sW  = target.stride(dim == 3 ? 4 : dim == 2 ? 3 : 2);
  trgt_ptr = target.data_ptr<scalar_t>();
  trgt_opt = target.options();
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_weight(const Tensor& weight)
{
  trgt_C   = weight.size(0);
  src_C    = weight.size(1) * G;
  wght_D   = dim == 3 ? weight.size(2) : static_cast<offset_t>(1);
  wght_H   = dim >= 2 ? weight.size(dim == 2 ? 2 : 3) : static_cast<offset_t>(1);
  wght_W   = weight.size(dim == 3 ? 4 : dim == 2 ? 3 : 2);
  wght_sCt = weight.stride(0);
  wght_sCs = weight.stride(1);
  wght_sD  = dim == 3 ? weight.stride(2) : static_cast<offset_t>(0);
  wght_sH  = dim >= 2 ? weight.stride(dim == 2 ? 2 : 3) : static_cast<offset_t>(0);
  wght_sW  = weight.stride(dim == 3 ? 4 : dim == 2 ? 3 : 2);
  wght_ptr = weight.data_ptr<scalar_t>;
  wght_opt = weight.options();

  iso      = wght_D == wght_H && wght_D == wght_W;
  if (offsetlow0>=center0*dilation0 && offsetup0>=(wght_W-center0-1)*dilation0) 
    bound0 = BoundType::NoCheck;
  if (offsetlow1>=center1*dilation1 && offsetup1>=(wght_H-center1-1)*dilation1) 
    bound1 = BoundType::NoCheck;
  if (offsetlow2>=center2*dilation2 && offsetup2>=(wght_D-center2-1)*dilation2) 
    bound2 = BoundType::NoCheck;
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_bias(const Tensor& bias)
{
  if (!bias.defined())
    return;
  bias_sC  = bias.stride(0);
  bias_ptr = bias.data_ptr<scalar_t>;
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_center(IntArrayRef center)
{
  center0 = center.size() > 0 ? center[0] : -1;
  center1 = center.size() > 1 ? center[1] : 
            center.size() > 0 ? center[0] : -1;
  center2 = center.size() > 2 ? center[2] : 
            center.size() > 1 ? center[1] : 
            center.size() > 0 ? center[0] : -1;
  if (center0 < 0)
    center0 = wght_W/2;
  if (center1 < 0)
    center1 = wght_H/2;
  if (center2 < 0)
    center2 = wght_D/2;
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_source()
{
  // Deconvolution: we assume that target and weight are known
  src_W = stride0*trgt_W + offsetlow0 - offsetup0;
  if (dim > 1) {
    src_H = stride1*trgt_H + offsetlow1 - offsetup1;
    if (dim > 2)
      src_D = stride2*trgt_D + offsetlow2 - offsetup2;
  }
}

template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_target()
{
  // Convolution: we assume that source and weight are known
  trgt_W = (src_W - offsetlow0 + offsetup0) / stride0;
  if (dim > 1) {
    trgt_H = (src_H - offsetlow1 + offsetup1) / stride1;
    if (dim > 2)
      trgt_D = (src_D - offsetlow2 + offsetup2) / stride2;
  }
}


template <typename scalar_t, typename offset_t> NI_HOST
void PushPullImpl<scalar_t,offset_t>::init_output()
{
  output.clear();
  if (do_conv) {
    if (dim == 1)
      output.push_back(at::empty({N, trgt_C, trgt_W}, src_opt));
    else if (dim == 2)
      output.push_back(at::empty({N, trgt_C, trgt_H, trgt_W}, src_opt));
    else
      output.push_back(at::empty({N, trgt_C, trgt_D, trgt_H, trgt_W}, src_opt));
    auto conv = output.back();
    out_sN   = conv.stride(0);
    out_sC   = conv.stride(1);
    out_sD   = dim == 3 ? conv.stride(2) : static_cast<offset_t>(0);
    out_sH   = dim >= 2 ? conv.stride(dim == 2 ? 2 : 3) : static_cast<offset_t>(0);
    out_sW   = conv.stride(dim == 3 ? 4 : dim == 2 ? 3 : 2);
    out_ptr  = conv.data_ptr<scalar_t>();

    if (!iso || !(wght_W == 3 || wght_W == 5))
      conv.zero_();
  }
  if (do_deconv) {
    if (dim == 1)
      output.push_back(at::empty({N, src_C, src_W}, trgt_opt));
    else if (dim == 2)
      output.push_back(at::empty({N, src_C, src_H, src_W}, trgt_opt));
    else
      output.push_back(at::empty({N, src_C, src_D, src_H, src_W}, trgt_opt));
    auto deconv = output.back();
    out_sN   = deconv.stride(0);
    out_sC   = deconv.stride(1);
    out_sD   = dim == 3 ? deconv.stride(2) : static_cast<offset_t>(0);
    out_sH   = dim >= 2 ? deconv.stride(dim == 2 ? 2 : 3) : static_cast<offset_t>(0);
    out_sW   = deconv.stride(dim == 3 ? 4 : dim == 2 ? 3 : 2);
    out_ptr  = deconv.data_ptr<scalar_t>();

    if (!iso || !(wght_W == 3 || wght_W == 5))
      conv.deconv();
  }
  if (do_grad) {
    if (dim == 1)
      output.push_back(at::zeros({C_trgt, C_src/G, wght_W}, wght_opt));
    else if (dim == 2)
      output.push_back(at::zeros({C_trgt, C_src/G, wght_H, wght_W}, wght_opt));
    else
      output.push_back(at::zeros({C_trgt, C_src/G, wght_D, wght_H, wght_W}, wght_opt));
    auto grad = output.back();
    grad_sCt  = grad.stride(0);
    grad_sCs  = grad.stride(1);
    grad_sD   = dim == 3 ? grad.stride(2) : static_cast<offset_t>(0);
    grad_sH   = dim >= 2 ? grad.stride(dim == 2 ? 2 : 3) : static_cast<offset_t>(0);
    grad_sW   = grad.stride(dim == 3 ? 4 : dim == 2 ? 3 : 2);
    grad_ptr  = grad.data_ptr<scalar_t>();
  }



}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             LOOP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if __CUDACC__

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::loop(
  int threadIdx, int blockIdx, int blockDim, int gridDim) const {

  int64_t index = blockIdx * blockDim + threadIdx;
  int64_t nthreads = voxcount(do_conv);
  offset_t D, H, W;
  if (do_conv) {
    W   = trgt_W;
    H   = trgt_H;
    D   = trgt_D;
  } else {
    W   = src_W;
    H   = src_H;
    D   = src_D;
  }
  offset_t HW  = W  * H;
  offset_t DHW = HW * D;
  offset_t n, d, h, w;
  for (offset_t i=index; index < nthreads; index += blockDim*gridDim, i=index) {
    // Convert index: linear to sub
    n  = (i/DHW);
    d  = (i/HW) % D;
    h  = (i/W)  % H;
    w  = i % W;

    if (iso) {
      if (wght_W == 3)
        if (dim == 1)
          return conv1d_3(w, n);
        else if (dim == 2)
          return conv2d_3x3(w, h, n);
        else
          return conv3d_3x3x3(w, h, d, n);
      else if (wght_W == 5)
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
void PushPullImpl<scalar_t,offset_t>::loop() const
{
  offset_t D, H, W;
  if (do_conv) {
    W   = trgt_W;
    H   = trgt_H;
    D   = trgt_D;
  } else {
    W   = src_W;
    H   = src_H;
    D   = src_D;
  }

# if !(AT_PARALLEL_OPENMP)
    if (do_grad)
    {
      // I do not have access to atomic operations so I cannot 
      // parallelize across voxels.  
      at::parallel_for(0, N, 0, [&](offset_t start, offset_t end) {
        for (offset_t n = start; n < end; ++n) {
          if (dim == 1) {
            for (offset_t w=0; w<W; ++w) {
              if (iso)
                if (wght_W == 3)
                  return conv1d_3(w, h, d, n);
                else if (wght_W == 5)
                  return conv1d_5(w, h, d, n);
              return conv1d(w, h, d, n);
            }
          } else if (dim == 2) {
            for (offset_t h=0; h<H; ++h)
            for (offset_t w=0; w<W; ++w) {
              if (iso)
                if (wght_W == 3)
                  return conv2d_3x3(w, h, d, n);
                else if (wght_W == 5)
                  return conv2d_5x5(w, h, d, n);
              return conv3d(w, h, d, n);
            }
          } else {
            for (offset_t d=0; d<D; ++d)
            for (offset_t h=0; h<H; ++h)
            for (offset_t w=0; w<W; ++w) {
              if (iso)
                if (wght_W == 3)
                  return conv3d_3x3x3(w, h, d, n);
                else if (wght_W == 5)
                  return conv3d_5x5x5(w, h, d, n);
              return conv3d(w, h, d, n);
            }
          }
        }
      }); 
      return
    }
#  endif

  // Parallelize across voxels   
  offset_t HW  = W  * H;
  offset_t DHW = HW * D;
  at::parallel_for(0, N * DHW, GRAIN_SIZE, 
                   [&](offset_t start, offset_t end) {
    offset_t n, d, h, w;
    for (offset_t i = start; i < end; ++i) {
      // Convert index: linear to sub
      n  = (i/DHW);
      d  = (i/HW) % D;
      h  = (i/W)  % H;
      w  = i % W;

      if (iso) {
        if (wght_W == 3)
          if (dim == 1)
            return conv1d_3(w, n);
          else if (dim == 2)
            return conv2d_3x3(w, h, n);
          else
            return conv3d_3x3x3(w, h, d, n);
        else if (wght_W == 5)
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
  }); 
}

#endif

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     GENERIC CONVOLUTION 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename offset_t> NI_DEVICE
void PushPullImpl<scalar_t,offset_t>::conv3d(
  offset_t w, offset_t h, offset_t d, offset_t n) const
{
  // Precompute pointers 
  scalar_t *trgt_ptr_NDHW = trgt_ptr + n * trgt_sN  + d * trgt_sD 
                                     + h * trgt_sH  + w * trgt_sW;

  // Compute coordinate of centre voxel in source volume
  offset_t x = w * stride0 + offsetlow0;
  offset_t y = h * stride1 + offsetlow1;
  offset_t z = d * stride2 + offsetlow2;

  if (do_conv) {

    scalar_t *out_ptr_NDHW  = out_ptr  + n * out_sN   + d * out_sD 
                                       + h * out_sH   + w * out_sW;
    // Convolve
    // Note that we can't pre-compute indices because we don't have 
    // an uppper bound on the kernel length (or we'd need to dynamically 
    // allocate memory and that's worse)

    for (offset_t ct = 0; ct < trgt_C; ++ct) {
      scalar_t target;
      if (do_grad) target = trgt_ptr_NDHW[ct*trgt_sC];
      scalar_t val2 = static_cast<scalar_t>(0);
      for (offset_t k = 0; k < wght_D; ++k) {
        offset_t owz = k * wght_sD;
        offset_t ogz = k * grad_sD;
        offset_t iz  = z + (k-center2)*dilation2;
        offset_t osz = bound::index(bound2, iz, src_D) * src_sD;
        uint8_t  sz  = bound::sign(bound2, iz, src_D);
        scalar_t val1 = static_cast<scalar_t>(0);
        for (offset_t j = 0; j < wght_D; ++k) {
          offset_t owyz = owz + j * wght_sH;
          offset_t ogyz = ogz + j * grad_sH;
          offset_t iy   = y + (j-center1)*dilation1;
          offset_t osyz = osz + bound::index(bound1, iy, src_H) * src_sH;
          uint8_t  syz  = sz * bound::sign(bound1, iy src_H);
          scalar_t val0 = static_cast<scalar_t>(0);
          for (offset_t i = 0; i < wght_D; ++k) {
            offset_t owxyz = owyz + i * wght_sW;
            offset_t ogxyz = ogyz + i * grad_sW;
            offset_t ix  = x + (i-center0)*dilation0;
            offset_t osxyz = osyz + bound::index(bound0, ix, src_W)* src_sW;
            uint8_t  sxyz  = syz * bound::sign(bound0, ix, src_W);
            scalar_t * src_ptr_NDHW = src_ptr_NC0 + osxyz;
            scalar_t * wght_ptr_DHW = wght_ptr + owxyz;
            scalar_t * grad_ptr_DHW = grad_ptr + ogxyz;
            scalar_t valc = static_cast<scalar_t>(0);
            for (offset_t cs = 0; cs < src_C; ++cs) {
              offset_t src = bound::get(src_ptr_NDHW, cs*src_sC, sxyz);
              valc += src * wght_ptr_NCDHW[(cs%G)*wght_sCs];
              if (do_grad)
                bound::add(grad_ptr_DHW, (cs%G)*grad_sCs, src * target);
                // ^ this is probably very bad in terms of floating point precision
            } // cs
            val0 += valc;
          } // x
          val1 += val0
        } // y
        val2 += val1;
      } // z
      if (bias_ptr) val2 += bias_ptr[ct*bias_sC];
      out_ptr_NDHW[ct*out_sC] = val2;
    } // ct



  } else if (do_deconv) {

    for (offset_t ct = 0; ct < trgt_C; ++ct) {
      scalar_t target;
      if (do_grad) target = trgt_ptr_NDHW[ct*trgt_sC];
      for (offset_t k = 0; k < wght_D; ++k) {
        offset_t owz = k * wght_sD;
        offset_t ooz = k * out_sD;
        offset_t ogz = k * grad_sD;
        offset_t iz  = z + (k-center2)*dilation2;
        offset_t osz = bound::index(bound2, iz, src_D) * src_sD;
        uint8_t  sz  = bound::sign(bound2, iz, src_D);
        for (offset_t j = 0; j < wght_D; ++k) {
          offset_t owyz = owz + j * wght_sH;
          offset_t ooyz = ooz + j * out_sH;
          offset_t ogyz = ogz + j * grad_sH;
          offset_t iy   = y + (j-center1)*dilation1;
          offset_t osyz = osz + bound::index(bound1, iy, src_H) * src_sH;
          uint8_t  syz  = sz * bound::sign(bound1, iy src_H);
          for (offset_t i = 0; i < wght_D; ++k) {
            offset_t owxyz = owyz + i * wght_sW;
            offset_t ooxyz = ooyz + i * out_sW;
            offset_t ogxyz = ogyz + i * grad_sW;
            offset_t ix  = x + (i-center0)*dilation0;
            offset_t osxyz = osyz + bound::index(bound0, ix, src_W)* src_sW;
            uint8_t  sxyz  = syz * bound::sign(bound0, ix, src_W);
            scalar_t * src_ptr_NDHW = src_ptr + n * src_sN + osxyz;
            scalar_t * out_ptr_NDHW = out_ptr + n * out_sN + ooxyz;
            scalar_t * wght_ptr_DHW = wght_ptr + owxyz;
            scalar_t * grad_ptr_DHW = grad_ptr + ogxyz;
            for (offset_t cs = 0; cs < src_C; ++cs) {
              scalar_t val = target * wght_ptr_NCDHW[(cs%G)*wght_sCs];
              bound::add(out_ptr_NDHW, cs*out_sC, val, sxyz);
              // ^ this is probably very bad in terms of floating point precision
              if (do_grad) {
                offset_t src = bound::get(src_ptr_NDHW, cs*src_sC, sxyz);
                bound::add(grad_ptr_DHW, (cs%G)*grad_sCs, src * target);
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
__global__ void conv_kernel(PushPullImpl<scalar_t,offset_t> f) {
  f.loop(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}
#endif

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    FUNCTIONAL FORM WITH DISPATCH
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define PUSHPULL_INSTANTIATE3(BoundType0, SourceType0) \
  template std::deque<Tensor> conv( \
    const SourceType0 &, const Tensor&, const Tensor&, \
    BoundType0, InterpolationType0, bool, bool, bool, bool, bool, bool); \
  template std::deque<Tensor> conv( \
    const SourceType0&, const Tensor&, \
    BoundType0, InterpolationType0, bool, bool, bool, bool, bool, bool)
#define PUSHPULL_INSTANTIATE2(BoundType0, InterpolationType0) \
  PUSHPULL_INSTANTIATE3(BoundType0, InterpolationType0, IntArrayRef); \
  PUSHPULL_INSTANTIATE3(BoundType0, InterpolationType0, Tensor)
#define PUSHPULL_INSTANTIATE1(BoundType0) \
  PUSHPULL_INSTANTIATE2(BoundType0, InterpolationType); \
  PUSHPULL_INSTANTIATE2(BoundType0, InterpolationVectorRef)
#define PUSHPULL_INSTANTIATE \
  PUSHPULL_INSTANTIATE1(BoundType); \
  PUSHPULL_INSTANTIATE1(BoundVectorRef)

#ifdef __CUDACC__

// ~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (source, grid)
// > `bound` and `interpolation` can be single arguments or vectors.
template <typename BoundType, typename InterpolationType, typename SourceType> 
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int64_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate, 
      do_pull, do_push, do_count, do_grad, do_sgrad);
    f.ioset(source, grid);
    pushpull_kernel<<<GET_BLOCKS(f.voxcount()), CUDA_NUM_THREADS, 0, 
                      at::cuda::getCurrentCUDAStream()>>>(f);
    return f.output;
  });
}

// Three arguments (source, grid, target)
// > `bound` and `interpolation` can be single arguments or vectors.
// > `source` can be a tensor or a vector of dimensions.
template <typename BoundType, typename InterpolationType, typename SourceType> 
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int64_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate,
      do_pull, do_push, do_count, do_grad, do_sgrad);
    f.ioset(source, grid, target);
    pushpull_kernel<<<GET_BLOCKS(f.voxcount()), CUDA_NUM_THREADS, 0, 
                      at::cuda::getCurrentCUDAStream()>>>(f);
    return f.output;
  });
}

#else

// ~~~ CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Two arguments (source, grid)
// > `bound` and `interpolation` can be single arguments or vectors.
template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  return AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int32_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate, 
      do_pull, do_push, do_count, do_grad, do_sgrad);
    f.ioset(source, grid);
    f.loop();
    return f.output;
  });
}

// Three arguments (source, grid, target)
// > `bound` and `interpolation` can be single arguments or vectors.
// > `source` can be a tensor or a vector of dimensions.
template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  return AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "pushpull", [&] {
    PushPullImpl<scalar_t,int32_t> 
    f(grid.dim()-2, bound, interpolation, extrapolate,
      do_pull, do_push, do_count, do_grad, do_sgrad);
    f.ioset(source, grid, target);
    f.loop();
    return f.output;
  });
}

#endif // __CUDACC__

PUSHPULL_INSTANTIATE;

} // namespace <device>

// ~~~ NOT IMPLEMENTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace notimplemented {

template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType& source, const Tensor& grid, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  throw std::logic_error("Function not implemented for this device.");
}

template <typename BoundType, typename InterpolationType, typename SourceType>
NI_HOST
std::deque<Tensor> pushpull(
  const SourceType & source, const Tensor& grid, const Tensor& target, 
  BoundType bound, InterpolationType interpolation, bool extrapolate, 
  bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad)
{
  throw std::logic_error("Function not implemented for this device.");
}

PUSHPULL_INSTANTIATE;

} // namespace notimplemented

} // namespace ni
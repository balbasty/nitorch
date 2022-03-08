#pragma once
#include "common.h"
#include "../defines.h"
#include "utils.h"
#include <cmath>

#define OnePlusTiny 1.000001;

namespace ni {
NI_NAMESPACE_DEVICE {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                Enum
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum class HessianType: uint8_t {
  None,         // No Hessian provided so nothing to do
  Eye,          // Scaled identity
  Diag,         // Diagonal matrix
  ESTATICS,     // (C-1) elements are independent conditioned on the last one
  Sym           // Symmetric matrix
};

static NI_HOST NI_INLINE
HessianType guess_hessian_type(int32_t C, int32_t CC)
{
  if (CC == 0)
    return HessianType::None;
  else if (CC == 1)
    return HessianType::Eye;
  else if (CC == C)
    return HessianType::Diag;
  else if (CC == 2*C-1)
    return HessianType::ESTATICS;
  else
    return HessianType::Sym;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             Cholesky
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace Cholesky {

  // Cholesky decomposition (Cholesky–Banachiewicz)
  //
  // @param[in]     C:  (u)int
  // @param[inout]  a:  CxC matrix
  //
  // https://en.wikipedia.org/wiki/Cholesky_decomposition
  template <typename reduce_t, typename offset_t> NI_INLINE NI_DEVICE static
  void decompose(offset_t C, reduce_t a[])
  {
    reduce_t sm, sm0;

    sm0  = 1e-40;
  #if 0
    for(offset_t c = 0; c < C; ++c) sm0 += a[c*C+c];
    sm0 *= 1e-7;
    sm0 *= sm0;
  #endif

    for (offset_t c = 0; c < C; ++c)
    {
      for (offset_t b = c; b < C; ++b)
      {
        sm = a[c*C+b];
        for(offset_t d = c-1; d >= 0; --d)
          sm -= a[c*C+d] * a[b*C+d];
        if (c == b) {
          a[c*C+c] = std::sqrt(MAX(sm, sm0));
        } else
          a[b*C+c] = sm / a[c*C+c];
      }
    }
    return;
  }

  // Cholesky solver (inplace)
  // @param[in]    C: (u)int
  // @param[in]    a: CxC matrix
  // @param[in]    p: C vector
  // @param[inout] x: C vector
  template <typename reduce_t, typename offset_t> NI_INLINE NI_DEVICE static
  void solve(offset_t C, const reduce_t a[], reduce_t x[])
  {
    reduce_t sm;
    for (offset_t c = 0; c < C; ++c)
    {
      sm = x[c];
      for (offset_t cc = c-1; cc >= 0; --cc)
        sm -= a[c*C+cc] * x[cc];
      x[c] = sm / a[c*C+c];
    }
    for(offset_t c = C-1; c >= 0; --c)
    {
      sm = x[c];
      for(offset_t cc = c+1; cc < C; ++cc)
        sm -= a[cc*C+c] * x[cc];
      x[c] = sm / a[c*C+c];
    }
  }
  
} // namespace Cholesky

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Static traits
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Function common to all Hessian types
template <typename Child>
struct HessianCommon 
{
  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE 
  void set(int32_t C, scalar_t * out, offset_t stride, const reduce_t * inp)
  {
    for (int32_t c = 0; c < C; ++c, out += stride)
     *out = inp[c];
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE 
  void add(int32_t C, scalar_t * out, offset_t stride, const reduce_t * inp)
  {
    for (int32_t c = 0; c < C; ++c, out += stride)
     *out += inp[c];
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_DEVICE 
  void invert(int32_t C, 
              scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
              reduce_t * v, const reduce_t * w)
  {
    reduce_t m[Child::max_size];
    Child::get(C, h, sh, m);
    Child::invert_(C, m, v, w);
    set(C, x, sx, v);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_DEVICE 
  void addinvert(int32_t C, 
                 scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
                 reduce_t * v, const reduce_t * w)
  {
    reduce_t m[Child::max_size];
    Child::get(C, h, sh, m);
    Child::submatvec_(C, x, sx, m, v);
    Child::invert_(C, m, v, w);
    add(C, x, sx, v);
  }
};

template <HessianType hessian_t, int32_t C>
struct HessianUtils: HessianCommon<HessianUtils<hessian_t, C> >  
{};

// aliases to make the following code less ugly
template <int32_t C> using HessianUtilsNone  = HessianUtils<HessianType::None, C>;
template <int32_t C> using HessianUtilsEye   = HessianUtils<HessianType::Eye, C>;
template <int32_t C> using HessianUtilsDiag  = HessianUtils<HessianType::Diag, C>;
template <int32_t C> using HessianUtilsEst   = HessianUtils<HessianType::ESTATICS, C>;
template <int32_t C> using HessianUtilsSym   = HessianUtils<HessianType::Sym, C>;
template <int32_t C> using HessianCommonNone = HessianCommon<HessianUtilsNone<C> >;
template <int32_t C> using HessianCommonEye  = HessianCommon<HessianUtilsEye<C> >;
template <int32_t C> using HessianCommonDiag = HessianCommon<HessianUtilsDiag<C> >;
template <int32_t C> using HessianCommonEst  = HessianCommon<HessianUtilsEst<C> >;
template <int32_t C> using HessianCommonSym  = HessianCommon<HessianUtilsSym<C> >;


template <int32_t MaxC>
struct HessianUtils<HessianType::None, MaxC>: HessianCommonNone<MaxC>
{ 
  static const int32_t max_length = MaxC;
  static const int32_t max_size   = 0; 

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  get(int32_t C, const scalar_t * inp, offset_t stride, reduce_t * out)
  {}

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  submatvec_(int32_t C, const scalar_t * inp, offset_t stride, const reduce_t * h, reduce_t * out)
  {}

  template <typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  invert_(int32_t C, reduce_t * h, reduce_t * v, const reduce_t * w) {
    for (int32_t c = 0; c < C; ++c)
      v[c] /= w[c];
  }

  // specialize parent functions to avoid defining zero-sized arrays

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_DEVICE 
  void invert(int32_t C, 
              scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
              reduce_t * v, const reduce_t * w)
  {
    reduce_t * m = static_cast<reduce_t*>(0);
    get(C, h, sh, m);
    invert_(C, m, v, w);
    HessianCommonNone<MaxC>::set(C, x, sx, v);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_DEVICE 
  void addinvert(int32_t C, 
                 scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
                 reduce_t * v, const reduce_t * w)
  {
    reduce_t * m = static_cast<reduce_t*>(0);
    get(C, h, sh, m);
    invert_(C, m, v, w);
    HessianCommonNone<MaxC>::add(C, x, sx, v);
  }
};

template <int32_t MaxC>
struct HessianUtils<HessianType::Eye, MaxC>: HessianCommonEye<MaxC>
{ 
  static const int32_t max_length = MaxC;
  static const int32_t max_size   = 1; 

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  get(int32_t C, const scalar_t * inp, offset_t stride, reduce_t * out)
  {
    *out = static_cast<reduce_t>(*inp);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  submatvec_(int32_t C, const scalar_t * inp, offset_t stride, const reduce_t * h, reduce_t * out)
  {
    reduce_t hh = *h;
    for (int32_t c = 0; c < C; ++c, inp += stride)
      out[c] -= hh * (*inp);
  }

  template <typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  invert_(int32_t C, reduce_t * h, reduce_t * v, const reduce_t * w) {
    reduce_t hh = *h;
    for (int32_t c = 0; c < C; ++c)
      v[c] /= hh + w[c];
  }
};

template <int32_t MaxC>
struct HessianUtils<HessianType::Diag, MaxC>: HessianCommonDiag<MaxC>
{ 
  static const int32_t max_length = MaxC;
  static const int32_t max_size   = MaxC; 

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  get(int32_t C, const scalar_t * inp, offset_t stride, reduce_t * out)
  {
    for (int32_t c = 0; c < C; ++c, inp += stride)
      out[c] = static_cast<reduce_t>(*inp);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  submatvec_(int32_t C, const scalar_t * inp, offset_t stride, const reduce_t * h, reduce_t * out)
  {
    for (int32_t c = 0; c < C; ++c, inp += stride)
      out[c] -= h[c] * (*inp);
  }

  template <typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  invert_(int32_t C, reduce_t * h, reduce_t * v, const reduce_t * w) {
    for (int32_t c = 0; c < C; ++c)
      v[c] /= h[c] + w[c];
  }
};

template <int32_t MaxC>
struct HessianUtils<HessianType::ESTATICS, MaxC>: HessianCommonEst<MaxC>
{ 
  static const int32_t max_length = MaxC;
  static const int32_t max_size   = 2*MaxC-1;

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  get(int32_t C, const scalar_t * inp, offset_t stride, reduce_t * out)
  {
    for (int32_t c = 0; c < 2*C-1; ++c, inp += stride)
      out[c] = static_cast<reduce_t>(*inp);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  submatvec_(int32_t C, const scalar_t * inp, offset_t stride, const reduce_t * h, reduce_t * out)
  {
    const reduce_t * o = h + C; // pointer to off-diagonal elements
    scalar_t r = inp[(C-1)*stride];
    for (int32_t c = 0; c < C-1; ++c, inp += stride) {
      out[c] -= h[c] * (*inp) + o[c] * r;
      out[C-1] -= o[c] * (*inp);
    }
    out[C-1] -= r * h[C-1];
  }

  template <typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  invert_(int32_t C, reduce_t * h, reduce_t * v, const reduce_t * w) {
    reduce_t * o = h + C;  // pointer to off-diagonal elements
    reduce_t oh = h[C-1] + w[C-1], ov = 0., tmp;
    for (int32_t c = 0; c < C-1; ++c) {
      h[c] += w[c];
      tmp = o[c] / h[c];
      oh -= o[c] * tmp;
      ov += v[c] * tmp;
    }
    oh = 1. / oh; // oh = 1/mini_inv, ov = sum(vec_norm * grad)
    //tmp = v[C-1];
    //for (int32_t c = 0; c < C-1; ++c)
    //  v[c] = (o[c] * (ov - tmp) + v[c]) / (oh * h[c]);
    //v[C-1] = (tmp - ov) / oh;
    v[C-1] = tmp = (v[C-1] - ov) * oh;
    for (int32_t c = 0; c < C-1; ++c)
      v[c] = (v[c] - tmp * o[c]) / h[c];
  }
};

template <int32_t MaxC>
struct HessianUtils<HessianType::Sym, MaxC>: HessianCommonSym<MaxC>
{ 
  static const int32_t max_length = MaxC;
  static const int32_t max_size   = MaxC*MaxC; //< How much we allocate on the stack!

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  get(int32_t C, const scalar_t * inp, offset_t stride, reduce_t * out)
  {
    for (int32_t c = 0; c < C; ++c, inp += stride)
      out[c+C*c] = (*inp) * OnePlusTiny;
    for (int32_t c = 0; c < C; ++c)
      for (int32_t cc = c+1; cc < C; ++cc, inp += stride)
        out[c+C*cc] = out[cc+C*c] = *inp;
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static NI_INLINE NI_DEVICE void 
  submatvec_(int32_t C, const scalar_t * inp, offset_t stride, const reduce_t * h, reduce_t * out)
  {
    reduce_t acc;
    for (int32_t c = 0; c < C; ++c) {
      acc = static_cast<reduce_t>(0);
      for (int32_t cc = 0; cc < C; ++cc)
        acc += h[c*C+cc] * inp[cc*stride];
      out[c] -= acc;
    }
  }

  template <typename reduce_t> 
  static NI_INLINE NI_DEVICE void invert_(
    int32_t C, reduce_t * h, reduce_t * v, const reduce_t * w) 
  {
    for (int32_t c = 0; c < C; ++c)
      h[c+C*c] += w[c];
    Cholesky::decompose(C, h);  // cholesky decomposition
    Cholesky::solve(C, h, v);   // solve linear system inplace
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Dispatcher
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define NI_CASE_ENUM_USING_HINT(value, HINT, ...)             \
  case value: {                                               \
    static const auto HINT = value;                           \
    return __VA_ARGS__();                                     \
  }

#define NI_CASE_HESSIAN(type, ...) \
  NI_CASE_ENUM_USING_HINT(type, hessian_t, __VA_ARGS__)

#define NI_DFLT_ENUM_USING_HINT(value, HINT, ...)             \
  default: {                                                  \
    static const auto HINT = value;                           \
    return __VA_ARGS__();                                     \
  }

#define NI_DFLT_HESSIAN(type, ...) \
  NI_DFLT_ENUM_USING_HINT(type, hessian_t, __VA_ARGS__)

#define NI_DISPATCH_HESSIAN_TYPE(TYPE, ...)                                 \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    switch (the_type) {                                                     \
      NI_CASE_HESSIAN(HessianType::None,     __VA_ARGS__)                   \
      NI_CASE_HESSIAN(HessianType::Eye,      __VA_ARGS__)                   \
      NI_CASE_HESSIAN(HessianType::Diag,     __VA_ARGS__)                   \
      NI_CASE_HESSIAN(HessianType::ESTATICS, __VA_ARGS__)                   \
      NI_CASE_HESSIAN(HessianType::Sym,      __VA_ARGS__)                   \
    }                                                                       \
  }()

// If 1 channel (logC = 0) -> only None and Eye needed
#define NI_DISPATCH_HESSIAN_TYPE0(TYPE, ...)                                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    switch (the_type) {                                                     \
      NI_CASE_HESSIAN(HessianType::None,     __VA_ARGS__)                   \
      NI_DFLT_HESSIAN(HessianType::Eye,      __VA_ARGS__)                   \
    }                                                                       \
  }()

// If 2 channels (logC = 1) -> only None, Eye and ESTATICS needed
#define NI_DISPATCH_HESSIAN_TYPE1(TYPE, ...)                                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    switch (the_type) {                                                     \
      NI_CASE_HESSIAN(HessianType::None,     __VA_ARGS__)                   \
      NI_CASE_HESSIAN(HessianType::Eye,      __VA_ARGS__)                   \
      NI_DFLT_HESSIAN(HessianType::ESTATICS, __VA_ARGS__)                   \
    }                                                                       \
  }()


} // namespace device
} // namespace ni

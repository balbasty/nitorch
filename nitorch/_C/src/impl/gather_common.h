#pragma once

// Separable sparse "gather" operators.
//
// A gather operator maps an input tensor (N, C, *Ispatial) to an output
// tensor (N, C, *Ospatial) where each output voxel is a weighted sum of
// input voxels and the weights are a separable (Kronecker) product of
// one 1D sparse operator per spatial dimension:
//
//   out[n, c, jx, jy, jz] = sum_{kz, ky, kx} wz[kz] * wy[ky] * wx[kx]
//                           * inp[n, c, ox[kx] + oy[ky] + oz[kz]]
//
// Row j of the 1D operator of dimension d holds up to K entries
// (offset of the input voxel [already multiplied by its stride], weight
// [already multiplied by the boundary sign]) stored at [j*K, j*K+len[j]).
// The kernel writes each output voxel exactly once (no atomics), and is
// templated on K (K > 0: compile-time row length, K == 0: runtime).
//
// Tables are built on the host (see `fold_taps` / `flatten_gather_rows`)
// and, on CUDA, copied to the device (`upload_gather_tables`).
//
// Used by resize_common.cpp (adjoint of resize, i.e., "restrict").
// multires_common.cpp (FMG restriction, always a factor of two) uses a
// fixed 4-tap stencil instead and only borrows NI_PRAGMA_UNROLL and
// gather_env_flag from this file.

#include "common.h"
#include "bounds_common.h"
#include <ATen/ATen.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#ifdef __CUDACC__
# define NI_PRAGMA_UNROLL _Pragma("unroll")
#else
# define NI_PRAGMA_UNROLL
#endif

// Expand `CASE(k)` for the statically unrolled row lengths (1..8), and
// `CASE(0)` (runtime row length) otherwise.
#define NI_GATHER_K_SWITCH(K, CASE) \
  switch (K) {                      \
    case 1: CASE(1)                 \
    case 2: CASE(2)                 \
    case 3: CASE(3)                 \
    case 4: CASE(4)                 \
    case 5: CASE(5)                 \
    case 6: CASE(6)                 \
    case 7: CASE(7)                 \
    case 8: CASE(8)                 \
    default: CASE(0)                \
  }

namespace ni {
NI_NAMESPACE_DEVICE { // cpu / cuda / ...
namespace {           // internal linkage

// Type of the tabulated weights / accumulator of the gather kernel.
// (float for float/half, double for double)
template <typename scalar_t> struct gather_weight { typedef float type; };
template <> struct gather_weight<double> { typedef double type; };

// True if the environment variable `name` is set to a non-empty value
// other than "0".
static NI_HOST NI_INLINE bool gather_env_flag(const char * name)
{
  const char * env = std::getenv(name);
  return env && *env && std::strcmp(env, "0") != 0;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                           HOST SIDE
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// One row of a 1D sparse operator: (unstrided index, weight)
template <typename weight_t>
using GatherRow = std::vector<std::pair<int64_t, weight_t> >;

// Fold the unfolded indices `b` (size `n`) into [0, size) with the
// boundary condition `bound`, apply the boundary sign to the weights `wt`,
// and merge taps that land on the same voxel (order of first occurrence
// is kept). Taps with a null sign, folded outside [0, size) (defensive)
// or with a null merged weight are dropped.
// Outputs are written in `jj` / `vv` (capacity >= n). Returns their count.
template <typename weight_t>
static NI_HOST int fold_taps(BoundType bound, int64_t size, int n,
                             const int64_t * b, const weight_t * wt,
                             int64_t * jj, weight_t * vv)
{
  int m = 0;
  for (int k = 0; k < n; ++k) {
    int8_t s = bound::sign(bound, b[k], size);
    if (!s) continue;
    int64_t j = bound::index(bound, b[k], size);
    if (j < 0 || j >= size) continue;
    weight_t v = s < 0 ? -wt[k] : wt[k];
    int l = 0;
    for (; l < m; ++l) if (jj[l] == j) break;
    if (l < m) vv[l] += v;
    else { jj[m] = j; vv[m] = v; ++m; }
  }
  int m2 = 0;
  for (int l = 0; l < m; ++l)
    if (vv[l] != static_cast<weight_t>(0)) {
      jj[m2] = jj[l]; vv[m2] = vv[l]; ++m2;
    }
  return m2;
}

// Flattened tables (host memory), rows padded to a common length K.
template <typename weight_t>
struct GatherTables {
  int K = 1;
  int64_t S[3] = {1, 1, 1};           // number of rows (output size)
  std::vector<int32_t>  len[3];       // [S]
  std::vector<int64_t>  off[3];       // [S*K] input offsets
  std::vector<weight_t> wgt[3];       // [S*K]
};

// Flatten sparse rows. `rows[d]` holds one row per output voxel of
// dimension d < dim; dimensions >= dim are set to the identity.
// Indices are multiplied by the input strides `stride`.
template <typename weight_t>
static NI_HOST GatherTables<weight_t> flatten_gather_rows(
  std::vector<GatherRow<weight_t> > rows[3], int dim, const int64_t stride[3])
{
  GatherTables<weight_t> tab;
  int K = 1;
  for (int d = 0; d < 3; ++d) {
    if (d >= dim)
      rows[d].assign(1, GatherRow<weight_t>(
                          1, std::make_pair(int64_t(0), weight_t(1))));
    tab.S[d] = static_cast<int64_t>(rows[d].size());
    for (const auto & row : rows[d])
      K = std::max<int>(K, static_cast<int>(row.size()));
  }
  tab.K = K;
  for (int d = 0; d < 3; ++d) {
    const int64_t S = tab.S[d];
    const int64_t st = d < dim ? stride[d] : 0;
    tab.len[d].assign(S, 0);
    tab.off[d].assign(S * K, 0);
    tab.wgt[d].assign(S * K, static_cast<weight_t>(0));
    for (int64_t j = 0; j < S; ++j) {
      const auto & row = rows[d][j];
      tab.len[d][j] = static_cast<int32_t>(row.size());
      for (size_t k = 0; k < row.size(); ++k) {
        tab.off[d][j * K + k] = row[k].first * st;
        tab.wgt[d][j * K + k] = row[k].second;
      }
    }
  }
  return tab;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    POINTERS (HOST OR DEVICE MEMORY)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Plain struct of pointers, copied by value into the Impl structs (and
// therefore to the device). Pointers must address memory accessible by
// the device that runs the kernel.
template <typename offset_t, typename weight_t>
struct GatherTablePtrs {
  int              K;
  const int32_t  * len[3];
  const offset_t * off[3];
  const weight_t * wgt[3];
};

template <typename offset_t, typename weight_t>
static NI_HOST GatherTablePtrs<offset_t, weight_t> null_gather_ptrs()
{
  GatherTablePtrs<offset_t, weight_t> ptrs;
  ptrs.K = 1;
  for (int d = 0; d < 3; ++d) {
    ptrs.len[d] = nullptr;
    ptrs.off[d] = nullptr;
    ptrs.wgt[d] = nullptr;
  }
  return ptrs;
}

// Point at the host tables (CPU, 64 bit offsets).
template <typename weight_t>
static NI_HOST GatherTablePtrs<int64_t, weight_t>
host_gather_ptrs(const GatherTables<weight_t> & tab)
{
  GatherTablePtrs<int64_t, weight_t> ptrs;
  ptrs.K = tab.K;
  for (int d = 0; d < 3; ++d) {
    ptrs.len[d] = tab.len[d].data();
    ptrs.off[d] = tab.off[d].data();
    ptrs.wgt[d] = tab.wgt[d].data();
  }
  return ptrs;
}

#ifdef __CUDACC__
// Copy the host tables to `device` (offsets converted to offset_t).
// The device tensors are appended to `keep`, which must outlive the kernel.
template <typename offset_t, typename weight_t>
static NI_HOST GatherTablePtrs<offset_t, weight_t>
upload_gather_tables(GatherTables<weight_t> & tab, at::Device device,
                     std::vector<at::Tensor> & keep)
{
  auto opt = at::TensorOptions().device(device);
  auto off_type = std::is_same<offset_t, int32_t>::value ? at::kInt : at::kLong;
  auto wgt_type = std::is_same<weight_t, double>::value ? at::kDouble : at::kFloat;
  GatherTablePtrs<offset_t, weight_t> ptrs;
  ptrs.K = tab.K;
  for (int d = 0; d < 3; ++d) {
    int64_t nlen = static_cast<int64_t>(tab.len[d].size());
    int64_t ntab = static_cast<int64_t>(tab.off[d].size());
    // from_blob does not own the host memory; .to() makes a (blocking)
    // device copy.
    at::Tensor tlen = at::from_blob(tab.len[d].data(), {nlen}, at::kInt)
                        .to(opt.dtype(at::kInt));
    at::Tensor toff = at::from_blob(tab.off[d].data(), {ntab}, at::kLong)
                        .to(opt.dtype(off_type));
    at::Tensor twgt = at::from_blob(tab.wgt[d].data(), {ntab}, wgt_type)
                        .to(opt.dtype(wgt_type));
    ptrs.len[d] = static_cast<const int32_t  *>(tlen.data_ptr());
    ptrs.off[d] = static_cast<const offset_t *>(toff.data_ptr());
    ptrs.wgt[d] = static_cast<const weight_t *>(twgt.data_ptr());
    keep.push_back(tlen);
    keep.push_back(toff);
    keep.push_back(twgt);
  }
  return ptrs;
}
#endif

// CPU grain size: each output voxel costs ~K^dim multiply-adds/channel
static NI_HOST NI_INLINE int64_t gather_grain(int64_t grain, int K, int dim)
{
  int64_t work = 1;
  for (int d = 0; d < dim; ++d) work *= K;
  return std::max<int64_t>(1, grain / std::max<int64_t>(1, work));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                              KERNEL
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compute all channels of output voxel (jx, jy, jz).
// `inp` points to inp[n, 0], `out` points to out[n, 0, jx, jy, jz].
// K > 0: compile-time row length (must equal tab.K). K == 0: tab.K.
template <int K, typename scalar_t, typename offset_t, typename weight_t>
static NI_INLINE NI_DEVICE void gather_separable(
  const GatherTablePtrs<offset_t, weight_t> & tab,
  offset_t jx, offset_t jy, offset_t jz, offset_t C,
  const scalar_t * inp, offset_t inp_sC,
  scalar_t * out, offset_t out_sC)
{
  const int KK = K > 0 ? K : tab.K;      // row stride == row capacity
  const int lx = tab.len[0][jx];
  const int ly = tab.len[1][jy];
  const int lz = tab.len[2][jz];
  const offset_t * ox = tab.off[0] + jx * KK;
  const offset_t * oy = tab.off[1] + jy * KK;
  const offset_t * oz = tab.off[2] + jz * KK;
  const weight_t * wx = tab.wgt[0] + jx * KK;
  const weight_t * wy = tab.wgt[1] + jy * KK;
  const weight_t * wz = tab.wgt[2] + jz * KK;

  for (offset_t c = 0; c < C; ++c, inp += inp_sC, out += out_sC) {
    weight_t acc = static_cast<weight_t>(0);
    NI_PRAGMA_UNROLL
    for (int kz = 0; kz < KK; ++kz) {
      if (kz >= lz) break;
      const scalar_t * inp_z = inp + oz[kz];
      weight_t accz = static_cast<weight_t>(0);
      NI_PRAGMA_UNROLL
      for (int ky = 0; ky < KK; ++ky) {
        if (ky >= ly) break;
        const scalar_t * inp_yz = inp_z + oy[ky];
        weight_t accy = static_cast<weight_t>(0);
        NI_PRAGMA_UNROLL
        for (int kx = 0; kx < KK; ++kx) {
          if (kx >= lx) break;
          accy += static_cast<weight_t>(inp_yz[ox[kx]]) * wx[kx];
        }
        accz += accy * wy[ky];
      }
      acc += accz * wz[kz];
    }
    *out = static_cast<scalar_t>(acc);
  }
}

} // namespace (anonymous)
} // namespace cpu / cuda
} // namespace ni

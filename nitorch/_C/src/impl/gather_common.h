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
#include <limits>

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
//
// All tables are packed into a single pinned host buffer, which is copied
// with a single non-blocking copy on the current stream (the stream the
// kernel is launched on), and the device pointers are carved out of the
// device copy. Both buffers are appended to `keep`, which must outlive
// the kernel launch (the caching allocators are stream-ordered).
template <typename offset_t, typename weight_t>
static NI_HOST GatherTablePtrs<offset_t, weight_t>
upload_gather_tables(const GatherTables<weight_t> & tab, at::Device device,
                     std::vector<at::Tensor> & keep)
{
  // byte layout: for each dimension, [len | off | wgt], 16-byte aligned
  auto align = [](int64_t nbytes) { return (nbytes + 15) / 16 * 16; };
  int64_t pos[3][3];
  int64_t total = 0;
  for (int d = 0; d < 3; ++d) {
    int64_t nlen = static_cast<int64_t>(tab.len[d].size());
    int64_t ntab = static_cast<int64_t>(tab.off[d].size());
    pos[d][0] = total; total += align(nlen * sizeof(int32_t));
    pos[d][1] = total; total += align(ntab * sizeof(offset_t));
    pos[d][2] = total; total += align(ntab * sizeof(weight_t));
  }

  at::Tensor host = at::empty({total}, at::TensorOptions()
                                         .dtype(at::kByte)
                                         .pinned_memory(true));
  uint8_t * hptr = host.data_ptr<uint8_t>();
  for (int d = 0; d < 3; ++d) {
    std::memcpy(hptr + pos[d][0], tab.len[d].data(),
                tab.len[d].size() * sizeof(int32_t));
    offset_t * hoff = reinterpret_cast<offset_t *>(hptr + pos[d][1]);
    for (size_t k = 0; k < tab.off[d].size(); ++k) {
      const int64_t o = tab.off[d][k];
      TORCH_INTERNAL_ASSERT(
        o >= static_cast<int64_t>(std::numeric_limits<offset_t>::min()) &&
        o <= static_cast<int64_t>(std::numeric_limits<offset_t>::max()),
        "gather table offset ", o, " does not fit in the index type");
      hoff[k] = static_cast<offset_t>(o);
    }
    std::memcpy(hptr + pos[d][2], tab.wgt[d].data(),
                tab.wgt[d].size() * sizeof(weight_t));
  }

  at::Tensor dev = host.to(at::TensorOptions().device(device),
                           /*non_blocking=*/true);
  keep.push_back(host);
  keep.push_back(dev);

  const uint8_t * dptr = static_cast<const uint8_t *>(dev.data_ptr());
  GatherTablePtrs<offset_t, weight_t> ptrs;
  ptrs.K = tab.K;
  for (int d = 0; d < 3; ++d) {
    ptrs.len[d] = reinterpret_cast<const int32_t  *>(dptr + pos[d][0]);
    ptrs.off[d] = reinterpret_cast<const offset_t *>(dptr + pos[d][1]);
    ptrs.wgt[d] = reinterpret_cast<const weight_t *>(dptr + pos[d][2]);
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
  // row bases in 64 bit (S*K may exceed the range of a 32 bit offset_t)
  const int64_t bx = static_cast<int64_t>(jx) * KK;
  const int64_t by = static_cast<int64_t>(jy) * KK;
  const int64_t bz = static_cast<int64_t>(jz) * KK;
  const offset_t * ox = tab.off[0] + bx;
  const offset_t * oy = tab.off[1] + by;
  const offset_t * oz = tab.off[2] + bz;
  const weight_t * wx = tab.wgt[0] + bx;
  const weight_t * wy = tab.wgt[1] + by;
  const weight_t * wz = tab.wgt[2] + bz;

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

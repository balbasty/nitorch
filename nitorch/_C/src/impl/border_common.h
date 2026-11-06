#pragma once

// Border correction of the quadratic regularisers (definition "F")
//
// Every energy is a sum of rows centred on the voxels of the field of
// view, built from the bound-extended field (no ghost rows, no row
// weights): A = E' D' F' W F D E. The stencils of the kernels apply
// L = D' W~ D to the extended field and read it at the voxels:
// S = R D' W~ D E. They sum all the rows of the extended lattice within
// their radius, so that A = S + B, where B
// - folds the taps of the kept rows that lie outside of the field of
//   view back onto their source voxel (with the sign of the extension);
// - removes the rows centred outside of the field of view (ghost rows).
// B only lives on the border voxels (`BorderBox`): the kernels run on
// the interior voxels unchanged, and the border voxels add B (gather
// form: each voxel computes its own row of B).
//
// The closed forms (and their dense check) are in
// nitorch/tests/_regulariser_reference.py (`border_correction`).
// Per axis `a` of length `n`, with the source voxel `p` and sign `s` of
// the extended sample `-1` (and `n` on the right):
// - membrane:  d = x[0] - x~[-1];  out[0] -= w~[-1] d / 2;
//              out[p] -= s w[0] d / 2
// - bending (central second differences):
//              out[0] -= x~[-2] - 2 x~[-1] + x[0];
//              out[p] += s (x~[-1] - 2 x[0] + x~[1])
// - bending (cross terms of the pair (a, b)): edges of axis a, where L_b
//   is the Neumann Laplacian along b of the plane d = x[0] - x~[-1]:
//              out[0] -= L_b d;  out[p] -= s L_b d
//   and corners (once per pair), with g = x[0,0] - x~[0,-1] - x~[-1,0]
//   + x~[-1,-1]: out[0,0] -= 3g/2, out[0,q] -= t g/2,
//   out[p,0] -= s g/2, out[p,q] += s t g/2
// - linear-elastic cross term (output component i, input component j,
//   scatter axis a, difference axis b), with g = C_b u_j (central first
//   difference, with the boundary condition of u_j):
//              out_i[0] -= s_j g(p_j) / 2;  out_i[p_i] -= s_i g(0) / 2
// The right border is the mirror image (and the sign of the
// linear-elastic terms flips).

#include "common.h"
#include "bounds_common.h"

namespace ni {
namespace border {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    Interior box and border slabs
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The interior box is [lo, hi) along each axis, with lo = min(r, X) and
// hi = max(X - r, lo) for a border of thickness r. The border voxels are
// enumerated as six slabs:
//   x < lo0                           (all y, z)
//   x >= hi0                          (all y, z)
//   lo0 <= x < hi0, y < lo1           (all z)
//   lo0 <= x < hi0, y >= hi1          (all z)
//   lo0 <= x < hi0, lo1 <= y < hi1, z < lo2
//   lo0 <= x < hi0, lo1 <= y < hi1, z >= hi2
template <typename offset_t>
struct BorderBox {
  offset_t X, Y, Z, lo0, lo1, lo2, hi0, hi1, hi2;

  NI_HOST NI_DEVICE void set(offset_t X_, offset_t Y_, offset_t Z_,
                             offset_t r0, offset_t r1, offset_t r2) {
    X = X_; Y = Y_; Z = Z_;
    lo0 = r0 < X ? r0 : X;  hi0 = X - r0 > lo0 ? X - r0 : lo0;
    lo1 = r1 < Y ? r1 : Y;  hi1 = Y - r1 > lo1 ? Y - r1 : lo1;
    lo2 = r2 < Z ? r2 : Z;  hi2 = Z - r2 > lo2 ? Z - r2 : lo2;
  }

  NI_HOST NI_DEVICE offset_t interior_count() const {
    return (hi0 - lo0) * (hi1 - lo1) * (hi2 - lo2);
  }

  NI_HOST NI_DEVICE offset_t border_count() const {
    return X * Y * Z - interior_count();
  }

  // linear index -> interior voxel
  NI_HOST NI_DEVICE void decode_interior(offset_t i, offset_t & x,
                                         offset_t & y, offset_t & z) const {
    offset_t NY = hi1 - lo1, NZ = hi2 - lo2;
    z = lo2 + i % NZ;  i /= NZ;
    y = lo1 + i % NY;  i /= NY;
    x = lo0 + i;
  }

  // linear index -> border voxel
  NI_HOST NI_DEVICE void decode(offset_t i, offset_t & x,
                                offset_t & y, offset_t & z) const {
    offset_t YZ = Y * Z, n;
    // x slabs
    n = lo0 * YZ;
    if (i < n) { x = i / YZ; y = (i / Z) % Y; z = i % Z; return; }
    i -= n;
    n = (X - hi0) * YZ;
    if (i < n) { x = hi0 + i / YZ; y = (i / Z) % Y; z = i % Z; return; }
    i -= n;
    // y slabs
    offset_t NX = hi0 - lo0;
    n = NX * lo1 * Z;
    if (i < n) {
      z = i % Z; i /= Z; y = i % lo1; x = lo0 + i / lo1; return;
    }
    i -= n;
    offset_t NYh = Y - hi1;
    n = NX * NYh * Z;
    if (i < n) {
      z = i % Z; i /= Z; y = hi1 + i % NYh; x = lo0 + i / NYh; return;
    }
    i -= n;
    // z slabs
    offset_t NY = hi1 - lo1;
    n = NX * NY * lo2;
    if (i < n) {
      z = i % lo2; i /= lo2; y = lo1 + i % NY; x = lo0 + i / NY; return;
    }
    i -= n;
    offset_t NZh = Z - hi2;
    z = hi2 + i % NZh; i /= NZh; y = lo1 + i % NY; x = lo0 + i / NY;
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                       Which terms are not zero
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// (B vanishes for these boundary conditions; see the reference)

static NI_INLINE NI_HOST NI_DEVICE bool
membrane_needs(BoundType b) {
  return !(b == BoundType::DFT || b == BoundType::DCT2 ||
           b == BoundType::DST2 || b == BoundType::Replicate);
}

static NI_INLINE NI_HOST NI_DEVICE bool
bending_diag_needs(BoundType b) {
  return !(b == BoundType::DFT || b == BoundType::DCT2 ||
           b == BoundType::DST2 || b == BoundType::DST1);
}

static NI_INLINE NI_HOST NI_DEVICE bool
bending_edge_needs(BoundType b) {
  return membrane_needs(b);
}

static NI_INLINE NI_HOST NI_DEVICE bool
bending_corner_needs(BoundType a, BoundType b) {
  if (a == BoundType::DCT2 || a == BoundType::Replicate ||
      b == BoundType::DCT2 || b == BoundType::Replicate)
    return false;
  if (a == b && (a == BoundType::DST2 || a == BoundType::DFT))
    return false;
  return true;
}

static NI_INLINE NI_HOST NI_DEVICE bool
lame_cross_needs(BoundType bi, BoundType bj) {
  bool zi = (bi == BoundType::DST1 || bi == BoundType::Zero);
  bool zj = (bj == BoundType::DST1 || bj == BoundType::Zero);
  if (zi && zj) return false;
  if (bi == BoundType::DFT && bj == BoundType::DFT) return false;
  if ((bi == BoundType::DCT2 && bj == BoundType::DST2) ||
      (bi == BoundType::DST2 && bj == BoundType::DCT2)) return false;
  return true;
}

// Boundary condition of component `comp` along `axis` (sliding axes:
// the normal component is antisymmetric, the others symmetric)
static NI_INLINE NI_HOST NI_DEVICE BoundType
component_bound(BoundType b, int axis, int comp) {
  if (b != BoundType::Sliding) return b;
  return axis == comp ? BoundType::DST2 : BoundType::DCT2;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    A channel seen from a voxel
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename reduce_t, typename scalar_t, typename offset_t>
struct View {
  const scalar_t * ptr;   // pointer to the voxel q
  offset_t q[3];          // coordinates of q
  offset_t n[3];          // size of each axis
  offset_t s[3];          // stride of each axis
  BoundType b[3];         // boundary condition of each axis

  NI_DEVICE View(const scalar_t * ptr,
                 offset_t x, offset_t y, offset_t z,
                 offset_t X, offset_t Y, offset_t Z,
                 offset_t sX, offset_t sY, offset_t sZ,
                 BoundType b0, BoundType b1, BoundType b2): ptr(ptr) {
    q[0] = x;  q[1] = y;  q[2] = z;
    n[0] = X;  n[1] = Y;  n[2] = Z;
    s[0] = sX; s[1] = sY; s[2] = sZ;
    b[0] = b0; b[1] = b1; b[2] = b2;
  }

  // source voxel `p` and sign of the extended sample `j` along `a`
  NI_DEVICE int8_t src(int a, offset_t j, offset_t & p) const {
    p = static_cast<offset_t>(bound::index(b[a], j, n[a]));
    return bound::sign(b[a], j, n[a]);
  }

  // whether the extended sample `j` along `a` is folded onto q
  // (returns its sign, 0 if it is not)
  NI_DEVICE int8_t folds(int a, offset_t j) const {
    offset_t p;
    int8_t sgn = src(a, j, p);
    return (sgn && p == q[a]) ? sgn : static_cast<int8_t>(0);
  }

  // extended sample `j` along `a` (the other coordinates are q's)
  NI_DEVICE reduce_t at(int a, offset_t j) const {
    offset_t p;
    int8_t sgn = src(a, j, p);
    if (!sgn) return static_cast<reduce_t>(0);
    reduce_t v = static_cast<reduce_t>(ptr[(p - q[a]) * s[a]]);
    return sgn > 0 ? v : -v;
  }

  // extended sample (ja, jb) along (a, b)
  NI_DEVICE reduce_t at(int a, offset_t ja, int b2, offset_t jb) const {
    offset_t pa, pb;
    int8_t sa = src(a, ja, pa);
    if (!sa) return static_cast<reduce_t>(0);
    int8_t sb = src(b2, jb, pb);
    if (!sb) return static_cast<reduce_t>(0);
    reduce_t v = static_cast<reduce_t>(
      ptr[(pa - q[a]) * s[a] + (pb - q[b2]) * s[b2]]);
    return sa * sb > 0 ? v : -v;
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    Row of B at the voxel q (gather form)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// (unit coefficients; `side` 0 is the left border, 1 the right one)

#define NI_BORDER_SIDE(V, a)                                     \
  offset_t n_ = V.n[a];                                          \
  offset_t j0 = side ? n_ - 1 : static_cast<offset_t>(0);        \
  offset_t jg = side ? n_     : static_cast<offset_t>(-1);       \
  bool ghost = V.q[a] == j0;                                     \
  int8_t fold = V.folds(a, jg);

// Membrane along `a` (W: weights, with their even boundary condition,
// or nullptr)
template <typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
membrane(const View<reduce_t, scalar_t, offset_t> & V,
         const View<reduce_t, scalar_t, offset_t> * W, int a) {
  reduce_t r = 0;
  for (int side = 0; side < 2; ++side) {
    NI_BORDER_SIDE(V, a)
    if (!ghost && !fold) continue;
    reduce_t d = V.at(a, j0) - V.at(a, jg);
    if (ghost) r -= 0.5 * (W ? W->at(a, jg) : static_cast<reduce_t>(1)) * d;
    if (fold)  r -= 0.5 * fold * (W ? W->at(a, j0) : static_cast<reduce_t>(1)) * d;
  }
  return r;
}

// Bending: central second differences along `a`
template <typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
bending_diag(const View<reduce_t, scalar_t, offset_t> & V, int a) {
  reduce_t r = 0;
  for (int side = 0; side < 2; ++side) {
    NI_BORDER_SIDE(V, a)
    if (!ghost && !fold) continue;
    offset_t j1  = side ? n_ - 2 : static_cast<offset_t>(1);
    offset_t jg2 = side ? n_ + 1 : static_cast<offset_t>(-2);
    reduce_t xg = V.at(a, jg), x0 = V.at(a, j0);
    if (ghost) r -= V.at(a, jg2) - 2 * xg + x0;
    if (fold)  r += fold * (xg - 2 * x0 + V.at(a, j1));
  }
  return r;
}

// Bending: cross term of the pair (a, b), edges of axis `a`
template <typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
bending_edge(const View<reduce_t, scalar_t, offset_t> & V, int a, int b) {
  reduce_t r = 0;
  offset_t qb = V.q[b], nb = V.n[b];
  for (int side = 0; side < 2; ++side) {
    NI_BORDER_SIDE(V, a)
    if (!ghost && !fold) continue;
    // Neumann Laplacian along b of d = x[j0] - x~[jg]
    reduce_t d = V.at(a, j0, b, qb) - V.at(a, jg, b, qb), Ld = 0;
    if (qb > 0)
      Ld += d - (V.at(a, j0, b, qb - 1) - V.at(a, jg, b, qb - 1));
    if (qb < nb - 1)
      Ld += d - (V.at(a, j0, b, qb + 1) - V.at(a, jg, b, qb + 1));
    if (ghost) r -= Ld;
    if (fold)  r -= fold * Ld;
  }
  return r;
}

// Bending: cross term of the pair (a, b), corners (once per pair)
template <typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
bending_corner(const View<reduce_t, scalar_t, offset_t> & V, int a, int b) {
  reduce_t r = 0;
  for (int sa = 0; sa < 2; ++sa) {
    offset_t ja = sa ? V.n[a] - 1 : static_cast<offset_t>(0);
    offset_t ga = sa ? V.n[a]     : static_cast<offset_t>(-1);
    bool    a0 = V.q[a] == ja;
    int8_t  ap = V.folds(a, ga);
    if (!a0 && !ap) continue;
    for (int sb = 0; sb < 2; ++sb) {
      offset_t jb = sb ? V.n[b] - 1 : static_cast<offset_t>(0);
      offset_t gb = sb ? V.n[b]     : static_cast<offset_t>(-1);
      bool    b0 = V.q[b] == jb;
      int8_t  bp = V.folds(b, gb);
      reduce_t acc = 0;
      if (a0 && b0) acc -= 1.5;
      if (a0 && bp) acc -= 0.5 * bp;
      if (ap && b0) acc -= 0.5 * ap;
      if (ap && bp) acc += 0.5 * ap * bp;
      if (acc == 0) continue;
      reduce_t g = V.at(a, ja, b, jb) - V.at(a, ja, b, gb)
                 - V.at(a, ga, b, jb) + V.at(a, ga, b, gb);
      r += acc * g;
    }
  }
  return r;
}

// Linear-elastic cross term E_i' C_a' F'F C_b E_j u_j at the voxel q of
// the component i (Vi: bounds of i at q; Vj: the component j at q)
template <typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
lame_cross(const View<reduce_t, scalar_t, offset_t> & Vi,
           const View<reduce_t, scalar_t, offset_t> & Vj, int a, int b) {
  reduce_t r = 0;
  offset_t qb = Vj.q[b];
  for (int side = 0; side < 2; ++side) {
    NI_BORDER_SIDE(Vi, a)
    reduce_t sgn = side ? 0.25 : -0.25;
    if (ghost)
      r += sgn * (Vj.at(a, jg, b, qb + 1) - Vj.at(a, jg, b, qb - 1));
    if (fold)
      r += sgn * fold * (Vj.at(a, j0, b, qb + 1) - Vj.at(a, j0, b, qb - 1));
  }
  return r;
}

#undef NI_BORDER_SIDE

} // namespace border
} // namespace ni

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
// B only lives on a few planes of each axis (`Targets`): the ghost rows
// target the planes 0 and n-1, the folds the source planes of the ghost
// samples. The regularisers run their kernels on all voxels unchanged,
// then a thin pass over the target planes adds B (gather form: each
// voxel of a plane computes its own row of the terms of that axis, from
// tables of offsets and signs computed once: `Axis`, `Plane`).
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
//                 Extended samples of an axis (tables)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Source voxel `p` and sign `s` of the samples that the border terms
// read along one axis of length n, for one boundary condition (computed
// once, on the host). Sample k of side 0 (left) / 1 (right):
//   k = 0: border voxel j0  (0  | n-1)
//   k = 1: ghost sample jg  (-1 | n)
//   k = 2: inward voxel j1  (1  | n-2)
//   k = 3: second ghost jg2 (-2 | n+1)
template <typename offset_t>
struct Axis {
  offset_t p[2][4];
  int8_t   s[2][4];

  NI_HOST NI_DEVICE void set(BoundType b, offset_t n) {
    for (int side = 0; side < 2; ++side)
    for (int k = 0; k < 4; ++k) {
      offset_t d = static_cast<offset_t>(k == 0 ? 0 : k == 1 ? -1 : k == 2 ? 1 : -2);
      offset_t j = side ? n - 1 - d : d;
      s[side][k] = bound::sign(b, j, n);
      // (a zero sample is never read: any index will do)
      p[side][k] = s[side][k] ? static_cast<offset_t>(bound::index(b, j, n))
                              : static_cast<offset_t>(0);
    }
  }
};

// A plane j of an axis, seen from its voxels: which border terms target
// it, and the offsets (from the plane) and signs of their samples.
template <typename offset_t>
struct Plane {
  bool     ghost[2];    // j is the border voxel of the side (ghost row)
  int8_t   fold[2];     // sign of the ghost sample folded onto j (or 0)
  int8_t   s[2][4];     // sign of the samples
  bool     self[2][4];  // whether the sample is the plane j itself
  offset_t o[2][4];     // offset of the samples from the plane j

  NI_HOST NI_DEVICE void set(const Axis<offset_t> & A, offset_t n,
                             offset_t j, offset_t stride) {
    for (int side = 0; side < 2; ++side) {
      ghost[side] = j == (side ? n - 1 : static_cast<offset_t>(0));
      fold[side]  = (A.s[side][1] && A.p[side][1] == j) ? A.s[side][1]
                                                         : static_cast<int8_t>(0);
      for (int k = 0; k < 4; ++k) {
        s[side][k]    = A.s[side][k];
        self[side][k] = A.s[side][k] && A.p[side][k] == j;
        o[side][k]    = (A.p[side][k] - j) * stride;
      }
    }
  }

  NI_HOST NI_DEVICE bool targeted() const {
    return ghost[0] || ghost[1] || fold[0] || fold[1];
  }
};

// Sample k of a side, at the offset `off` in the plane (with the sign
// `sgn`). DIAG: the coefficient of the voxel itself in that sample
// (`same`: the in-plane offset is the voxel itself).
template <bool DIAG, typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
sample(const scalar_t * x, const Plane<offset_t> & P, int side, int k,
       offset_t off = 0, int8_t sgn = 1, bool same = true) {
  int8_t s = P.s[side][k] * sgn;
  if (DIAG)
    return (same && P.self[side][k]) ? static_cast<reduce_t>(s)
                                     : static_cast<reduce_t>(0);
  if (!s) return static_cast<reduce_t>(0);
  reduce_t v = static_cast<reduce_t>(x[P.o[side][k] + off]);
  return s > 0 ? v : -v;
}

// Sample (ka, kb) of the sides (sa, sb) of two axes
template <bool DIAG, typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
sample2(const scalar_t * x,
        const Plane<offset_t> & Pa, int sa, int ka,
        const Plane<offset_t> & Pb, int sb, int kb) {
  int8_t s = Pa.s[sa][ka] * Pb.s[sb][kb];
  if (DIAG)
    return (Pa.self[sa][ka] && Pb.self[sb][kb]) ? static_cast<reduce_t>(s)
                                                : static_cast<reduce_t>(0);
  if (!s) return static_cast<reduce_t>(0);
  reduce_t v = static_cast<reduce_t>(x[Pa.o[sa][ka] + Pb.o[sb][kb]]);
  return s > 0 ? v : -v;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//            Row of B at a voxel q of the plane j (gather form)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Terms of the axis `a` of the plane P (unit coefficients). `x` points to
// the voxel q. DIAG: the diagonal of B at q (same closed form, every
// sample replaced by its coefficient of q).

// Membrane along a (w: weights at q, with the plane PW of their even
// boundary condition, or nullptr)
template <bool DIAG, typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
membrane(const scalar_t * x, const Plane<offset_t> & P,
         const scalar_t * w, const Plane<offset_t> & PW) {
  reduce_t r = 0;
  for (int side = 0; side < 2; ++side) {
    if (!P.ghost[side] && !P.fold[side]) continue;
    reduce_t d = sample<DIAG,reduce_t>(x, P, side, 0)
               - sample<DIAG,reduce_t>(x, P, side, 1);
    if (P.ghost[side])
      r -= 0.5 * (w ? static_cast<reduce_t>(w[PW.o[side][1]]) : 1.) * d;
    if (P.fold[side])
      r -= 0.5 * P.fold[side] *
           (w ? static_cast<reduce_t>(w[PW.o[side][0]]) : 1.) * d;
  }
  return r;
}

// Bending: central second differences along a
template <bool DIAG, typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
bending_diag(const scalar_t * x, const Plane<offset_t> & P) {
  reduce_t r = 0;
  for (int side = 0; side < 2; ++side) {
    if (!P.ghost[side] && !P.fold[side]) continue;
    reduce_t x0 = sample<DIAG,reduce_t>(x, P, side, 0),
             xg = sample<DIAG,reduce_t>(x, P, side, 1);
    if (P.ghost[side])
      r -= sample<DIAG,reduce_t>(x, P, side, 3) - 2 * xg + x0;
    if (P.fold[side])
      r += P.fold[side] * (xg - 2 * x0 + sample<DIAG,reduce_t>(x, P, side, 2));
  }
  return r;
}

// Bending: cross term of the pair (a, b), edges of axis a. The Neumann
// Laplacian along b (in the plane, at qb, with stride sb) of
// d = x[j0] - x~[jg].
template <bool DIAG, typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
bending_edge(const scalar_t * x, const Plane<offset_t> & P,
             offset_t qb, offset_t nb, offset_t sb) {
  reduce_t r = 0;
  for (int side = 0; side < 2; ++side) {
    if (!P.ghost[side] && !P.fold[side]) continue;
    reduce_t d = sample<DIAG,reduce_t>(x, P, side, 0)
               - sample<DIAG,reduce_t>(x, P, side, 1), Ld = 0;
    // (the in-plane neighbours are never q itself)
    if (qb > 0)
      Ld += d - (sample<DIAG,reduce_t>(x, P, side, 0, -sb, 1, false)
               - sample<DIAG,reduce_t>(x, P, side, 1, -sb, 1, false));
    if (qb < nb - 1)
      Ld += d - (sample<DIAG,reduce_t>(x, P, side, 0, sb, 1, false)
               - sample<DIAG,reduce_t>(x, P, side, 1, sb, 1, false));
    if (P.ghost[side]) r -= Ld;
    if (P.fold[side])  r -= P.fold[side] * Ld;
  }
  return r;
}

// Bending: cross term of the pair (a, b), corners (once per pair).
// Pb: the plane qb of the axis b.
template <bool DIAG, typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
bending_corner(const scalar_t * x, const Plane<offset_t> & Pa,
               const Plane<offset_t> & Pb) {
  reduce_t r = 0;
  for (int sa = 0; sa < 2; ++sa) {
    bool   a0 = Pa.ghost[sa];
    int8_t ap = Pa.fold[sa];
    if (!a0 && !ap) continue;
    for (int sb = 0; sb < 2; ++sb) {
      bool   b0 = Pb.ghost[sb];
      int8_t bp = Pb.fold[sb];
      reduce_t acc = 0;
      if (a0 && b0) acc -= 1.5;
      if (a0 && bp) acc -= 0.5 * bp;
      if (ap && b0) acc -= 0.5 * ap;
      if (ap && bp) acc += 0.5 * ap * bp;
      if (acc == 0) continue;
      reduce_t g = sample2<DIAG,reduce_t>(x, Pa, sa, 0, Pb, sb, 0)
                 - sample2<DIAG,reduce_t>(x, Pa, sa, 0, Pb, sb, 1)
                 - sample2<DIAG,reduce_t>(x, Pa, sa, 1, Pb, sb, 0)
                 + sample2<DIAG,reduce_t>(x, Pa, sa, 1, Pb, sb, 1);
      r += acc * g;
    }
  }
  return r;
}

// Linear-elastic cross term E_i' C_a' F'F C_b E_j u_j at the voxel q of
// the component i (no diagonal: it reads another component).
// Pi, Pj: the plane of the components i and j; uj: u_j at q; Ab: the
// samples of u_j along b (qb, nb, sb: coordinate, size and stride).
template <typename reduce_t, typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
lame_cross(const scalar_t * uj, const Plane<offset_t> & Pi,
           const Plane<offset_t> & Pj, const Axis<offset_t> & Ab,
           offset_t qb, offset_t nb, offset_t sb) {
  // samples qb + 1 and qb - 1 along b
  offset_t op = sb, om = -sb;
  int8_t   sp = 1,  sm = 1;
  if (qb == nb - 1) { op = (Ab.p[1][1] - qb) * sb; sp = Ab.s[1][1]; }
  if (qb == 0)      { om = (Ab.p[0][1] - qb) * sb; sm = Ab.s[0][1]; }
  reduce_t r = 0;
  for (int side = 0; side < 2; ++side) {
    reduce_t sgn = side ? 0.25 : -0.25;
    if (Pi.ghost[side])
      r += sgn * (sample<false,reduce_t>(uj, Pj, side, 1, op, sp)
                - sample<false,reduce_t>(uj, Pj, side, 1, om, sm));
    if (Pi.fold[side])
      r += sgn * Pi.fold[side] *
           (sample<false,reduce_t>(uj, Pj, side, 0, op, sp)
          - sample<false,reduce_t>(uj, Pj, side, 0, om, sm));
  }
  return r;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                       Target planes of an axis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// B only lives on the planes 0, n-1 (ghost rows) and on the source
// planes of the ghost samples -1 and n (folds). The regularisers run
// their plain kernel on all voxels, then one job per target plane of
// each axis (in sequence: two jobs of different axes may share voxels).
template <typename offset_t>
struct Targets {
  offset_t j[4];
  int      n;

  NI_HOST NI_DEVICE void clear() { n = 0; }

  NI_HOST NI_DEVICE void add(offset_t jj) {
    for (int k = 0; k < n; ++k) if (j[k] == jj) return;
    j[n++] = jj;
  }

  // the planes targeted by the samples of A
  NI_HOST NI_DEVICE void add(const Axis<offset_t> & A, offset_t size) {
    add(static_cast<offset_t>(0));
    add(size - 1);
    for (int side = 0; side < 2; ++side)
      if (A.s[side][1]) add(A.p[side][1]);
  }
};

// One job of the border pass: the plane j of the axis a
template <typename offset_t, int NC>
struct PlaneJob {
  int             a;
  offset_t        j;
  Plane<offset_t> P[NC];  // per component (grids) or one (fields)
  Plane<offset_t> PW;     // weights (even boundary condition)
};

// linear index -> voxel of the plane j of the axis a (and batch n)
template <typename offset_t>
static NI_INLINE NI_DEVICE void
plane_decode(offset_t i, int a, offset_t j,
             offset_t X, offset_t Y, offset_t Z,
             offset_t & x, offset_t & y, offset_t & z, offset_t & n) {
  if (a == 0) {
    z = i % Z; i /= Z; y = i % Y; n = i / Y; x = j;
  } else if (a == 1) {
    z = i % Z; i /= Z; x = i % X; n = i / X; y = j;
  } else {
    y = i % Y; i /= Y; x = i % X; n = i / X; z = j;
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                    Border terms of a regulariser
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Which terms of B are not zero, their coefficients, and the tables of
// extended samples. Decided once, on the host.

// Displacement grids: component c has its own boundary conditions
// (sliding axes), and its energy is scaled by its squared voxel size.
template <typename reduce_t, typename offset_t>
struct GridTerms {
  int            dim;
  bool           needs;            // whether B is not zero
  offset_t       n[3];             // size of each axis
  reduce_t       kmem[3][3];       // membrane-like coefficients [comp][axis]
  reduce_t       kbd[3][3];        // bending (second differences)
  reduce_t       kbc[3][3][3];     // bending (cross terms) [comp][axis][axis]
  reduce_t       mu, lam;          // linear-elastic cross terms
  bool           fmem[3][3];       // non-zero border terms
  bool           fbd[3][3];
  bool           fedge[3][3];
  bool           fcorner[3][3][3];
  bool           fshear[3][3];     // [output comp][input comp]
  bool           fdiv[3][3];
  bool           fany[3][3];       // any term of the axis [comp][axis]
  Axis<offset_t> tab[3][3];        // extended samples [comp][axis]
  Axis<offset_t> wtab[3];          // extended samples of the weights

  // ivx: 1 / voxel_size**2
  NI_HOST void set(int dim_, const BoundType b[3], const offset_t n_[3],
                   const double ivx[3], double membrane, double bending,
                   double mu_, double lam_) {
    dim = dim_; mu = mu_; lam = lam_;
    needs = false;
    BoundType cb[3][3];
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a) {
      n[a] = n_[a];
      cb[c][a] = component_bound(b[a], a, c);
      bool on = a < dim && c < dim;
      double r = on ? ivx[a] / ivx[c] : 0.;  // (vx_c / vx_a)**2
      kmem[c][a] = static_cast<reduce_t>(membrane * r + mu * (a == c ? 2. : r)
                                         + (a == c ? lam : 0.));
      kbd[c][a]  = static_cast<reduce_t>(bending * ivx[a] * r);
      fmem[c][a] = on && kmem[c][a] && membrane_needs(cb[c][a]);
      fbd[c][a]  = on && bending && bending_diag_needs(cb[c][a]);
      fedge[c][a] = on && bending && bending_edge_needs(cb[c][a]);
      tab[c][a].set(cb[c][a], n[a]);
    }
    for (int a = 0; a < 3; ++a)
      wtab[a].set(bound::even(b[a]), n[a]);
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
    for (int bb = 0; bb < 3; ++bb) {
      bool on = a < dim && bb < dim && c < dim;
      kbc[c][a][bb] = static_cast<reduce_t>(
        on ? bending * ivx[a] * ivx[bb] / ivx[c] : 0.);
      fcorner[c][a][bb] = on && bending && a < bb &&
        bending_corner_needs(cb[c][a], cb[c][bb]);
    }
    // linear-elastic cross terms of the output component i and input
    // component j: shear (scatter axis j, difference axis i) and
    // divergence (scatter axis i, difference axis j)
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      bool on = i < dim && j < dim && i != j;
      fshear[i][j] = on && mu && lame_cross_needs(cb[i][j], cb[j][j]);
      fdiv[i][j] = on && lam && lame_cross_needs(cb[i][i], cb[j][i]);
    }
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a) {
      bool any = fmem[c][a] || fbd[c][a] || fedge[c][a]
              || (a != c && fshear[c][a]);
      for (int k = 0; k < 3; ++k)
        any = any || fcorner[c][a][k] || (a == c && fdiv[c][k]);
      fany[c][a] = any;
      needs = needs || any;
    }
  }

  // Terms of the axis a at the voxel q of the component c: B (DIAG =
  // false) or its diagonal (DIAG = true). x: the input at q (component
  // 0), sC: its channel stride, s: its spatial strides; w: the weights at
  // q (or nullptr) and PW their plane q[a]; P: the plane q[a] of each
  // component along a.
  template <bool DIAG, typename scalar_t>
  NI_DEVICE reduce_t axis(int a, int c, const offset_t q[3],
                          const offset_t s[3], const scalar_t * x,
                          offset_t sC, const scalar_t * w,
                          const Plane<offset_t> & PW,
                          const Plane<offset_t> * P) const {
    const scalar_t * xc = x + c * sC;
    reduce_t B = 0;
    if (fmem[c][a])
      B += kmem[c][a] * membrane<DIAG,reduce_t>(xc, P[c], w, PW);
    if (fbd[c][a])
      B += kbd[c][a] * bending_diag<DIAG,reduce_t>(xc, P[c]);
    for (int bb = 0; bb < dim; ++bb) {
      if (bb == a) continue;
      if (fedge[c][a])
        B += kbc[c][a][bb] * bending_edge<DIAG,reduce_t>(
          xc, P[c], q[bb], n[bb], s[bb]);
      if (fcorner[c][a][bb]) {
        Plane<offset_t> Pb;
        Pb.set(tab[c][bb], n[bb], q[bb], s[bb]);
        if (Pb.targeted())
          B += kbc[c][a][bb] * bending_corner<DIAG,reduce_t>(xc, P[c], Pb);
      }
    }
    if (DIAG) return B;
    // linear-elastic cross terms: shear (input component a, difference
    // axis c) and divergence (scatter axis c, difference axis j)
    if (a != c && fshear[c][a])
      B += mu * lame_cross<reduce_t>(x + a * sC, P[c], P[a], tab[a][c],
                                     q[c], n[c], s[c]);
    if (a == c)
      for (int j = 0; j < dim; ++j)
        if (fdiv[c][j])
          B += lam * lame_cross<reduce_t>(x + j * sC, P[c], P[j], tab[j][j],
                                          q[j], n[j], s[j]);
    return B;
  }
};

// Fields: every channel has the same boundary conditions (and its own
// membrane and bending factors, applied by the caller)
template <typename reduce_t, typename offset_t>
struct FieldTerms {
  int            dim;
  bool           needs;            // whether B is not zero
  offset_t       n[3];             // size of each axis
  reduce_t       kmem[3];          // membrane coefficient of each axis
  reduce_t       kbd[3];           // bending (second differences)
  reduce_t       kbc[3][3];        // bending (cross terms)
  bool           fmem[3];          // non-zero border terms
  bool           fbd[3];
  bool           fedge[3];
  bool           fcorner[3][3];
  bool           fany[3];          // any term of the axis
  Axis<offset_t> tab[3];           // extended samples of each axis
  Axis<offset_t> wtab[3];          // extended samples of the weights

  // ivx: 1 / voxel_size**2
  NI_HOST void set(int dim_, const BoundType b[3], const offset_t n_[3],
                   const double ivx[3], bool has_membrane, bool has_bending) {
    dim = dim_;
    needs = false;
    for (int a = 0; a < 3; ++a) {
      bool on = a < dim;
      n[a] = n_[a];
      kmem[a] = static_cast<reduce_t>(ivx[a]);
      kbd[a]  = static_cast<reduce_t>(ivx[a] * ivx[a]);
      fmem[a] = on && has_membrane && membrane_needs(b[a]);
      fbd[a]  = on && has_bending && bending_diag_needs(b[a]);
      fedge[a] = on && has_bending && bending_edge_needs(b[a]);
      fany[a] = fmem[a] || fbd[a] || fedge[a];
      for (int bb = 0; bb < 3; ++bb) {
        kbc[a][bb] = static_cast<reduce_t>(ivx[a] * ivx[bb]);
        fcorner[a][bb] = on && bb < dim && a < bb && has_bending &&
                         bending_corner_needs(b[a], b[bb]);
        fany[a] = fany[a] || fcorner[a][bb];
      }
      needs = needs || fany[a];
      tab[a].set(b[a], n[a]);
      wtab[a].set(bound::even(b[a]), n[a]);
    }
  }

  // Terms of the axis a at the voxel q of one channel (factors m and b),
  // B (DIAG = false) or its diagonal (DIAG = true). x: the channel at q,
  // s: its spatial strides; w: its weights at q (or nullptr) and PW their
  // plane q[a]; P: the plane q[a] of the axis a.
  template <bool DIAG, typename scalar_t>
  NI_DEVICE reduce_t axis(int a, const offset_t q[3], const offset_t s[3],
                          const scalar_t * x, reduce_t m, reduce_t b,
                          const scalar_t * w, const Plane<offset_t> & PW,
                          const Plane<offset_t> & P) const {
    reduce_t B = 0;
    if (m && fmem[a])
      B += m * kmem[a] * membrane<DIAG,reduce_t>(x, P, w, PW);
    if (!b) return B;
    if (fbd[a])
      B += b * kbd[a] * bending_diag<DIAG,reduce_t>(x, P);
    for (int bb = 0; bb < dim; ++bb) {
      if (bb == a) continue;
      if (fedge[a])
        B += b * kbc[a][bb] * bending_edge<DIAG,reduce_t>(
          x, P, q[bb], n[bb], s[bb]);
      if (fcorner[a][bb]) {
        Plane<offset_t> Pb;
        Pb.set(tab[bb], n[bb], q[bb], s[bb]);
        if (Pb.targeted())
          B += b * kbc[a][bb] * bending_corner<DIAG,reduce_t>(x, P, Pb);
      }
    }
    return B;
  }
};

} // namespace border
} // namespace ni

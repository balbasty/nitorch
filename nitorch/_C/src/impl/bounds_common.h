#pragma once

// This file contains static functions for handling out-of-bound indices.
// They implement typical boundary conditions (those of standard discrete 
// transforms) + a few other cases (replicated border, zeros, sliding)
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . ni::bound::index -> wrap out-of-bound indices
// . ni::bound::sign  -> optional out-of-bound sign change (sine transforms)
// . ni::BoundType    -> enumerated boundary type
//
// Everything in this file should have internal linkage (static) except
// the BoundType/BoundVectorRef types.

#include <ATen/ATen.h> // Do I really need ATen her?
#include "common.h"
#include "../bounds.h"
#include <iostream>

namespace ni {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             INDEXING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _index {
 
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE size_t inbounds(size_t coord, size_t size) {
  return coord;
}

// Boundary condition of a DCT-I (periodicity: (n-1)*2)
// Indices are reflected about the centre of the border elements:
//    -1 --> 1
//     n --> n-2 
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE size_t reflect1c(size_t coord, size_t size) {
  // size == 1: the field is constant (and the period, 0, is unusable)
  if (size == 1) return 0;
  size_t size_twice = (size-1)*2;
  coord = coord < 0 ? -coord : coord;
  coord = coord % size_twice;
  coord = coord >= size ? size_twice - coord : coord;
  return coord;
}

// Boundary condition of a DST-I (periodicity: (n+1)*2)
// Indices are reflected about the centre of the first out-of-bound 
// element:
//    -1 --> undefined [0]
//    -2 --> 0
//     n --> undefined [n-1]
//   n+1 --> n-1
// The zeros (-1 and n, and their periodic copies) are mapped to a valid
// index so that callers may compute an offset; their sign is 0.
// This also holds for size == 1 (period 4: `0, x0, 0, -x0`), where every
// index maps to 0 (it is *not* a replicated border: see `periodic1`).
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE size_t reflect1s(size_t coord, size_t size) {
  size_t size_twice = (size+1)*2;
  coord = coord == -1 ? 0 : coord < 0 ? -coord-2 : coord;
  coord = coord % size_twice;
  coord = coord == size ? size-1                 // zero at n
        : coord == size_twice-1 ? 0              // zero at -1 (mod period)
        : coord > size ? size_twice-coord-2 : coord;
  return coord;
}

// Boundary condition of a DCT/DST-II (periodicity: n*2)
// Indices are reflected about the edge of the border elements:
//    -1 --> 0
//     n --> n-1
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE size_t reflect2(size_t coord, size_t size) {
  size_t size_twice = size*2;
  coord = coord < 0 ? size_twice - ((-coord-1) % size_twice) - 1
                    : coord % size_twice;
  coord = coord >= size ? size_twice - coord - 1 : coord;
  return coord;
}

// Boundary condition of a DFT (periodicity: n)
// Indices wrap about the edges:
//    -1 --> n-1
//     n --> 0
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE size_t circular(size_t coord, size_t size) {
  coord = coord < 0 ? (size + coord%size) % size : coord % size;
  return coord;
}

// Replicate edge values:
//    -1 --> 0
//    -2 --> 0
//     n --> n-1
//   n+1 --> n-1
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE size_t replicate(size_t coord, size_t size) {
  coord = coord <= 0 ? 0 : coord >= size ? size - 1 : coord;
  return coord;
}

} // namespace index


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          SIGN MODIFICATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _sign {

template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE int8_t inbounds(size_t coord, size_t size) {
  return coord < 0 || coord >= size ? 0 : 1;
}

// Boundary condition of a DCT/DFT
// No sign modification based on coordinates
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE int8_t constant(size_t coord, size_t size) {
  return static_cast<int8_t>(1);
}

// Boundary condition of a DST-I
// Periodic sign change based on coordinates
// (no special case for size == 1: the sign is +1 at 0 mod 4, 0 at odd
// coordinates and -1 at 2 mod 4)
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE int8_t periodic1(size_t coord, size_t size) {
  size_t size_twice = (size+1)*2;
  coord = coord < 0 ? size - coord - 1 : coord;
  coord = coord % size_twice; 
  if (coord % (size+1) == size)   return  static_cast<int8_t>(0);
  else if ((coord/(size+1)) % 2)  return  static_cast<int8_t>(-1);
  else                            return  static_cast<int8_t>(1);
}

// Boundary condition of a DST-II
// Periodic sign change based on coordinates
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE int8_t periodic2(size_t coord, size_t size) {
  coord = (coord < 0 ? size - coord - 1 : coord);
  return static_cast<int8_t>((coord/size) % 2 ? -1 : 1);
}

} // namespace sign

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                BOUND
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static NI_INLINE NI_HOST 
std::ostream& operator<<(std::ostream& os, const BoundType & bound) {
  switch (bound) {
    case BoundType::Replicate:  return os << "Replicate";
    case BoundType::DCT1:       return os << "DCT1";
    case BoundType::DCT2:       return os << "DCT2";
    case BoundType::DST1:       return os << "DST1";
    case BoundType::DST2:       return os << "DST2";
    case BoundType::DFT:        return os << "DFT";
    case BoundType::Zero:       return os << "Zero";
    case BoundType::Sliding:    return os << "Sliding";
    case BoundType::NoCheck:    return os << "NoCheck";
  }
   return os << "Unknown bound";
}

// Check if coordinates within bounds
template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE bool inbounds(size_t coord, size_t size) {
  return coord >= 0 && coord < size;
}

template <typename scalar_t, typename size_t>
static NI_INLINE NI_HOST NI_DEVICE bool inbounds(scalar_t coord, size_t size, scalar_t tol) {
  return coord >= -tol && coord < (scalar_t)(size-1)+tol;
}


namespace bound {

template <typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE scalar_t 
get(const scalar_t * ptr, offset_t offset, 
    int8_t sign = static_cast<int8_t>(1)) {
  if (sign == -1)  return -ptr[offset];
  else if (sign)   return  ptr[offset];
  else             return  static_cast<scalar_t>(0);
}

template <typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE void 
add(scalar_t *ptr, offset_t offset, scalar_t val, 
    int8_t sign = static_cast<int8_t>(1)) {
  if (sign == -1)  NI_ATOMIC_ADD(ptr, offset, -val);
  else if (sign)   NI_ATOMIC_ADD(ptr, offset,  val);
}

template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE int64_t index(BoundType bound_type, size_t coord, size_t size) {
  switch (bound_type) {
    case BoundType::Replicate:  return _index::replicate(coord, size);
    case BoundType::DCT1:       return _index::reflect1c(coord, size);
    case BoundType::DCT2:       return _index::reflect2(coord, size);
    case BoundType::DST1:       return _index::reflect1s(coord, size);
    case BoundType::DST2:       return _index::reflect2(coord, size);
    case BoundType::Sliding:    return _index::reflect2(coord, size);
    case BoundType::DFT:        return _index::circular(coord, size);
    case BoundType::Zero:       return _index::inbounds(coord, size);
    default:                    return _index::inbounds(coord, size);
  }
}

template <typename size_t>
static NI_INLINE NI_HOST NI_DEVICE int8_t sign(BoundType bound_type, size_t coord, size_t size) {
  switch (bound_type) {
    case BoundType::Replicate:  return _sign::constant(coord, size);
    case BoundType::DCT1:       return _sign::constant(coord, size);
    case BoundType::DCT2:       return _sign::constant(coord, size);
    case BoundType::DST1:       return _sign::periodic1(coord, size);
    case BoundType::DST2:       return _sign::periodic2(coord, size);
    // Sliding: sign of the component normal to the axis (see `slide`)
    case BoundType::Sliding:    return _sign::periodic2(coord, size);
    case BoundType::DFT:        return _sign::constant(coord, size);
    case BoundType::Zero:       return _sign::inbounds(coord, size);
    default:                    return _sign::inbounds(coord, size);
  }
}

// ---------------------------------------------------------------------
//   Boundary handling of the quadratic regularisers
// ---------------------------------------------------------------------
// The regularisers apply their stencil `L = K'K` to the bound-extended
// field and read it at the voxels. This equals the exact operator
// `D' diag(omega) D` ("Exact operators" in nitorch/spatial/_regularisers.py) for dft,
// dct2, dst2, dst1 and zero, but not for:
// - dct1, which mirrors the field about its border voxels: they appear
//   once per period and the other voxels twice, so that their rows must
//   be weighted by 1/2 (`row_weight`);
// - replicate, where the central second difference of a border voxel
//   folds its ghost sample back onto it, which the stencil does not:
//   the bending energy must add `(x[inward] - x[border]) / h^4`
//   (`replicate_inward`).

// Weight of the row (equation) of a voxel along one axis
template <typename reduce_t, typename offset_t>
static NI_INLINE NI_DEVICE reduce_t
row_weight(BoundType bound_type, offset_t coord, offset_t size) {
  return (bound_type == BoundType::DCT1 && size > 1 &&
          (coord == 0 || coord == size - 1))
         ? static_cast<reduce_t>(0.5) : static_cast<reduce_t>(1);
}

// Direction (+1/-1) of the inward neighbour of a border voxel of a
// replicate axis, or 0 if no bending correction is needed
template <typename offset_t>
static NI_INLINE NI_DEVICE int8_t
replicate_inward(BoundType bound_type, offset_t coord, offset_t size) {
  if (bound_type != BoundType::Replicate || size < 2) return 0;
  if (coord == 0)        return  1;
  if (coord == size - 1) return -1;
  return 0;
}

// Sliding boundary conditions (displacement grids) reflect the vector
// field geometrically: along a sliding axis, the component normal to the
// border is antisymmetric (DST2) and the other components symmetric
// (DCT2). Both have the same indices (`index`). `sign` is the sign of
// the normal component; this returns the sign of component `comp` along
// axis `axis`. The stencil applied to this extension is the exact
// operator (see nitorch/spatial/_quadform.py), including the
// cross-component terms of the linear-elastic energy.
static NI_INLINE NI_DEVICE int8_t
slide(BoundType bound_type, int8_t sign, int axis, int comp) {
  return (bound_type == BoundType::Sliding && axis != comp)
         ? static_cast<int8_t>(1) : sign;
}

// Boundary condition of the weights (RLS), which are always even: they
// are never negated (dst2/sliding -> dct2) and, where the field is zero
// (dst1/zero), take the value of the nearest voxel (replicate), as
// `weight_bound` in nitorch/spatial/_quadform.py. Their sign is always 1.
static NI_INLINE NI_HOST NI_DEVICE BoundType
even(BoundType bound_type) {
  switch (bound_type) {
    case BoundType::DST2:
    case BoundType::Sliding:    return BoundType::DCT2;
    case BoundType::DST1:
    case BoundType::Zero:       return BoundType::Replicate;
    default:                    return bound_type;
  }
}

} // namespace bound
} // namespace ni

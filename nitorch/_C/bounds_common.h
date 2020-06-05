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
#include "bounds.h"
#include <iostream>

namespace ni {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             INDEXING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _index {
 
template <typename size_t>
static NI_INLINE NI_DEVICE size_t inbounds(size_t coord, size_t size) {
  return coord;
}

// Boundary condition of a DCT-I (periodicity: (n-1)*2)
// Indices are reflected about the centre of the border elements:
//    -1 --> 1
//     n --> n-2 
template <typename size_t>
static NI_INLINE NI_DEVICE size_t reflect1c(size_t coord, size_t size) {
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
template <typename size_t>
static NI_INLINE NI_DEVICE size_t reflect1s(size_t coord, size_t size) {
  if (size == 1) return 0;
  size_t size_twice = (size+1)*2;
  coord = coord == -1 ? 0 : coord < 0 ? -coord-2 : coord;
  coord = coord % size_twice;
  coord = coord == size ? size-1 : coord > size ? size_twice-coord-2 : coord;
  return coord;
}

// Boundary condition of a DCT/DST-II (periodicity: n*2)
// Indices are reflected about the edge of the border elements:
//    -1 --> 0
//     n --> n-1
template <typename size_t>
static NI_INLINE NI_DEVICE size_t reflect2(size_t coord, size_t size) {
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
static NI_INLINE NI_DEVICE size_t circular(size_t coord, size_t size) {
  coord = coord < 0 ? (size + coord%size) % size : coord % size;
  return coord;
}

// Replicate edge values:
//    -1 --> 0
//    -2 --> 0
//     n --> n-1
//   n+1 --> n-1
template <typename size_t>
static NI_INLINE NI_DEVICE size_t replicate(size_t coord, size_t size) {
  coord = coord <= 0 ? 0 : coord >= size ? size - 1 : coord;
  return coord;
}

} // namespace index

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                           INVERSE INDEXING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Indexing functions are (usually) not invertible. However, they 
// are always based on some periodicity. These 'inverse'  functions 
// therefore return the period of the forward indexing functions.
// NOTE: 0 means infinite periodicity

namespace _indexinv {
 
template <typename size_t>
static NI_INLINE NI_DEVICE size_t inbounds(size_t coord, size_t size) {
  return static_cast<size_t>(0);
}

// Boundary condition of a DCT-I (periodicity: (n-1)*2)
// Indices are reflected about the centre of the border elements:
//    -1 --> 1
//     n --> n-2 
template <typename size_t>
static NI_INLINE NI_DEVICE size_t reflect1c(size_t coord, size_t size) {
  return (size-1)*2;
}

// Boundary condition of a DST-I (periodicity: (n+1)*2)
// Indices are reflected about the centre of the first out-of-bound 
// element:
//    -1 --> undefined [0]
//    -2 --> 0
//     n --> undefined [n-1]
//   n+1 --> n-1
template <typename size_t>
static NI_INLINE NI_DEVICE size_t reflect1s(size_t coord, size_t size) {
  return (size+1)*2;
}

// Boundary condition of a DCT/DST-II (periodicity: n*2)
// Indices are reflected about the edge of the border elements:
//    -1 --> 0
//     n --> n-1
template <typename size_t>
static NI_INLINE NI_DEVICE size_t reflect2(size_t coord, size_t size) {
  return size*2;
}

// Boundary condition of a DFT (periodicity: n)
// Indices wrap about the edges:
//    -1 --> n-1
//     n --> 0
template <typename size_t>
static NI_INLINE NI_DEVICE size_t circular(size_t coord, size_t size) {
  return size;
}

// Replicate edge values:
//    -1 --> 0
//    -2 --> 0
//     n --> n-1
//   n+1 --> n-1
template <typename size_t>
static NI_INLINE NI_DEVICE size_t replicate(size_t coord, size_t size) {
  if (coord > 0 || coord < size - 1)
    return static_cast<size_t>(0);
  else if (coord == 0)
    return static_cast<size_t>(-1);
  else
    return static_cast<size_t>(1);
}

} // namespace indexinv

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          SIGN MODIFICATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _sign {

template <typename size_t>
static NI_INLINE NI_DEVICE int8_t inbounds(size_t coord, size_t size) {
  return coord < 0 || coord >= size ? 0 : 1;
}

// Boundary condition of a DCT/DFT
// No sign modification based on coordinates
template <typename size_t>
static NI_INLINE NI_DEVICE int8_t constant(size_t coord, size_t size) {
  return static_cast<int8_t>(1);
}

// Boundary condition of a DST-I
// Periodic sign change based on coordinates
template <typename size_t>
static NI_INLINE NI_DEVICE int8_t periodic1(size_t coord, size_t size) {
  if (size == 1) return 1;
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
static NI_INLINE NI_DEVICE int8_t periodic2(size_t coord, size_t size) {
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
  }
   return os << "Unknown bound";
}

// Check if coordinates within bounds
template <typename size_t>
static NI_INLINE NI_DEVICE bool inbounds(size_t coord, size_t size) {
  return coord >= 0 && coord < size;
}

template <typename scalar_t, typename size_t>
static NI_INLINE NI_DEVICE bool inbounds(scalar_t coord, size_t size, scalar_t tol) {
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
static NI_INLINE NI_DEVICE int64_t index(BoundType bound_type, size_t coord, size_t size) {
  switch (bound_type) {
    case BoundType::NoCheck:    return _index::inbounds(coord, size);
    case BoundType::Zero:       return _index::inbounds(coord, size);
    case BoundType::Replicate:  return _index::replicate(coord, size);
    case BoundType::DFT:        return _index::circular(coord, size);
    case BoundType::DCT2:       return _index::reflect2(coord, size);
    case BoundType::DST2:       return _index::reflect2(coord, size);
    case BoundType::DCT1:       return _index::reflect1c(coord, size);
    case BoundType::DST1:       return _index::reflect1s(coord, size);
    default:                    return _index::inbounds(coord, size);
  }
}

template <typename size_t>
static NI_INLINE NI_DEVICE int8_t sign(BoundType bound_type, size_t coord, size_t size) {
  switch (bound_type) {
    case BoundType::NoCheck:    return _sign::constant(coord, size);
    case BoundType::Zero:       return _sign::inbounds(coord, size);
    case BoundType::Replicate:  return _sign::constant(coord, size);
    case BoundType::DFT:        return _sign::constant(coord, size);
    case BoundType::DCT2:       return _sign::constant(coord, size);
    case BoundType::DST2:       return _sign::periodic2(coord, size);
    case BoundType::DCT1:       return _sign::constant(coord, size);
    case BoundType::DST1:       return _sign::periodic1(coord, size);
    default:                    return _sign::inbounds(coord, size);
  }
}

} // namespace bound
} // namespace ni
#pragma once

// This file contains static functions for handling (0-7 order) 
// interpolation weights.
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . ni::interpolation::weight -> node weight based on distance
// . ni::interpolation::grad   -> weight derivative // oriented distance
// . ni::InterpolationType     -> enumerated type
//
// Everything in this file should have internal linkage (static) except
// the BoundType/BoundVectorRef types.


#include "common.h"
#include "interpolation.h"
#include <cstdint>
#include <iostream>

namespace ni {

static NI_INLINE NI_HOST 
std::ostream& operator<<(std::ostream& os, const InterpolationType & itp) {
  switch (itp) {
    case InterpolationType::Nearest:      return os << "Nearest";
    case InterpolationType::Linear:       return os << "Linear";
    case InterpolationType::Quadratic:    return os << "Quadratic";
    case InterpolationType::Cubic:        return os << "Cubic";
    case InterpolationType::FourthOrder:  return os << "FourthOrder";
    case InterpolationType::FifthOrder:   return os << "FifthOrder";
    case InterpolationType::SixthOrder:   return os << "SixthOrder";
    case InterpolationType::SeventhOrder: return os << "SeventhOrder";
  }
   return os << "Unknown interpolation order";
}

} // namespace ni
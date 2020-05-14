#pragma once

// This file contains static functions for handling (0-7 order) 
// interpolation weights.
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . ni::interpolation::weight -> node weight based on distance
// . ni::interpolation::grad   -> weight derivative // oriented distance
// . ni::InterpolationType     -> enumerated type

#include "include_first.h"
#include <cstdint>
#include <iostream>

namespace ni {

enum class InterpolationType : int64_t
    {Nearest, Linear, Quadratic, Cubic, 
     FourthOrder, FifthOrder, SixthOrder, SeventhOrder};

using InterpolationVectorRef = c10::ArrayRef<InterpolationType>;

static NI_INLINE std::ostream& operator<<(std::ostream& os, const InterpolationType & itp) {
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
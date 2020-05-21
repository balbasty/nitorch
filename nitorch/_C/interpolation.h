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

#include <ATen/ATen.h>
namespace ni {

enum class InterpolationType : int64_t
    {Nearest, Linear, Quadratic, Cubic, 
     FourthOrder, FifthOrder, SixthOrder, SeventhOrder};
using InterpolationVectorRef = c10::ArrayRef<InterpolationType>;

} // namespace ni
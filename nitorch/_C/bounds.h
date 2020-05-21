#pragma once

#include <ATen/ATen.h>

namespace ni {

enum class BoundType : int64_t
  {Replicate, DCT1, DCT2, DST1, DST2, DFT, Sliding, Zero};

using BoundVectorRef = c10::ArrayRef<BoundType>;

} // namespace ni
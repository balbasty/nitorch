#pragma once

#include <ATen/ATen.h>

namespace ni {

enum class BoundType : int64_t{
  Replicate,    // Replicate last inbound value = clip coordinates
  DCT1,         // Symetric w.r.t. center of the last inbound voxel
  DCT2,         // Symetric w.r.t. edge of the last inbound voxel (=Neuman)
  DST1,         // Antisymetric w.r.t. center of the last inbound voxel 
  DST2,         // Antisymetric w.r.t. edge of the last inbound voxel (=Dirichlet)
  DFT,          // Circular / Wrap arounf the FOV
  Sliding,      // For deformation-fields only: mixture of DCT2 and DST2
  Zero,         // Zero outside of the FOV
  NoCheck       // /!\ Checks disabled: assume coordinates are inbound
};

using BoundVectorRef = c10::ArrayRef<BoundType>;

} // namespace ni
#pragma once

#include <ATen/ATen.h>

namespace ni {

enum class GridAlignType : int64_t{
  Edge,    		// Align outer edges of the corner voxels
  Center,       // Align centers of the corner voxels
  First,        // Align center of the first voxel (factor is preserved)
  Last          // Align center of the last voxel  (factor is preserved)
};

using GridAlignVectorRef = c10::ArrayRef<GridAlignType>;

}
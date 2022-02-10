

class Allocator {

  static constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();

private:

  // Copied from aten/src/ATen/native/IndexingUtils.cpp in PyTorch 1.6.
  // It is used to decide to which pointer type we should dispatch to.
  // Basically, we need to make sure that the "furthest" element we need
  // to reach is less than max_elem away.
  static bool tensorCanUse32BitIndexMath(
    const Tensor &t, int64_t max_elem=max_int32)
  {
    int64_t elements = t.numel();
    if (elements >= max_elem) {
      return false;
    }
    if (elements == 0) {
      return max_elem > 0;
    }

    int64_t offset = 0;
    int64_t linearId = elements - 1;

    // NOTE: Assumes all strides are positive, which is true for now
    for (int i = t.dim() - 1; i >= 0; --i) {
      int64_t curDimIndex = linearId % t.size(i);
      int64_t curDimOffset = curDimIndex * t.stride(i);
      offset += curDimOffset;
      linearId /= t.size(i);
    }

    if (offset >= max_elem) {
      return false;
    }

    return true;
  }

};
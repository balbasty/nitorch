#pragma once

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// We need to define AT_PARALLEL_OPENMP (even if -fopenmp is 
// not used) so that at::parallel_for is defined somewhere.
// This must be done before <ATen/Parallel.h> is included.
//
// Note that if AT_PARALLEL_OPENMP = 1 but compilation does not use 
// -fopenmp, omp pragmas will be ignored. In that case, the code will
// be effectively sequential, and we don't have to worry about 
// operations being atomic.
#if !(AT_PARALLEL_OPENMP)
#  if !(AT_PARALLEL_NATIVE)
#    if !(AT_PARALLEL_NATIVE_TBB)
#      error No parallel backend specified
#    endif
#  endif
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// These are defines that help writing generic code for both GPU and CPU
#ifdef __CUDACC__
#  include <ATen/cuda/CUDAApplyUtils.cuh>
#  include <THC/THCAtomics.cuh>
#  define NI_INLINE __forceinline__
#  define NI_DEVICE __device__
#  define NI_HOST   __host__
#  define NI_ATOMIC_ADD ni::gpuAtomicAdd
#  define NI_NAMESPACE_DEVICE namespace cuda
namespace ni {
  // atomicAdd API changed between pytorch 1.4 and 1.5. 
  template <typename scalar_t, typename offset_t>
  static __forceinline__ __device__ 
  void gpuAtomicAdd(scalar_t * ptr, offset_t offset, scalar_t value) {
#if NI_TORCH_VERSION >= 10500
    ::gpuAtomicAdd(ptr+offset, value);
#else
    ::atomicAdd(ptr+offset, value);
#endif
  }
}
#else
#  define NI_INLINE inline
#  define NI_DEVICE
#  define NI_HOST
#  define NI_ATOMIC_ADD ni::cpuAtomicAdd
#  define NI_NAMESPACE_DEVICE namespace cpu
namespace ni {
  template <typename scalar_t, typename offset_t>
  static inline void cpuAtomicAdd(scalar_t * ptr, offset_t offset, scalar_t value) {
#   if AT_PARALLEL_OPENMP
#     if _OPENMP
#       pragma omp atomic
#     endif
#   endif
    ptr[offset] += value;
  }
}
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

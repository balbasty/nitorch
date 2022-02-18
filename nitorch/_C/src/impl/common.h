#pragma once
#include <type_traits>

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
// === CUDA ============================================================
#ifdef __CUDACC__
#  include <ATen/cuda/CUDAApplyUtils.cuh>
#  include <THC/THCAtomics.cuh>
// --- DEFINES ---------------------------------------------------------
#  define NI_INLINE __forceinline__
#  define NI_DEVICE __device__
#  define NI_HOST   __host__
#  define NI_NAMESPACE_DEVICE namespace cuda
// --- ATOMIC ADD ------------------------------------------------------
#  define NI_ATOMIC_ADD ni::gpuAtomicAdd
namespace ni {
  // compile-time helper used in pushpull_common.cpp
  template <typename scalar_t>
  struct has_atomic_add { enum { value = true }; };
  // atomicAdd API changed between pytorch 1.4 and 1.5. 
  template <typename scalar_t, typename offset_t>
  static __forceinline__ __device__ 
  void gpuAtomicAdd(scalar_t * ptr, offset_t offset, scalar_t value) {
#   if NI_TORCH_VERSION >= 10500
      ::gpuAtomicAdd(ptr+offset, value);
#   else
      ::atomicAdd(ptr+offset, value);
#   endif
  }
}
namespace ni {
template <typename T>
NI_HOST NI_INLINE 
T * alloc_on_device(T & obj)
{
  T * pointer_device;
  cudaMalloc((void **)&pointer_device, sizeof(T));
  return pointer_device;
}
template <typename T, typename Stream>
NI_HOST NI_INLINE 
T * copy_to_device(T & obj, T * pointer_device, Stream stream)
{
  cudaMemcpyAsync(pointer_device, &obj, sizeof(T), cudaMemcpyHostToDevice, stream);
  return pointer_device;
}
template <typename T, typename Stream>
NI_HOST NI_INLINE 
T * alloc_and_copy_to_device(T & obj, Stream stream)
{
  T * pointer_device = alloc_on_device(obj);
  copy_to_device(obj, pointer_device, stream);
  return pointer_device;
}
}
// === CPU =============================================================
#else
// --- DEFINES ---------------------------------------------------------
#  define NI_INLINE inline
#  define NI_DEVICE
#  define NI_HOST
#  define NI_NAMESPACE_DEVICE namespace cpu
// --- ATOMIC ADD ------------------------------------------------------
#  define NI_ATOMIC_ADD ni::cpuAtomicAdd
#  if AT_PARALLEL_NATIVE
#    include <atomic>
template <typename T>
class has_fetch_add
{
    // This class helps us check if atomic += is defined for
    // floating point types.
    // https://stackoverflow.com/questions/257288
    typedef char one;
    struct two { char x[2]; };

    template <typename C> static one test( decltype(&C::fetch_add) ) ;
    template <typename C> static two test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(char) };
};
#  elif AT_PARALLEL_NATIVE_TBB
#    include <tbb/atomic.h>
#  endif
namespace ni {
  template <typename scalar_t>
  struct has_atomic_add
  {
# if AT_PARALLEL_OPENMP
    enum { value = true };
# elif AT_PARALLEL_NATIVE_TBB
    // Only implemented for integral types
    enum { value = !std::is_floating_point<float>::value };
# else // AT_PARALLEL_NATIVE
    // Implemented in c++ 11+ for integral types
    // Implemented in c++ 20+ for floating types
    enum { value = has_fetch_add<std::atomic<scalar_t> >::value };
# endif
  };

  template <bool has_atom>
  struct AtomicAdd {
    template <typename scalar_t, typename offset_t>
    static inline void atomic_add(scalar_t * ptr, offset_t offset, scalar_t value) {
      ptr[offset] += value;
    }
  };

  template <>
  struct AtomicAdd<true> {
    template <typename scalar_t, typename offset_t>
    static inline void atomic_add(scalar_t * ptr, offset_t offset, scalar_t value) {
  #   if AT_PARALLEL_NATIVE
        // Implemented in c++ 11+ for integral types
        // Implemented in c++ 20+ for floating types
        std::atomic<scalar_t> *aptr;
        aptr->store(ptr + offset);
        aptr->fetch_add(value);
        return;
#     elif AT_PARALLEL_NATIVE_TBB
        // Only implemented for integral types
        tbb::atomic<scalar_t> *aptr;
        aptr = ptr + offset;
        aptr->fetch_and_add(value);
        return;
#     elif _OPENMP
#       pragma omp atomic
#     endif
      ptr[offset] += value;
    }
  };

  template <typename scalar_t, typename offset_t>
  static inline void cpuAtomicAdd(scalar_t * ptr, offset_t offset, scalar_t value) {
    return AtomicAdd<has_atomic_add<scalar_t>::value>::atomic_add(ptr, offset, value);
  }
} // namespace ni
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

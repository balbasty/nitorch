#pragma once
#include "common.h"
#include "../defines.h"

namespace ni {

template <typename T>
class Pair {
public:

  Pair(const T & l, const T & r): left(l), right(r) {}

  T left;
  T right;
};

template <typename T>
class Triplet {
public:

  Triplet(const T & x, const T & y, const T & z): x(x), y(y), z(z) {}

  T x;
  T y;
  T z;
};

template <typename Fn, typename offset_t>
static NI_INLINE NI_DEVICE void for_unroll(offset_t L, Fn fn) 
{
  switch (L) {
    case 0:
      break;
    case 10:
      fn(9);
    case 9:
      fn(8);
    case 8:
      fn(7);
    case 7:
      fn(6);
    case 6:
      fn(5);
    case 5:
      fn(4);
    case 4:
      fn(3);
    case 3:
      fn(2);
    case 2:
      fn(1);
    case 1:
      fn(0);
      break;
    default:
      for (offset_t l=L; l > 9; --l)
        fn(l);
      fn(9);
      fn(8);
      fn(7);
      fn(6);
      fn(5);
      fn(4);
      fn(3);
      fn(2);
      fn(1);
      fn(0);
  }
}

// Returns a value such that: index <= 2 ** log2_ceil(index)
// Adapted from https://stackoverflow.com/questions/994593
template <typename T>
static NI_HOST  
T log2_ceil(T index)
{
  if (index == 0)
    return static_cast<T>(-1); // assumes signed T
  --index;
  T targetlevel = 0;
  while (index) {
    index >>= 1;
    ++targetlevel;
  }
  return targetlevel;
}

// https://stackoverflow.com/questions/101439
template <typename BaseType, typename ExpType>
static NI_HOST  
BaseType pow_int(BaseType base, ExpType exp)
{
    BaseType result = 1;
    for (;;)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }
    return result;
}

// The following structures are used to compute powers of 2 at compile time
// Adapted from https://stackoverflow.com/questions/27270541
template <int32_t N>
struct power_of_two
{
    static const int32_t value = 2 * power_of_two<N-1>::value;
};
template <>
struct power_of_two<0>
{
    static const int32_t value = 1;
};

// The folowing structures are used to dynamically dispatch integer values
// to integer-templated functions
// Adapted from https://stackoverflow.com/questions/7089284
template<int32_t min, int32_t... Indices>
struct indices {
    typedef indices<min, Indices..., sizeof...(Indices) + min> next;
};
template<int32_t min, int32_t N>
struct build_indices {
    typedef typename build_indices<min, N - 1>::type::next type;
};
template<int32_t min>
struct build_indices<min, min> {
    typedef indices<min> type;
};

}

#include "common.h"
#include "../defines.h"

namespace ni {

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

}
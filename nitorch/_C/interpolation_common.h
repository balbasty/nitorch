#pragma once

// This file contains static functions for handling (0-7 order) 
// interpolation weights.
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . ni::interpolation::weight     -> node weight based on distance
// . ni::interpolation::fastweight -> same, assuming x lies in support
// . ni::interpolation::grad       -> weight derivative // oriented distance
// . ni::interpolation::fastgrad   -> same, assuming x lies in support
// . ni::interpolation::bounds     -> min/max nodes

// TODO: 
// . second order derivatives

#include "common.h"
#include "interpolation.h"
#include <cstdint>
#include <cmath>
#include <iostream>

namespace ni {

static NI_INLINE NI_HOST 
std::ostream& operator<<(std::ostream& os, const InterpolationType & itp) {
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

namespace _interpolation {

  // --- order 0 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight0(scalar_t x) {
    x = std::abs(x);
    return x < 0.5 ? static_cast<scalar_t>(1) : static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight0(scalar_t x) {
    x = std::abs(x);
    return static_cast<scalar_t>(1);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t grad0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastgrad0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds0(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::round(x));
    upp = low;
  }

  // --- order 1 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight1(scalar_t x) {
    x = std::abs(x);
    return x < 1 ? static_cast<scalar_t>(1) - x : static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight1(scalar_t x) {
    return static_cast<scalar_t>(1) - std::abs(x);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t grad1(scalar_t x) {
    if (std::abs(x) < 1) return static_cast<scalar_t>(0);
    return fastgrad1(x);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastgrad1(scalar_t x) {
    return x < static_cast<scalar_t>(0) ? static_cast<scalar_t>(1) 
                                        : static_cast<scalar_t>(-1);
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds1(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::floor(x));
    upp = low + 1;
  }

  // --- order 2 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight2(scalar_t x) {
    x = std::abs(x);
    if ( x < 0.5 ) 
    {
      return 0.75 - x * x;
    } 
    else if ( x < 1.5 ) 
    {
      x = 1.5 - x;
      return 0.5 * x * x;
    } 
    else 
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight2(scalar_t x) {
    x = std::abs(x);
    if ( x < 0.5 ) 
    {
      return 0.75 - x * x;
    } 
    else 
    {
      x = 1.5 - x;
      return 0.5 * x * x;
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t grad2(scalar_t x) {
    if ( x >= 0 ) 
    {
      if ( x < 0.5 ) 
      {
        return -2. * x;
      } 
      else if ( x < 1.5 ) 
      {
        return x - 1.5;
      } 
      else 
      {
        return static_cast<scalar_t>(0);
      }
    } 
    else 
    {
      if ( x > -0.5 ) 
      {
        return 2. * x;
      } 
      else if ( x > -1.5 ) 
      {
        return 1.5 - x;
      } 
      else 
      {
        return static_cast<scalar_t>(0);
      }
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastgrad2(scalar_t x) {
    if ( x >= 0 ) 
    {
      if ( x < 0.5 ) 
      {
        return -2. * x;
      } 
      else 
      {
        return x - 1.5;
      }
    } 
    else 
    {
      if ( x > -0.5 ) 
      {
        return 2. * x;
      } 
      else 
      {
        return 1.5 - x;
      }
    }
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds2(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::floor(x))-2;
    upp = low + 3;
  }

  // --- order 3 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight3(scalar_t x) {
    x = std::abs(x);
    if ( x < 1. ) 
    {
      return ( x * x * (x - 2.) * 3. + 4. ) / 6.;
    } 
    else if ( x < 2. ) 
    {
      x = 2. - x;
      return ( x * x * x ) / 6.;
    } 
    else 
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight3(scalar_t x) {
    x = std::abs(x);
    if ( x < 1. ) 
    {
      return ( x * x * (x - 2.) * 3. + 4. ) / 6.;
    } 
    else 
    {
      x = 2. - x;
      return ( x * x * x ) / 6.;
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t grad3(scalar_t x) {
    if ( x >= 0 ) 
    {
      if ( x < 1. ) 
      {
        return x * x * 1.5 - 2. * x;
      } 
      else if ( x < 2. ) 
      {
        x = 2. - x;
        return - ( x * x ) * 0.5;
      } 
      else 
      {
        return static_cast<scalar_t>(0);
      }
    } 
    else 
    {
      if ( x > -1. ) 
      {
        return 2. * x - x * x * 1.5;
      } 
      else if ( x > -2. ) 
      {
        return ( x * x ) * 0.5;
      } 
      else 
      {
        return static_cast<scalar_t>(0);
      }
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastgrad3(scalar_t x) {
    if ( x >= 0 ) 
    {
      if ( x < 1. ) 
      {
        return x * x * 1.5 - 2. * x;
      } 
      else 
      {
        x = 2. - x;
        return - ( x * x ) * 0.5;
      }
    } 
    else 
    {
      if ( x > -1. ) 
      {
        return 2. * x - x * x * 1.5;
      } 
      else 
      {
        return ( x * x ) * 0.5;
      }
    }
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds3(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::floor(x))-3;
    upp = low + 5;
  }

  // --- order 4 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight4(scalar_t x) {
    x = std::abs(x);
    if ( x < 0.5 ) 
    {
      x *= x;
      return x * ( x * 0.25 - 0.625 ) + 115. / 192.;
    } 
    else if ( x < 1.5 ) 
    {
      return x * ( x * ( x * ( 5. - x ) / 6. - 1.25 ) + 5. / 24. ) + 55. / 96.;
    } 
    else if ( x < 2.5 ) 
    {
      x -= 2.5;
      x *= x;
      return ( x * x ) / 24.;
    } 
    else 
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight4(scalar_t x) {
    x = std::abs(x);
    if ( x < 0.5 ) 
    {
      x *= x;
      return x * ( x * 0.25 - 0.625 ) + 115. / 192.;
    } 
    else if ( x < 1.5 ) 
    {
      return x * ( x * ( x * ( 5. - x ) / 6. - 1.25 ) + 5. / 24. ) + 55. / 96.;
    } 
    else 
    {
      x -= 2.5;
      x *= x;
      return ( x * x ) / 24.;
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t grad4(scalar_t x) {
    if ( x >= 0 ) 
    {
      if ( x < 0.5 ) 
      {
        scalar_t x2 = x * x;
        return x * ( x2 * x2 - 2.5 );
      } 
      else if ( x < 1.5 ) 
      {
        return x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
      } 
      else if ( x < 2.5 ) 
      {
        scalar_t xm = x - 2.5;
        return x * ( xm * xm * xm ) / 6.;
      } 
      else 
      {
        return static_cast<scalar_t>(0);
      }
    } 
    else 
    {
      if ( x > -0.5 ) 
      {
        scalar_t x2 = x * x;
        return - x * ( x2 * x2 - 2.5 );
      } 
      else if ( x > -1.5 ) 
      {
        return - x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
      } 
      else if ( x > -2.5 ) 
      {
        scalar_t xm = x - 2.5;
        return - x * ( xm * xm * xm ) / 6.;
      } 
      else 
      {
        return static_cast<scalar_t>(0);
      }
    }
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastgrad4(scalar_t x) {
    if ( x >= 0 ) 
    {
      if ( x < 0.5 ) 
      {
        scalar_t x2 = x * x;
        return x * ( x2 * x2 - 2.5 );
      } 
      else if ( x < 1.5 ) 
      {
        return x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
      } 
      else 
      {
        scalar_t xm = x - 2.5;
        return x * ( xm * xm * xm ) / 6.;
      }
    } 
    else 
    {
      if ( x > -0.5 ) 
      {
        scalar_t x2 = x * x;
        return - x * ( x2 * x2 - 2.5 );
      } 
      else if ( x > -1.5 ) 
      {
        return - x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
      } 
      else 
      {
        scalar_t xm = x - 2.5;
        return - x * ( xm * xm * xm ) / 6.;
      }
    }
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds4(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::floor(x))-4;
    upp = low + 7;
  }

  // --- order 5 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight5(scalar_t x) {
    x = std::abs(x);
    if ( x < 1.0 )
    {
      scalar_t f = x * x;
      return f * ( f * ( 0.25 - x * ( 1.0 / 12.0 ) ) - 0.5 ) + 0.55;
    }
    else if ( x < 2.0 )
    {
      return x * ( x * ( x * ( x * ( x * ( 1.0 / 24.0 ) - 0.375 ) + 1.25 ) -
             1.75 ) + 0.625 ) + 0.425;
    }
    else if ( x < 3.0 )
    {
      scalar_t f = 3.0 - x;
      x = f * f;
      return f * x * x * ( 1.0 / 120.0 );
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight5(scalar_t x) {
    x = std::abs(x);
    if ( x < 1.0 )
    {
      scalar_t f = x * x;
      return f * ( f * ( 0.25 - x * ( 1.0 / 12.0 ) ) - 0.5 ) + 0.55;
    }
    else if ( x < 2.0 )
    {
      return x * ( x * ( x * ( x * ( x * ( 1.0 / 24.0 ) - 0.375 ) + 1.25 ) -
             1.75 ) + 0.625 ) + 0.425;
    }
    else
    {
      scalar_t f = 3.0 - x;
      x = f * f;
      return f * x * x * ( 1.0 / 120.0 );
    }
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds5(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::floor(x))-5;
    upp = low + 9;
  }

  // --- order 6 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight6(scalar_t x) {
    x = std::abs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * ( 7.0 / 48.0 - x * ( 1.0 / 36.0 ) ) - 77.0 / 192.0 ) +
             5887.0 / 11520.0;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 1.0 / 48.0 ) - 7.0 / 48.0 ) +
             0.328125 ) - 35.0 / 288.0 ) - 91.0 / 256.0 ) - 7.0 / 768.0 ) +
             7861.0 / 15360.0;
    }
    else if ( x < 2.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( 7.0 / 60.0 - x * ( 1.0 / 120.0 ) ) -
             0.65625 ) + 133.0 / 72.0 ) - 2.5703125 ) + 1267.0 / 960.0 ) +
             1379.0 / 7680.0;
    }
    else if ( x < 3.5 )
    {
      x -= 3.5;
      x *= x * x;
      return x * x * ( 1.0 / 720.0 );
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight6(scalar_t x) {
    x = std::abs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * ( 7.0 / 48.0 - x * ( 1.0 / 36.0 ) ) - 77.0 / 192.0 ) +
             5887.0 / 11520.0;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 1.0 / 48.0 ) - 7.0 / 48.0 ) +
             0.328125 ) - 35.0 / 288.0 ) - 91.0 / 256.0 ) - 7.0 / 768.0 ) +
             7861.0 / 15360.0;
    }
    else if ( x < 2.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( 7.0 / 60.0 - x * ( 1.0 / 120.0 ) ) -
             0.65625 ) + 133.0 / 72.0 ) - 2.5703125 ) + 1267.0 / 960.0 ) +
             1379.0 / 7680.0;
    }
    else
    {
      x -= 3.5;
      x *= x * x;
      return x * x * ( 1.0 / 720.0 );
    }
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds6(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::floor(x))-6;
    upp = low + 11;
  }

  // --- order 7 -------------------------------------------------------

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t weight7(scalar_t x) {
    x = std::abs(x);
    if ( x < 1.0 )
    {
      scalar_t f = x * x;
      return f * ( f * ( f * ( x * ( 1.0 / 144.0 ) - 1.0 / 36.0 ) + 1.0 / 9.0 ) -
             1.0 / 3.0 ) + 151.0 / 315.0;
    }
    else if ( x < 2.0 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 0.05 - x * ( 1.0 / 240.0 ) ) -
             7.0 / 30.0 ) + 0.5 ) - 7.0 / 18.0 ) - 0.1 ) - 7.0 / 90.0 ) +
             103.0 / 210.0;
    }
    else if ( x < 3.0 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( x * ( 1.0 / 720.0 ) -
             1.0 / 36.0 ) + 7.0 / 30.0 ) - 19.0 / 18.0 ) + 49.0 / 18.0 ) -
             23.0 / 6.0 ) + 217.0 / 90.0 ) - 139.0 / 630.0;
    }
    else if ( x < 4.0 )
    {
      scalar_t f = 4.0 - x;
      x = f * f * f;
      return x * x * f * ( 1.0 / 5040.0 );
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static NI_INLINE NI_DEVICE scalar_t fastweight7(scalar_t x) {
    x = std::abs(x);
    if ( x < 1.0 )
    {
      scalar_t f = x * x;
      return f * ( f * ( f * ( x * ( 1.0 / 144.0 ) - 1.0 / 36.0 ) + 1.0 / 9.0 ) -
             1.0 / 3.0 ) + 151.0 / 315.0;
    }
    else if ( x < 2.0 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 0.05 - x * ( 1.0 / 240.0 ) ) -
             7.0 / 30.0 ) + 0.5 ) - 7.0 / 18.0 ) - 0.1 ) - 7.0 / 90.0 ) +
             103.0 / 210.0;
    }
    else if ( x < 3.0 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( x * ( 1.0 / 720.0 ) -
             1.0 / 36.0 ) + 7.0 / 30.0 ) - 19.0 / 18.0 ) + 49.0 / 18.0 ) -
             23.0 / 6.0 ) + 217.0 / 90.0 ) - 139.0 / 630.0;
    }
    else
    {
      scalar_t f = 4.0 - x;
      x = f * f * f;
      return x * x * f * ( 1.0 / 5040.0 );
    }
  }

  template <typename scalar_t, typename offset_t>
  static NI_INLINE NI_DEVICE void bounds7(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(std::floor(x))-7;
    upp = low + 13;
  }


} // namespace _interpolation

namespace interpolation {

template <typename scalar_t>
static NI_INLINE NI_DEVICE scalar_t 
weight(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:      return _interpolation::weight0(x);
    case InterpolationType::Linear:       return _interpolation::weight1(x);
    case InterpolationType::Quadratic:    return _interpolation::weight2(x);
    case InterpolationType::Cubic:        return _interpolation::weight3(x);
    case InterpolationType::FourthOrder:  return _interpolation::weight4(x);
    case InterpolationType::FifthOrder:   return _interpolation::weight5(x);
    case InterpolationType::SixthOrder:   return _interpolation::weight6(x);
    case InterpolationType::SeventhOrder: return _interpolation::weight7(x);
    default:                              return _interpolation::weight1(x);
  }
}

template <typename scalar_t>
static NI_INLINE NI_DEVICE scalar_t 
fastweight(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:      return _interpolation::fastweight0(x);
    case InterpolationType::Linear:       return _interpolation::fastweight1(x);
    case InterpolationType::Quadratic:    return _interpolation::fastweight2(x);
    case InterpolationType::Cubic:        return _interpolation::fastweight3(x);
    case InterpolationType::FourthOrder:  return _interpolation::fastweight4(x);
    case InterpolationType::FifthOrder:   return _interpolation::fastweight5(x);
    case InterpolationType::SixthOrder:   return _interpolation::fastweight6(x);
    case InterpolationType::SeventhOrder: return _interpolation::fastweight7(x);
    default:                              return _interpolation::fastweight1(x);
  }
}

template <typename scalar_t>
static NI_INLINE NI_DEVICE scalar_t 
grad(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:      return _interpolation::grad0(x);
    case InterpolationType::Linear:       return _interpolation::grad1(x);
    case InterpolationType::Quadratic:    return _interpolation::grad2(x);
    case InterpolationType::Cubic:        return _interpolation::grad3(x);
    case InterpolationType::FourthOrder:  return _interpolation::grad4(x);
    case InterpolationType::FifthOrder:   return _interpolation::grad0(x); // notimplemented
    case InterpolationType::SixthOrder:   return _interpolation::grad0(x); // notimplemented
    case InterpolationType::SeventhOrder: return _interpolation::grad0(x); // notimplemented
    default:                              return _interpolation::grad1(x);
  }
}

template <typename scalar_t>
static NI_INLINE NI_DEVICE scalar_t 
fastgrad(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:      return _interpolation::fastgrad0(x);
    case InterpolationType::Linear:       return _interpolation::fastgrad1(x);
    case InterpolationType::Quadratic:    return _interpolation::fastgrad2(x);
    case InterpolationType::Cubic:        return _interpolation::fastgrad3(x);
    case InterpolationType::FourthOrder:  return _interpolation::fastgrad4(x);
    case InterpolationType::FifthOrder:   return _interpolation::fastgrad0(x); // notimplemented
    case InterpolationType::SixthOrder:   return _interpolation::fastgrad0(x); // notimplemented
    case InterpolationType::SeventhOrder: return _interpolation::fastgrad0(x); // notimplemented
    default:                              return _interpolation::fastgrad1(x);
  }
}

template <typename scalar_t, typename offset_t>
static NI_INLINE NI_DEVICE void 
bounds(InterpolationType interpolation_type, scalar_t x, offset_t & low, offset_t & upp) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:      return _interpolation::bounds0(x, low, upp);
    case InterpolationType::Linear:       return _interpolation::bounds1(x, low, upp);
    case InterpolationType::Quadratic:    return _interpolation::bounds2(x, low, upp);
    case InterpolationType::Cubic:        return _interpolation::bounds3(x, low, upp);
    case InterpolationType::FourthOrder:  return _interpolation::bounds4(x, low, upp);
    case InterpolationType::FifthOrder:   return _interpolation::bounds5(x, low, upp);
    case InterpolationType::SixthOrder:   return _interpolation::bounds6(x, low, upp);
    case InterpolationType::SeventhOrder: return _interpolation::bounds7(x, low, upp);
    default:                              return _interpolation::bounds1(x, low, upp);
  }
}


} // namespace interpolation

} // namespace ni
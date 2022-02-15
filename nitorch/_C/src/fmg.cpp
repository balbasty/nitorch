#include <ATen/ATen.h>
#include <vector>
#include "defines.h"
#include "bounds.h"
#include "resize.h"
#include "relax.h"
#include "regulariser.h"
#include "relax_grid.h"
#include "regulariser_grid.h"


using at::Tensor;
using c10::IntArrayRef;
using c10::ArrayRef;
using std::vector;
using at::indexing::Slice;



namespace ni {


/* =========================================================== */
/*                            UTILS                            */
/* =========================================================== */

namespace {

  inline void get_shape(const Tensor & x, int64_t s[])
  {
    int64_t dim = x.dim() - 2;
    s[0] = x.size(0);
    s[1] = x.size(1);
    switch (dim) {  // there are no breaks on purpose
      case 3:
        s[4] = x.size(4);
      case 2:
        s[3] = x.size(3);
      case 1:
        s[2] = x.size(2);
      default:
        break;
    }
  }

  inline int64_t restrict_shape1(int64_t s0)
  {
    // fast version of (int)ceil((double)s0 / 2.)
    // assumes s0 > 0
    return 1 + ((s0 - 1) / 2);
  }

  inline bool restrict_shape(const int64_t s0[], int64_t s1[], int64_t dim)
  {
    bool all_ones = true;
    s1[0] = s0[0];  // batch
    s1[1] = s0[1];  // channels
    for (int64_t i = 2; i < dim+2; ++i) {
      s1[i] = restrict_shape1(s0[i]);
      all_ones = all_ones && (s1[i] == 1);
    }
    return all_ones;
  }


  inline Tensor init_solution(const Tensor & solution, const Tensor & gradient)
  {
    if (solution.defined() && solution.numel() > 0)
      return solution;
    return at::zeros_like(gradient);
  }

  template <typename T>
  inline vector<T> slice(const vector<T> & v, int64_t i0, int64_t i1)
  {
    return vector<T>(v.cbegin() + i0, v.cbegin() + i1);
  }

  template <typename RestrictFn>
  inline int64_t prepare_tensors(
    const Tensor & h, 
    const Tensor & g, 
    const Tensor & x, 
    const Tensor & w,
    Tensor out[],
    RestrictFn restrict,
    int64_t max_levels)
  {
    int64_t dim = g.dim() - 2;

    Tensor * hh = out;
    Tensor * gg = out + max_levels;
    Tensor * xx = out + max_levels * 2;
    Tensor * ww = out + max_levels * 3;
    hh[0] = h;
    gg[0] = g;
    xx[0] = x;
    ww[0] = w;

    bool has_h = h.defined() && h.numel();
    bool has_w = w.defined() && w.numel();

    vector<int64_t> shapes(5*max_levels);
    get_shape(g, shapes.data());

    int64_t n;
    for (n = 1; n < max_levels; ++n)
    {
      // compute shape
      if (restrict_shape(shapes.data() + (n-1)*5, shapes.data() + n*5, dim))
        break;
      vector<int64_t> shape1 = slice(shapes, n*5, n*5 + dim);
      // hessian
      if (has_h) {
        shape1[1] = h.size(1);
        hh[n] = at::empty(shape1, h.options());
        restrict(hh[n-1], hh[n]);
      }
      // gradient
      shape1[1] = g.size(1);
      gg[n] = at::empty(shape1, g.options());
      restrict(gg[n-1], gg[n]);
      // solution
      shape1[1] = x.size(1);
      xx[n] = at::empty(shape1, x.options());
      restrict(xx[n-1], xx[n]);
      // weights
      if (has_w) {
        shape1[1] = w.size(1);
        ww[n] = at::empty(shape1, w.options());
        restrict(ww[n-1], ww[n]);
      }
    }
    
    return n; // nb_levels
  }
                  

} // anonymous namespace


/* =========================================================== */
/*                            FIELD                            */
/* =========================================================== */

namespace {

  inline void residuals(
    const Tensor & h, 
    const Tensor & g, 
    const Tensor & x, 
    const Tensor & w, 
          Tensor & r,
    const vector<double> &  absolute, 
    const vector<double> &  membrane, 
    const vector<double> &  bending,
    const vector<double> &  voxel_size, 
    const vector<BoundType> & bound)
  {
    regulariser(x, r, w, h,
                absolute, membrane, bending,
                voxel_size, bound);
    at::sub_out(r, g, r);
  }

}

Tensor fmg(const Tensor & hessian, 
           const Tensor & gradient,
           Tensor solution,
           const Tensor & weight,
           const vector<double> &  absolute, 
           const vector<double> &  membrane, 
           const vector<double> &  bending,
           const vector<double> & voxel_size, 
           const vector<BoundType> & bound,
           int64_t nb_cycles,
           int64_t nb_iter,
           int64_t max_levels)
{

  /* ---------------- function handles ---------------------- */
  auto relax_ = [absolute, membrane, bending, bound, voxel_size, nb_iter]
                (const Tensor & hessian, const Tensor & gradient,
                 const Tensor & solution, const Tensor & weight)
  {
    relax(hessian, gradient, solution, weight, 
          absolute, membrane, bending, voxel_size, bound, nb_iter);
  };
  auto prolong_ = [bound](const Tensor & x, const Tensor & o)
  {
    prolong(x, o, bound);
  };
  auto restrict_ = [bound](const Tensor & x, const Tensor & o)
  {
    restrict(x, o, bound);
  };
  auto residuals_ = [absolute, membrane, bending, bound, voxel_size]
                    (const Tensor & hessian,  const Tensor & gradient,
                     const Tensor & solution, const Tensor & weight,
                           Tensor & res)
  {
    residuals(hessian, gradient, solution, weight, res,
              absolute, membrane, bending, voxel_size, bound);
  };


  /* ---------------- initialize pyramid -------------------- */
  solution = init_solution(solution, gradient);
  vector<Tensor> tensors(max_levels*5);
  auto N = prepare_tensors(hessian, gradient, solution, weight, 
                           tensors.data(), restrict_, max_levels);
  Tensor * h = tensors.data();
  Tensor * g = h + max_levels;
  Tensor * x = g + max_levels;
  Tensor * w = x + max_levels;
  Tensor * r = w + max_levels;

  /* ------------------------ FMG algorithm ------------------ */
  relax_(h[N-1], g[N-1], x[N-1], w[N-1]);

  for (int64_t n_base = N - 2; n_base >= 0; --n_base)
  {
    prolong_(x[n_base+1], x[n_base]);

    for (int64_t n_cycle = 0; n_cycle < nb_cycles; ++n_cycle) 
    {

      for (int64_t n = n_base; n < N-1; ++n) 
      {
        relax_(h[n], g[n], x[n], w[n]);
        residuals_(h[n], g[n], x[n], w[n], r[n]);
        restrict_(r[n], g[n+1]);
        x[n+1].zero_();
      }

      relax_(h[N-1], g[N-1], x[N-1], w[N-1]);

      for (int64_t n = N-2; n >= n_base; --n) 
      {
        prolong_(x[n+1], r[n]);
        at::add_out(x[n], x[n], r[n]);
        relax_(h[n], g[n], x[n], w[n]);
      }
    }
  }

  solution.copy_(x[0]);
  return solution;
}

/* =========================================================== */
/*                            GRID                             */
/* =========================================================== */

namespace {

  inline void residuals_grid(
    const Tensor & h, 
    const Tensor & g, 
    const Tensor & x, 
    const Tensor & w, 
          Tensor & r,
          double    absolute, 
          double    membrane, 
          double    bending,
          double    lame_shear,
          double    lame_div,
    const vector<double> &  voxel_size, 
    const vector<BoundType> & bound)
  {
    regulariser_grid(x, r, w, h,
                     absolute, membrane, bending, lame_shear, lame_div,
                     voxel_size, bound);
    at::sub_out(r, g, r);
  }

}

Tensor fmg_grid(const Tensor & hessian, 
                const Tensor & gradient,
                      Tensor   solution,
                const Tensor & weight,
                      double    absolute, 
                      double    membrane, 
                      double    bending,
                      double    lame_shear,
                      double    lame_div,
                const vector<double> & voxel_size, 
                const vector<BoundType> & bound,
                int64_t nb_cycles,
                int64_t nb_iter,
                int64_t max_levels)
{
  int64_t dim = gradient.dim() - 2;

  /* ---------------- function handles ---------------------- */
  auto relax_ = [absolute, membrane, bending, lame_shear, lame_div, 
                 bound, voxel_size, nb_iter]
                (const Tensor & hessian, const Tensor & gradient,
                 const Tensor & solution, const Tensor & weight)
  {
    relax_grid(hessian, gradient, solution, weight, 
               absolute, membrane, bending, lame_shear, lame_div,
               voxel_size, bound, nb_iter);
  };
  auto prolong_ = [bound, dim](const Tensor & x, const Tensor & o)
  {
    prolong(x, o, bound);
    Tensor view;
    switch (dim) {  // there are no breaks on purpose
      case 3:
        view  = o.index({Slice(), 2});
        view *= static_cast<double>(o.size(4)) / static_cast<double>(x.size(4));
      case 2:
        view  = o.index({Slice(), 1});
        view *= static_cast<double>(o.size(3)) / static_cast<double>(x.size(3));
      case 1:
        view  = o.index({Slice(), 0});
        view *= static_cast<double>(o.size(2)) / static_cast<double>(x.size(2));
      default:
        break;
    }

  };
  auto restrict_ = [bound, dim](const Tensor & x, const Tensor & o)
  {
    restrict(x, o, bound);
    Tensor view;
    switch (dim) {  // there are no breaks on purpose
      case 3:
        view  = o.index({Slice(), 2});
        view *= static_cast<double>(o.size(4)) / static_cast<double>(x.size(4));
      case 2:
        view  = o.index({Slice(), 1});
        view *= static_cast<double>(o.size(3)) / static_cast<double>(x.size(3));
      case 1:
        view  = o.index({Slice(), 0});
        view *= static_cast<double>(o.size(2)) / static_cast<double>(x.size(2));
      default:
        break;
    }
  };
  auto residuals_ = [absolute, membrane, bending, lame_shear, lame_div,
                     bound, voxel_size]
                    (const Tensor & hessian,  const Tensor & gradient,
                     const Tensor & solution, const Tensor & weight,
                           Tensor & res)
  {
    residuals_grid(hessian, gradient, solution, weight, res,
                   absolute, membrane, bending, lame_shear, lame_div, 
                   voxel_size, bound);
  };


  /* ---------------- initialize pyramid -------------------- */
  solution = init_solution(solution, gradient);
  vector<Tensor> tensors(max_levels*5);
  auto N = prepare_tensors(hessian, gradient, solution, weight, 
                           tensors.data(), restrict_, max_levels);
  Tensor * h = tensors.data();
  Tensor * g = h + max_levels;
  Tensor * x = g + max_levels;
  Tensor * w = x + max_levels;
  Tensor * r = w + max_levels;

  /* ------------------------ FMG algorithm ------------------ */
  relax_(h[N-1], g[N-1], x[N-1], w[N-1]);

  for (int64_t n_base = N - 2; n_base >= 0; --n_base)
  {
    prolong_(x[n_base+1], x[n_base]);

    for (int64_t n_cycle = 0; n_cycle < nb_cycles; ++n_cycle) 
    {

      for (int64_t n = n_base; n < N-1; ++n) 
      {
        relax_(h[n], g[n], x[n], w[n]);
        residuals_(h[n], g[n], x[n], w[n], r[n]);
        restrict_(r[n], g[n+1]);
        x[n+1].zero_();
      }

      relax_(h[N-1], g[N-1], x[N-1], w[N-1]);

      for (int64_t n = N-2; n >= n_base; --n) 
      {
        prolong_(x[n+1], r[n]);
        at::add_out(x[n], x[n], r[n]);
        relax_(h[n], g[n], x[n], w[n]);
      }
    }
  }

  solution.copy_(x[0]);
  return solution;
}


} // namespace ni
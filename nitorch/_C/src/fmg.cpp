#include <ATen/ATen.h>
#include <vector>
#include "defines.h"
#include "bounds.h"
#include "resize.h"
#include "pcg.h"
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

  inline void restrict_vx(double * vx0, double * vx1, 
                          int64_t * shape0, int64_t * shape1, int64_t dim)
  {
    for (int64_t d = 0; d < dim; ++d)
      vx1[d] = vx0[d] * (static_cast<double>(shape0[d]) / static_cast<double>(shape1[d]));
  }

#ifdef NI_DEBUG
  inline bool allfinite(const Tensor & x) {
    return at::all(at::isfinite(x)).item<bool>();
  }
#endif

  template <typename RestrictGFn, typename RestrictHFn, typename RestrictWFn>
  inline int64_t prepare_tensors(
    const Tensor & h, 
    const Tensor & g, 
    const Tensor & x, 
    const Tensor & w,
    Tensor out[],
    double vx[],
    RestrictGFn restrict_g,
    RestrictHFn restrict_h,
    RestrictWFn restrict_w,
    int64_t max_levels)
  {
    int64_t dim = g.dim() - 2;

    Tensor * hh = out;
    Tensor * gg = hh + max_levels;
    Tensor * xx = gg + max_levels;
    Tensor * ww = xx + max_levels;
    Tensor * rr = ww + max_levels;
    hh[0] = h;
    gg[0] = g;
    xx[0] = x;
    ww[0] = w;
    rr[0] = at::empty_like(g);

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
      auto shape1 = slice(shapes, n*5, n*5 + dim + 2);
      // voxel size
      restrict_vx(vx + (n-1)*3, vx + n*3, 
                  shapes.data() + (n-1)*5 + 2, shapes.data() + n*5 + 2, dim);
      // hessian
      if (has_h) {
        shape1[1] = h.size(1);
        hh[n] = at::empty(shape1, h.options());
        restrict_h(hh[n-1], hh[n]);
      }
      // gradient
      shape1[1] = g.size(1);
      gg[n] = at::empty(shape1, g.options());
      restrict_g(gg[n-1], gg[n]);
      // residuals
      rr[n] = at::empty_like(gg[n]);
      // solution
      shape1[1] = x.size(1);
      xx[n] = at::empty(shape1, x.options());
      restrict_g(xx[n-1], xx[n]);
      // weights
      if (has_w) {
        shape1[1] = w.size(1);
        ww[n] = at::empty(shape1, w.options());
        restrict_w(ww[n-1], ww[n]);
      }
    }
    
    return n; // nb_levels
  }
                
template <typename RelaxFn, typename ProlongFn, typename RestrictFn, typename ResidualsFn>
inline void do_fmg(Tensor * h, Tensor * g, Tensor * x, Tensor * w, Tensor * r,
                   double * vx, int64_t N, int64_t nb_cycles,
                   RelaxFn relax, ProlongFn prolongation, RestrictFn restriction, ResidualsFn residuals) 
{
  int64_t dim = h[0].dim() - 2;
  auto v = [vx, dim](int64_t n) { return vector<double>(vx + 3*n, vx + 3*n + dim); };

  relax(h[N-1], g[N-1], x[N-1], w[N-1], v(N-1));

  for (int64_t n_base = N - 2; n_base >= 0; --n_base)
  {
    prolongation(x[n_base+1], x[n_base]);

    for (int64_t n_cycle = 0; n_cycle < nb_cycles; ++n_cycle) 
    {

      for (int64_t n = n_base; n < N-1; ++n) 
      {
        relax(h[n], g[n], x[n], w[n], v(n));
        residuals(h[n], g[n], x[n], w[n], r[n], v(n));
        restriction(r[n], g[n+1]);
        x[n+1].zero_();
      }

      relax(h[N-1], g[N-1], x[N-1], w[N-1], v(N-1));

      for (int64_t n = N-2; n >= n_base; --n) 
      {
        prolongation(x[n+1], r[n]);
        at::add_out(x[n], x[n], r[n]);
        relax(h[n], g[n], x[n], w[n], v(n));
      }
    }
  }
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
           int64_t max_levels,
           bool use_cg)
{
  int64_t dim = gradient.dim() - 2;

  /* ---------------- function handles ---------------------- */
  auto relax_ = [absolute, membrane, bending, bound, nb_iter, use_cg]
                (const Tensor & hessian, const Tensor & gradient,
                 const Tensor & solution, const Tensor & weight,
                 const vector<double> & voxel_size)
  {
    if (use_cg)
      pcg(hessian, gradient, solution, weight, 
          absolute, membrane, bending, voxel_size, bound, nb_iter);
    else
      relax(hessian, gradient, solution, weight, 
            absolute, membrane, bending, voxel_size, bound, nb_iter);
  };
  auto prolong_ = [bound](const Tensor & x, const Tensor & o)
  {
    prolongation(x, o, bound);
  };
  auto restrict_ = [bound](const Tensor & x, const Tensor & o)
  {
    restriction(x, o, bound);
  };
  auto residuals_ = [absolute, membrane, bending, bound]
                    (const Tensor & hessian,  const Tensor & gradient,
                     const Tensor & solution, const Tensor & weight,
                           Tensor & res,
                     const vector<double> & voxel_size)
  {
    residuals(hessian, gradient, solution, weight, res,
              absolute, membrane, bending, voxel_size, bound);
  };


  /* ---------------- initialize pyramid -------------------- */
  solution = init_solution(solution, gradient);
  vector<Tensor> tensors(max_levels*5);
  vector<double> vx(max_levels*3);
  for (size_t d=0; d<(size_t)dim; ++d) 
    vx[d] = (voxel_size.size() == 0 ? 1.0 :
             voxel_size.size() > d  ? voxel_size[d] : vx[d-1]);

  int64_t N = prepare_tensors(hessian, gradient, solution, weight, 
                              tensors.data(), vx.data(), 
                              restrict_, restrict_, restrict_,
                              max_levels);

  Tensor * h = tensors.data();
  Tensor * g = h + max_levels;
  Tensor * x = g + max_levels;
  Tensor * w = x + max_levels;
  Tensor * r = w + max_levels;

  /* ------------------------ FMG algorithm ------------------ */
  do_fmg(h, g, x, w, r, vx.data(), N, nb_cycles, 
         relax_, prolong_, restrict_, residuals_);

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
                int64_t max_levels,
                bool use_cg)
{
  int64_t dim = gradient.dim() - 2;

  /* ---------------- function handles ---------------------- */
  auto relax_ = [absolute, membrane, bending, lame_shear, lame_div, 
                 bound, nb_iter, use_cg]
                (const Tensor & hessian, const Tensor & gradient,
                 const Tensor & solution, const Tensor & weight,
                 const vector<double> & voxel_size)
  {
    if (use_cg)
      pcg_grid(hessian, gradient, solution, weight, 
               absolute, membrane, bending, lame_shear, lame_div,
               voxel_size, bound, nb_iter);
    else
      relax_grid(hessian, gradient, solution, weight, 
                 absolute, membrane, bending, lame_shear, lame_div,
                 voxel_size, bound, nb_iter);
  };
  auto prolong_ = [bound, dim](const Tensor & x, const Tensor & o)
  {
    prolongation(x, o, bound);
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
  auto restrict_w_ = [bound, dim](const Tensor & x, const Tensor & o)
  {
    if (!o.defined() || o.numel() == 0)
      return;
    restriction(x, o, bound);
  };
  auto restrict_g_ = [bound, dim](const Tensor & x, const Tensor & o)
  {
    if (!o.defined() || o.numel() == 0)
      return;;
    restriction(x, o, bound);
    Tensor view;
    switch (dim) {  // there are no breaks on purpose
      case 3:
        view  = o.index({Slice(), 2});
        view *= static_cast<double>(x.size(4)) / static_cast<double>(o.size(4));
      case 2:
        view  = o.index({Slice(), 1});
        view *= static_cast<double>(x.size(3)) / static_cast<double>(o.size(3));
      case 1:
        view  = o.index({Slice(), 0});
        view *= static_cast<double>(x.size(2)) / static_cast<double>(o.size(2));
      default:
        break;
    }
  };
  auto restrict_h_ = [bound, dim](const Tensor & x, const Tensor & o)
  {
    if (!o.defined() || o.numel() == 0)
      return;
    restriction(x, o, bound);
    Tensor view;
    double f0 = 1., f1 = 1., f2 = 1.;
    int64_t CC = x.size(1);
    switch (dim) {  // there are no breaks on purpose
      case 3:
        f2 = static_cast<double>(x.size(4)) / static_cast<double>(o.size(4));
      case 2:
        f1 = static_cast<double>(x.size(3)) / static_cast<double>(o.size(3));
      case 1:
        f0 = static_cast<double>(x.size(2)) / static_cast<double>(o.size(2));
      default:
        break;
    }
    switch (dim) {  // there are no breaks on purpose
      case 3:
        if (CC > 1) {
          view   = o.index({Slice(), 2});
          view  *= (f2 * f2);
          if (CC > dim) {
            view   = o.index({Slice(), 5});
            view  *= (f1 * f2);
            view   = o.index({Slice(), 4});
            view  *= (f0 * f2);
          }
        }
      case 2:
        if (CC > 1) {
          view   = o.index({Slice(), 1});
          view  *= (f1 * f1);
          if (CC > dim) {
            view   = o.index({Slice(), dim});
            view  *= (f0 * f1);
          }
        }
      case 1:
        view   = o.index({Slice(), 0});
        view  *= (f0 * f0);
      default:
        break;
    }
  };
  auto residuals_ = [absolute, membrane, bending, lame_shear, lame_div,
                     bound]
                    (const Tensor & hessian,  const Tensor & gradient,
                     const Tensor & solution, const Tensor & weight,
                           Tensor & res, const vector<double> & voxel_size)
  {
    residuals_grid(hessian, gradient, solution, weight, res,
                   absolute, membrane, bending, lame_shear, lame_div, 
                   voxel_size, bound);
  };


  /* ---------------- initialize pyramid -------------------- */
  solution = init_solution(solution, gradient);
  vector<Tensor> tensors(max_levels*5);
  vector<double> vx(max_levels*3);
  for (size_t d=0; d<(size_t)dim; ++d) 
    vx[d] = (voxel_size.size() == 0 ? 1.0 :
             voxel_size.size() > d  ? voxel_size[d] : vx[d-1]);

  auto N = prepare_tensors(hessian, gradient, solution, weight, 
                           tensors.data(), vx.data(), 
                           restrict_g_, restrict_h_, restrict_w_,
                           max_levels);

  Tensor * h = tensors.data();
  Tensor * g = h + max_levels;
  Tensor * x = g + max_levels;
  Tensor * w = x + max_levels;
  Tensor * r = w + max_levels;

  /* ------------------------ FMG algorithm ------------------ */
  do_fmg(h, g, x, w, r, vx.data(), N, nb_cycles, 
         relax_, prolong_, restrict_g_, residuals_);

  return solution;
}


} // namespace ni

#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "common.h"
#include "../defines.h"
#include "../bounds.h"
#include "precond_common.h"
#include "precond_grid_common.h"
#include "regulariser_common.h"
#include "regulariser_grid_common.h"

using c10::IntArrayRef;
using c10::ArrayRef;
using at::Tensor;
using std::vector;

namespace ni {
NI_NAMESPACE_DEVICE {

namespace {

  inline Tensor init_solution(const Tensor & solution, const Tensor & gradient)
  {
    if (solution.defined() && solution.numel() > 0)
      return solution;
    return at::zeros_like(gradient);
  }

  inline Tensor dotprod(Tensor a, Tensor b)
  {
    int64_t dim = a.dim() - 2;
    a = a.reshape({a.size(0), 1, -1});
    b = b.reshape({b.size(0), -1, 1});
    a = at::matmul(a, b);
    while (a.dim() < dim+2) a = a.unsqueeze(-1);
    return a;
  }

  // Early stopping: same criterion as the TorchScript solver
  // (`core.optim.cg(..., stop='max_gain')`), which is invariant to the
  // scale of the system. With `obj_n = alpha * (alpha * pAp + 2 * rz)`
  // at iteration n (`rz` after the update), stop when
  //     |obj_{n-1} - obj_n| / (max_k obj_k - min_k obj_k) < tol
  // (The previous criterion, `sum(obj_n) < tol * numel`, was absolute: it
  //  stopped after a few iterations when the gradient and Hessian are
  //  small, e.g., with the normalised factors of `nitorch register`.)
  template <typename ForwardFn, typename PrecondFn>
  inline void do_pcg(const Tensor & h, const Tensor & g, 
                     Tensor & x, const Tensor & w,
                     int64_t nb_iter, double tol,
                     ForwardFn forward, PrecondFn precond)
  {
      Tensor alpha, beta, rz, rz0, pAp; // "scalar" tensors (can have a batch dim)
      Tensor r = at::empty_like(g), z = at::empty_like(g);

      // objective values (for early stopping)
      double obj = 0, obj_prev = std::numeric_limits<double>::infinity();
      double obj_min = std::numeric_limits<double>::infinity();
      double obj_max = -std::numeric_limits<double>::infinity();

      // Initialisation
      forward(h, x, w, r);       // r  = (H + KWK) * x
      at::sub_out(r, g, r);      // r  = g - r
      precond(h, r, w, z);       // z  = (H + diag(W)) \ r
      rz = dotprod(r, z);        // rz = r' * z
      Tensor p = z.clone();      // Initial conjugate directions p

      int64_t n = 0;
      for (;  n < nb_iter; ++n)
      {
        forward(h, p, w, z);                     // Ap = (H + KWK) * p
        pAp = dotprod(p, z);                     // alpha = p' * Ap
        alpha = rz / at::clamp_min(pAp, 1e-12);  // alpha = (r' * z) / (p' * Ap)

        at::addcmul_out(x, x, p, alpha);          // x += alpha * p
        at::addcmul_out(r, r, z, alpha, -1);      // r -= alpha * Ap
        precond(h, r, w, z);                      // z  = (H + diag(W)) \ r
        rz0 = rz;
        rz = dotprod(r, z);
        beta = rz / at::clamp_min_(rz0, 1e-12);

        if (tol) {
          obj = at::sum(alpha * (alpha * pAp + 2 * rz)).item<double>();
          if (std::isfinite(obj)) {
            obj_min = std::min(obj_min, obj);
            obj_max = std::max(obj_max, obj);
          }
          double gain = (obj_prev - obj)
                      / std::max(obj_max - obj_min, 1e-12);
          obj_prev = obj;
          if (std::abs(gain) < tol) {
            ++n;
            break;
          }
        }

        at::addcmul_out(p, z, p, beta);           // p = z + beta * p
      }

      if (tol)
        NI_TRACE("PCG: %d/%d, obj = %f, tol = %f\n", 
                 n, nb_iter, obj, tol);
 
  }

} // anonymous namespace


Tensor pcg_impl(const Tensor & hessian, 
                const Tensor & gradient,
                      Tensor   solution,
                const Tensor & weight,
                const ArrayRef<double> &  absolute, 
                const ArrayRef<double> &  membrane, 
                const ArrayRef<double> &  bending,
                const ArrayRef<double> &  voxel_size, 
                const BoundVectorRef   & bound,
                int64_t nb_iter, double tol)
{

  /* ---------------- function handles ---------------------- */
  auto forward_ = [&]
                  (const Tensor & hessian, const Tensor & input,
                   const Tensor & weight,  const Tensor & output)
  {
    regulariser_impl(input, output, weight, hessian,
                     absolute, membrane, bending, voxel_size, bound);
  };
  auto precond_ = [&]
                  (const Tensor & hessian, const Tensor & gradient,
                   const Tensor & weight, const Tensor & output)
  {
    precond_impl(hessian, gradient, output, weight, 
                 absolute, membrane, bending, voxel_size, bound);
  };

  /* ------------------------ PCG algorithm ------------------ */
  solution = init_solution(solution, gradient);
  do_pcg(hessian, gradient, solution, weight, nb_iter, tol, forward_, precond_);

  return solution;
}

Tensor pcg_grid_impl(
           const Tensor & hessian, 
           const Tensor & gradient,
                 Tensor   solution,
           const Tensor & weight,
           double  absolute, 
           double  membrane, 
           double  bending,
           double  lame_shear,
           double  lame_div,
           const ArrayRef<double> &  voxel_size, 
           const BoundVectorRef   & bound,
           int64_t nb_iter, double tol)
{

  /* ---------------- function handles ---------------------- */
  auto forward_ = [&]
                  (const Tensor & hessian, const Tensor & input,
                   const Tensor & weight,  const Tensor & output)
  {
    regulariser_grid_impl(input, output, weight, hessian,
                          absolute, membrane, bending, lame_shear, lame_div, 
                          voxel_size, bound);
  };
  auto precond_ = [&]
                  (const Tensor & hessian, const Tensor & gradient,
                   const Tensor & weight, const Tensor & output)
  {
    precond_grid_impl(hessian, gradient, output, weight, 
                      absolute, membrane, bending, lame_shear, lame_div, 
                      voxel_size, bound);
  };

  /* ------------------------ PCG algorithm ------------------ */
  solution = init_solution(solution, gradient);
  do_pcg(hessian, gradient, solution, weight, nb_iter, tol, forward_, precond_);

  return solution;
}

} // namespace device

namespace notimplemented {
  Tensor pcg_impl(const Tensor & hessian, 
                  const Tensor & gradient,
                        Tensor   solution,
                  const Tensor & weight,
                  const ArrayRef<double> &  absolute, 
                  const ArrayRef<double> &  membrane, 
                  const ArrayRef<double> &  bending,
                  const ArrayRef<double> &  voxel_size, 
                  const BoundVectorRef   & bound,
                  int64_t nb_iter, double tol) 
  {
    return solution;
  }

  Tensor pcg_grid_impl(
             const Tensor & hessian, 
             const Tensor & gradient,
                   Tensor   solution,
             const Tensor & weight,
             double  absolute, 
             double  membrane, 
             double  bending,
             double  lame_shear,
             double  lame_div,
             const ArrayRef<double> &  voxel_size, 
             const BoundVectorRef   & bound,
             int64_t nb_iter, double tol) 
  {
    return solution;
  }
}

} // namespace ni

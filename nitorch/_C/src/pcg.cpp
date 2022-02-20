#include <ATen/ATen.h>
#include <vector>
#include "defines.h"
#include "bounds.h"
#include "precond.h"
#include "regulariser.h"
#include "regulariser_grid.h"

using c10::IntArrayRef;
using at::Tensor;
using std::vector;

namespace ni {

namespace {

  inline Tensor init_solution(const Tensor & solution, const Tensor & gradient)
  {
    if (solution.defined() && solution.numel() > 0)
      return solution;
    return at::zeros_like(gradient);
  }

  inline Tensor dot(Tensor a, Tensor b)
  {
    int64_t dim = a.dim() - 2;
    a = a.reshape({a.size(0), 1, -1});
    b = b.reshape({b.size(0), -1, 1});
    a = at::matmul(a, b);
    while (a.dim() < dim+2) a = a.unsqueeze(-1);
    return a;
  }

  template <typename ForwardFn, typename PrecondFn>
  inline void do_pcg(const Tensor & h, const Tensor & g, 
                     Tensor & x, const Tensor & w,
                     int64_t nb_iter, ForwardFn forward, PrecondFn precond)
  {
      Tensor alpha, beta, rz, rz0; // "scalar" tensors (can have a batch dim)
      Tensor r = at::empty_like(g), z = at::empty_like(g);

      // Initialisation
      forward(h, x, w, r);       // r  = (H + KWK) * x
      at::sub_out(r, g, r);      // r  = g - r
      precond(h, r, w, z);       // z  = (H + diag(W)) \ r
      rz = ni::dot(r, z);        // rz = r' * z
      Tensor p = z.clone();      // Initial conjugate directions p

      for (int64_t n = 0;  n < nb_iter; ++n)
      {
        forward(h, p, w, z);                      // Ap = (H + KWK) * p
        alpha = ni::dot(p, z);                    // alpha = p' * Ap
        alpha = rz / at::clamp_min(alpha, 1e-12); // alpha = (r' * z) / (p' * Ap)
        at::addcmul_out(x, x, p, alpha);          // x += alpha * p
        at::addcmul_out(r, r, z, alpha, -1);      // r -= alpha * Ap
        precond(h, r, w, z);                      // z  = (H + diag(W)) \ r
        rz0 = rz;
        rz = ni::dot(r, z);
        beta = rz / at::clamp_min_(rz0, 1e-12);
        at::addcmul_out(p, z, p, beta);           // p = z + beta * p
      }
  }

}


Tensor pcg(const Tensor & hessian, 
           const Tensor & gradient,
                 Tensor   solution,
           const Tensor & weight,
           const vector<double> &  absolute, 
           const vector<double> &  membrane, 
           const vector<double> &  bending,
           const vector<double> &  voxel_size, 
           const vector<BoundType> & bound,
           int64_t nb_iter)
{

  /* ---------------- function handles ---------------------- */
  auto forward_ = [absolute, membrane, bending, bound, voxel_size]
                  (const Tensor & hessian, const Tensor & input,
                   const Tensor & weight,  const Tensor & output)
  {
    regulariser(input, output, weight, hessian,
                absolute, membrane, bending, voxel_size, bound);
  };
  auto precond_ = [absolute, membrane, bending, bound, voxel_size]
                  (const Tensor & hessian, const Tensor & gradient,
                   const Tensor & weight, const Tensor & output)
  {
    precond(hessian, gradient, output, weight, 
            absolute, membrane, bending, voxel_size, bound);
  };

  /* ------------------------ PCG algorithm ------------------ */
  solution = init_solution(solution, gradient);
  do_pcg(hessian, gradient, solution, weight, nb_iter, forward_, precond_);

  return solution;
}

}

/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <torch/torch.h>
#include "libmesh/int_range.h"

#define pti(tensor) MooseTensor::printTensorInfo(#tensor, tensor)

#define pez(tensor) MooseTensor::printElementZero(#tensor, tensor)

namespace MooseTensor
{

/// Passkey pattern key template
template <typename T>
class Key
{
  friend T;
  Key() {}
  Key(Key const &) {}
};

void printTensorInfo(const torch::Tensor & x);
void printTensorInfo(const std::string & name, const torch::Tensor & x);

void printElementZero(const torch::Tensor & tensor);
void printElementZero(const std::string & name, const torch::Tensor & tensor);

void printBuffer(const torch::Tensor & t,
                 const unsigned int & precision = 5,
                 const unsigned int & index = 0);

const torch::TensorOptions floatTensorOptions();
const torch::TensorOptions complexFloatTensorOptions();
const torch::TensorOptions intTensorOptions();

/// unsqueeze(0) ndim times
torch::Tensor unsqueeze0(const torch::Tensor & t, unsigned int ndim);

torch::Tensor trans2(const torch::Tensor & A2);
torch::Tensor ddot42(const torch::Tensor & A4, const torch::Tensor & B2);
torch::Tensor ddot44(const torch::Tensor & A4, const torch::Tensor & B4);
torch::Tensor dot22(const torch::Tensor & A2, const torch::Tensor & B2);
torch::Tensor dot24(const torch::Tensor & A2, const torch::Tensor & B4);
torch::Tensor dot42(const torch::Tensor & A4, const torch::Tensor & B2);
torch::Tensor dyad22(const torch::Tensor & A2, const torch::Tensor & B2);

template <typename T1, typename T2>
std::tuple<torch::Tensor, unsigned int, double>
conjugateGradientSolve(T1 A, torch::Tensor b, torch::Tensor x0, double tol, int64_t maxiter, T2 M)
{
  // initialize solution guess
  torch::Tensor x = x0.defined() ? x0.clone() : torch::zeros_like(b);

  // norm of b (for relative tolerance)
  const double b_norm = torch::norm(b).cpu().template item<double>();
  if (b_norm == 0.0)
    // solution is zero if b is zero
    return {x, 0u, 0.0};

  // default max iterations
  if (!maxiter)
    maxiter = b.numel();

  // initial residual
  torch::Tensor r = b - A(x);

  // Apply preconditioner (or identity)
  torch::Tensor z = M(r); // z = M^{-1} r

  // initial search direction p
  torch::Tensor p = z.clone();

  // dot product (r, z)
  double rz_old = torch::sum(r * z).cpu().template item<double>();

  // CG iteration
  double res_norm;
  for (const auto k : libMesh::make_range(maxiter))
  {
    // compute matrix-vector product
    const auto Ap = A(p);

    // step size alpha
    double alpha = rz_old / torch::sum(p * Ap).cpu().template item<double>();

    // update solution
    x = x + alpha * p;

    // update residual
    r = r - alpha * Ap;
    res_norm = torch::norm(r).cpu().template item<double>(); // ||r||

    // std::cout << res_norm << '\n';

    // Converged to desired tolerance
    if (res_norm <= tol * b_norm)
      return {x, k + 1, res_norm};

    // apply preconditioner to new residual
    z = M(r);
    const auto rz_new = torch::sum(r * z).cpu().template item<double>();

    // update scalar beta
    double beta = rz_new / rz_old;

    // update search direction
    p = z + beta * p;

    // prepare for next iteration
    rz_old = rz_new;
  }

  // Reached max iterations without full convergence
  return {x, maxiter, res_norm};
}

template <typename T>
std::tuple<torch::Tensor, unsigned int, double>
conjugateGradientSolve(
    T A, torch::Tensor b, torch::Tensor x0 = {}, double tol = 1e-6, int64_t maxiter = 0)
{
  return conjugateGradientSolve(A, b, x0, tol, maxiter, [](const torch::Tensor r) { return r; });
}

} // namespace MooseTensor

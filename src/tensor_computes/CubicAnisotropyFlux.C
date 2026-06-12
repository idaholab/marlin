/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "CubicAnisotropyFlux.h"

registerMooseObject("MarlinApp", CubicAnisotropyFlux);

InputParameters
CubicAnisotropyFlux::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Compute the full anisotropic gradient energy flux for cubic symmetry, "
      "including the corner correction term essential for dendritic growth.");
  params.addRequiredParam<TensorInputBufferName>("gradient",
                                                  "Gradient vector buffer (value_dimensions = '3')");
  params.addRequiredParam<TensorInputBufferName>("rotation",
                                                  "Rotation matrix buffer (value_dimensions = '3 3')");
  params.addRequiredParam<Real>("eps_a", "Anisotropy strength parameter");
  params.addParam<Real>("eps_n", 1e-8, "Small regularization to avoid division by zero");
  return params;
}

CubicAnisotropyFlux::CubicAnisotropyFlux(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _gradient(getInputBuffer("gradient")),
    _rotation(getInputBuffer("rotation")),
    _eps_a(getParam<Real>("eps_a")),
    _eps_n(getParam<Real>("eps_n"))
{
}

void
CubicAnisotropyFlux::computeBuffer()
{
  // Validate input dimensions
  if (_gradient.dim() < static_cast<int64_t>(_dim) + 1 || _gradient.size(_dim) != 3)
    mooseError("CubicAnisotropyFlux: gradient must have value_dimensions = '3'");
  if (_rotation.dim() < static_cast<int64_t>(_dim) + 2 || _rotation.size(_dim) != 3 ||
      _rotation.size(_dim + 1) != 3)
    mooseError("CubicAnisotropyFlux: rotation must have value_dimensions = '3 3'");
  if (_u.dim() < static_cast<int64_t>(_dim) + 1 || _u.size(_dim) != 3)
    mooseError("CubicAnisotropyFlux: output buffer must have value_dimensions = '3'");

  // Step 1: Rotate gradient to crystal frame
  // q = R^T * grad, where R is [... x 3 x 3] and grad is [... x 3]
  // We compute q_j = sum_i R_ij * grad_i = (R^T * grad)_j
  auto grad_col = _gradient.unsqueeze(-1);               // [... x 3 x 1]
  auto rot_t = _rotation.transpose(-2, -1);              // R^T: [... x 3 x 3]
  auto q = torch::matmul(rot_t, grad_col).squeeze(-1);   // [... x 3]

  // Step 2: Compute anisotropy function components
  // q2 = |q|^2 = sum(q_i^2)
  // q4 = sum(q_i^4) (4-fold cubic term)
  auto q_sq = q * q;                         // [... x 3]
  auto q2 = q_sq.sum(-1, /*keepdim=*/true);  // [... x 1]
  auto q4 = (q_sq * q_sq).sum(-1, /*keepdim=*/true);  // [... x 1]

  // d = sqrt(q2) + eps_n (regularized magnitude)
  auto sqrt_q2 = torch::sqrt(q2 + _eps_n * _eps_n);  // [... x 1]
  auto d = sqrt_q2 + _eps_n;                          // [... x 1]
  auto d4 = d * d * d * d;                            // [... x 1]

  // a = 1 + eps_a * (q4/d^4 - 0.6)
  // The 0.6 offset makes a=1 isotropic for cubic in 3D
  auto a = 1.0 + _eps_a * (q4 / d4 - 0.6);  // [... x 1]
  auto a2 = a * a;                           // [... x 1]
  auto a2m1 = a2 - 1.0;                      // a^2 - 1: [... x 1]

  // Step 3: Compute the standard flux term (a^2 - 1) * grad
  auto flux_standard = a2m1 * _gradient;  // [... x 3]

  // Step 4: Compute the corner correction term
  // This is: (1/2) * |grad|^2 * d(a^2)/d(grad)
  //
  // d(a^2)/d(grad) = 2*a * d(a)/d(grad)
  // d(a)/d(grad) = eps_a * d(q4/d^4)/d(grad)
  //
  // Using chain rule through q = R^T * grad:
  // d(a)/d(grad)_i = eps_a * sum_j [d(q4/d^4)/d(q_j)] * R_ij
  //
  // d(q4/d^4)/d(q_j) = 4*q_j^3/d^4 - 4*q4*q_j/(d^5 * sqrt(q2))
  //                  = (4/d^4) * [q_j^3 - q4*q_j/(d*sqrt(q2))]
  //
  // Define w_j = q_j^3 - q4*q_j/(d*sqrt(q2))
  // Then: d(a)/d(grad) = eps_a * (4/d^4) * R * w
  // And:  d(a^2)/d(grad) = 2*a * eps_a * (4/d^4) * R * w = 8*a*eps_a/d^4 * R * w
  //
  // The correction flux is:
  // F_corr = (1/2) * |grad|^2 * d(a^2)/d(grad)
  //        = (1/2) * q2 * 8*a*eps_a/d^4 * R * w
  //        = 4 * a * eps_a * q2 / d^4 * R * w

  // Compute w = q^3 - (q4/(d*sqrt_q2)) * q
  auto q_cubed = q * q * q;                           // [... x 3]
  auto coeff = q4 / (d * sqrt_q2 + _eps_n);           // [... x 1]
  auto w = q_cubed - coeff * q;                       // [... x 3]

  // Compute R * w (rotate back to lab frame)
  auto w_col = w.unsqueeze(-1);                       // [... x 3 x 1]
  auto Rw = torch::matmul(_rotation, w_col).squeeze(-1);  // [... x 3]

  // Compute correction flux factor: 4 * a * eps_a * q2 / d^4
  auto corr_factor = 4.0 * _eps_a * a * q2 / d4;     // [... x 1]

  // Corner correction flux
  auto flux_correction = corr_factor * Rw;           // [... x 3]

  // Step 5: Total flux = standard + correction
  _u = flux_standard + flux_correction;
}

/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMPressureCorrectedEquilibrium.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMPressureCorrectedEquilibrium);

InputParameters
LBMPressureCorrectedEquilibrium::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription(
      "Compute object for correcting the equilibrium distribution function for phase field model.");
  params.addRequiredParam<TensorInputBufferName>("rho", "LBM density");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM fluid velocity");
  params.addRequiredParam<TensorInputBufferName>("pressure",
                                                 "Pressure computed from phase field model");
  return params;
}

LBMPressureCorrectedEquilibrium::LBMPressureCorrectedEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _rho(getInputBuffer("rho", _radius)),
    _velocity(getInputBuffer("velocity", _radius)),
    _pressure(getInputBuffer("pressure", _radius))
{
}

void
LBMPressureCorrectedEquilibrium::computeBuffer()
{
  const unsigned int dim = _domain.getDim();

  const int64_t N = _u.numel() / _stencil._q;
  const int Q = _stencil._q;

  auto vel_flat = _velocity.slice(/*dim=*/-1, /*start=*/0, /*end=*/dim).reshape({N, dim});
  auto rho_flat = _rho.reshape({N, 1});
  auto p_flat = _pressure.reshape({N, 1});
  auto u_flat = _u.view({N, Q});
  auto w_flat = _w.view({1, Q});

  auto edotu = torch::matmul(vel_flat, _e_mat.t());                // [N, Q]
  auto usqr = vel_flat.square().sum(/*dim=*/-1, /*keepdim=*/true); // [N, 1]

  // rho * s_i(u) directly into u_flat
  u_flat.copy_(edotu).square_().div_(2.0 * _lb_problem._cs4);
  u_flat.add_(edotu, /*alpha=*/1.0 / _lb_problem._cs2);
  u_flat.sub_(usqr, /*alpha=*/1.0 / (2.0 * _lb_problem._cs2));
  u_flat.mul_(w_flat);
  u_flat.mul_(rho_flat);

  // pressure term: (p / cs2) * w_i
  auto p_scaled = p_flat / _lb_problem._cs2;

  // addcmul_ computes: u_flat + (p_scaled * w_flat) without allocating the [N, Q] intermediate!
  u_flat.addcmul_(p_scaled, w_flat, /*value=*/1.0);

  // correct i = 0: subtract p / cs2
  u_flat.select(/*dim=*/1, /*index=*/0).sub_(p_scaled.squeeze(-1));

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

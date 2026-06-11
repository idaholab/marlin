/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMPhaseFieldPressure.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMPhaseFieldPressure);

InputParameters
LBMPhaseFieldPressure::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute pressure for phase field model.");
  params.addRequiredParam<TensorInputBufferName>("f", "LBM distribution function");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM fluid velocity");
  params.addRequiredParam<TensorInputBufferName>("grad_phi",
                                                 "Gradient of LBM phase field parameter");
  params.addRequiredParam<TensorInputBufferName>("rho", "LBM density");
  params.addRequiredParam<std::string>("rho_l", "Density of the liquid (high density) phase");
  params.addRequiredParam<std::string>("rho_g", "Density of the gas (low density) phase");

  return params;
}

LBMPhaseFieldPressure::LBMPhaseFieldPressure(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _f(getInputBuffer("f", _radius)),
    _velocity(getInputBuffer("velocity", _radius)),
    _grad_phi(getInputBuffer("grad_phi", _radius)),
    _rho(getInputBuffer("rho", _radius)),
    _rho_l(_lb_problem.getConstant<Real>(getParam<std::string>("rho_l"))),
    _rho_g(_lb_problem.getConstant<Real>(getParam<std::string>("rho_g")))
{
}

void
LBMPhaseFieldPressure::computeBuffer()
{
  const int64_t N = _u.numel();
  const int Q = _f.size(-1);

  auto f_flat = _f.view({N, Q});
  auto vel_flat = _velocity.view({N, _velocity.size(-1)});
  auto grad_phi_flat = _grad_phi.view({N, _grad_phi.size(-1)});
  auto rho_flat = _rho.view({N});
  auto u_flat = _u.view({N});

  auto f_nonzero_sum = torch::sum(f_flat.slice(-1, 1, Q), -1);
  auto u_dot_grad_phi = torch::sum(vel_flat * grad_phi_flat, -1);
  auto usqr = vel_flat.square().sum(-1);

  auto w0 = _stencil._weights[0].item<Real>();

  // assembly
  u_flat.copy_(f_nonzero_sum);
  u_flat.add_(u_dot_grad_phi, /*alpha=*/0.5 * (_rho_l - _rho_g));
  usqr.mul_(rho_flat); // [N] * [N] -> [N]
  u_flat.add_(usqr, /*alpha=*/-0.5 * w0 / _lb_problem._cs2);
  u_flat.mul_(_lb_problem._cs2 / (1.0 - w0));

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

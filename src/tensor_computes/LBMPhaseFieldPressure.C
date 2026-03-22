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
  // p = \frac{c_s^2}{(1 - \omega_0)} \left[ \sum_{i \neq 0} g_i +
  // \frac{\delta_t}{2}(\rho_l - \rho_g)\mathbf{u} \cdot \nabla\phi +
  // \rho s_0(\mathbf{u}) \right]
  if (_rho.dim() < 3)
    _rho.unsqueeze_(2);

  auto f_nonzero_sum = torch::sum(_f.slice(-1, 1, _f.size(-1)), -1);
  auto u_dot_grad_phi = torch::sum(_velocity * _grad_phi, -1);
  auto usqr = torch::sum(_velocity * _velocity, -1);
  auto rho_s_0 = _stencil._weights[0].item<Real>() * _rho * (-0.5 * usqr / _lb_problem._cs2);

  _u = (f_nonzero_sum + 0.5 * (_rho_l - _rho_g) * u_dot_grad_phi + rho_s_0) * _lb_problem._cs2 /
       (1.0 - _stencil._weights[0]);

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

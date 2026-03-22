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
  // g_i^{eq} = p / cs2 * (w_0 - 1) + rho * s_0(u),   i = 0
  // g_i^{eq} = p / cs2 * w_i       + rho * s_i(u),    i != 0
  // where s_i(u) = kind_of_eq_i / rho

  const unsigned int & dim = _domain.getDim();

  if (_rho.dim() < 3)
    _rho.unsqueeze_(2);

  // Compute s_i(u) inline
  torch::Tensor rho_unsqueezed = _rho.unsqueeze(-1);
  torch::Tensor ux = _velocity.select(-1, 0).unsqueeze(-1);
  torch::Tensor uy = _velocity.select(-1, 1).unsqueeze(-1);
  torch::Tensor uz;

  switch (dim)
  {
    case 3:
      uz = _velocity.select(-1, 2).unsqueeze(-1);
      break;
    case 2:
      uz = torch::zeros_like(rho_unsqueezed, MooseTensor::floatTensorOptions());
      break;
    default:
      mooseError("Unsupported dimensions for LBMPressureCorrectedEquilibrium");
  }

  torch::Tensor second_order;
  torch::Tensor third_order;
  {
    auto edotu = _ex * ux + _ey * uy + _ez * uz;
    auto edotu_sqr = edotu * edotu;
    auto usqr = ux * ux + uy * uy + uz * uz;
    second_order = edotu / _lb_problem._cs2 + 0.5 * edotu_sqr / _lb_problem._cs4;
    third_order = 0.5 * usqr / _lb_problem._cs2;
  }
  torch::Tensor rho_s_i = _w * rho_unsqueezed * (second_order - third_order);

  // g_i^eq = p / cs2 * w_i + rho_s_i  for all i
  _u = _pressure.unsqueeze(-1) / _lb_problem._cs2 * _w + rho_s_i;

  // Correct i = 0: subtract p / cs2 to get p / cs2 * (w_0 - 1) instead of p / cs2 * w_0
  _u.select(-1, 0) -= _pressure / _lb_problem._cs2;

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

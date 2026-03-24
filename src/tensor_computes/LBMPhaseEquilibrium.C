/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMPhaseEquilibrium.h"

registerMooseObject("MarlinApp", LBMPhaseEquilibrium);

InputParameters
LBMPhaseEquilibrium::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription(
      "Compute LB equilibrium distribution object for phase field parameter.");
  params.addRequiredParam<TensorInputBufferName>("phi", "LBM phase field parameter");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM fluid velocity");

  return params;
}

LBMPhaseEquilibrium::LBMPhaseEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _phi(getInputBuffer("phi", _radius)),
    _velocity(getInputBuffer("velocity", _radius))
{
}

void
LBMPhaseEquilibrium::computeBuffer()
{
  const unsigned int & dim = _domain.getDim();
  if (_phi.dim() < 3)
    _phi.unsqueeze_(2);

  torch::Tensor phi_unsqueezed = _phi.unsqueeze(-1);
  torch::Tensor ux = _velocity.select(-1, 0).unsqueeze(-1);
  torch::Tensor uy = _velocity.select(-1, 1).unsqueeze(-1);
  torch::Tensor uz;
  switch (dim)
  {
    case 3:
      uz = _velocity.select(-1, 2).unsqueeze(-1);
      break;
    case 2:
      uz = torch::zeros_like(phi_unsqueezed, MooseTensor::floatTensorOptions());
      break;
    default:
      mooseError("Unsupported dimension for LBMPhaseEquilibrium");
  }

  torch::Tensor ci_dot_u = _ex * ux + _ey * uy + _ez * uz;
  _u = _w * phi_unsqueezed * (1.0 + ci_dot_u / _lb_problem._cs2);

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

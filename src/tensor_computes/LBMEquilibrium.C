/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMEquilibrium.h"

registerMooseObject("MarlinApp", LBMEquilibrium);

InputParameters
LBMEquilibrium::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute LB equilibrium distribution object.");
  params.addRequiredParam<TensorInputBufferName>(
      "bulk", "LBM bluk macroscpic parameter, e.g density or temperature");
  params.addRequiredParam<TensorInputBufferName>("velocity",
                                                 "LBM Velocty in x, y and z directions");
  return params;
}

LBMEquilibrium::LBMEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _rho(getInputBuffer("bulk", _radius)),
    _velocity(getInputBuffer("velocity", _radius))
{
}

void
LBMEquilibrium::computeBuffer()
{
  // prepping
  const unsigned int & dim = _domain.getDim();

  if (_rho.dim() < 3)
    _rho.unsqueeze_(2);

  torch::Tensor rho_unsqueezed = _rho.unsqueeze(3);
  torch::Tensor ux = _velocity.select(3, 0).unsqueeze(3);
  torch::Tensor uy = _velocity.select(3, 1).unsqueeze(3);
  torch::Tensor uz;

  switch (dim)
  {
    case 3:
      uz = _velocity.select(3, 2).unsqueeze(3);
      break;
    case 2:
      uz = torch::zeros_like(rho_unsqueezed, MooseTensor::floatTensorOptions());
      break;
    default:
      mooseError("Unsupported dimensions for buffer _u");
  }

  // compute equilibrium
  torch::Tensor second_order;
  torch::Tensor third_order;

  {
    auto edotu = _ex * ux + _ey * uy + _ez * uz;
    auto edotu_sqr = edotu * edotu;
    auto usqr = ux * ux + uy * uy + uz * uz;
    second_order = edotu / _lb_problem._cs2 + 0.5 * edotu_sqr / _lb_problem._cs4;
    third_order = 0.5 * usqr / _lb_problem._cs2;
  }

  _u = _w * rho_unsqueezed * (1.0 + second_order - third_order);
  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

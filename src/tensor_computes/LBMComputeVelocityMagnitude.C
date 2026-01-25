/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeVelocityMagnitude.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("MarlinApp", LBMComputeVelocityMagnitude);

InputParameters
LBMComputeVelocityMagnitude::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("LBMComputeVelocityMagnitude object.");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM velocity");
  return params;
}

LBMComputeVelocityMagnitude::LBMComputeVelocityMagnitude(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), _velocity(getInputBuffer("velocity", _radius))
{
}

void
LBMComputeVelocityMagnitude::computeBuffer()
{
  const unsigned int & dim = _domain.getDim();
  switch (dim)
  {
    case 2:
      _u = torch::sqrt(_velocity.select(3, 0).pow(2) + _velocity.select(3, 1).pow(2));
      break;
    case 3:
      _u = torch::sqrt(_velocity.select(3, 0).pow(2) + _velocity.select(3, 1).pow(2) +
                       _velocity.select(3, 2).pow(2));
      break;
    default:
      mooseError("Unsupported dimension");
  }
  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

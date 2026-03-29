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
  auto active_velocity = _velocity.narrow(3, 0, _domain.getDim());
  _u.copy_(torch::norm(active_velocity, /*p=*/2, /*dim=*/3));

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

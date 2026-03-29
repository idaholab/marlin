/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeResidual.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("MarlinApp", LBMComputeResidual);

InputParameters
LBMComputeResidual::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute object for LBM residual.");
  params.addRequiredParam<TensorInputBufferName>("speed", "LB speed");
  return params;
}

LBMComputeResidual::LBMComputeResidual(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), _speed(getInputBuffer("speed", _radius))
{
}

void
LBMComputeResidual::computeBuffer()
{
  auto speed_owned = ownedView(_speed);

  if (_speed_previous.numel() == 0)
  {
    _speed_previous = torch::empty_like(speed_owned);
    _speed_previous.copy_(speed_owned);
    _lb_problem.setSolverResidual(1.0);
  }
  else
  {
    Real sumUsquare = speed_owned.sum().item<Real>();

    // in-place absolute difference avoids temporary tensor allocations
    _speed_previous.sub_(speed_owned).abs_();
    Real sumUsqareMinusUsqareOld = _speed_previous.sum().item<Real>();

    _domain.comm().sum(sumUsqareMinusUsqareOld);
    _domain.comm().sum(sumUsquare);

    Real residual = (sumUsquare == 0 || sumUsqareMinusUsqareOld == 0)
                        ? 1.0
                        : sumUsqareMinusUsqareOld / sumUsquare;

    _lb_problem.setSolverResidual(residual);

    // zero-allocation save for the next substep
    _speed_previous.copy_(speed_owned);
  }
}

/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
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
  : LatticeBoltzmannOperator(parameters),
    _speed(getInputBuffer("speed")),
    _speed_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("speed"), 1))
{
}

void
LBMComputeResidual::computeBuffer()
{
  const auto & n_old = _speed_old.size();
  if (n_old == 0)
  {
    Real residual = 1.0;
    _lb_problem.setSolverResidual(residual);
  }
  else
  {
    Real sumUsqareMinusUsqareOld = torch::sum(torch::abs(_speed - _speed_old[0])).item<Real>();
    Real sumUsquare = torch::sum(_speed).item<Real>();
    Real residual = (sumUsquare == 0 || sumUsqareMinusUsqareOld == 0)
                        ? 1.0
                        : sumUsqareMinusUsqareOld / sumUsquare;
    _lb_problem.setSolverResidual(residual);
  }
}

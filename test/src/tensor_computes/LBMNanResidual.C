/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMNanResidual.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("MarlinApp", LBMNanResidual);

InputParameters
LBMNanResidual::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Set a NaN residual at a specified step.");
  params.addRequiredParam<int>("step", "Global step to issue NaN residual.");
  return params;
}

LBMNanResidual::LBMNanResidual(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _nan_step(getParam<int>("step")),
    _step(_lb_problem.getTotalSteps())
{
}

void
LBMNanResidual::computeBuffer()
{
  _lb_problem.setSolverResidual(_nan_step < _step ? std::numeric_limits<Real>::quiet_NaN() : 1.0);
}

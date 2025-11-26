/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeDensity.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("MarlinApp", LBMComputeDensity);

InputParameters
LBMComputeDensity::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addClassDescription("Compute object for macroscopic density reconstruction.");
  return params;
}

LBMComputeDensity::LBMComputeDensity(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), _f(getInputBuffer("f"))
{
}

void
LBMComputeDensity::computeBuffer()
{
  _u = torch::sum(_f, 3);
  _lb_problem.maskedFillSolids(_u, 0);
}

/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ForwardEulerSolver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", ForwardEulerSolver);

InputParameters
ForwardEulerSolver::validParams()
{
  InputParameters params = ExplicitSolverBase::validParams();
  params.addClassDescription("Semi-implicit time integration solver.");
  return params;
}

ForwardEulerSolver::ForwardEulerSolver(const InputParameters & parameters)
  : ExplicitSolverBase(parameters)
{
}

void
ForwardEulerSolver::substep()
{
  // re-evaluate the solve compute
  _compute->computeBuffer();
  forwardBuffers();

  // integrate all variables
  for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
    u = _domain.ifft(reciprocal_buffer + _sub_dt * time_derivative_reciprocal);
}

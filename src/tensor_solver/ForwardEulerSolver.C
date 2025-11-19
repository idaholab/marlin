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
  params.addParam<unsigned int>("substeps", 1, "semi-implicit substeps per time step.");
  return params;
}

ForwardEulerSolver::ForwardEulerSolver(const InputParameters & parameters)
  : ExplicitSolverBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
}

void
ForwardEulerSolver::computeBuffer()
{
  torch::Tensor ubar;
  _sub_dt = _dt / _substeps;

  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    // re-evaluate the solve compute
    _compute->computeBuffer();
    forwardBuffers();

    // integrate all variables
    for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
    {
      ubar = reciprocal_buffer + _sub_dt * time_derivative_reciprocal;

      u = _domain.ifft(ubar);
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();

    // increment substep time
    _sub_time += _sub_dt;
  }
}

/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// MOOSE includes
#include "TensorSolveIterationAdaptiveDT.h"
#include "TensorProblem.h"
#include "IterativeTensorSolverInterface.h"
#include "TensorSolver.h"

#include <limits>

registerMooseObject("MooseApp", TensorSolveIterationAdaptiveDT);

InputParameters
TensorSolveIterationAdaptiveDT::validParams()
{
  InputParameters params = TimeStepper::validParams();
  params.addClassDescription(
      "Adjust the timestep based on the number of internal TensorSolve iterations");
  params.addParam<unsigned int>(
      "min_iterations",
      "If the solve takes less than 'min_iterations', dt is increased by 'growth_factor'");
  params.addParam<unsigned int>(
      "max_iterations",
      "If the solve takes more than 'max_iterations', dt is decreased by 'cutback_factor'");

  params.addParam<std::vector<PostprocessorName>>("timestep_limiting_postprocessor", {},
                                                  "If specified, a list of postprocessor values "
                                                  "used as an upper limit for the "
                                                  "current time step length");
  params.addRequiredParam<Real>("dt", "The default timestep size between solves");
  params.addParam<Real>("growth_factor",
                        2.0,
                        "Factor to apply to timestep if easy convergence or if recovering "
                        "from failed solve");
  params.addParam<Real>("cutback_factor",
                        0.5,
                        "Factor to apply to timestep if difficult convergence "
                        "occurs. "
                        "For failed solves, use cutback_factor_at_failure");

  params.declareControllable("growth_factor cutback_factor");

  return params;
}

TensorSolveIterationAdaptiveDT::TensorSolveIterationAdaptiveDT(const InputParameters & parameters)
  : TimeStepper(parameters),
    PostprocessorInterface(this),
    _dt_old(declareRestartableData<Real>("dt_old", 0.0)),
    _input_dt(getParam<Real>("dt")),
    _min_iterations(getParam<unsigned int>("min_iterations")),
    _max_iterations(getParam<unsigned int>("max_iterations")),
    _growth_factor(getParam<Real>("growth_factor")),
    _cutback_factor(getParam<Real>("cutback_factor")),
    _cutback_occurred(declareRestartableData<bool>("cutback_occurred", false)),
    _tensor_problem(TensorProblem::cast(this, _fe_problem))
{
  for (const auto & name :
       getParam<std::vector<PostprocessorName>>("timestep_limiting_postprocessor"))
    _pps_value.push_back(&getPostprocessorValueByName(name));
}

Real
TensorSolveIterationAdaptiveDT::computeInitialDT()
{
  return _input_dt;
}

Real
TensorSolveIterationAdaptiveDT::computeDT()
{
  Real dt = _dt_old;

  if (_cutback_occurred)
  {
    _cutback_occurred = false;

    // Don't allow it to grow this step, but shrink if needed
    computeAdaptiveDT(dt, /* allowToGrow = */ false);
  }
  else
    computeAdaptiveDT(dt);

  return dt;
}

bool
TensorSolveIterationAdaptiveDT::constrainStep(Real & dt)
{
  bool at_sync_point = TimeStepper::constrainStep(dt);

  // Limit the timestep to postprocessor value
  limitDTToPostprocessorValue(dt);

  return at_sync_point;
}

bool
TensorSolveIterationAdaptiveDT::converged() const
{
  const auto & iterative_solver = _tensor_problem.getSolver<IterativeTensorSolverInterface>();
  return iterative_solver.isConverged();
}

Real
TensorSolveIterationAdaptiveDT::computeFailedDT()
{
  _cutback_occurred = true;

  // Can't cut back any more
  if (_dt <= _dt_min)
    mooseError("Solve failed and timestep already at dtmin, cannot continue!");

  if (_verbose)
  {
    _console << "\nSolve failed with dt: " << std::setw(9) << _dt
             << "\nRetrying with reduced dt: " << std::setw(9) << _dt * _cutback_factor_at_failure
             << std::endl;
  }
  else
    _console << "\nSolve failed, cutting timestep." << std::endl;

  return _dt * _cutback_factor_at_failure;
}

void
TensorSolveIterationAdaptiveDT::limitDTToPostprocessorValue(Real & limitedDT) const
{
  if (_pps_value.empty() || _t_step <= 1)
    return;

  Real limiting_pps_value = *_pps_value[0];
  unsigned int i_min = 0;
  for (const auto i : index_range(_pps_value))
    if (*_pps_value[i] < limiting_pps_value)
    {
      limiting_pps_value = *_pps_value[i];
      i_min = i;
    }

  if (limitedDT > limiting_pps_value)
  {
    if (limiting_pps_value < 0)
      mooseWarning(
          "Negative timestep limiting postprocessor '" +
          getParam<std::vector<PostprocessorName>>("timestep_limiting_postprocessor")[i_min] +
          "': " + std::to_string(limiting_pps_value));
    limitedDT = std::max(_dt_min, limiting_pps_value);

    if (_verbose)
      _console << "Limiting dt to postprocessor value. dt = " << limitedDT << std::endl;
  }
}

void
TensorSolveIterationAdaptiveDT::computeAdaptiveDT(Real & dt, bool allowToGrow, bool allowToShrink)
{
  const auto & iterative_solver = _tensor_problem.getSolver<IterativeTensorSolverInterface>();
  const auto previous_iterations = iterative_solver.getIterations();

  if (allowToGrow && previous_iterations < _min_iterations)
    // Grow the timestep
    dt *= _growth_factor;
  else if (allowToShrink && previous_iterations > _max_iterations)
    // Shrink the timestep
    dt *= _cutback_factor;
}

void
TensorSolveIterationAdaptiveDT::acceptStep()
{
  TimeStepper::acceptStep();
  _dt_old = _dt;
}


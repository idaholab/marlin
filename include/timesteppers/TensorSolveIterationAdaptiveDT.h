/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TimeStepper.h"
#include "PostprocessorInterface.h"

class IterativeTensorSolverInterface;
class TensorProblem;

/**
 * Adjust the timestep based on the number of internal TensorSolve iterations.
 */
class TensorSolveIterationAdaptiveDT : public TimeStepper, public PostprocessorInterface
{
public:
  static InputParameters validParams();

  TensorSolveIterationAdaptiveDT(const InputParameters & parameters);

  virtual bool constrainStep(Real & dt) override;
  virtual void acceptStep() override;

protected:
  virtual Real computeInitialDT() override;
  virtual Real computeDT() override;
  virtual bool converged() const override;
  virtual Real computeFailedDT() override;

  void computeAdaptiveDT(Real & dt, bool allowToGrow = true, bool allowToShrink = true);
  void limitDTToPostprocessorValue(Real & limitedDT) const;

  Real & _dt_old;

  /// The dt from the input file.
  const Real _input_dt;

  /// Adapt the timestep to maintain this non-linear iteration count...
  const unsigned int _min_iterations;
  const unsigned int _max_iterations;

  /// if specified, the postprocessor values used to determine an upper limit for the time step length
  std::vector<const PostprocessorValue *> _pps_value;

  /// grow the timestep by this factor
  const Real & _growth_factor;
  /// cut the timestep by by this factor
  const Real & _cutback_factor;
  bool & _cutback_occurred;

  const TensorProblem & _tensor_problem;

  /// Number of iterations in previous solve
  unsigned int _previous_iterations;
};

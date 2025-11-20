/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "GeneralPostprocessor.h"

class ComputeGroup;
class TensorProblem;

/**
 * Get number of cumulative compute group executions
 */
class ComputeGroupExecutionCount : public GeneralPostprocessor
{
public:
  static InputParameters validParams();

  ComputeGroupExecutionCount(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override {}
  virtual void finalize() override {}
  virtual PostprocessorValue getValue() const override;

protected:
  TensorProblem & _tensor_problem;
  const ComputeGroup & _compute_group;
};

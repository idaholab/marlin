/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorSolver.h"

/**
 * TensorTimeIntegrator object (this is mostly a compute object)
 */
class SplitOperatorBase : public TensorSolver
{
public:
  static InputParameters validParams();

  SplitOperatorBase(const InputParameters & parameters);

protected:
  /// couple the variables (call from derived class)
  void getVariables(unsigned int history_size);

  struct Variable
  {
    torch::Tensor & _buffer;
    const torch::Tensor & _reciprocal_buffer;
    const torch::Tensor * _linear_reciprocal;
    const torch::Tensor & _nonlinear_reciprocal;
    const std::vector<torch::Tensor> & _old_nonlinear_reciprocal;
  };

  std::vector<Variable> _variables;
};

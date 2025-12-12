/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorSolver.h"

/**
 * Simple forward Euler integrator for real space finite difference solves
 */
class RealSpaceForwardEuler : public TensorSolver
{
public:
  static InputParameters validParams();

  RealSpaceForwardEuler(const InputParameters & parameters);

protected:
  struct Variable
  {
    torch::Tensor & _buffer;
    const torch::Tensor & _time_derivative;
  };

  virtual void substep();

  std::vector<Variable> _variables;
};

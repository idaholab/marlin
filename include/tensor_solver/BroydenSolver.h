/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SplitOperatorBase.h"
#include "IterativeTensorSolverInterface.h"

/**
 * BroydenSolver object
 */
class BroydenSolver : public SplitOperatorBase, public IterativeTensorSolverInterface
{
public:
  static InputParameters validParams();

  BroydenSolver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  void broydenSolve();

  unsigned int _substep;
  unsigned int _substeps;
  unsigned int _max_iterations;

  const Real _relative_tolerance;
  const Real _absolute_tolerance;

  /// approximation of the Jacobian inverse
  torch::Tensor _M;

  const bool _verbose;
  const Real _damping;
  const Real _eye_factor;
  unsigned int _dim;
  torch::TensorOptions _options;
};

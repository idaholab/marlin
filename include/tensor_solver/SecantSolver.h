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
 * SecantSolver object
 */
class SecantSolver : public SplitOperatorBase, public IterativeTensorSolverInterface
{
public:
  static InputParameters validParams();

  SecantSolver(const InputParameters & parameters);

protected:
  virtual void substep();

  unsigned int _max_iterations;

  const Real _relative_tolerance;
  const Real _absolute_tolerance;

  const bool _verbose;
  const Real _damping;
};

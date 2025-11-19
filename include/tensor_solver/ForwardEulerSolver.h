/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "ExplicitSolverBase.h"
#include "SplitOperatorBase.h"

/**
 * ForwardEulerSolver object
 */
class ForwardEulerSolver : public ExplicitSolverBase
{
public:
  static InputParameters validParams();

  ForwardEulerSolver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  unsigned int _substeps;
  Real & _sub_dt;
  Real & _sub_time;
};

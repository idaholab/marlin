/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SplitOperatorBase.h"

/**
 * Exponential Time Differencing Runge-Kutta solver (fourth order).
 */
class ETDRK4Solver : public SplitOperatorBase
{
public:
  static InputParameters validParams();

  ETDRK4Solver(const InputParameters & parameters);

protected:
  virtual void substep() override;
};

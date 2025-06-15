/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Current simulation time
 */
class TimeTensorCompute : public TensorOperator<>
{
public:
  static InputParameters validParams();

  TimeTensorCompute(const InputParameters & parameters);

  virtual void computeBuffer() override;
};

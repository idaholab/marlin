/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "FFTGradientBase.h"

/**
 * Gradient of a tensor field returned as a stacked torch tensor.
 */
class GradientVector : public FFTGradientBase<>
{
public:
  static InputParameters validParams();

  GradientVector(const InputParameters & parameters);

  virtual void computeBuffer() override;
};

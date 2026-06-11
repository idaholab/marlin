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
 * Tensor gradient component.
 */
class FFTGradient : public FFTGradientBase<>
{
public:
  static InputParameters validParams();

  FFTGradient(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  unsigned int _direction;
};

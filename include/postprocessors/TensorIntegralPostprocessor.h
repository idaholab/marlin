/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorAveragePostprocessor.h"

/**
 * Compute the integral of a Tensor buffer
 */
class TensorIntegralPostprocessor : public TensorAveragePostprocessor
{
public:
  static InputParameters validParams();

  TensorIntegralPostprocessor(const InputParameters & parameters);

  virtual void finalize() override;
  virtual PostprocessorValue getValue() const override;

protected:
  Real _integral;
};

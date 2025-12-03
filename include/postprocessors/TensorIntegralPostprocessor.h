/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorPostprocessor.h"

/**
 * Compute the integral of a Tensor buffer
 */
class TensorIntegralPostprocessor : public TensorPostprocessor
{
public:
  static InputParameters validParams();

  TensorIntegralPostprocessor(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override;
  virtual PostprocessorValue getValue() const override;

protected:
  Real _integral;
};

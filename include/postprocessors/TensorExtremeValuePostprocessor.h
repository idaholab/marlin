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
 * Compute the average of a Tensor buffer
 */
class TensorExtremeValuePostprocessor : public TensorPostprocessor
{
public:
  static InputParameters validParams();

  TensorExtremeValuePostprocessor(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override {}
  virtual PostprocessorValue getValue() const override;

protected:
  enum class ValueType
  {
    MIN,
    MAX
  } _value_type;

  Real _value;
};

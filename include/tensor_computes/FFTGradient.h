/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Constant Tensor
 */
class FFTGradient : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FFTGradient(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const bool _input_is_reciprocal;

  unsigned int _direction;

  /// imaginary unit i
  const torch::Tensor _i;
};

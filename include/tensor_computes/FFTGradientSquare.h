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
class FFTGradientSquare : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FFTGradientSquare(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const bool _input_is_reciprocal;
  const Real _factor;

  /// mesh dimension
  const unsigned int & _dim;

  /// imaginary unit i
  const torch::Tensor _i;
};

/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Compute the gradient of a scalar tensor and write into a vector tensor.
 */
class GradientVector : public TensorOperator<>
{
public:
  static InputParameters validParams();
  GradientVector(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const bool _input_is_reciprocal;
  const torch::Tensor _zero;
};

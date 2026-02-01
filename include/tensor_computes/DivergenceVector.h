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
 * Compute the divergence of a vector tensor.
 */
class DivergenceVector : public TensorOperator<>
{
public:
  static InputParameters validParams();
  DivergenceVector(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const Real _factor;
};

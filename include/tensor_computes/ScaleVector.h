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
 * Scale a vector tensor by a scalar field.
 */
class ScaleVector : public TensorOperator<>
{
public:
  static InputParameters validParams();
  ScaleVector(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _scalar;
  const torch::Tensor & _vector;
};

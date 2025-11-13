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
 * Compute updated displacements from the deformation gradient tensor
 */
class ComputeDisplacements : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ComputeDisplacements(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _deformation_gradient_tensor;
};

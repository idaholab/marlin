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
 * Compute vonMises stress
 */
class ComputeVonMisesStress : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ComputeVonMisesStress(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _stress;
};

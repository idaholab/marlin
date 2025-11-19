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
 * Sinusoidal IC
 */
class ReciprocalLaplacianSquareFactor : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ReciprocalLaplacianSquareFactor(const InputParameters & parameters);

  virtual void computeBuffer() override;

  const Real _factor;
  const torch::Tensor & _k2;
};

/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 * LBMComputeVelocityMagnitude object
 */
class LBMComputeVelocityMagnitude : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeVelocityMagnitude(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _velocity;
};

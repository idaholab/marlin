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
 * Compute object for macroscopic density reconstruction
 */
class LBMComputeDensity : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeDensity(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _f;
};

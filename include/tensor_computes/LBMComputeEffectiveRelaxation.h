/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 * Compute object for calculating effective relaxation times based on local pore size and Knudsen
 * number.
 */
class LBMComputeEffectiveRelaxation : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeEffectiveRelaxation(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _local_pore_size;
  const torch::Tensor & _local_Knudsen_number;

  const Real _mfp; // mean free path
  const Real _dx;  // domain resolution
  const Real _A2;  // second order slip boundary constant
  // precomputed Constants
  Real _C1;
  Real _C2;
};

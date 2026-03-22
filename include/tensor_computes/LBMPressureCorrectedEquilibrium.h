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
 * Compute object for correcting the equilibrium distribution function for phase field model.
 */
class LBMPressureCorrectedEquilibrium : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMPressureCorrectedEquilibrium(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _rho;
  const torch::Tensor & _velocity;
  const torch::Tensor & _pressure;
};

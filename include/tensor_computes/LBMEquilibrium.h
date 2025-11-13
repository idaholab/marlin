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
 * Compute LB equilibrium distribution
 */
class LBMEquilibrium : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMEquilibrium(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _rho;
  const torch::Tensor & _velocity;
};

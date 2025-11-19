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
 * Compute LB equilibrium distribution for phase field
 */
class LBMPhaseEquilibrium : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMPhaseEquilibrium(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _phi;
  const torch::Tensor & _grad_phi;
  const Real _tau_phi;
  const Real _D;
};

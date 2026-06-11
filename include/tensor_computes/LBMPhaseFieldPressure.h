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
 * Compute hydrodynamic pressure for phase field model.
 */
class LBMPhaseFieldPressure : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMPhaseFieldPressure(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _f;
  const torch::Tensor & _velocity;
  const torch::Tensor & _grad_phi;
  const torch::Tensor & _rho;

  const Real _rho_l;
  const Real _rho_g;
};

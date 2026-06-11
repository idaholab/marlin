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
 * Compute Allen-Cahn source term for phase field model.
 */
class LBMAllenCahnSource : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMAllenCahnSource(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  void computeSourceTerm();

  const torch::Tensor & _phi;
  const torch::Tensor & _velocity;
  const torch::Tensor & _grad_phi;

  torch::Tensor _phi_u_old;
  torch::Tensor _source_term;
  torch::Tensor _P_mat;

  const Real _tau;
  const Real _D;
};

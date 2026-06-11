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
 * Compute object for the force distribution function (source term) for phase field model.
 */
class LBMForceDistribution : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMForceDistribution(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  void computeSourceTerm();

  const torch::Tensor & _grad_phi;
  const torch::Tensor & _velocity;
  const torch::Tensor & _forces;
  const torch::Tensor & _tau_tensor;

  torch::Tensor _source_term;

  const Real _tau;
  const Real _rho_l;
  const Real _rho_g;
  const bool _is_dynamic_relaxation;
};

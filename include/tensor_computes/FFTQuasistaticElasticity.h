/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

/**
 * Monolithic mechanics solve for small strain elasticity
 */
class FFTQuasistaticElasticity : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  FFTQuasistaticElasticity(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  std::vector<torch::Tensor *> _displacements;
  const torch::Tensor _two_pi_i;
  const Real _mu;
  const Real _lambda;
  const Real _e0;
  const torch::Tensor & _cbar;
};

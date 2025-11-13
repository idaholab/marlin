/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Chemical potential for small strain elasticity volumetric Eigenstrain solute
 */
class FFTElasticChemicalPotential : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FFTElasticChemicalPotential(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  std::vector<const torch::Tensor *> _displacements;
  const torch::Tensor _two_pi_i;
  const Real _mu;
  const Real _lambda;
  const Real _e0;
  const torch::Tensor & _cbar;
};

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
 * Compute surface tension forces
 */
class LBMComputeSurfaceForces : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeSurfaceForces(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _chemical_potential;
  const torch::Tensor & _grad_phi;
};

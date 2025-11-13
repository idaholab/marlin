/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMIsotropicGradient.h"

/**
 * Compute gradient with isotropic discretization scheme
 */
class LBMIsotropicLaplacian : public LBMIsotropicGradient
{
public:
  static InputParameters validParams();

  LBMIsotropicLaplacian(const InputParameters & parameters);

  virtual void computeBuffer() override;
};

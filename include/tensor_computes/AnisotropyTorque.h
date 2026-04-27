/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "FFTGradientBase.h"

/**
 * Compute the anisotropy torque from dmu_dn and a scalar weighting field g.
 */
class AnisotropyTorque : public FFTGradientBase<>
{
public:
  static InputParameters validParams();

  AnisotropyTorque(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _dmu_dn;
  const torch::Tensor & _g;
};

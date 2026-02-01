/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Compute latent heat source term L * (s - s_old) / dt.
 */
class LatentHeatSource : public TensorOperator<>
{
public:
  static InputParameters validParams();
  LatentHeatSource(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _s;
  const std::vector<torch::Tensor> & _s_old;
  const Real _L;
};

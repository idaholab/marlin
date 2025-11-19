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
 * LBMComputeResidual object
 */
class LBMComputeResidual : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeResidual(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _speed;
  const std::vector<torch::Tensor> & _speed_old;
};

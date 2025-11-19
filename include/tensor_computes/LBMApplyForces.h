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
 * Apply forces forces
 */
class LBMApplyForces : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMApplyForces(const InputParameters & parameters);

  void computeSourceTerm();
  void computeBuffer() override;

protected:
  const torch::Tensor & _velocity;
  const torch::Tensor & _density;
  const torch::Tensor & _forces;

  torch::Tensor _source_term = torch::zeros(_shape_q, MooseTensor::floatTensorOptions());

  const Real _tau;
};

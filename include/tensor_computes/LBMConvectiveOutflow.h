/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMBoundaryCondition.h"

/**
 * Convective outflow boundary condition for lattice Boltzmann method.
 */
class LBMConvectiveOutflow : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMConvectiveOutflow(const InputParameters & parameters);

  void topBoundary() override;
  void bottomBoundary() override;
  void leftBoundary() override;
  void rightBoundary() override;
  void frontBoundary() override;
  void backBoundary() override;
  void computeBuffer() override;

protected:
  void applyConvectiveOutflow(int dim, int64_t b_idx, int64_t n_idx, int normal_component);
  torch::Tensor computeMeanNormalVelocity(const torch::Tensor & f_slice, int normal_component);

  const std::vector<torch::Tensor> & _f_old;
  torch::Tensor _f_old_owned;
  const bool _auto_velocity;
  const Real _uc_value;
};

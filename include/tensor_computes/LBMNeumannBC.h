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
 * LBMNeumannBC object that fixes the value at the walls
 */
class LBMNeumannBC : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMNeumannBC(const InputParameters & parameters);

  void topBoundary() override;
  void bottomBoundary() override;
  void leftBoundary() override;
  void rightBoundary() override;
  void frontBoundary() override;
  void backBoundary() override;
  void wallBoundary() override;
  void regionalBoundary() override;

  void computeBoundaryEquilibrium();

  void computeBuffer() override;

protected:
  const std::vector<torch::Tensor> & _f_old;
  torch::Tensor _f_old_owned;

  const torch::Tensor & _feq;
  const torch::Tensor & _rho;
  const torch::Tensor & _velocity;
  const Real & _gradient_value;
  int _region_id = 0;

  torch::Tensor _feq_boundary;
  // Cached active views for the current timestep
  torch::Tensor _feq_owned;
  torch::Tensor _feq_boundary_owned;
  // Precomputed direction indices for vectorized boundary assignments
  torch::Tensor _left_dirs, _right_dirs;
  torch::Tensor _bottom_dirs, _top_dirs;
  torch::Tensor _front_dirs, _back_dirs;
};

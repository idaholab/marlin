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

  /// boundary equilibrium at the given nodes only: velocity [..., dim], rho [...] -> [..., Q]
  torch::Tensor boundaryEquilibrium(const torch::Tensor & velocity, const torch::Tensor & rho);
  /// boundary equilibrium on the face index of dimension dim
  torch::Tensor faceEquilibrium(int64_t dim, int64_t index);
  /// boundary equilibrium at the indexed nodes
  torch::Tensor nodeEquilibrium(const std::vector<torch::indexing::TensorIndex> & index);

  void computeBuffer() override;

protected:
  const std::vector<torch::Tensor> & _f_old;
  torch::Tensor _f_old_owned;

  const torch::Tensor & _feq;
  const torch::Tensor & _rho;
  const torch::Tensor & _velocity;
  const Real & _gradient_value;
  int _region_id = 0;

  // Cached active views for the current timestep
  torch::Tensor _feq_owned;
  // Precomputed direction indices for vectorized boundary assignments
  torch::Tensor _left_dirs, _right_dirs;
  torch::Tensor _bottom_dirs, _top_dirs;
  torch::Tensor _front_dirs, _back_dirs;
};

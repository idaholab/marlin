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
 * LBMDirichletBC object that fixes the value at the walls
 */
class LBMDirichletBC : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMDirichletBC(const InputParameters & parameters);

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
  const Real & _boundary_value;
  int _region_id = 0;

  torch::Tensor _feq_owned;
};

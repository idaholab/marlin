/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMBoundaryCondition.h"

/**
 * LBMDirichletWallBC object that fixes the value at the complex walls
 */
class LBMDirichletWallBC : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMDirichletWallBC(const InputParameters & parameters);

  void topBoundary() override {};
  void bottomBoundary() override {};
  void leftBoundary() override {};
  void rightBoundary() override {};
  void frontBoundary() override {};
  void backBoundary() override {};
  void wallBoundary() override;

  void computeBoundaryNormals();

protected:
  const std::vector<torch::Tensor> & _f_old;

  torch::Tensor _binary_mesh;
  torch::Tensor _boundary_mask;
  torch::Tensor _boundary_normals;
  torch::Tensor _boundary_tangent_vectors;
  torch::Tensor _e_xyz;

  const torch::Tensor _velocity;
  bool _is_value_tensor;
  torch::Tensor _value_tensor;
  Real _value;
};

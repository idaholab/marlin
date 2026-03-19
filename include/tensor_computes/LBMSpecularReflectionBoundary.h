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
 * LBM combination of bounce-back and specular reflection boundary condition.
 *
 * Uses a precomputed lookup table to determine the
 * specular reflection direction for each lattice velocity at each boundary
 * node type. Boundary node types are identified by a binary encoding of the
 * local connectivity pattern (which neighbors are solid vs fluid).
 *
 * At each boundary node the incoming distribution is split:
 *   f[opposite]      += r     * f_old[incoming]    (bounce-back part)
 *   f[specular_dir]  += (1-r) * f_old[incoming]    (specular reflection part)
 */
class LBMSpecularReflectionBoundary : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMSpecularReflectionBoundary(const InputParameters & parameters);

  void topBoundary() override {}
  void bottomBoundary() override {}
  void leftBoundary() override {}
  void rightBoundary() override {}
  void frontBoundary() override {}
  void backBoundary() override {}
  void wallBoundary() override;

  void computeBuffer() override;

protected:
  /// Build the per-node boundary type classification and lookup indices.
  /// Called once on the first timestep.
  void buildSpecularIndices();

  const std::vector<torch::Tensor> & _f_old;

  /// Local Knudsen number field
  const torch::Tensor & _local_Knudsen_number;

  /// Per-node combination coefficient: 0 = pure specular, 1 = pure bounce-back
  torch::Tensor _r;

  /// Per boundary-direction entry: the specular reflection target direction
  torch::Tensor _specular_directions;

  /// Per boundary-direction entry: spatial indices (x, y, z) and incoming direction
  torch::Tensor _boundary_entries;

  bool _indices_built;
};

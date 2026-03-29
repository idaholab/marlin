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
 * LBMBounceBack object
 */
class LBMBounceBack : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMBounceBack(const InputParameters & parameters);

  void topBoundary() override;
  void bottomBoundary() override;
  void leftBoundary() override;
  void rightBoundary() override;
  void frontBoundary() override;
  void backBoundary() override;
  void wallBoundary() override;
  void computeBuffer() override;

protected:
  const std::vector<torch::Tensor> & _f_old;
  torch::Tensor _f_old_owned;

  // whether or not apply bounce back in the corners
  const bool _exclude_corners_x;
  const bool _exclude_corners_y;
  const bool _exclude_corners_z;

  // Replaced heavy index tensors with zero-cost slice bounds
  int64_t _x_start, _x_end;
  int64_t _y_start, _y_end;
  int64_t _z_start, _z_end;

  // Precomputed direction indices for O(1) vectorized assignments
  torch::Tensor _left_dirs, _left_opp_dirs;
  torch::Tensor _bottom_dirs, _bottom_opp_dirs;
  torch::Tensor _front_dirs, _front_opp_dirs;

  /// Pre-computed opposite direction indices for vectorized wall bounce-back
  torch::Tensor _op_indices;
};

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

  torch::Tensor _x_indices;
  torch::Tensor _y_indices;
  torch::Tensor _z_indices;
};

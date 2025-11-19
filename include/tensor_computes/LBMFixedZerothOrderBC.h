/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMBoundaryCondition.h"

/**
 * LBMFixedZerothOrderBC object
 */
class LBMFixedZerothOrderBC : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMFixedZerothOrderBC(const InputParameters & parameters);

  void topBoundary() override;
  void topBoundaryD2Q9();
  void bottomBoundary() override;
  void bottomBoundaryD2Q9();
  void leftBoundary() override;
  void leftBoundaryD2Q9();
  void rightBoundary() override;
  void rightBoundaryD2Q9();
  void frontBoundary() override;
  void backBoundary() override;
  void computeBuffer() override;

protected:
  const torch::Tensor & _f;
  const std::array<int64_t, 3> _grid_size;
  const Real _value;
};

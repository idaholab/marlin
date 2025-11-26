
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
class LBMMicroscopicZeroGradientBC : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMMicroscopicZeroGradientBC(const InputParameters & parameters);

  void topBoundary() override {};
  void bottomBoundary() override {};
  void leftBoundary() override;
  void rightBoundary() override;
  void frontBoundary() override {};
  void backBoundary() override {};
  void wallBoundary() override {};
  void computeBuffer() override;
};

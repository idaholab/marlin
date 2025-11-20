/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMBoundaryCondition.h"

#if 0

/**
 * LBM combination of bounce-back and specular reflection boundary condition
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

  void buildBoundaryIndices() override;
  void computeBuffer() override;

protected:
  const std::vector<torch::Tensor> & _f_old;
  torch::Tensor _specular_reflection_indices;
  const Real _r; // combination coefficient
};

#endif

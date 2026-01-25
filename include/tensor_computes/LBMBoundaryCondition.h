/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"
#include "MooseEnum.h"

/**
 * LBMBoundaryCondition object
 */
class LBMBoundaryCondition : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMBoundaryCondition(const InputParameters & parameters);

  /**
   * Conventions:
   * Left boundary is at x = 0
   * Right is at x = Nx - 1
   * Bottom is at y = 0
   * Top is at y = Ny - 1
   * Front is at z = 0
   * Back is at z = Nz - 1
   *
   * Wall boundary refers to any obstacles in the domain
   * It is entirely possible to put an obstacle at left, right, front, back, top or
   * bottom boundary and use wall boundary to replace any of them.
   */

  virtual void topBoundary() = 0;
  virtual void bottomBoundary() = 0;
  virtual void leftBoundary() = 0;
  virtual void rightBoundary() = 0;
  virtual void frontBoundary() = 0;
  virtual void backBoundary() = 0;
  virtual void wallBoundary() {};
  virtual void regionalBoundary() {};

  void maskBoundary();
  bool isBoundaryOwned(const int &);
  virtual void computeBuffer() override;

protected:
  enum class Boundary
  {
    top,
    bottom,
    left,
    right,
    front,
    back,
    wall,
    regional
  } _boundary;

  torch::Tensor _boundary_indices;
  torch::Tensor _boundary_types;
  // maps the rank to the boundary node indices
  uint8_t _boundary_rank = 0;

  torch::Tensor _binary_mesh;
  torch::Tensor _boundary_mask;
};

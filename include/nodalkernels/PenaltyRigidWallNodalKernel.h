/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "NodalKernel.h"

/**
 * One-sided rigid-wall penalty applied node-wise (the standard explicit
 * rigid-wall contact): any node whose current position along the variable's
 * direction drops below the wall plane receives a restoring force
 * F = -penalty * penetration. Unlike a sideset traction, this also holds
 * interior nodes, which is what prevents mesh fold-through when a crushed
 * element row collapses at the contact face.
 */
class PenaltyRigidWallNodalKernel : public NodalKernel
{
public:
  static InputParameters validParams();

  PenaltyRigidWallNodalKernel(const InputParameters & parameters);

protected:
  virtual Real computeQpResidual() override;
  virtual Real computeQpJacobian() override;

  /// Penalty stiffness per node (N/m)
  const Real _penalty;

  /// Wall plane position along the variable's coordinate direction
  const Real _wall_position;

  /// Coordinate component the variable displaces (0=x, 1=y, 2=z)
  const unsigned int _component;

  /// Current penetration (negative when penetrated), updated per node
  Real penetration() const;
};

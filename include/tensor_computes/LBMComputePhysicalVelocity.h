/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 * LBMComputePhysicalVelocity object
 */
class LBMComputePhysicalVelocity : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputePhysicalVelocity(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  /// Lattice Boltzmann velocity (dimensionless)
  const torch::Tensor & _u_lb;
  /// Relaxation parameter
  const Real & _tau;
  /// Physical kinematic viscosity (area/time)
  const Real & _nu;

  /// Size of cells, i.e. C_x
  const RealVectorValue & _grid_spacing;
};

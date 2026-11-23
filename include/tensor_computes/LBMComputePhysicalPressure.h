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
 * LBMComputePhysicalPressure object
 */
class LBMComputePhysicalPressure : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputePhysicalPressure(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  /// Lattice Boltzmann density (dimensionless)
  const torch::Tensor & _rho;
  /// LBM reference density
  const Real & _rho0;
  /// Physical reference density
  const Real & _rho0_phys;
  /// Relaxation parameter
  const Real & _tau;
  /// Physical kinematic viscosity (area/time)
  const Real & _nu;

  /// Size of cells, i.e. C_x
  const RealVectorValue & _grid_spacing;
};

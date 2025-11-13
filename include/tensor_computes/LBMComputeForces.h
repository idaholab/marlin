/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 * Compute forces
 */
class LBMComputeForces : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  void computeBuoyancy();
  void computeGravity();
  void computeSurfaceForces();
  /// more forces can be added ?

  LBMComputeForces(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const Real & _reference_density;
  const Real & _reference_temperature;

  const bool _enable_gravity;
  const bool _enable_buoyancy;
  const bool _enable_surface_forces;

  const Real _g; // gravitational acceleration

  const int64_t _gravity_direction; // direction of gravitational acceleration

  const torch::Tensor & _density_tensor;
  const torch::Tensor & _temperature;
};

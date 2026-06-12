/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Compute the full anisotropic gradient energy flux for cubic symmetry.
 *
 * This computes the flux vector F such that div(F) gives the anisotropic
 * gradient energy contribution: W^2 * div(a^2 * grad(phi) + corner_correction)
 *
 * The corner correction term (1/2)|grad(phi)|^2 * d(a^2)/d(grad(phi)) is
 * essential for dendritic morphology formation with strong anisotropy.
 *
 * Output is (a^2 - 1) * grad(phi) + corner_correction, which when combined
 * with the isotropic Laplacian from the linear solver gives the full
 * anisotropic gradient energy term.
 */
class CubicAnisotropyFlux : public TensorOperator<>
{
public:
  static InputParameters validParams();
  CubicAnisotropyFlux(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  /// Gradient vector in lab frame (value_dimensions = '3')
  const torch::Tensor & _gradient;
  /// Rotation matrix from lab to crystal frame (value_dimensions = '3 3')
  const torch::Tensor & _rotation;
  /// Anisotropy strength
  const Real _eps_a;
  /// Small regularization parameter to avoid division by zero
  const Real _eps_n;
};

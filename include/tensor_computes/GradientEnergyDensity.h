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
 * Gradient energy density 0.5 * kappa * sum_i |grad eta_i|^2 from a set of
 * gradient vector buffers (e.g. as produced by GradientVector).
 */
class GradientEnergyDensity : public TensorOperator<>
{
public:
  static InputParameters validParams();

  GradientEnergyDensity(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const std::vector<TensorInputBufferName> & _gradient_buffer_names;
  const Real _kappa;
};

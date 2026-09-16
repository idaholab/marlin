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
 * Compute the anisotropy torque from dmu_dn and a scalar weighting field g.
 */
class AnisotropyTorque : public TensorOperator<>
{
public:
  static InputParameters validParams();

  AnisotropyTorque(const InputParameters & parameters);

  /// Parallel FFT uses MPI communication which cannot be JIT traced
  virtual bool supportsJIT() const override { return !usesParallelFFT(); }

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _dmu_dn;
  const torch::Tensor & _g;

  /// imaginary unit i
  const torch::Tensor _imaginary_unit;
};

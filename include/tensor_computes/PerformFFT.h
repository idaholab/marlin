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
 * PerformFFT object
 */
template <bool forward>
class PerformFFTTempl : public TensorOperator<>
{
public:
  static InputParameters validParams();

  PerformFFTTempl(const InputParameters & parameters);

  void computeBuffer() override;

  /// Parallel FFT uses MPI communication which cannot be JIT traced
  virtual bool supportsJIT() const override { return !this->usesParallelFFT(); }

protected:
  const torch::Tensor & _input;

  using TensorOperator::_tensor_problem;
  using TensorOperator::_u;
};

typedef PerformFFTTempl<true> ForwardFFT;
typedef PerformFFTTempl<false> InverseFFT;

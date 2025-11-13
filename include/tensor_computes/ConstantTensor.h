/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Constant Tensor
 */
template <bool reciprocal>
class ConstantTensorTempl : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ConstantTensorTempl(const InputParameters & parameters);

  virtual void computeBuffer() override;

  /// mesh dimension
  const unsigned int & _dim;

  /// constants
  const Real & _real;
  const Real & _imaginary;

  using TensorOperator<>::_domain;
  using TensorOperator<>::_u;
};

typedef ConstantTensorTempl<false> ConstantTensor;
typedef ConstantTensorTempl<true> ConstantReciprocalTensor;

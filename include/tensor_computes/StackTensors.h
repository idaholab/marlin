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
 * Stack scalar tensors along a new dimension to build a vector tensor.
 */
class StackTensors : public TensorOperator<>
{
public:
  static InputParameters validParams();
  StackTensors(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const std::vector<TensorInputBufferName> _buffer_names;
  const int _stack_dim;
};
